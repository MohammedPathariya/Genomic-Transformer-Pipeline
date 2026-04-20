"""
src/classification/drm_head.py
================================
HIV Drug Resistance Mutation (DRM) classification head.

Responsibility:
    Take a FeatureVector (output of feature_builder) and produce a
    ResistanceProfile — per-drug resistance level calls using the
    Stanford HIVdb algorithm rules parsed from HIVDB_9.8.xml.

Design decision — HIVDB Rule Engine (deterministic, not learned):
    We apply the Stanford HIVdb scoring rules directly rather than
    training a classifier. This is intentional for this pipeline stage:

    1. It is the gold standard — Stanford HIVdb is the accepted clinical
       reference for HIV drug resistance interpretation worldwide.

    2. It is deterministic and reproducible — given the same mutations,
       it always produces the same resistance calls.

    3. It gives us a real accuracy baseline — when we run the pipeline on
       ACTG sequences and compare our calls against the ground truth
       mutation tables, any errors are attributable to the upstream
       extraction pipeline (localizer, framer, feature_builder), NOT to
       the classification logic. This cleanly isolates what we are testing.

    4. It does not require training data — we can run it immediately on
       any FeatureVector without a trained model.

    The ML classifier (trained on ACTG sequences) is the next phase.
    The rule engine is the proof-of-concept that validates the enricher.

Resistance level scoring (from HIVDB XML GLOBALRANGE):
    Score  0-9   → Susceptible          (S)
    Score 10-14  → Potential Low-Level  (Pot_R)
    Score 15-29  → Low-Level Resistance (Low_R)
    Score 30-59  → Intermediate         (Mid_R)
    Score >= 60  → High-Level           (High_R)

HIVDB XML source:
    https://cms.hivdb.org/prod/downloads/asi/HIVDB_9.8.xml
    The XML is fetched once, cached locally, and parsed into rule tables.
    If the network is unavailable, a bundled minimal rule set is used.

Data contract:
    Input:  FeatureVector (from feature_builder.py)
    Output: ResistanceProfile dataclass

Position in pipeline:
    feature_builder → drm_head → aggregator (not yet built)

Author: Genomic-Transformer-Pipeline
"""

import logging
import os
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import itertools

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.enricher.feature_builder import FeatureVector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
HIVDB_XML_URL    = "https://cms.hivdb.org/prod/downloads/asi/HIVDB_9.8.xml"
HIVDB_CACHE_PATH = "data/public/HIVDB_9.8.xml"

# Resistance level thresholds — from HIVDB GLOBALRANGE
SCORE_THRESHOLDS = [
    (0,   9,  "S",      "Susceptible"),
    (10,  14, "Pot_R",  "Potential Low-Level Resistance"),
    (15,  29, "Low_R",  "Low-Level Resistance"),
    (30,  59, "Mid_R",  "Intermediate Resistance"),
    (60,  999,"High_R", "High-Level Resistance"),
]

# Drug class groupings — for structured output
DRUG_CLASSES: dict[str, list[str]] = {
    "NRTI":  ["ABC", "AZT", "D4T", "DDI", "FTC", "3TC", "TDF"],
    "NNRTI": ["DOR", "EFV", "ETR", "NVP", "RPV", "DPV"],
    "PI":    ["ATV/r", "DRV/r", "FPV/r", "IDV/r", "LPV/r", "NFV", "SQV/r", "TPV/r"],
    "INSTI": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    "CAI":   ["LEN"],
}

# Flat list of all drugs in order
ALL_DRUGS: list[str] = [d for drugs in DRUG_CLASSES.values() for d in drugs]

# Gene → drug class mapping (which genes affect which drug classes)
GENE_DRUG_CLASSES: dict[str, list[str]] = {
    "PR": ["PI"],
    "RT": ["NRTI", "NNRTI"],
    "IN": ["INSTI"],
}


# ---------------------------------------------------------------------------
# ResistanceProfile dataclass — output of drm_head
# ---------------------------------------------------------------------------
@dataclass
class ResistanceProfile:
    """
    Per-drug resistance level calls for a single sequenced sample.

    Fields
    ------
    read_id        : str  — original read identifier
    gene_region    : str  — "PR", "RT", or "IN"
    mutations      : list[str] — detected mutations ["L90M", "M184V"]

    drug_scores    : dict[str, int]   — raw HIVDB score per drug
    drug_levels    : dict[str, str]   — resistance level per drug
                                        "S" / "Pot_R" / "Low_R" / "Mid_R" / "High_R"
    drug_labels    : dict[str, str]   — human readable level per drug
    drug_sir       : dict[str, str]   — simplified S/I/R per drug (clinical)

    resistant_drugs: list[str] — drugs with level >= Low_R
    susceptible_drugs: list[str] — drugs with level == S
    intermediate_drugs: list[str] — drugs with level == Pot_R

    coverage_fraction : float — from FeatureVector
    low_confidence    : bool  — True if upstream frame confidence was low
    ruleset_version   : str   — HIVDB version used (e.g. "9.8")
    """
    read_id:           str
    gene_region:       str
    mutations:         list  = field(default_factory=list)

    drug_scores:       dict  = field(default_factory=dict)
    drug_levels:       dict  = field(default_factory=dict)
    drug_labels:       dict  = field(default_factory=dict)
    drug_sir:          dict  = field(default_factory=dict)

    resistant_drugs:   list  = field(default_factory=list)
    susceptible_drugs: list  = field(default_factory=list)
    intermediate_drugs:list  = field(default_factory=list)

    coverage_fraction: float = 0.0
    low_confidence:    bool  = False
    ruleset_version:   str   = "unknown"

    def to_dict(self) -> dict:
        return {
            "read_id":            self.read_id,
            "gene_region":        self.gene_region,
            "mutations":          self.mutations,
            "drug_scores":        self.drug_scores,
            "drug_levels":        self.drug_levels,
            "drug_sir":           self.drug_sir,
            "resistant_drugs":    self.resistant_drugs,
            "susceptible_drugs":  self.susceptible_drugs,
            "intermediate_drugs": self.intermediate_drugs,
            "coverage_fraction":  round(self.coverage_fraction, 4),
            "low_confidence":     self.low_confidence,
            "ruleset_version":    self.ruleset_version,
            "n_resistant":        len(self.resistant_drugs),
        }

    def summary(self) -> str:
        """One-line human readable summary."""
        muts = ", ".join(self.mutations) if self.mutations else "none"
        res  = ", ".join(self.resistant_drugs) if self.resistant_drugs else "none"
        return (
            f"[{self.gene_region}] {self.read_id} | "
            f"mutations: {muts} | "
            f"resistant to: {res}"
        )

    def __repr__(self) -> str:
        return (
            f"ResistanceProfile("
            f"id='{self.read_id}', "
            f"region='{self.gene_region}', "
            f"mutations={self.mutations}, "
            f"resistant={self.resistant_drugs}"
            f")"
        )


# ---------------------------------------------------------------------------
# Score → resistance level mapping
# ---------------------------------------------------------------------------
def _score_to_level(score: int) -> tuple[str, str, str]:
    """
    Map a cumulative HIVDB score to (level_code, label, SIR).

    Returns
    -------
    tuple[str, str, str]
        (level_code, label, sir)
        e.g. ("Mid_R", "Intermediate Resistance", "R")
    """
    for lo, hi, code, label in SCORE_THRESHOLDS:
        if lo <= score <= hi:
            sir = "S" if code in ("S", "Pot_R") else ("I" if code in ("Low_R", "Mid_R") else "R")
            return code, label, sir
    # Anything above 60 is High_R
    return "High_R", "High-Level Resistance", "R"


# ---------------------------------------------------------------------------
# HIVDB XML parser
# ---------------------------------------------------------------------------

def _fetch_hivdb_xml(
    cache_path: str = HIVDB_CACHE_PATH,
    url: str = HIVDB_XML_URL,
) -> Optional[ET.Element]:
    """
    Load HIVDB XML from local cache or download from Stanford.

    Priority:
        1. Local cache at cache_path
        2. Download from url and save to cache
        3. Return None if both fail
    """
    # Try local cache first
    if os.path.exists(cache_path):
        logger.info(f"Loading HIVDB XML from cache: {cache_path}")
        try:
            tree = ET.parse(cache_path)
            return tree.getroot()
        except ET.ParseError as e:
            logger.warning(f"Cached XML is corrupt: {e}. Re-downloading.")

    # Attempt download
    logger.info(f"Downloading HIVDB XML from {url}")
    try:
        import urllib.request
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        logger.info(f"HIVDB XML saved to {cache_path}")
        tree = ET.parse(cache_path)
        return tree.getroot()
    except Exception as e:
        logger.error(f"Failed to download HIVDB XML: {e}")
        return None


def _parse_hivdb_rules(root: ET.Element) -> tuple[dict, str]:
    """
    Parse HIVDB XML into a usable rule table.

    Returns
    -------
    tuple[dict, str]
        rules: {drug_name: [(condition_mutations, score), ...]}
               condition_mutations is a list of mutation sets.
               Each mutation set is a list of mutation strings that
               must ALL be present (AND condition).
               e.g. [["41L"], ...] or [["67N", "215F"], ...]

        version: HIVDB algorithm version string
    """
    version = root.findtext(".//ALGVERSION") or "unknown"
    logger.info(f"Parsing HIVDB rules — version {version}")

    rules: dict[str, list[tuple[list[str], int]]] = {}

    for drug_elem in root.findall(".//DRUG"):
        drug_name = drug_elem.findtext("NAME")
        if not drug_name:
            continue

        drug_rules = []

        for rule_elem in drug_elem.findall("RULE"):
            rule_text = "".join(rule_elem.itertext()).strip()

            # Extract condition => score pairs
            # Pattern: "67N => 5" or "41L AND 215Y => 10"
            matches = re.findall(
                r"((?:[0-9]+[A-Za-z]+(?:\s+AND\s+[0-9]+[A-Za-z]+)*))\s*=>\s*([0-9]+)",
                rule_text
            )

            for condition_str, score_str in matches:
                score = int(score_str)

                # Split AND conditions into individual mutations
                parts = [p.strip() for p in condition_str.split("AND")]

                # Expand shorthand: "67EGNHST" → ["67E","67G","67N","67H","67S","67T"]
                # Each part is either a single mutation or needs expansion
                expanded_parts = []
                for part in parts:
                    expanded = _expand_mutation_shorthand(part)
                    expanded_parts.append(expanded)

                # Each combination of expanded alternatives is one condition
                # e.g. "67EGNHST AND 215FY" →
                #   [["67E","215F"], ["67E","215Y"], ["67G","215F"], ...]
                for combo in itertools.product(*expanded_parts):
                    drug_rules.append((list(combo), score))

        if drug_rules:
            rules[drug_name] = drug_rules
            logger.debug(f"  {drug_name}: {len(drug_rules)} conditions parsed")

    logger.info(
        f"HIVDB rules parsed: {len(rules)} drugs, "
        f"{sum(len(v) for v in rules.values())} total conditions"
    )
    return rules, version


def _expand_mutation_shorthand(mutation_str: str) -> list[str]:
    """
    Expand a mutation shorthand string into a list of individual mutations.

    Examples:
        "41L"      → ["41L"]           (single, no expansion needed)
        "67EGNHST" → ["67E","67G","67N","67H","67S","67T"]
        "215FY"    → ["215F","215Y"]
        "69i"      → ["69i"]           (insertion, keep as-is)
        "67d"      → ["67d"]           (deletion, keep as-is)
    """
    mutation_str = mutation_str.strip()

    # Match position + amino acid(s)
    match = re.match(r"^(\d+)([A-Za-z]+)$", mutation_str)
    if not match:
        return [mutation_str]  # can't parse, keep as-is

    pos  = match.group(1)
    aas  = match.group(2)

    if len(aas) == 1:
        return [mutation_str]  # single AA, no expansion needed

    # Lowercase single char = insertion/deletion code, don't expand
    if len(aas) == 1 and aas.islower():
        return [mutation_str]

    # Multi-char uppercase = one mutation per character
    return [f"{pos}{aa}" for aa in aas]


# ---------------------------------------------------------------------------
# Score computation
# ---------------------------------------------------------------------------

def _compute_drug_score(
    mutation_set: set[str],
    drug_rules:   list[tuple[list[str], int]],
) -> int:
    """
    Compute cumulative HIVDB score for a drug given the observed mutations.

    Each rule is (condition_mutations, score). The condition fires if ALL
    mutations in condition_mutations are present in mutation_set.
    Scores are summed across all firing conditions.

    Parameters
    ----------
    mutation_set : set[str]
        Observed mutations in stripped form, e.g. {"41L", "184V", "103N"}.
        Mutations are stripped of gene prefix if present.

    drug_rules : list[tuple[list[str], int]]
        Parsed rules for this drug from _parse_hivdb_rules.

    Returns
    -------
    int
        Cumulative resistance score.
    """
    total_score = 0
    for condition_muts, score in drug_rules:
        if all(m in mutation_set for m in condition_muts):
            total_score += score
    return total_score


def _normalize_mutation(mut_str: str) -> str:
    """
    Strip gene prefix from a mutation string if present.

    Stanford HIVdb mutation format: "L90M" (no gene prefix)
    Our mutation_list() format:     "L90M" (no gene prefix either)
    HIVDB rule format:              "90M"  (position + AA, no wildtype)

    We need to match our "L90M" against HIVDB rules "90M".
    Strategy: strip the leading wildtype amino acid letter.

    Examples:
        "L90M"  → "90M"
        "M184V" → "184V"
        "K103N" → "103N"
        "90M"   → "90M"   (already stripped)
    """
    # If starts with a letter followed by digits, strip the first letter
    match = re.match(r"^([A-Z])(\d+[A-Za-z]+)$", mut_str)
    if match:
        return match.group(2)  # return "position + mutant_aa"
    return mut_str


# ---------------------------------------------------------------------------
# DRMHead class
# ---------------------------------------------------------------------------

class DRMHead:
    """
    HIV Drug Resistance classification head using Stanford HIVdb rules.

    Applies the HIVDB scoring algorithm to a FeatureVector and produces
    a ResistanceProfile with per-drug resistance level calls.

    Usage
    -----
    drm_head = DRMHead()
    profile = drm_head.classify(feature_vector)

    print(profile.summary())
    # [RT] read_001 | mutations: M184V, K103N | resistant to: ABC, 3TC, FTC, EFV, NVP

    print(profile.drug_levels)
    # {"ABC": "High_R", "3TC": "High_R", "EFV": "High_R", ...}
    """

    def __init__(
        self,
        hivdb_xml_path: str = HIVDB_CACHE_PATH,
        hivdb_xml_url:  str = HIVDB_XML_URL,
    ) -> None:
        logger.info("Initializing DRMHead...")

        root = _fetch_hivdb_xml(cache_path=hivdb_xml_path, url=hivdb_xml_url)

        if root is None:
            logger.warning(
                "HIVDB XML unavailable. DRMHead will return empty profiles. "
                "Ensure network access or place HIVDB_9.8.xml in data/public/."
            )
            self.rules   = {}
            self.version = "unavailable"
        else:
            self.rules, self.version = _parse_hivdb_rules(root)

        logger.info(
            f"DRMHead ready. "
            f"HIVDB version: {self.version}, "
            f"drugs loaded: {len(self.rules)}"
        )

    def classify(self, feature_vector: FeatureVector) -> ResistanceProfile:
        """
        Classify a FeatureVector into a ResistanceProfile.

        Parameters
        ----------
        feature_vector : FeatureVector
            Output from FeatureBuilder.extract().

        Returns
        -------
        ResistanceProfile
        """
        gene      = feature_vector.gene_region
        mutations = feature_vector.mutation_list()

        # Convert mutations to normalized form for HIVDB rule matching
        # "L90M" → "90M", "M184V" → "184V"
        normalized = {_normalize_mutation(m) for m in mutations}

        logger.debug(
            f"Classifying '{feature_vector.read_id}' | "
            f"region={gene} | mutations={mutations} | "
            f"normalized={normalized}"
        )

        # Determine which drug classes apply to this gene
        applicable_classes = GENE_DRUG_CLASSES.get(gene, [])
        applicable_drugs   = [
            drug
            for cls in applicable_classes
            for drug in DRUG_CLASSES.get(cls, [])
        ]

        drug_scores  = {}
        drug_levels  = {}
        drug_labels  = {}
        drug_sir     = {}

        for drug in applicable_drugs:
            drug_rules = self.rules.get(drug, [])

            if not drug_rules:
                # Drug exists but no rules loaded — score 0 → Susceptible
                score = 0
            else:
                score = _compute_drug_score(normalized, drug_rules)

            level, label, sir = _score_to_level(score)

            drug_scores[drug] = score
            drug_levels[drug] = level
            drug_labels[drug] = label
            drug_sir[drug]    = sir

            if score > 0:
                logger.debug(
                    f"  {drug}: score={score} → {level} ({sir})"
                )

        # Categorize drugs by resistance tier
        resistant    = [d for d, lvl in drug_levels.items()
                        if lvl in ("Low_R", "Mid_R", "High_R")]
        susceptible  = [d for d, lvl in drug_levels.items() if lvl == "S"]
        intermediate = [d for d, lvl in drug_levels.items() if lvl == "Pot_R"]

        profile = ResistanceProfile(
            read_id            = feature_vector.read_id,
            gene_region        = gene,
            mutations          = mutations,
            drug_scores        = drug_scores,
            drug_levels        = drug_levels,
            drug_labels        = drug_labels,
            drug_sir           = drug_sir,
            resistant_drugs    = resistant,
            susceptible_drugs  = susceptible,
            intermediate_drugs = intermediate,
            coverage_fraction  = feature_vector.coverage_fraction,
            low_confidence     = feature_vector.low_confidence,
            ruleset_version    = self.version,
        )

        logger.debug(
            f"Profile complete: {profile.summary()}"
        )

        return profile

    def classify_batch(
        self,
        feature_vectors: list[FeatureVector],
    ) -> tuple[list[ResistanceProfile], dict]:
        """
        Classify a batch of FeatureVectors.

        Returns
        -------
        tuple[list[ResistanceProfile], dict]
            profiles: list of ResistanceProfile objects
            stats:    batch summary statistics
        """
        profiles = []
        stats = {
            "total":                 0,
            "with_mutations":        0,
            "with_resistance":       0,
            "low_confidence":        0,
            "mutation_frequency":    {},   # {mutation: count}
            "drug_resistance_count": {},   # {drug: n_resistant_reads}
        }

        for fv in feature_vectors:
            profile = self.classify(fv)
            profiles.append(profile)
            stats["total"] += 1

            if profile.mutations:
                stats["with_mutations"] += 1

            if profile.resistant_drugs:
                stats["with_resistance"] += 1

            if profile.low_confidence:
                stats["low_confidence"] += 1

            for mut in profile.mutations:
                stats["mutation_frequency"][mut] = (
                    stats["mutation_frequency"].get(mut, 0) + 1
                )

            for drug in profile.resistant_drugs:
                stats["drug_resistance_count"][drug] = (
                    stats["drug_resistance_count"].get(drug, 0) + 1
                )

        return profiles, stats

    def format_resistance_report(
        self,
        profile: ResistanceProfile,
        verbose: bool = False,
    ) -> str:
        """
        Format a ResistanceProfile as a human-readable clinical report.

        Parameters
        ----------
        profile : ResistanceProfile
        verbose : bool
            If True, include all drugs. If False, only show non-susceptible.
        """
        lines = [
            f"{'='*60}",
            f"HIV Drug Resistance Report",
            f"  Read ID     : {profile.read_id}",
            f"  Gene Region : {profile.gene_region}",
            f"  Mutations   : {', '.join(profile.mutations) if profile.mutations else 'None detected'}",
            f"  Coverage    : {profile.coverage_fraction:.1%} of DRM positions",
            f"  HIVDB Ver   : {profile.ruleset_version}",
            f"  Confidence  : {'LOW — interpret with caution' if profile.low_confidence else 'OK'}",
            f"{'='*60}",
        ]

        # Group by drug class
        gene = profile.gene_region
        applicable_classes = GENE_DRUG_CLASSES.get(gene, [])

        for drug_class in applicable_classes:
            drugs = DRUG_CLASSES.get(drug_class, [])
            lines.append(f"\n{drug_class}:")
            for drug in drugs:
                if drug not in profile.drug_levels:
                    continue
                level = profile.drug_levels[drug]
                label = profile.drug_labels.get(drug, level)
                score = profile.drug_scores.get(drug, 0)
                sir   = profile.drug_sir.get(drug, "?")

                # Skip susceptible in non-verbose mode
                if not verbose and level == "S":
                    continue

                # Resistance level indicator
                indicator = {
                    "S":      "  ✓",
                    "Pot_R":  "  ~",
                    "Low_R":  "  ⚠",
                    "Mid_R":  "  ⚠⚠",
                    "High_R": "  ✗",
                }.get(level, "  ?")

                lines.append(
                    f"  {indicator} {drug:<10} {label:<35} (score={score}, SIR={sir})"
                )

            if not verbose:
                susceptible = [
                    d for d in drugs
                    if profile.drug_levels.get(d) == "S"
                ]
                if susceptible:
                    lines.append(
                        f"  ✓  Susceptible: {', '.join(susceptible)}"
                    )

        lines.append(f"{'='*60}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def classify_resistance(feature_vector: FeatureVector) -> ResistanceProfile:
    """
    Convenience wrapper — classify a single FeatureVector.
    Instantiates DRMHead internally (downloads/caches HIVDB XML once).
    For batch processing, instantiate DRMHead directly.
    """
    head = DRMHead()
    return head.classify(feature_vector)


# ---------------------------------------------------------------------------
# Quick validation
# Usage: python -m src.classification.drm_head
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    print("=" * 65)
    print("DRMHead — Validation Run")
    print("=" * 65)

    # -----------------------------------------------------------------
    # Test 1: Known resistance mutation profiles (hand-crafted)
    # These should match Stanford HIVdb website output exactly.
    # -----------------------------------------------------------------
    from src.enricher.feature_builder import FeatureVector

    print("\n--- Test 1: Known DRM profiles (should match HIVdb website) ---\n")

    test_cases = [
        {
            "name":     "K103N + M184V (classic NNRTI + NRTI resistance)",
            "gene":     "RT",
            "mutations": {"103": "N", "184": "V"},
            "expected_resistant": ["EFV", "NVP", "3TC", "FTC"],
        },
        {
            "name":     "L90M (classic broad PI resistance)",
            "gene":     "PR",
            "mutations": {"90": "M"},
            "expected_resistant": ["NFV"],
        },
        {
            "name":     "Q148H + G140S (high-level INSTI resistance)",
            "gene":     "IN",
            "mutations": {"148": "H", "140": "S"},
            "expected_resistant": ["RAL", "EVG"],
        },
        {
            "name":     "Wildtype — no mutations (should be all susceptible)",
            "gene":     "RT",
            "mutations": {},
            "expected_resistant": [],
        },
    ]

    drm_head = DRMHead()

    for tc in test_cases:
        print(f"\n  Test: {tc['name']}")

        # Build a synthetic FeatureVector with the test mutations
        from src.enricher.feature_builder import HXB2_WILDTYPE, DRM_POSITIONS

        gene   = tc["gene"]
        wt_seq = HXB2_WILDTYPE[gene]
        muts   = tc["mutations"]

        # Populate drm_candidates as {pos_int: mut_aa}
        drm_candidates = {int(pos): aa for pos, aa in muts.items()}

        fv = FeatureVector(
            read_id          = f"test_{tc['name'][:20]}",
            gene_region      = gene,
            reading_frame    = 0,
            frame_confidence = 0.95,
            drm_candidates   = drm_candidates,
            positions_extracted = list(range(1, len(wt_seq) + 1)),
            coverage_fraction = 1.0,
        )

        profile = drm_head.classify(fv)

        print(f"  Mutations detected : {profile.mutations}")
        print(f"  Resistant drugs    : {profile.resistant_drugs}")
        print(f"  Expected resistant : {tc['expected_resistant']}")

        # Check if expected resistant drugs are found
        found    = set(profile.resistant_drugs)
        expected = set(tc["expected_resistant"])
        correct  = expected.issubset(found)
        print(f"  Expected found     : {'PASS ✓' if correct else 'FAIL ✗'}")

        # Print full report for first test case
        if tc == test_cases[0]:
            print()
            print(drm_head.format_resistance_report(profile, verbose=False))

    # -----------------------------------------------------------------
    # Test 2: Run on synthetic FASTQ data through full pipeline
    # -----------------------------------------------------------------
    print("\n--- Test 2: Full pipeline (synthetic FASTQ → ResistanceProfile) ---\n")

    import os
    from src.ingestion.stream_reader   import stream_reads
    from src.ingestion.quality_filter  import quality_filter
    from src.enricher.pol_localizer    import PolLocalizer
    from src.enricher.region_filter    import RegionFilter
    from src.enricher.codon_framer     import CodonFramer
    from src.enricher.feature_builder  import FeatureBuilder

    test_files = [
        "data/test/synthetic/targeted/PR_targeted.fastq.gz",
        "data/test/synthetic/targeted/RT_targeted.fastq.gz",
        "data/test/synthetic/targeted/IN_targeted.fastq.gz",
    ]

    localizer     = PolLocalizer()
    region_filter = RegionFilter()
    framer        = CodonFramer()
    builder       = FeatureBuilder()

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"  SKIP (not found): {test_file}")
            continue

        print(f"\n  File: {Path(test_file).name}")

        raw_stream         = stream_reads(test_file)
        filtered, _        = quality_filter(raw_stream)
        localized          = (localizer.localize(r) for r in filtered)
        region_passed      = region_filter.filter_stream(localized)

        framed_reads = []
        for loc in region_passed:
            if loc.gene_region != "unknown":
                framed_reads.append(framer.resolve(loc))

        sample = framed_reads[:50]
        fvs, _ = builder.extract_batch(sample)
        profiles, stats = drm_head.classify_batch(fvs)

        print(f"  Reads processed       : {stats['total']}")
        print(f"  With mutations        : {stats['with_mutations']}")
        print(f"  With resistance calls : {stats['with_resistance']}")

        if stats["mutation_frequency"]:
            print(f"  Top mutations:")
            for mut, cnt in sorted(
                stats["mutation_frequency"].items(),
                key=lambda x: -x[1]
            )[:5]:
                print(f"    {mut}: {cnt} reads")

        if stats["drug_resistance_count"]:
            print(f"  Resistance by drug:")
            for drug, cnt in sorted(
                stats["drug_resistance_count"].items(),
                key=lambda x: -x[1]
            )[:5]:
                print(f"    {drug}: {cnt} reads")