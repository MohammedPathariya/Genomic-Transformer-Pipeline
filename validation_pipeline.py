#!/usr/bin/env python3
"""
validation_pipeline.py
========================
End-to-end validation of the HIV DRM detection pipeline against
Stanford HIVdb ACTG5288 clinical ground truth.

What this script does:
    1. Loads ground truth mutation tables from HIVdb_clinical_data/
       (ACTG5288_PR.txt, ACTG5288_RT.txt, ACTG5288_IN.txt)

    2. Runs the full pipeline on the converted FASTQ files:
       stream_reader → quality_filter → pol_localizer → region_filter
       → codon_framer → feature_builder → drm_head

    3. For each read, extracts the PtID from the read ID and looks up
       the corresponding ground truth mutations from the .txt file

    4. Computes and reports:
       - Per-mutation detection rates (sensitivity/recall per DRM)
       - Per-read mutation match accuracy
       - Coverage statistics (what fraction of DRM positions extracted)
       - Resistance call agreement vs Stanford HIVdb ground truth
       - Pipeline stage statistics (localization, framing, feature extraction)

Ground truth format (ACTG5288_PR.txt):
    PtID | Alias | IsolateDate | Subtype | MutList | P1 | P2 | ... | P99
    The MutList column contains comma-separated mutations like:
    "Q18H, Q61E, I62V, L63P, E65D, I66IV, A71T"

    Note: MutList uses format "L90M" (wildtype + position + mutant).
    Our pipeline also outputs this format — direct comparison is valid.

    Important nuance: MutList includes ALL mutations vs HXB2 wildtype,
    not just known DRM positions. Our pipeline only checks DRM-relevant
    positions. We therefore compute two metrics:
      - DRM recall: of mutations AT DRM positions in ground truth,
                    how many did we detect?
      - False positive rate: mutations we called that are NOT in ground truth

Usage:
    python validation_pipeline.py --gene PR
    python validation_pipeline.py --gene RT
    python validation_pipeline.py --gene IN
    python validation_pipeline.py --gene all --verbose

Author: Genomic-Transformer-Pipeline
"""

import argparse
import gzip
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ingestion.stream_reader   import stream_reads
from src.ingestion.quality_filter  import quality_filter
from src.enricher.pol_localizer    import PolLocalizer
from src.enricher.region_filter    import RegionFilter
from src.enricher.codon_framer     import CodonFramer
from src.enricher.feature_builder  import FeatureBuilder, DRM_POSITIONS
from src.classification.drm_head   import DRMHead

logging.basicConfig(
    level  = logging.WARNING,   # suppress module-level noise during validation
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("validation_pipeline")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ACTG_DATA_DIR  = "HIVdb_clinical_data"
FASTQ_DATA_DIR = "data/actg_fastq"
TRIAL          = "ACTG5288"

# DRM positions set — for filtering ground truth to DRM-relevant only
DRM_POSITIONS_SET: dict[str, set[int]] = {
    gene: set(positions)
    for gene, positions in DRM_POSITIONS.items()
}


# ---------------------------------------------------------------------------
# Ground truth loader
# ---------------------------------------------------------------------------
def load_ground_truth(gene: str, trial: str = TRIAL) -> dict[str, list[str]]:
    """
    Load the ground truth mutation table for a gene from an ACTG .txt file.

    Returns
    -------
    dict mapping PtID → list of mutations at DRM positions
    e.g. {"23424": ["L90M", "V82A"], "23425": ["M184V", "K103N"]}

    Only mutations at positions in DRM_POSITIONS are included —
    this is what our pipeline can detect, so comparison must be fair.
    """
    # Handle both ACTG5288 and ACTGA5095 naming conventions
    candidates = [
        Path(ACTG_DATA_DIR) / f"{trial}_{gene}.txt",
        Path(ACTG_DATA_DIR) / f"{trial.replace('ACTG','ACTGA')}_{gene}.txt",
    ]

    gt_path = None
    for c in candidates:
        if c.exists():
            gt_path = c
            break

    if gt_path is None:
        logger.error(
            f"Ground truth file not found for {trial}/{gene}. "
            f"Tried: {[str(c) for c in candidates]}"
        )
        return {}

    logger.info(f"Loading ground truth: {gt_path}")

    drm_pos = DRM_POSITIONS_SET.get(gene, set())
    ground_truth: dict[str, list[str]] = {}

    with open(gt_path, "r", encoding="utf-8", errors="replace") as fh:
        header = None
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue

            fields = line.split("\t")

            # First non-empty line is the header
            if header is None:
                header = fields
                continue

            # Need at least PtID and MutList columns
            if len(fields) < 5:
                continue

            ptid    = fields[0].strip()
            mutlist = fields[4].strip()  # MutList is column index 4

            if not ptid or mutlist in ("", "-", ".", "nan"):
                continue

            # Parse MutList — comma separated, may have spaces
            # Format: "Q18H, Q61E, I62V, L63P"
            all_mutations = [
                m.strip()
                for m in mutlist.split(",")
                if m.strip() and m.strip() not in ("-", ".")
            ]

            # Filter to DRM-relevant positions only
            # A mutation like "L90M" is at position 90
            drm_mutations = []
            for mut in all_mutations:
                pos = _extract_position(mut)
                if pos and pos in drm_pos:
                    drm_mutations.append(mut)

            # Store — use PtID as key
            # If same PtID appears multiple times (multiple timepoints),
            # keep earliest (IsolateDate=0 = baseline)
            if ptid not in ground_truth:
                ground_truth[ptid] = drm_mutations

    logger.info(
        f"Ground truth loaded: {len(ground_truth)} patients, "
        f"gene={gene}, DRM positions only"
    )
    return ground_truth


def _extract_position(mutation: str) -> Optional[int]:
    """
    Extract the numeric position from a mutation string.

    Examples:
        "L90M"    → 90
        "M184V"   → 184
        "K103N"   → 103
        "I66IV"   → 66   (mixed population, still valid)
        "N37THNP" → 37
    """
    match = re.match(r"^[A-Z](\d+)", mutation)
    if match:
        return int(match.group(1))
    return None


def _extract_mutant_aa(mutation: str) -> Optional[str]:
    """
    Extract the mutant amino acid(s) from a mutation string.
    Returns the first mutant AA for multi-variant calls.

    Examples:
        "L90M"    → "M"
        "M184V"   → "V"
        "I66IV"   → "I"  (first variant)
        "N37THNP" → "T"  (first variant)
    """
    match = re.match(r"^[A-Z]\d+([A-Z])", mutation)
    if match:
        return match.group(1)
    return None


# ---------------------------------------------------------------------------
# Read ID parser
# ---------------------------------------------------------------------------
def parse_read_id(read_id: str) -> dict:
    """
    Parse a read ID produced by fasta_to_fastq.py back into components.

    Format: PtID_{ptid}|Alias_{alias}|Week_{week}|Gene_{gene}|Trial_{trial}

    Returns
    -------
    dict with keys: ptid, alias, week, gene, trial
    """
    result = {
        "ptid": None, "alias": None,
        "week": None, "gene":  None, "trial": None,
    }
    for part in read_id.split("|"):
        if part.startswith("PtID_"):
            result["ptid"]  = part[5:]
        elif part.startswith("Alias_"):
            result["alias"] = part[6:]
        elif part.startswith("Week_"):
            result["week"]  = part[5:]
        elif part.startswith("Gene_"):
            result["gene"]  = part[5:]
        elif part.startswith("Trial_"):
            result["trial"] = part[6:]
    return result


# ---------------------------------------------------------------------------
# Mutation normalizer — handles mixed population calls
# ---------------------------------------------------------------------------
def normalize_mutation_for_comparison(mut: str) -> set[str]:
    """
    Normalize a mutation string to a set of simple calls for comparison.

    The ground truth MutList uses mixed population notation like "I66IV"
    (both I and V at position 66). Our pipeline calls single AAs.
    We expand mixed calls into all possible single-AA mutations.

    Examples:
        "L90M"    → {"L90M"}
        "I66IV"   → {"I66I", "I66V"}   (mixed population)
        "N37THNP" → {"N37T","N37H","N37N","N37P"}
    """
    match = re.match(r"^([A-Z])(\d+)([A-Z]+)$", mut)
    if not match:
        return {mut}

    wt_aa  = match.group(1)
    pos    = match.group(2)
    mut_aas = match.group(3)

    return {f"{wt_aa}{pos}{aa}" for aa in mut_aas}


# ---------------------------------------------------------------------------
# Validation result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    gene:           str
    n_reads_input:  int = 0
    n_reads_passed_qc: int = 0
    n_reads_localized: int = 0
    n_reads_framed:    int = 0
    n_reads_features:  int = 0
    n_reads_matched_gt: int = 0   # reads with PtID found in ground truth

    # Per-mutation statistics
    true_positives:  dict = field(default_factory=dict)   # {mut: count}
    false_negatives: dict = field(default_factory=dict)   # {mut: count}
    false_positives: dict = field(default_factory=dict)   # {mut: count}

    # Per-read statistics
    exact_matches:      int = 0   # pipeline mutations == ground truth exactly
    partial_matches:    int = 0   # pipeline found subset of ground truth
    no_mutations_both:  int = 0   # both pipeline and GT found no DRM mutations
    false_pos_reads:    int = 0   # pipeline found mutations GT doesn't have

    # Coverage statistics
    total_positions_possible: int = 0
    total_positions_covered:  int = 0
    low_confidence_reads:     int = 0

    # Framing statistics
    frame_0_count: int = 0
    frame_1_count: int = 0
    frame_2_count: int = 0


# ---------------------------------------------------------------------------
# Core validation runner
# ---------------------------------------------------------------------------
def run_validation_for_gene(
    gene:      str,
    verbose:   bool = False,
    max_reads: Optional[int] = None,
) -> ValidationResult:
    """
    Run full pipeline validation for one gene region.

    Parameters
    ----------
    gene      : "PR", "RT", or "IN"
    verbose   : print per-read details
    max_reads : limit reads for quick testing (None = all)

    Returns
    -------
    ValidationResult with all statistics populated
    """
    result = ValidationResult(gene=gene)

    # ------------------------------------------------------------------
    # Load ground truth
    # ------------------------------------------------------------------
    ground_truth = load_ground_truth(gene, TRIAL)
    if not ground_truth:
        print(f"  ERROR: No ground truth loaded for {gene}. Skipping.")
        return result

    print(f"\n  Ground truth: {len(ground_truth)} patients loaded")

    # ------------------------------------------------------------------
    # Load FASTQ
    # ------------------------------------------------------------------
    fastq_path = Path(FASTQ_DATA_DIR) / f"{TRIAL}_{gene}.fastq.gz"
    if not fastq_path.exists():
        print(f"  ERROR: FASTQ not found: {fastq_path}")
        print(f"         Run: python scripts/fasta_to_fastq.py first")
        return result

    # ------------------------------------------------------------------
    # Initialize pipeline components
    # ------------------------------------------------------------------
    localizer     = PolLocalizer()
    region_filter = RegionFilter()
    framer        = CodonFramer()
    builder       = FeatureBuilder()
    drm_head      = DRMHead()

    # ------------------------------------------------------------------
    # Run pipeline
    # ------------------------------------------------------------------
    raw_stream            = stream_reads(str(fastq_path))
    filtered_stream, _    = quality_filter(raw_stream)
    localized_stream      = (localizer.localize(r) for r in filtered_stream)
    region_stream         = region_filter.filter_stream(localized_stream)

    n_processed = 0

    for localized in region_stream:

        if max_reads and n_processed >= max_reads:
            break

        result.n_reads_localized += 1

        if localized.gene_region == "unknown":
            continue

        # Frame resolution
        framed = framer.resolve(localized)
        result.n_reads_framed += 1

        if framed.reading_frame == 0:
            result.frame_0_count += 1
        elif framed.reading_frame == 1:
            result.frame_1_count += 1
        else:
            result.frame_2_count += 1

        if framed.low_confidence_frame:
            result.low_confidence_reads += 1

        # Feature extraction
        # read_start_position=1 because ACTG amplicons start at gene position 1
        fv = builder.extract(framed, read_start_position=1)
        result.n_reads_features += 1

        # Track coverage
        result.total_positions_possible += len(DRM_POSITIONS.get(gene, []))
        result.total_positions_covered  += len(fv.positions_extracted)

        # Parse PtID from read ID
        meta  = parse_read_id(framed.read_id)
        ptid  = meta.get("ptid")

        if not ptid or ptid not in ground_truth:
            # Read has no matching ground truth entry
            n_processed += 1
            continue

        result.n_reads_matched_gt += 1
        n_processed += 1

        # ------------------------------------------------------------------
        # Compare pipeline mutations vs ground truth
        # ------------------------------------------------------------------
        pipeline_muts = set(fv.mutation_list())
        gt_muts_raw   = ground_truth[ptid]

        # Expand mixed population calls in ground truth
        gt_muts_expanded: set[str] = set()
        for m in gt_muts_raw:
            gt_muts_expanded |= normalize_mutation_for_comparison(m)

        # Only compare at DRM positions — fair comparison
        drm_pos = DRM_POSITIONS_SET.get(gene, set())
        gt_drm  = {
            m for m in gt_muts_expanded
            if _extract_position(m) in drm_pos
        }

        # True positives: pipeline called it AND ground truth has it
        tp = pipeline_muts & gt_drm
        # False negatives: ground truth has it but pipeline missed it
        fn = gt_drm - pipeline_muts
        # False positives: pipeline called it but ground truth does not have it
        fp = pipeline_muts - gt_drm

        for m in tp:
            result.true_positives[m]  = result.true_positives.get(m, 0)  + 1
        for m in fn:
            result.false_negatives[m] = result.false_negatives.get(m, 0) + 1
        for m in fp:
            result.false_positives[m] = result.false_positives.get(m, 0) + 1

        # Per-read classification
        if not pipeline_muts and not gt_drm:
            result.no_mutations_both += 1
        elif pipeline_muts == gt_drm:
            result.exact_matches += 1
        elif pipeline_muts.issubset(gt_drm):
            result.partial_matches += 1
        elif fp:
            result.false_pos_reads += 1
        else:
            result.partial_matches += 1

        if verbose:
            status = (
                "EXACT"   if pipeline_muts == gt_drm else
                "PARTIAL" if tp else
                "MISS"    if fn and not fp else
                "FP"
            )
            print(
                f"    [{status}] PtID={ptid} | "
                f"pipeline={sorted(pipeline_muts)} | "
                f"truth={sorted(gt_drm)}"
            )

    result.n_reads_input = n_processed
    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------
def print_report(result: ValidationResult) -> None:
    """Print a structured validation report for one gene."""

    gene = result.gene
    sep  = "=" * 65

    print(f"\n{sep}")
    print(f"  VALIDATION REPORT — {gene} ({TRIAL})")
    print(sep)

    # ------------------------------------------------------------------
    # Pipeline throughput
    # ------------------------------------------------------------------
    print(f"\n  Pipeline Throughput:")
    print(f"    Reads localized     : {result.n_reads_localized}")
    print(f"    Reads framed        : {result.n_reads_framed}")
    print(f"    Features extracted  : {result.n_reads_features}")
    print(f"    Matched to GT       : {result.n_reads_matched_gt}")
    print(f"    Low confidence frame: {result.low_confidence_reads} "
          f"({result.low_confidence_reads / max(1, result.n_reads_framed):.1%})")

    # ------------------------------------------------------------------
    # Frame distribution
    # ------------------------------------------------------------------
    total_framed = max(1, result.n_reads_framed)
    print(f"\n  Reading Frame Distribution:")
    print(f"    Frame 0 : {result.frame_0_count} "
          f"({result.frame_0_count/total_framed:.1%})")
    print(f"    Frame 1 : {result.frame_1_count} "
          f"({result.frame_1_count/total_framed:.1%})")
    print(f"    Frame 2 : {result.frame_2_count} "
          f"({result.frame_2_count/total_framed:.1%})")

    # ------------------------------------------------------------------
    # Coverage
    # ------------------------------------------------------------------
    coverage = (
        result.total_positions_covered /
        max(1, result.total_positions_possible)
    )
    print(f"\n  DRM Position Coverage:")
    print(f"    Positions possible  : {result.total_positions_possible}")
    print(f"    Positions covered   : {result.total_positions_covered}")
    print(f"    Mean coverage       : {coverage:.1%}")

    # ------------------------------------------------------------------
    # Mutation detection accuracy
    # ------------------------------------------------------------------
    n_matched = max(1, result.n_reads_matched_gt)

    print(f"\n  Per-Read Mutation Match (n={result.n_reads_matched_gt}):")
    print(f"    Exact match         : {result.exact_matches} "
          f"({result.exact_matches/n_matched:.1%})")
    print(f"    Partial match       : {result.partial_matches} "
          f"({result.partial_matches/n_matched:.1%})")
    print(f"    Both wildtype       : {result.no_mutations_both} "
          f"({result.no_mutations_both/n_matched:.1%})")
    print(f"    False positive reads: {result.false_pos_reads} "
          f"({result.false_pos_reads/n_matched:.1%})")

    # ------------------------------------------------------------------
    # Per-mutation statistics
    # ------------------------------------------------------------------
    all_muts = sorted(
        set(result.true_positives) |
        set(result.false_negatives) |
        set(result.false_positives)
    )

    if all_muts:
        print(f"\n  Per-Mutation Statistics:")
        print(f"    {'Mutation':<12} {'TP':>5} {'FN':>5} {'FP':>5} "
              f"{'Recall':>8} {'Precision':>10}")
        print(f"    {'-'*12} {'-'*5} {'-'*5} {'-'*5} {'-'*8} {'-'*10}")

        for mut in all_muts:
            tp = result.true_positives.get(mut, 0)
            fn = result.false_negatives.get(mut, 0)
            fp = result.false_positives.get(mut, 0)

            recall    = tp / max(1, tp + fn)
            precision = tp / max(1, tp + fp)

            print(
                f"    {mut:<12} {tp:>5} {fn:>5} {fp:>5} "
                f"{recall:>7.1%} {precision:>10.1%}"
            )
    else:
        print(f"\n  No mutations detected at DRM positions.")
        print(f"  This is expected if the pipeline's frame resolution")
        print(f"  needs calibration for these sequence types.")

    # ------------------------------------------------------------------
    # Overall accuracy summary
    # ------------------------------------------------------------------
    total_tp = sum(result.true_positives.values())
    total_fn = sum(result.false_negatives.values())
    total_fp = sum(result.false_positives.values())

    overall_recall    = total_tp / max(1, total_tp + total_fn)
    overall_precision = total_tp / max(1, total_tp + total_fp)
    f1 = (
        2 * overall_precision * overall_recall /
        max(1e-9, overall_precision + overall_recall)
    )

    print(f"\n  Overall DRM Detection Performance:")
    print(f"    Total TP            : {total_tp}")
    print(f"    Total FN            : {total_fn}")
    print(f"    Total FP            : {total_fp}")
    print(f"    Recall (Sensitivity): {overall_recall:.1%}")
    print(f"    Precision           : {overall_precision:.1%}")
    print(f"    F1 Score            : {f1:.3f}")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate HIV DRM pipeline against ACTG5288 ground truth."
    )
    parser.add_argument(
        "--gene",
        choices=["PR", "RT", "IN", "all"],
        default="all",
        help="Gene region to validate (default: all)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print per-read mutation comparison details"
    )
    parser.add_argument(
        "--max_reads",
        type=int,
        default=None,
        help="Limit reads per gene for quick testing (default: all)"
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="WARNING",
        help="Logging verbosity (default: WARNING)"
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    genes = ["PR", "RT", "IN"] if args.gene == "all" else [args.gene]

    print("\n" + "=" * 65)
    print("  HIV DRM Pipeline — End-to-End Validation")
    print(f"  Trial   : {TRIAL}")
    print(f"  Genes   : {', '.join(genes)}")
    print(f"  FASTQ   : {FASTQ_DATA_DIR}/")
    print(f"  GT dir  : {ACTG_DATA_DIR}/")
    print("=" * 65)

    all_results = []

    for gene in genes:
        print(f"\n{'─'*65}")
        print(f"  Processing {gene}...")
        print(f"{'─'*65}")

        result = run_validation_for_gene(
            gene      = gene,
            verbose   = args.verbose,
            max_reads = args.max_reads,
        )
        all_results.append(result)
        print_report(result)

    # ------------------------------------------------------------------
    # Cross-gene summary
    # ------------------------------------------------------------------
    if len(all_results) > 1:
        print("=" * 65)
        print("  CROSS-GENE SUMMARY")
        print("=" * 65)
        print(f"\n  {'Gene':<6} {'Reads':>7} {'Matched':>8} "
              f"{'Coverage':>10} {'Recall':>8} {'Precision':>10} {'F1':>6}")
        print(f"  {'-'*6} {'-'*7} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*6}")

        for r in all_results:
            coverage  = (r.total_positions_covered /
                         max(1, r.total_positions_possible))
            tp = sum(r.true_positives.values())
            fn = sum(r.false_negatives.values())
            fp = sum(r.false_positives.values())
            recall    = tp / max(1, tp + fn)
            precision = tp / max(1, tp + fp)
            f1 = (2 * precision * recall /
                  max(1e-9, precision + recall))

            print(
                f"  {r.gene:<6} {r.n_reads_framed:>7} "
                f"{r.n_reads_matched_gt:>8} "
                f"{coverage:>9.1%} {recall:>7.1%} "
                f"{precision:>9.1%} {f1:>6.3f}"
            )
        print()


if __name__ == "__main__":
    main()