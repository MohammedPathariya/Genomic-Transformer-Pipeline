"""
src/enricher/feature_builder.py
================================
Codon-level feature extraction for HIV-1 Drug Resistance Mutation (DRM) detection.

Responsibility:
    Take a FramedRead (output of codon_framer) and produce a structured
    FeatureVector — a numerical representation of the amino acid sequence
    at clinically relevant positions, suitable for consumption by drm_head.

Why this is needed:
    The codon_framer produces a raw amino acid string like "PQITLWQRPLVT...".
    No classifier or rule engine can consume a raw string. This module:
      1. Extracts the amino acid at each resistance-relevant position
      2. Compares it against the HXB2 wildtype amino acid at that position
      3. Flags positions where the amino acid differs (candidate DRMs)
      4. Encodes the result as a binary one-hot feature vector

Design decision — Option A (codon one-hot encoding):
    For each clinically relevant position, encode the amino acid present
    as a 20-dimensional one-hot vector (one dimension per standard AA).
    Only resistance-relevant positions are encoded — not all 99/240/288
    positions — to keep the feature vector compact and interpretable.

    PR:  30 resistance positions × 20 AA = 600 features
    RT:  40 resistance positions × 20 AA = 800 features
    IN:  20 resistance positions × 20 AA = 400 features

    In practice, feature vectors are stored as dicts keyed by
    "{gene}_{position}_{amino_acid}" for human readability and sparse
    storage efficiency. The drm_head consumes the mutation_flags dict
    directly — it does not need the full one-hot encoding.

Data contract:
    Input:  FramedRead (from codon_framer.py)
    Output: FeatureVector dataclass

Position in pipeline:
    codon_framer → feature_builder → drm_head

Author: Genomic-Transformer-Pipeline
"""

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.enricher.codon_framer import FramedRead

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Standard amino acid alphabet (20 standard AAs, alphabetical order)
# ---------------------------------------------------------------------------
AA_ALPHABET: list[str] = [
    "A", "C", "D", "E", "F", "G", "H", "I", "K", "L",
    "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y",
]
AA_INDEX: dict[str, int] = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# ---------------------------------------------------------------------------
# HXB2 wildtype amino acid sequences
# Position index is 1-based (matching Stanford HIVdb convention)
# These are sliced from the full HXB2 protein sequences in codon_framer.py
# ---------------------------------------------------------------------------
HXB2_WILDTYPE: dict[str, str] = {
    # Protease — 99 amino acids
    "PR": (
        "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKA"
        "IGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    ),
    # Reverse Transcriptase — 240 amino acids (p51 domain, DRM-relevant region)
    "RT": (
        "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTK"
        "WLKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDEDFRKYTAFTIPSINNETPG"
        "IRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLR"
        "WGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCK"
    ),
    # Integrase — 288 amino acids
    "IN": (
        "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
        "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
        "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
        "QKQITKIQNFRVYYRDSRNPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
        "RQDED"
    ),
}

# ---------------------------------------------------------------------------
# Clinically relevant DRM positions per gene (Stanford HIVdb derived)
# These are 1-based positions where resistance mutations are documented.
# Only these positions are encoded in the feature vector.
#
# Sources:
#   PR positions: HIVdb PI resistance mutation list
#   RT positions: HIVdb NRTI + NNRTI resistance mutation list
#   IN positions: HIVdb INSTI resistance mutation list
# ---------------------------------------------------------------------------
DRM_POSITIONS: dict[str, list[int]] = {
    "PR": [
        10, 11, 13, 20, 23, 24, 30, 32, 33, 35,
        36, 43, 46, 47, 48, 50, 53, 54, 58, 60,
        63, 71, 73, 74, 76, 77, 82, 83, 84, 85,
        88, 89, 90, 93,
    ],
    "RT": [
        # NRTI positions
        41,  44,  62,  65,  67,  68,  69,  70,  74,  75,
        77,  98, 100, 101, 103, 106, 108, 115, 116, 118,
        # NNRTI positions
        138, 151, 179, 181, 184, 188, 190, 210, 215, 219,
        221, 225, 227, 230, 234, 236, 238, 318, 348, 369,
    ],
    "IN": [
        # Core INSTI resistance positions — Stanford HIVdb validated
        # Removed positions with precision <15% on resistant dataset
        # (natural HXB2 polymorphisms, not drug-selected resistance):
        # 51, 66, 92, 95, 114, 118, 121, 128, 145, 146, 147, 149, 153, 232, 263
        74, 97, 138, 140, 143, 148, 151, 155, 157, 163, 230,
    ],
}


# ---------------------------------------------------------------------------
# FeatureVector dataclass — output of feature_builder
# ---------------------------------------------------------------------------
@dataclass
class FeatureVector:
    """
    Structured feature representation of a FramedRead at DRM positions.

    Fields
    ------
    read_id         : str   — original read identifier
    gene_region     : str   — "PR", "RT", or "IN"
    reading_frame   : int   — 0, 1, or 2 (from codon_framer)
    frame_confidence: float — frame confidence score

    mutation_flags  : dict  — {position: amino_acid} for ALL extracted positions
                              e.g. {90: "M", 46: "I", 184: "V"}
                              Wildtype positions ARE included for completeness.

    drm_candidates  : dict  — {position: amino_acid} for positions that DIFFER
                              from HXB2 wildtype. These are DRM candidates.
                              e.g. {90: "M"} if wildtype is "L" at position 90
                              → L90M mutation detected

    wildtype_calls  : dict  — {position: amino_acid} for positions matching
                              HXB2 wildtype exactly.

    positions_extracted : list[int]  — positions successfully extracted from
                                       the amino acid sequence (may be fewer
                                       than DRM_POSITIONS if read is short)

    positions_missing   : list[int]  — positions that could not be extracted
                                       (read too short or ambiguous AA)

    one_hot_vector  : list[int] — flat binary vector encoding AA at each
                                   DRM position. Length = n_positions × 20.
                                   Primarily used for ML model input.

    coverage_fraction : float — fraction of DRM positions covered by this read

    low_confidence    : bool  — True if frame_confidence was below threshold
    has_iupac         : bool  — True if sequence contained IUPAC ambiguity codes
                                (relevant for Sanger sequences from ACTG data)
    """
    read_id:              str
    gene_region:          str
    reading_frame:        int
    frame_confidence:     float

    mutation_flags:       dict   = field(default_factory=dict)
    drm_candidates:       dict   = field(default_factory=dict)
    wildtype_calls:       dict   = field(default_factory=dict)

    positions_extracted:  list   = field(default_factory=list)
    positions_missing:    list   = field(default_factory=list)

    one_hot_vector:       list   = field(default_factory=list)

    coverage_fraction:    float  = 0.0
    low_confidence:       bool   = False
    has_iupac:            bool   = False

    def to_dict(self) -> dict:
        return {
            "read_id":            self.read_id,
            "gene_region":        self.gene_region,
            "reading_frame":      self.reading_frame,
            "frame_confidence":   round(self.frame_confidence, 4),
            "mutation_flags":     self.mutation_flags,
            "drm_candidates":     self.drm_candidates,
            "wildtype_calls":     self.wildtype_calls,
            "positions_extracted": self.positions_extracted,
            "positions_missing":  self.positions_missing,
            "coverage_fraction":  round(self.coverage_fraction, 4),
            "low_confidence":     self.low_confidence,
            "has_iupac":          self.has_iupac,
            "n_drm_candidates":   len(self.drm_candidates),
            "n_positions_covered": len(self.positions_extracted),
        }

    def mutation_list(self) -> list[str]:
        """
        Return DRM candidates in Stanford HIVdb format: ["L90M", "M184V", ...].
        Format: {wildtype_aa}{position}{mutant_aa}
        """
        wt = HXB2_WILDTYPE.get(self.gene_region, "")
        result = []
        for pos, mut_aa in sorted(self.drm_candidates.items()):
            if pos <= len(wt):
                wt_aa = wt[pos - 1]  # 1-based → 0-based
                result.append(f"{wt_aa}{pos}{mut_aa}")
        return result

    def __repr__(self) -> str:
        muts = self.mutation_list()
        return (
            f"FeatureVector("
            f"id='{self.read_id}', "
            f"region='{self.gene_region}', "
            f"frame={self.reading_frame}, "
            f"coverage={self.coverage_fraction:.2f}, "
            f"drm_candidates={muts}"
            f")"
        )


# ---------------------------------------------------------------------------
# IUPAC ambiguity code resolver
# Maps ambiguous IUPAC nucleotide codes to the most common base.
# For amino acid sequences from Sanger data, ambiguity shows up as 'X'
# after translation — we flag but do not discard these reads.
# ---------------------------------------------------------------------------
IUPAC_AA_AMBIGUOUS = {"X", "B", "Z", "J", "U", "O"}


def _is_iupac_ambiguous(aa_sequence: str) -> bool:
    """Return True if the AA sequence contains IUPAC ambiguity codes."""
    return any(aa in IUPAC_AA_AMBIGUOUS for aa in aa_sequence)


# ---------------------------------------------------------------------------
# Core extraction logic
# ---------------------------------------------------------------------------

def _extract_aa_at_position(
    amino_acid_sequence: str,
    position: int,
    gene_region: str,
) -> Optional[str]:
    """
    Extract the amino acid at a given 1-based position from the AA sequence.

    The AA sequence from codon_framer starts at the beginning of the read,
    which for targeted amplicon reads (ACTG FASTA) starts at position 1 of
    the gene. For full-pol reads (Nanopore), the offset requires the
    genomic_start coordinate — handled by the caller if available.

    Parameters
    ----------
    amino_acid_sequence : str
        The full translated amino acid string from FramedRead.
    position : int
        1-based position in the HXB2 reference (e.g. 90 for L90M).
    gene_region : str
        "PR", "RT", or "IN" — used for bounds checking.

    Returns
    -------
    str or None
        The amino acid character at that position, or None if out of range.
    """
    idx = position - 1  # convert 1-based → 0-based
    if idx < 0 or idx >= len(amino_acid_sequence):
        return None
    aa = amino_acid_sequence[idx]
    # Stop codons in the sequence are not valid amino acids
    if aa == "*":
        return None
    return aa


def _build_one_hot(
    aa: Optional[str],
    positions_in_order: list[int],
    current_position: int,
) -> list[int]:
    """
    Build a 20-dimensional one-hot vector for a single amino acid.
    Unknown or missing AAs produce an all-zero vector.
    """
    vector = [0] * 20
    if aa and aa in AA_INDEX:
        vector[AA_INDEX[aa]] = 1
    return vector


# ---------------------------------------------------------------------------
# FeatureBuilder class
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Extracts codon-level DRM features from a FramedRead.

    Usage
    -----
    builder = FeatureBuilder()
    feature_vector = builder.extract(framed_read)

    # Access mutation candidates
    print(feature_vector.mutation_list())   # ["L90M", "M184V"]
    print(feature_vector.drm_candidates)    # {90: "M", 184: "V"}
    print(feature_vector.coverage_fraction) # 0.85

    Notes on genomic offset
    -----------------------
    For ACTG targeted amplicon sequences (used in validation):
        The FASTA sequences start at position 1 of the gene.
        No offset is needed. The AA at index 0 = position 1.

    For full-pol Nanopore reads:
        The read may start anywhere in the gene. The genomic_start
        coordinate from pol_localizer is needed to compute the offset.
        This is handled via the `read_start_position` parameter.
        If not provided, the builder assumes the read starts at position 1
        (correct for targeted amplicons, incorrect for full-pol reads).
    """

    def __init__(self) -> None:
        logger.info("FeatureBuilder initialized.")
        logger.info(
            f"DRM positions loaded — "
            f"PR: {len(DRM_POSITIONS['PR'])}, "
            f"RT: {len(DRM_POSITIONS['RT'])}, "
            f"IN: {len(DRM_POSITIONS['IN'])}"
        )

    def extract(
        self,
        framed_read: FramedRead,
        read_start_position: int = 1,
    ) -> FeatureVector:
        """
        Extract DRM features from a single FramedRead.

        Parameters
        ----------
        framed_read : FramedRead
            Output from CodonFramer.resolve().
        read_start_position : int
            1-based genomic position where this read begins within the gene.
            Default = 1 (correct for ACTG targeted amplicons).
            For full-pol Nanopore reads, pass the genomic_start from
            pol_localizer to correctly offset position lookups.

        Returns
        -------
        FeatureVector
        """
        gene   = framed_read.gene_region
        aa_seq = framed_read.amino_acid_sequence
        wt_seq = HXB2_WILDTYPE.get(gene, "")

        if not aa_seq:
            logger.warning(
                f"Empty AA sequence for read '{framed_read.read_id}' "
                f"(region={gene}). Returning empty FeatureVector."
            )
            return FeatureVector(
                read_id          = framed_read.read_id,
                gene_region      = gene,
                reading_frame    = framed_read.reading_frame,
                frame_confidence = framed_read.frame_confidence,
                low_confidence   = framed_read.low_confidence_frame,
            )

        positions      = DRM_POSITIONS.get(gene, [])
        mutation_flags = {}
        drm_candidates = {}
        wildtype_calls = {}
        extracted      = []
        missing        = []
        one_hot_parts  = []

        for pos in positions:
            # Adjust for read start offset
            # If read starts at position 5 in the gene, then
            # gene position 10 is at AA index (10 - 5) = 5
            aa_idx = pos - read_start_position

            # Extract AA at this adjusted index
            if aa_idx < 0 or aa_idx >= len(aa_seq):
                missing.append(pos)
                one_hot_parts.extend([0] * 20)
                continue

            aa = aa_seq[aa_idx]

            # Stop codons are not valid — treat as missing
            if aa == "*":
                missing.append(pos)
                one_hot_parts.extend([0] * 20)
                continue

            # Record extraction
            mutation_flags[pos] = aa
            extracted.append(pos)

            # Compare against HXB2 wildtype
            wt_aa = wt_seq[pos - 1] if pos <= len(wt_seq) else None

            if wt_aa and aa != wt_aa:
                drm_candidates[pos] = aa
                logger.debug(
                    f"  DRM candidate: {gene} pos {pos} — "
                    f"wildtype={wt_aa}, observed={aa} → {wt_aa}{pos}{aa}"
                )
            else:
                wildtype_calls[pos] = aa

            # Build one-hot encoding for this position
            vec = [0] * 20
            if aa in AA_INDEX:
                vec[AA_INDEX[aa]] = 1
            one_hot_parts.extend(vec)

        # Coverage fraction
        n_total   = len(positions)
        n_covered = len(extracted)
        coverage  = n_covered / n_total if n_total > 0 else 0.0

        has_iupac = _is_iupac_ambiguous(aa_seq)

        fv = FeatureVector(
            read_id              = framed_read.read_id,
            gene_region          = gene,
            reading_frame        = framed_read.reading_frame,
            frame_confidence     = framed_read.frame_confidence,
            mutation_flags       = mutation_flags,
            drm_candidates       = drm_candidates,
            wildtype_calls       = wildtype_calls,
            positions_extracted  = extracted,
            positions_missing    = missing,
            one_hot_vector       = one_hot_parts,
            coverage_fraction    = round(coverage, 4),
            low_confidence       = framed_read.low_confidence_frame,
            has_iupac            = has_iupac,
        )

        logger.debug(
            f"FeatureVector built: '{framed_read.read_id}' | "
            f"region={gene} | covered={n_covered}/{n_total} | "
            f"DRM candidates={fv.mutation_list()}"
        )

        return fv

    def extract_batch(
        self,
        framed_reads: list[FramedRead],
        read_start_position: int = 1,
    ) -> tuple[list[FeatureVector], dict]:
        """
        Extract features from a batch of FramedReads.

        Returns
        -------
        tuple[list[FeatureVector], dict]
            feature_vectors: list of FeatureVector objects
            stats: summary statistics for the batch
        """
        feature_vectors = []
        stats = {
            "total":              0,
            "empty_aa":           0,
            "low_confidence":     0,
            "has_iupac":          0,
            "drm_candidates_found": 0,
            "mean_coverage":      0.0,
            "by_region":          {"PR": 0, "RT": 0, "IN": 0},
        }

        coverages = []

        for read in framed_reads:
            fv = self.extract(read, read_start_position)
            feature_vectors.append(fv)
            stats["total"] += 1

            if not fv.positions_extracted:
                stats["empty_aa"] += 1
                continue

            if fv.low_confidence:
                stats["low_confidence"] += 1
            if fv.has_iupac:
                stats["has_iupac"] += 1
            if fv.drm_candidates:
                stats["drm_candidates_found"] += 1

            coverages.append(fv.coverage_fraction)
            region = fv.gene_region
            if region in stats["by_region"]:
                stats["by_region"][region] += 1

        if coverages:
            stats["mean_coverage"] = round(
                sum(coverages) / len(coverages), 4
            )

        return feature_vectors, stats


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def build_features(
    framed_read: FramedRead,
    read_start_position: int = 1,
) -> FeatureVector:
    """
    Convenience wrapper — build features for a single FramedRead.
    Instantiates a FeatureBuilder internally.
    For batch processing, instantiate FeatureBuilder directly.
    """
    builder = FeatureBuilder()
    return builder.extract(framed_read, read_start_position)


# ---------------------------------------------------------------------------
# Quick validation
# Usage: python -m src.enricher.feature_builder
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    import os
    from src.ingestion.stream_reader  import stream_reads
    from src.ingestion.quality_filter import quality_filter
    from src.enricher.pol_localizer   import PolLocalizer
    from src.enricher.region_filter   import RegionFilter
    from src.enricher.codon_framer    import CodonFramer

    print("=" * 65)
    print("FeatureBuilder — Validation Run")
    print("=" * 65)

    localizer     = PolLocalizer()
    region_filter = RegionFilter()
    framer        = CodonFramer()
    builder       = FeatureBuilder()

    test_files = [
        ("data/test/synthetic/targeted/PR_targeted.fastq.gz", "PR"),
        ("data/test/synthetic/targeted/RT_targeted.fastq.gz", "RT"),
        ("data/test/synthetic/targeted/IN_targeted.fastq.gz", "IN"),
    ]

    for test_file, expected_region in test_files:
        if not os.path.exists(test_file):
            print(f"\nSKIP (not found): {test_file}")
            continue

        print(f"\n{'='*65}")
        print(f"Processing: {Path(test_file).name}")
        print(f"{'='*65}")

        raw_stream           = stream_reads(test_file)
        filtered_stream, _   = quality_filter(raw_stream)
        localized_stream     = (localizer.localize(r) for r in filtered_stream)
        region_stream        = region_filter.filter_stream(localized_stream)

        framed_reads = []
        for loc in region_stream:
            if loc.gene_region != "unknown":
                framed_reads.append(framer.resolve(loc))

        print(f"Framed reads: {len(framed_reads)}")

        # Extract features for first 100 reads
        sample = framed_reads[:100]
        feature_vectors, stats = builder.extract_batch(sample)

        print(f"\nFeature Extraction Stats (n={stats['total']}):")
        print(f"  Mean coverage       : {stats['mean_coverage']:.1%}")
        print(f"  Low confidence      : {stats['low_confidence']}")
        print(f"  DRM candidates found: {stats['drm_candidates_found']}")
        print(f"  IUPAC ambiguous     : {stats['has_iupac']}")

        # Collect all DRM candidates across reads
        all_mutations: dict[str, int] = {}
        for fv in feature_vectors:
            for mut in fv.mutation_list():
                all_mutations[mut] = all_mutations.get(mut, 0) + 1

        if all_mutations:
            print(f"\n  DRM candidates detected (top 10):")
            for mut, count in sorted(
                all_mutations.items(), key=lambda x: -x[1]
            )[:10]:
                print(f"    {mut}: {count} reads")
        else:
            print(f"\n  No DRM candidates detected in sample.")

        # Show first 3 feature vectors in detail
        print(f"\n  Sample FeatureVectors (first 3):")
        for i, fv in enumerate(feature_vectors[:3]):
            print(f"\n    FV #{i+1}: {fv}")
            print(f"      Coverage    : {fv.coverage_fraction:.1%} "
                  f"({len(fv.positions_extracted)}/{len(DRM_POSITIONS[fv.gene_region])} positions)")
            print(f"      DRM calls   : {fv.mutation_list()}")
            print(f"      WT calls    : {len(fv.wildtype_calls)} positions")
            print(f"      Missing     : {fv.positions_missing[:5]}"
                  f"{'...' if len(fv.positions_missing) > 5 else ''}")
            print(f"      1-hot len   : {len(fv.one_hot_vector)} "
                  f"(expected {len(DRM_POSITIONS[fv.gene_region]) * 20})")