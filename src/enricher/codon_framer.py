"""
src/enricher/codon_framer.py
=============================
Reading frame resolver for HIV-1 pol gene sequences.

Responsibility:
    Take a LocalizedRead (a read assigned to PR, RT, or IN by pol_localizer)
    and determine the correct reading frame — 0, 1, or 2. Translate the
    read into an amino acid sequence using that frame. Return a FramedRead
    containing the original read data plus the frame assignment, amino acid
    sequence, and frame confidence scores.

Why this is needed:
    DNA encodes proteins in triplets called codons. There are three possible
    reading frames depending on which base you start from. The wrong frame
    produces nonsense amino acids and false stop codons. Without correct
    frame assignment, codon-level mutation detection (K103N, M184V, etc.)
    is impossible.

How frame selection works (v1.1):
    1. Translate only the first 100 codons (300bp) of each frame for scoring.
    2. Score each frame by stop codon density on the 100-codon window.
    3. If the best frame's stop density advantage >= 0.03, pick immediately.
    4. Only when stop density margins < 0.03, run alignment as tiebreaker.
    5. After frame selection, translate the full read in the winning frame.

Design note — why no coordinate-aware framing here:
    A coordinate-aware approach (deriving frame from genomic_start) was
    explored but rejected because it overfits to the synthetic test data
    (where reads are generated from HXB2 with known coordinates) and does
    not generalise to real clinical reads from HIV-1 subtypes B/C/D/CRF.

    The correct model for targeted amplicon data is that all reads from the
    same amplicon start at the same genomic position (fixed primer design).
    With the synthetic data generator using fixed primer offsets (offset=0),
    the statistical scorer will see ~100% of reads in frame 0 — which is
    the correct and generalisable behaviour.

    Coordinate extraction for long reads belongs in the future pol_extractor
    module, not here.

Position in pipeline:
    quality_filter → pol_localizer → region_filter → codon_framer

Data contract:
    Input:  LocalizedRead (from pol_localizer.py, passed through region_filter)
    Output: FramedRead (LocalizedRead + frame fields)

Author: Genomic-Transformer-Pipeline
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.enricher.pol_localizer import LocalizedRead

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "src/config/pipeline_config.yaml"

# Codon scoring window — only first N codons used for frame scoring.
SCORE_WINDOW_CODONS = 100

# Stop density margin threshold for skipping alignment.
# 0.03 = 3 fewer stop codons per 100 codons (e.g. best=0.02, second=0.05)
STOP_MARGIN_THRESHOLD = 0.03

# Alignment window size in amino acids (tiebreaker only).
ALIGNMENT_WINDOW = 10

# Weights for the combined frame score (used only when alignment runs)
STOP_CODON_WEIGHT = 0.70
ALIGNMENT_WEIGHT  = 0.30

# Minimum frame confidence to accept a frame assignment.
# The realistic margin between frames on noisy Nanopore reads is 0.007-0.097
# (observed across 1500 synthetic reads). 0.10 rejects nearly everything.
# Lowered to 0.02 — flags only genuinely ambiguous cases where all three
# frames score within 2% of each other.
MIN_FRAME_CONFIDENCE = 0.02

# Standard genetic code
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# HXB2 reference protein sequences for PR, RT, IN
HXB2_PROTEINS: dict[str, str] = {
    "PR": (
        "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKA"
        "IGTVLVGPTPVNIIGRNLLTQIGCTLNF"
    ),
    "RT": (
        "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTK"
        "WLKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDEDFRKYTAFTIPSINNETPG"
        "IRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLR"
        "WGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCK"
        "LLRGTKALTEVIPLTEEAELELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKT"
        "GKYARMRGAHTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTPPL"
        "VKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQAIHLALQDSGSE"
        "VNIVTDSQYALGIIQAQPDKSESELVSQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVL"
    ),
    "IN": (
        "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
        "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
        "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
        "QKQITKIQNFRVYYRDSRNPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
        "RQDED"
    ),
}


# ---------------------------------------------------------------------------
# FramedRead dataclass
# ---------------------------------------------------------------------------
@dataclass
class FramedRead:
    """
    A LocalizedRead annotated with reading frame resolution results.

    Fields (inherited from LocalizedRead)
    --------------------------------------
    read_id, sequence, quality, quality_is_inferred,
    source_format, source_file, raw_header,
    gene_region, localization_confidence, seed_hits, hit_breakdown

    Fields (added by codon_framer)
    --------------------------------
    reading_frame        : int   — 0, 1, or 2
    frame_confidence     : float — margin between winner and runner-up (0-1)
    amino_acid_sequence  : str   — full translation in winning frame
    codon_sequence       : str   — nucleotides trimmed to codon boundaries
    frame_scores         : dict  — raw scores per frame for debugging
    low_confidence_frame : bool  — True if frame_confidence < MIN_FRAME_CONFIDENCE
    used_alignment       : bool  — True if alignment tiebreaker was needed
    """

    # LocalizedRead fields
    read_id:                 str
    sequence:                str
    quality:                 list
    quality_is_inferred:     bool
    source_format:           str
    source_file:             str
    raw_header:              str
    gene_region:             str
    localization_confidence: float
    seed_hits:               int
    hit_breakdown:           dict

    # Frame fields
    reading_frame:        int   = 0
    frame_confidence:     float = 0.0
    amino_acid_sequence:  str   = ""
    codon_sequence:       str   = ""
    frame_scores:         dict  = field(default_factory=lambda: {
        "frame_0": 0.0, "frame_1": 0.0, "frame_2": 0.0,
    })
    low_confidence_frame: bool  = False
    used_alignment:       bool  = False

    @classmethod
    def from_localized(
        cls,
        read:                LocalizedRead,
        reading_frame:       int,
        frame_confidence:    float,
        amino_acid_sequence: str,
        codon_sequence:      str,
        frame_scores:        dict,
        low_confidence_frame: bool = False,
        used_alignment:      bool = False,
    ) -> "FramedRead":
        return cls(
            read_id                 = read.read_id,
            sequence                = read.sequence,
            quality                 = read.quality,
            quality_is_inferred     = read.quality_is_inferred,
            source_format           = read.source_format,
            source_file             = read.source_file,
            raw_header              = read.raw_header,
            gene_region             = read.gene_region,
            localization_confidence = read.localization_confidence,
            seed_hits               = read.seed_hits,
            hit_breakdown           = read.hit_breakdown,
            reading_frame           = reading_frame,
            frame_confidence        = frame_confidence,
            amino_acid_sequence     = amino_acid_sequence,
            codon_sequence          = codon_sequence,
            frame_scores            = frame_scores,
            low_confidence_frame    = low_confidence_frame,
            used_alignment          = used_alignment,
        )

    @property
    def length(self) -> int:
        return len(self.sequence)

    @property
    def mean_quality(self) -> float:
        if not self.quality:
            return 0.0
        return sum(self.quality) / len(self.quality)

    @property
    def protein_length(self) -> int:
        return len(self.amino_acid_sequence)

    @property
    def stop_codon_count(self) -> int:
        return self.amino_acid_sequence.count("*")

    def to_dict(self) -> dict:
        return {
            "read_id":                self.read_id,
            "sequence":               self.sequence,
            "quality":                self.quality,
            "quality_is_inferred":    self.quality_is_inferred,
            "source_format":          self.source_format,
            "source_file":            self.source_file,
            "raw_header":             self.raw_header,
            "length":                 self.length,
            "mean_quality":           round(self.mean_quality, 2),
            "gene_region":            self.gene_region,
            "localization_confidence": round(self.localization_confidence, 4),
            "seed_hits":              self.seed_hits,
            "hit_breakdown":          self.hit_breakdown,
            "reading_frame":          self.reading_frame,
            "frame_confidence":       round(self.frame_confidence, 4),
            "amino_acid_sequence":    self.amino_acid_sequence,
            "codon_sequence":         self.codon_sequence,
            "frame_scores":           {k: round(v, 4) for k, v in self.frame_scores.items()},
            "low_confidence_frame":   self.low_confidence_frame,
            "used_alignment":         self.used_alignment,
            "protein_length":         self.protein_length,
            "stop_codon_count":       self.stop_codon_count,
        }

    def __repr__(self) -> str:
        return (
            f"FramedRead("
            f"id='{self.read_id}', "
            f"region='{self.gene_region}', "
            f"frame={self.reading_frame}, "
            f"frame_conf={self.frame_confidence:.3f}, "
            f"aa_len={self.protein_length}, "
            f"stops={self.stop_codon_count}, "
            f"used_aln={self.used_alignment}"
            f")"
        )


# ---------------------------------------------------------------------------
# Translation utilities
# ---------------------------------------------------------------------------

def _translate(nucleotide_sequence: str, frame: int) -> str:
    sequence    = nucleotide_sequence[frame:]
    amino_acids = []
    for i in range(0, len(sequence) - 2, 3):
        codon = sequence[i:i + 3]
        if len(codon) < 3:
            break
        amino_acids.append(CODON_TABLE.get(codon, "X"))
    return "".join(amino_acids)


def _get_codon_sequence(nucleotide_sequence: str, frame: int) -> str:
    sequence       = nucleotide_sequence[frame:]
    trimmed_length = (len(sequence) // 3) * 3
    return sequence[:trimmed_length]


def _stop_codon_density(amino_acid_sequence: str) -> float:
    if not amino_acid_sequence:
        return 1.0
    return amino_acid_sequence.count("*") / len(amino_acid_sequence)


# ---------------------------------------------------------------------------
# Alignment scoring (tiebreaker only)
# ---------------------------------------------------------------------------

def _score_alignment(
    query_protein:     str,
    reference_protein: str,
    window_size:       int = ALIGNMENT_WINDOW,
) -> float:
    if len(query_protein) < window_size or len(reference_protein) < window_size:
        min_len = min(len(query_protein), len(reference_protein))
        if min_len == 0:
            return 0.0
        matches = sum(q == r for q, r in zip(query_protein[:min_len], reference_protein[:min_len]))
        return matches / min_len

    best_score = 0.0
    for ref_start in range(0, len(reference_protein) - window_size + 1):
        ref_window = reference_protein[ref_start:ref_start + window_size]
        for q_start in range(0, len(query_protein) - window_size + 1):
            q_window = query_protein[q_start:q_start + window_size]
            matches  = sum(q == r for q, r in zip(q_window, ref_window))
            score    = matches / window_size
            if score > best_score:
                best_score = score
            if best_score > 0.85:
                return best_score
    return best_score


# ---------------------------------------------------------------------------
# Frame scoring — two-stage: stop density first, alignment as tiebreaker
# ---------------------------------------------------------------------------

def _score_all_frames(
    nucleotide_sequence: str,
    gene_region:         str,
) -> tuple[dict, bool]:
    """
    Score all three frames and determine if alignment tiebreaker is needed.

    Stage 1 — Stop density on first SCORE_WINDOW_CODONS codons.
    Stage 2 — Alignment tiebreaker if stop margins are ambiguous.

    Returns
    -------
    tuple[dict, bool]
        frame_results: {0: {score, aa_seq, codon_seq, stop_density}, ...}
        used_alignment: True if alignment tiebreaker was run
    """
    frame_results: dict[int, dict] = {}

    # Stage 1: translate and score by stop density
    for frame in [0, 1, 2]:
        full_aa    = _translate(nucleotide_sequence, frame)
        full_codon = _get_codon_sequence(nucleotide_sequence, frame)

        if not full_aa:
            frame_results[frame] = {
                "score": 0.0, "aa_seq": "", "codon_seq": "",
                "stop_density": 1.0,
            }
            continue

        scoring_aa   = full_aa[:SCORE_WINDOW_CODONS]
        stop_density = _stop_codon_density(scoring_aa)
        stop_score   = 1.0 - stop_density

        combined_score = stop_score * STOP_CODON_WEIGHT + 0.5 * ALIGNMENT_WEIGHT

        frame_results[frame] = {
            "score":        combined_score,
            "aa_seq":       full_aa,
            "codon_seq":    full_codon,
            "stop_density": stop_density,
        }

    # Check if stop density alone is discriminating enough
    stop_densities = sorted(
        [(f, frame_results[f]["stop_density"]) for f in [0, 1, 2]],
        key=lambda x: x[1]
    )
    best_stop   = stop_densities[0][1]
    second_stop = stop_densities[1][1]
    stop_margin = second_stop - best_stop

    used_alignment = False

    if stop_margin < STOP_MARGIN_THRESHOLD:
        reference_protein = HXB2_PROTEINS.get(gene_region, "")
        if reference_protein:
            used_alignment = True
            for frame in [0, 1, 2]:
                aa_seq = frame_results[frame]["aa_seq"]
                if not aa_seq:
                    continue
                scoring_aa      = aa_seq[:SCORE_WINDOW_CODONS]
                alignment_score = _score_alignment(scoring_aa, reference_protein)
                stop_density    = frame_results[frame]["stop_density"]
                stop_score      = 1.0 - stop_density
                combined_score  = (
                    stop_score      * STOP_CODON_WEIGHT +
                    alignment_score * ALIGNMENT_WEIGHT
                )
                frame_results[frame]["score"]     = combined_score
                frame_results[frame]["alignment"] = alignment_score

    return frame_results, used_alignment


# ---------------------------------------------------------------------------
# CodonFramer
# ---------------------------------------------------------------------------

class CodonFramer:
    """
    Reading frame resolver for HIV-1 pol gene sequences.

    v1.1 changes:
    - Translate only first 100 codons for scoring (was full length)
    - Stop codon density as primary discriminator
    - Alignment runs only when stop margins are ambiguous (<3% difference)
    - ~27x faster for RT reads; most reads skip alignment entirely

    v1.2 changes:
    - MIN_FRAME_CONFIDENCE lowered from 0.10 to 0.02
      (0.10 was rejecting 99%+ of reads on noisy Nanopore data;
       realistic margin between frames is 0.007-0.097)

    Usage
    -----
    framer = CodonFramer()
    for localized_read in region_filter.filter_stream(localizer_output):
        framed = framer.resolve(localized_read)
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        logger.info("Initializing CodonFramer...")
        self.config_path          = config_path
        self.min_frame_confidence = MIN_FRAME_CONFIDENCE
        logger.info(
            f"CodonFramer ready. "
            f"Scoring window: {SCORE_WINDOW_CODONS} codons, "
            f"stop margin threshold: {STOP_MARGIN_THRESHOLD}, "
            f"alignment window: {ALIGNMENT_WINDOW}aa, "
            f"min confidence: {self.min_frame_confidence}"
        )

    def resolve(self, read: LocalizedRead) -> FramedRead:
        """
        Resolve the reading frame for a single localized read.
        """
        sequence    = read.sequence.upper()
        gene_region = read.gene_region

        if gene_region == "unknown":
            logger.warning(
                f"resolve() called on unlocalized read '{read.read_id}'. "
                f"Frame assignment will be unreliable."
            )

        logger.debug(
            f"Resolving frame for '{read.read_id}' "
            f"(region={gene_region}, len={len(sequence)}bp)"
        )

        # Score all three frames
        frame_results, used_alignment = _score_all_frames(sequence, gene_region)

        # Build output frame_scores dict
        frame_scores = {f"frame_{f}": frame_results[f]["score"] for f in [0, 1, 2]}

        # Winning frame
        winning_frame = max(frame_results, key=lambda f: frame_results[f]["score"])
        winning_score = frame_results[winning_frame]["score"]

        # Runner-up score for confidence
        other_scores     = [frame_results[f]["score"] for f in [0, 1, 2] if f != winning_frame]
        runner_up_score  = max(other_scores) if other_scores else 0.0
        frame_confidence = min(1.0, max(0.0, winning_score - runner_up_score))
        low_confidence   = frame_confidence < self.min_frame_confidence

        if low_confidence:
            logger.warning(
                f"Low frame confidence for '{read.read_id}': "
                f"frame={winning_frame}, confidence={frame_confidence:.3f}, "
                f"scores={frame_scores}, used_alignment={used_alignment}"
            )
        else:
            logger.debug(
                f"Frame resolved: '{read.read_id}' → frame {winning_frame} "
                f"(confidence={frame_confidence:.3f}, "
                f"used_alignment={used_alignment})"
            )

        return FramedRead.from_localized(
            read                 = read,
            reading_frame        = winning_frame,
            frame_confidence     = round(frame_confidence, 4),
            amino_acid_sequence  = frame_results[winning_frame]["aa_seq"],
            codon_sequence       = frame_results[winning_frame]["codon_seq"],
            frame_scores         = frame_scores,
            low_confidence_frame = low_confidence,
            used_alignment       = used_alignment,
        )

    def resolve_batch(self, reads: list) -> tuple[list, dict]:
        results = []
        stats   = {
            "total": 0, "skipped_unknown": 0,
            "frame_0": 0, "frame_1": 0, "frame_2": 0,
            "low_confidence": 0, "used_alignment": 0,
        }

        for read in reads:
            if read.gene_region == "unknown":
                stats["skipped_unknown"] += 1
                continue
            framed = self.resolve(read)
            results.append(framed)
            stats["total"]                         += 1
            stats[f"frame_{framed.reading_frame}"] += 1
            if framed.low_confidence_frame:
                stats["low_confidence"] += 1
            if framed.used_alignment:
                stats["used_alignment"] += 1

        total = max(1, stats["total"])
        stats["frame_0_rate"]        = round(stats["frame_0"] / total, 4)
        stats["frame_1_rate"]        = round(stats["frame_1"] / total, 4)
        stats["frame_2_rate"]        = round(stats["frame_2"] / total, 4)
        stats["low_confidence_rate"] = round(stats["low_confidence"] / total, 4)
        stats["alignment_rate"]      = round(stats["used_alignment"] / total, 4)

        return results, stats


# ---------------------------------------------------------------------------
# Quick Validation
# Usage: python -m src.enricher.codon_framer
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    from src.ingestion.stream_reader  import stream_reads
    from src.ingestion.quality_filter import quality_filter
    from src.enricher.pol_localizer   import PolLocalizer
    from src.enricher.region_filter   import RegionFilter

    print("Initializing pipeline components...")
    localizer     = PolLocalizer()
    region_filter = RegionFilter()
    framer        = CodonFramer()

    test_files = [
        "data/test/synthetic/targeted/PR_targeted.fastq.gz",
        "data/test/synthetic/targeted/RT_targeted.fastq.gz",
        "data/test/synthetic/targeted/IN_targeted.fastq.gz",
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"SKIP (not found): {test_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Framing: {Path(test_file).name}")
        print(f"{'='*60}")

        raw_stream                    = stream_reads(test_file)
        filtered_stream, filter_stats = quality_filter(raw_stream)
        localized_stream              = (localizer.localize(r) for r in filtered_stream)
        region_passed_stream          = region_filter.filter_stream(localized_stream)

        framed_reads    = []
        skipped_unknown = 0
        low_conf_count  = 0
        used_aln_count  = 0
        frame_counts    = {0: 0, 1: 0, 2: 0}

        for localized in region_passed_stream:
            if localized.gene_region == "unknown":
                skipped_unknown += 1
                continue
            framed = framer.resolve(localized)
            framed_reads.append(framed)
            frame_counts[framed.reading_frame] += 1
            if framed.low_confidence_frame:
                low_conf_count += 1
            if framed.used_alignment:
                used_aln_count += 1

        total_framed = len(framed_reads)

        print(f"\nFraming Results:")
        print(f"  Total reads after region filter : {total_framed + skipped_unknown}")
        print(f"  Successfully framed             : {total_framed}")
        print(f"  Skipped (unknown region)        : {skipped_unknown}")
        print(f"  Low confidence frames           : {low_conf_count} ({low_conf_count/max(1,total_framed)*100:.1f}%)")
        print(f"  Used alignment tiebreaker       : {used_aln_count} ({used_aln_count/max(1,total_framed)*100:.1f}%)")

        print(f"\nReading Frame Distribution:")
        for f in [0, 1, 2]:
            count = frame_counts[f]
            pct   = count / max(1, total_framed) * 100
            print(f"  Frame {f}: {count} reads ({pct:.1f}%)")

        if framed_reads:
            print(f"\nSample framed reads (first 3):")
            for i, r in enumerate(framed_reads[:3]):
                print(f"\n  Read #{i+1}: {r.read_id}")
                print(f"    Region          : {r.gene_region}")
                print(f"    Reading Frame   : {r.reading_frame}")
                print(f"    Frame Confidence: {r.frame_confidence:.4f}")
                print(f"    Used Alignment  : {r.used_alignment}")
                print(f"    Protein Length  : {r.protein_length} aa")
                print(f"    Stop Codons     : {r.stop_codon_count}")
                print(f"    AA Sequence[:30]: {r.amino_acid_sequence[:30]}")

            confidences = [r.frame_confidence for r in framed_reads]
            stop_counts = [r.stop_codon_count  for r in framed_reads]

            print(f"\nFrame Confidence Distribution:")
            print(f"  Min  : {min(confidences):.4f}")
            print(f"  Max  : {max(confidences):.4f}")
            print(f"  Mean : {sum(confidences)/len(confidences):.4f}")

            print(f"\nStop Codon Distribution (winning frame, full read):")
            print(f"  Min  : {min(stop_counts)}")
            print(f"  Max  : {max(stop_counts)}")
            print(f"  Mean : {sum(stop_counts)/len(stop_counts):.2f}")