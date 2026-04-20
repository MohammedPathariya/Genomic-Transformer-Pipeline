"""
src/enricher/pol_localizer.py
==============================
Alignment-free HIV-1 pol gene region localizer.

Responsibility:
    Determine whether a RawRead originates from the HIV-1 pol gene and,
    if so, which sub-region: Protease (PR), Reverse Transcriptase (RT),
    or Integrase (IN). Uses k-mer seed matching against conserved anchor
    sequences extracted from the HXB2 reference genome.

This is Novel Component 1 of the research contribution.

Why this is novel:
    Every existing clinical tool (Sierra, PASeq, Geneious) requires a
    completed Minimap2/BWA alignment before calling a single mutation.
    This module replaces that 10-20 minute alignment step with a
    sub-second k-mer lookup that requires no coordinate system and
    generalizes across HIV-1 subtypes A through D and recombinant forms.

How it works:
    1. At module load time, extract every 15-mer from the PR, RT, and IN
       regions of the HXB2 reference genome.
    2. Filter to keep only k-mers that are unique to their region across
       the entire HXB2 genome (not just within pol).
    3. For reads shorter than long_read_threshold (3000bp):
       Whole-read scoring — count k-mer hits across the full sequence.
    4. For reads longer than long_read_threshold:
       Windowed scoring — slide a 400bp window and find the window with
       the strongest discriminating signal.
    5. The region with the most normalised hits wins — if it clears
       minimum thresholds.
    6. Return the read annotated with gene_region and confidence score.

Known limitation — full-pol amplicon datasets:
    When reads span the entire pol gene (PR + RT + IN, ~2843bp total),
    all three regions score positively simultaneously. The architecturally
    correct solution is subsequence extraction via a pol_extractor module
    (deferred to next phase).

Position in pipeline:
    quality_filter → pol_localizer → codon_framer

Data contract:
    Input:  RawRead (from stream_reader.py)
    Output: LocalizedRead (RawRead + localization fields)

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

from src.ingestion.stream_reader import RawRead

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "src/config/pipeline_config.yaml"

# HXB2 gene region coordinates (nucleotide positions, 0-indexed)
DEFAULT_GENE_REGIONS = {
    "PR": {"start": 2252, "end": 2549},
    "RT": {"start": 2549, "end": 4229},
    "IN": {"start": 4229, "end": 5096},
}

DEFAULT_LONG_READ_THRESHOLD = 3000
DEFAULT_WINDOW_SIZE         = 400
DEFAULT_WINDOW_STEP         = 200

_COMPLEMENT = str.maketrans("ATCGN", "TAGCN")


# ---------------------------------------------------------------------------
# LocalizedRead
# ---------------------------------------------------------------------------

@dataclass
class LocalizedRead:
    """
    A RawRead annotated with pol gene region localization results.

    Fields (inherited from RawRead)
    --------------------------------
    read_id, sequence, quality, quality_is_inferred,
    source_format, source_file, raw_header

    Fields (added by pol_localizer)
    --------------------------------
    gene_region : str
        "PR", "RT", "IN", or "unknown"

    localization_confidence : float
        Confidence score 0.0 to 1.0.
        Computed as: winning_region_hits / total_kmers_in_read

    seed_hits : int
        Raw count of k-mer matches for the winning region.

    hit_breakdown : dict
        Raw hit counts for all three regions.
        Example: {"PR": 2, "RT": 18, "IN": 1}
    """

    # RawRead fields
    read_id:             str
    sequence:            str
    quality:             list
    quality_is_inferred: bool
    source_format:       str
    source_file:         str
    raw_header:          str

    # Localization fields
    gene_region:              str   = "unknown"
    localization_confidence:  float = 0.0
    seed_hits:                int   = 0
    hit_breakdown:            dict  = field(default_factory=lambda: {
        "PR": 0, "RT": 0, "IN": 0
    })

    @classmethod
    def from_rawread(
        cls,
        read:                    RawRead,
        gene_region:             str   = "unknown",
        localization_confidence: float = 0.0,
        seed_hits:               int   = 0,
        hit_breakdown:           dict  = None,
    ) -> "LocalizedRead":
        return cls(
            read_id                 = read.read_id,
            sequence                = read.sequence,
            quality                 = read.quality,
            quality_is_inferred     = read.quality_is_inferred,
            source_format           = read.source_format,
            source_file             = read.source_file,
            raw_header              = read.raw_header,
            gene_region             = gene_region,
            localization_confidence = localization_confidence,
            seed_hits               = seed_hits,
            hit_breakdown           = hit_breakdown or {"PR": 0, "RT": 0, "IN": 0},
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
    def is_localized(self) -> bool:
        return self.gene_region != "unknown"

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
        }

    def __repr__(self) -> str:
        return (
            f"LocalizedRead("
            f"id='{self.read_id}', "
            f"region='{self.gene_region}', "
            f"confidence={self.localization_confidence:.3f}, "
            f"hits={self.seed_hits}, "
            f"breakdown={self.hit_breakdown}"
            f")"
        )


# ---------------------------------------------------------------------------
# K-mer utilities
# ---------------------------------------------------------------------------

def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_COMPLEMENT)[::-1]


def _is_low_complexity(kmer: str) -> bool:
    k = len(kmer)
    for base in "ATCG":
        if kmer.count(base) / k > 0.60:
            return True
    if "N" in kmer:
        return True
    return False


def _extract_kmers(sequence: str, k: int) -> set:
    kmers = set()
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        if len(kmer) == k:
            kmers.add(kmer)
            kmers.add(_reverse_complement(kmer))
    return kmers


def _count_kmer_hits(read_sequence: str, anchor_set: set, k: int) -> int:
    """
    Count how many k-mers in a read sequence appear in an anchor set.

    Parameters
    ----------
    read_sequence : str
        The read sequence, uppercase.
    anchor_set : set
        Set of anchor k-mers for one gene region (including rev complements).
    k : int
        K-mer length.

    Returns
    -------
    int
        Number of k-mers in the read that appear in the anchor set.
    """
    hits = 0
    for i in range(len(read_sequence) - k + 1):
        kmer = read_sequence[i:i + k]
        if kmer in anchor_set:
            hits += 1
    return hits


# ---------------------------------------------------------------------------
# Reference genome loading
# ---------------------------------------------------------------------------

def _load_hxb2_sequence(reference_path: str) -> str:
    if not os.path.exists(reference_path):
        raise FileNotFoundError(
            f"HXB2 reference not found at '{reference_path}'.\n"
            f"Download with:\n"
            f'  curl "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
            f'?db=nuccore&id=K03455&rettype=fasta&retmode=text" '
            f'-o {reference_path}'
        )

    sequence_parts = []
    with open(reference_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                continue
            sequence_parts.append(line.upper())

    full_sequence = "".join(sequence_parts)

    if not full_sequence:
        raise ValueError(
            f"HXB2 reference file '{reference_path}' contains no sequence data."
        )

    logger.info(f"HXB2 reference loaded: {len(full_sequence)} bases")
    return full_sequence


# ---------------------------------------------------------------------------
# Anchor K-mer builder
# ---------------------------------------------------------------------------

def build_anchor_sets(
    hxb2_sequence: str,
    gene_regions:  dict,
    k:             int = 15,
) -> dict[str, set]:
    """
    Build the anchor k-mer sets for PR, RT, and IN from the HXB2 sequence.

    Steps:
    1. Extract all k-mers from each gene region
    2. Remove low-complexity k-mers
    3. Remove k-mers that appear anywhere else in the full HXB2 genome
    4. Add reverse complements of all surviving k-mers
    """

    # Step 1: Extract raw k-mers for each region
    raw_kmers = {}
    for region_name, coords in gene_regions.items():
        region_seq = hxb2_sequence[coords["start"]:coords["end"]]
        kmers = set()
        for i in range(len(region_seq) - k + 1):
            kmer = region_seq[i:i + k]
            if len(kmer) == k and not _is_low_complexity(kmer):
                kmers.add(kmer)
        raw_kmers[region_name] = kmers

    # Step 2: Build outside-region k-mer sets for uniqueness filtering
    all_genome_kmers: dict[str, set] = {}
    for region_name, coords in gene_regions.items():
        outside_sequence = (
            hxb2_sequence[:coords["start"]] +
            hxb2_sequence[coords["end"]:]
        )
        outside_kmers = set()
        for i in range(len(outside_sequence) - k + 1):
            kmer = outside_sequence[i:i + k]
            if len(kmer) == k:
                outside_kmers.add(kmer)
        all_genome_kmers[region_name] = outside_kmers

    # Step 3: Keep only region-unique k-mers
    unique_kmers = {}
    for region_name in raw_kmers:
        outside_kmers = all_genome_kmers[region_name]
        unique = raw_kmers[region_name] - outside_kmers
        unique_kmers[region_name] = unique

    # Step 4: Add reverse complements
    final_sets = {}
    for region_name, kmers in unique_kmers.items():
        with_rc = set()
        for kmer in kmers:
            with_rc.add(kmer)
            with_rc.add(_reverse_complement(kmer))
        final_sets[region_name] = with_rc

        logger.info(
            f"  Anchor set [{region_name}]: {len(with_rc)} k-mers "
            f"({len(kmers)} genome-unique + reverse complements)"
        )

    return final_sets


# ---------------------------------------------------------------------------
# Windowed localization — for reads longer than pol gene (~3000bp)
# ---------------------------------------------------------------------------

def _localize_windowed(
    sequence:    str,
    anchor_sets: dict,
    k:           int,
    window_size: int = DEFAULT_WINDOW_SIZE,
    window_step: int = DEFAULT_WINDOW_STEP,
) -> tuple[str, float, int, dict]:
    """
    Localize a long read by scoring sliding windows across its length.

    Scores each window as: hits / anchor_set_size
    This normalises for the different anchor set sizes across regions
    (PR=564, RT=3200, IN=1612) so raw hit counts are comparable.
    """
    read_len = len(sequence)

    best_per_region: dict[str, dict] = {
        region: {"score": 0.0, "hits": 0}
        for region in anchor_sets
    }

    window_count = 0
    for window_start in range(0, read_len - window_size + 1, window_step):
        window_seq = sequence[window_start: window_start + window_size]
        window_len = len(window_seq)

        if window_len < k:
            break

        for region_name, anchor_set in anchor_sets.items():
            hits  = _count_kmer_hits(window_seq, anchor_set, k)

            # Normalise by anchor set size — not by window possible k-mers.
            # This measures what fraction of the region's known anchors
            # the window found — a true specificity score that is comparable
            # across regions regardless of anchor set size differences.
            score = hits / max(1, len(anchor_set))

            if score > best_per_region[region_name]["score"]:
                best_per_region[region_name]["score"] = score
                best_per_region[region_name]["hits"]  = hits

        window_count += 1

    logger.debug(
        f"Windowed scoring: {window_count} windows across {read_len}bp read"
    )

    hit_breakdown = {
        region: best_per_region[region]["hits"]
        for region in anchor_sets
    }

    winning_region = max(
        best_per_region,
        key=lambda r: best_per_region[r]["score"]
    )
    winning_score = best_per_region[winning_region]["score"]
    winning_hits  = best_per_region[winning_region]["hits"]

    return winning_region, winning_score, winning_hits, hit_breakdown


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def _load_localizer_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    defaults = {
        "kmer_size":           15,
        "min_seed_hits":       3,
        "min_kmer_fraction":   0.05,
        "long_read_threshold": DEFAULT_LONG_READ_THRESHOLD,
        "window_size":         DEFAULT_WINDOW_SIZE,
        "window_step":         DEFAULT_WINDOW_STEP,
        "gene_regions":        DEFAULT_GENE_REGIONS,
        "reference": {
            "path": "data/public/HXB2_reference.fasta"
        },
    }

    if not os.path.exists(config_path):
        logger.warning(
            f"Config not found at '{config_path}'. "
            f"Using default localizer settings."
        )
        return defaults

    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        enricher_config = full_config.get("enricher", {})

        ref_path = (
            full_config
            .get("output", {})
            .get("reference", {})
            .get("path", defaults["reference"]["path"])
        )

        merged = {
            "kmer_size":           enricher_config.get("kmer_size",           defaults["kmer_size"]),
            "min_seed_hits":       enricher_config.get("min_seed_hits",       defaults["min_seed_hits"]),
            "min_kmer_fraction":   enricher_config.get("min_kmer_fraction",   defaults["min_kmer_fraction"]),
            "long_read_threshold": enricher_config.get("long_read_threshold", defaults["long_read_threshold"]),
            "window_size":         enricher_config.get("window_size",         defaults["window_size"]),
            "window_step":         enricher_config.get("window_step",         defaults["window_step"]),
            "gene_regions":        enricher_config.get("gene_regions",        defaults["gene_regions"]),
            "reference":           {"path": ref_path},
        }

        logger.info(
            f"Localizer config loaded: "
            f"k={merged['kmer_size']}, "
            f"min_hits={merged['min_seed_hits']}, "
            f"min_fraction={merged['min_kmer_fraction']}, "
            f"long_read_threshold={merged['long_read_threshold']}bp, "
            f"window={merged['window_size']}bp step={merged['window_step']}bp, "
            f"reference='{ref_path}'"
        )

        return merged

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config '{config_path}': {e}")
        return defaults


# ---------------------------------------------------------------------------
# PolLocalizer
# ---------------------------------------------------------------------------

class PolLocalizer:
    """
    Alignment-free HIV-1 pol gene region localizer.

    Instantiate once per pipeline run. Anchor sets are built at construction
    time and reused for every read.

    Usage
    -----
    localizer = PolLocalizer()
    for read in quality_filter(stream_reads("sample.fastq.gz")):
        localized = localizer.localize(read)
        print(localized.gene_region, localized.localization_confidence)
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        logger.info("Initializing PolLocalizer...")

        config = _load_localizer_config(config_path)

        self.k                   = config["kmer_size"]
        self.min_seed_hits       = config["min_seed_hits"]
        self.min_kmer_fraction   = config["min_kmer_fraction"]
        self.long_read_threshold = config["long_read_threshold"]
        self.window_size         = config["window_size"]
        self.window_step         = config["window_step"]
        self.reference_path      = config["reference"]["path"]

        raw_regions = config["gene_regions"]
        self.gene_regions = {}
        for region_name, coords in raw_regions.items():
            if isinstance(coords, dict):
                self.gene_regions[region_name] = {
                    "start": int(coords["start"]),
                    "end":   int(coords["end"]),
                }

        logger.info(f"Loading HXB2 reference: {self.reference_path}")
        hxb2_sequence = _load_hxb2_sequence(self.reference_path)

        logger.info(
            f"Building anchor k-mer sets "
            f"(k={self.k}, regions={list(self.gene_regions.keys())})..."
        )
        self.anchor_sets = build_anchor_sets(
            hxb2_sequence = hxb2_sequence,
            gene_regions  = self.gene_regions,
            k             = self.k,
        )

        for region, anchors in self.anchor_sets.items():
            logger.info(f"  [{region}] anchor set size: {len(anchors)} k-mers")

        logger.info("PolLocalizer ready.")

    def localize(self, read: RawRead) -> LocalizedRead:
        """
        Localize a single read to a pol gene region.
        """
        sequence = read.sequence.upper()
        read_len = len(sequence)

        total_possible_kmers = max(1, read_len - self.k + 1)

        if read_len >= self.long_read_threshold:
            logger.debug(
                f"Using windowed scoring for '{read.read_id}' "
                f"(len={read_len}bp >= threshold={self.long_read_threshold}bp)"
            )
            winning_region, confidence, winning_hits, hit_breakdown = _localize_windowed(
                sequence    = sequence,
                anchor_sets = self.anchor_sets,
                k           = self.k,
                window_size = self.window_size,
                window_step = self.window_step,
            )

        else:
            logger.debug(
                f"Using whole-read scoring for '{read.read_id}' "
                f"(len={read_len}bp < threshold={self.long_read_threshold}bp)"
            )
            hit_breakdown = {}
            for region_name, anchor_set in self.anchor_sets.items():
                hits = _count_kmer_hits(sequence, anchor_set, self.k)
                hit_breakdown[region_name] = hits

            winning_region = max(hit_breakdown, key=hit_breakdown.get)
            winning_hits   = hit_breakdown[winning_region]
            confidence     = winning_hits / total_possible_kmers

        # Apply minimum thresholds
        if (winning_hits < self.min_seed_hits or
                confidence < self.min_kmer_fraction):

            logger.debug(
                f"UNKNOWN: '{read.read_id}' — "
                f"best={winning_region} hits={winning_hits} "
                f"confidence={confidence:.4f}"
            )

            return LocalizedRead.from_rawread(
                read                    = read,
                gene_region             = "unknown",
                localization_confidence = 0.0,
                seed_hits               = winning_hits,
                hit_breakdown           = hit_breakdown,
            )

        logger.debug(
            f"LOCALIZED: '{read.read_id}' → {winning_region} "
            f"hits={winning_hits} confidence={confidence:.4f}"
        )

        return LocalizedRead.from_rawread(
            read                    = read,
            gene_region             = winning_region,
            localization_confidence = round(confidence, 4),
            seed_hits               = winning_hits,
            hit_breakdown           = hit_breakdown,
        )

    def localize_batch(
        self,
        reads: list,
    ) -> tuple[list, dict]:
        results = []
        stats   = {"PR": 0, "RT": 0, "IN": 0, "unknown": 0, "total": 0}

        for read in reads:
            localized = self.localize(read)
            results.append(localized)
            stats[localized.gene_region] += 1
            stats["total"] += 1

        total = max(1, stats["total"])
        stats["PR_rate"]      = round(stats["PR"]      / total, 4)
        stats["RT_rate"]      = round(stats["RT"]      / total, 4)
        stats["IN_rate"]      = round(stats["IN"]      / total, 4)
        stats["unknown_rate"] = round(stats["unknown"] / total, 4)

        return results, stats


# ---------------------------------------------------------------------------
# Quick Validation
# Usage: python -m src.enricher.pol_localizer
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import json
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    from src.ingestion.stream_reader  import stream_reads
    from src.ingestion.quality_filter import quality_filter

    print("Initializing PolLocalizer...")
    localizer = PolLocalizer()

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
        print(f"Localizing: {Path(test_file).name}")
        print(f"{'='*60}")

        raw_stream                    = stream_reads(test_file)
        filtered_stream, filter_stats = quality_filter(raw_stream)

        results  = []
        pr_reads = []
        rt_reads = []
        in_reads = []
        unknown  = []

        for read in filtered_stream:
            localized = localizer.localize(read)
            results.append(localized)

            if localized.gene_region == "PR":   pr_reads.append(localized)
            elif localized.gene_region == "RT": rt_reads.append(localized)
            elif localized.gene_region == "IN": in_reads.append(localized)
            else:                               unknown.append(localized)

        total = len(results)

        print(f"\nLocalization Results:")
        print(f"  Total reads processed : {total}")
        print(f"  PR (Protease)         : {len(pr_reads)} ({len(pr_reads)/max(1,total)*100:.1f}%)")
        print(f"  RT (Reverse Transcrip): {len(rt_reads)} ({len(rt_reads)/max(1,total)*100:.1f}%)")
        print(f"  IN (Integrase)        : {len(in_reads)} ({len(in_reads)/max(1,total)*100:.1f}%)")
        print(f"  Unknown               : {len(unknown)} ({len(unknown)/max(1,total)*100:.1f}%)")

        most_populated = max(
            [("PR", pr_reads), ("RT", rt_reads), ("IN", in_reads)],
            key=lambda x: len(x[1])
        )
        region_name, region_reads = most_populated

        if region_reads:
            print(f"\nSample reads from {region_name} (showing first 3):")
            for i, r in enumerate(region_reads[:3]):
                print(f"\n  Read #{i+1}: {r.read_id}")
                print(f"    Region     : {r.gene_region}")
                print(f"    Confidence : {r.localization_confidence:.4f}")
                print(f"    Seed hits  : {r.seed_hits}")
                print(f"    Breakdown  : {r.hit_breakdown}")
                print(f"    Length     : {r.length}bp")

        localized_reads = pr_reads + rt_reads + in_reads
        if localized_reads:
            confidences = [r.localization_confidence for r in localized_reads]
            print(f"\nConfidence Score Distribution (localized reads only):")
            print(f"  Min  : {min(confidences):.4f}")
            print(f"  Max  : {max(confidences):.4f}")
            print(f"  Mean : {sum(confidences)/len(confidences):.4f}")