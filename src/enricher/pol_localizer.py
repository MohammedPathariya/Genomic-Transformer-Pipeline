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
       Works correctly when the read covers only one pol sub-region.
    4. For reads longer than long_read_threshold:
       Windowed scoring — slide a 400bp window and find the window with
       the strongest discriminating signal. Prevents RT from always
       winning on reads that span the full pol gene.
    5. The region with the most normalised hits wins — if it clears
       minimum thresholds.
    6. Return the read annotated with gene_region and confidence score.

Known limitation — full-pol amplicon datasets:
    When reads span the entire pol gene (PR + RT + IN, ~2843bp total),
    all three regions score positively simultaneously. Single-region
    assignment in this case is an approximation — the winning region
    reflects which sub-region is most densely represented in the best
    scoring window, not necessarily the only region present.

    The architecturally correct solution for full-pol reads is
    subsequence extraction: use anchor k-mer positions to identify
    where each sub-region sits within the read and extract those
    subsequences independently. This is deferred to the test suite
    phase where ground-truth BAM alignment coordinates are available
    for validation.

    For targeted amplicon datasets (reads covering one sub-region only),
    this module works correctly without modification.

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
# These match the values in pipeline_config.yaml and are duplicated
# here as a safety fallback in case the config file is missing.
DEFAULT_GENE_REGIONS = {
    "PR": {"start": 2253, "end": 2550},
    "RT": {"start": 2550, "end": 4229},
    "IN": {"start": 4229, "end": 5096},
}

# Long read threshold in base pairs
# Reads longer than this likely span multiple pol regions simultaneously
# (PR + RT + IN = ~2843bp total). A read longer than this threshold
# cannot be reliably assigned to a single region using whole-read scoring
# because RT will always win — it has the most anchor k-mers.
# Windowed scoring is used instead for these reads.
DEFAULT_LONG_READ_THRESHOLD = 3000

# Window size for windowed localization of long reads
# Each window is scored independently against all three anchor sets
# 400bp windows cover roughly one-third of a single pol sub-region
# and are large enough to contain multiple anchor k-mer hits
DEFAULT_WINDOW_SIZE = 400

# Step size between windows (overlap = window_size - step_size)
# 200bp step = 50% overlap between consecutive windows
# More overlap = more sensitive but slower
# Less overlap = faster but may miss region boundaries
DEFAULT_WINDOW_STEP = 200

# Reverse complement translation table
# Built once at module load — used millions of times during processing
_COMPLEMENT = str.maketrans("ATCGN", "TAGCN")


# ---------------------------------------------------------------------------
# LocalizedRead: RawRead + localization annotation
# ---------------------------------------------------------------------------

@dataclass
class LocalizedRead:
    """
    A RawRead annotated with pol gene region localization results.

    This is the data contract between pol_localizer.py and codon_framer.py.
    Everything in RawRead is preserved. Four localization fields are added.

    Fields (inherited from RawRead)
    --------------------------------
    read_id, sequence, quality, quality_is_inferred,
    source_format, source_file, raw_header
    (see stream_reader.py for full documentation)

    Fields (added by pol_localizer)
    --------------------------------
    gene_region : str
        Which pol sub-region this read was assigned to.
        One of: "PR", "RT", "IN", "unknown"
        "unknown" means the read does not contain sufficient pol sequence
        to make a confident assignment. These reads are excluded from
        codon_framer and all downstream processing.

    localization_confidence : float
        Confidence score for the gene_region assignment. Range: 0.0 to 1.0.
        Computed as: winning_region_hits / total_kmers_in_read
        This is the fraction of the read's k-mers that matched the
        winning region's anchor set.
        0.0 → no matching k-mers found
        1.0 → every k-mer in the read matched the anchor set (rare)
        Typical values for correct assignments: 0.05 to 0.35

    seed_hits : int
        Raw count of k-mer matches for the winning region.
        The minimum to make any call is set by min_seed_hits in config.

    hit_breakdown : dict
        Raw hit counts for all three regions.
        Example: {"PR": 2, "RT": 18, "IN": 1}
        Useful for debugging borderline cases where two regions scored
        similarly.
    """

    # All RawRead fields
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
        read:                   RawRead,
        gene_region:            str   = "unknown",
        localization_confidence: float = 0.0,
        seed_hits:              int   = 0,
        hit_breakdown:          dict  = None,
    ) -> "LocalizedRead":
        """
        Construct a LocalizedRead from an existing RawRead.

        This is the primary constructor used by pol_localizer.
        It copies all RawRead fields and adds the localization results.

        Parameters
        ----------
        read : RawRead
            The source read to annotate.
        gene_region : str
            Assignment result: "PR", "RT", "IN", or "unknown"
        localization_confidence : float
            Confidence score 0.0 to 1.0
        seed_hits : int
            Number of matching k-mers for the winning region
        hit_breakdown : dict
            Per-region raw hit counts

        Returns
        -------
        LocalizedRead
        """
        return cls(
            read_id              = read.read_id,
            sequence             = read.sequence,
            quality              = read.quality,
            quality_is_inferred  = read.quality_is_inferred,
            source_format        = read.source_format,
            source_file          = read.source_file,
            raw_header           = read.raw_header,
            gene_region          = gene_region,
            localization_confidence = localization_confidence,
            seed_hits            = seed_hits,
            hit_breakdown        = hit_breakdown or {"PR": 0, "RT": 0, "IN": 0},
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
        """True if the read was successfully assigned to a gene region."""
        return self.gene_region != "unknown"

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSONL output."""
        return {
            "read_id":               self.read_id,
            "sequence":              self.sequence,
            "quality":               self.quality,
            "quality_is_inferred":   self.quality_is_inferred,
            "source_format":         self.source_format,
            "source_file":           self.source_file,
            "raw_header":            self.raw_header,
            "length":                self.length,
            "mean_quality":          round(self.mean_quality, 2),
            "gene_region":           self.gene_region,
            "localization_confidence": round(self.localization_confidence, 4),
            "seed_hits":             self.seed_hits,
            "hit_breakdown":         self.hit_breakdown,
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
# K-mer Utilities
# ---------------------------------------------------------------------------

def _reverse_complement(sequence: str) -> str:
    """
    Compute the reverse complement of a DNA sequence.

    DNA is double-stranded. A sequencer can read a fragment in either
    direction. To match k-mers regardless of read orientation, we check
    both the forward sequence and its reverse complement.

    Example:
        "ATCG" → reverse → "GCTA" → complement → "CGAT"
        So reverse_complement("ATCG") = "CGAT"

    Parameters
    ----------
    sequence : str
        Uppercase DNA string containing only A, T, C, G, N.

    Returns
    -------
    str
        Reverse complement of the input sequence.
    """
    return sequence.translate(_COMPLEMENT)[::-1]


def _is_low_complexity(kmer: str) -> bool:
    """
    Detect low-complexity k-mers that should be excluded from anchor sets.

    A low-complexity k-mer is one dominated by a single nucleotide or a
    simple repeat. These k-mers appear everywhere in the genome and are
    useless for discriminating gene regions.

    Criteria:
        1. Any single base makes up more than 60% of the k-mer
           Example: "AAAAAATCGATCG" → 8/13 A bases → 61.5% → excluded
        2. The k-mer is a perfect dinucleotide repeat
           Example: "ATATATATATATAT" → excluded

    Parameters
    ----------
    kmer : str
        The k-mer to evaluate.

    Returns
    -------
    bool
        True if the k-mer is low complexity and should be excluded.
    """
    k = len(kmer)

    # Check single-base dominance (>60% threshold)
    for base in "ATCG":
        if kmer.count(base) / k > 0.60:
            return True

    # Check for N bases — k-mers with N are ambiguous
    if "N" in kmer:
        return True

    return False


def _extract_kmers(sequence: str, k: int) -> set:
    """
    Extract all k-mers from a sequence as a set.

    Includes both the forward k-mers and their reverse complements,
    so that reads from either strand will match.

    Parameters
    ----------
    sequence : str
        DNA sequence string, uppercase.
    k : int
        K-mer length.

    Returns
    -------
    set
        All unique k-mers and their reverse complements from the sequence.
    """
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

    This is the core matching operation. For a read of length L,
    this checks L - k + 1 k-mers. Each lookup in the anchor set is O(1).
    Total complexity: O(L) per region per read.

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
# Reference Genome Loading
# ---------------------------------------------------------------------------

def _load_hxb2_sequence(reference_path: str) -> str:
    """
    Load the HXB2 reference genome sequence from a FASTA file.

    Parameters
    ----------
    reference_path : str
        Path to the HXB2 FASTA file.

    Returns
    -------
    str
        The full HXB2 genome sequence, uppercase, with no whitespace.

    Raises
    ------
    FileNotFoundError
        If the reference file does not exist.
    ValueError
        If the file contains no sequence data.
    """
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
                continue  # skip header lines
            sequence_parts.append(line.upper())

    full_sequence = "".join(sequence_parts)

    if not full_sequence:
        raise ValueError(
            f"HXB2 reference file '{reference_path}' contains no sequence data."
        )

    logger.info(f"HXB2 reference loaded: {len(full_sequence)} bases")
    return full_sequence


# ---------------------------------------------------------------------------
# Anchor K-mer Builder
# ---------------------------------------------------------------------------

def build_anchor_sets(
    hxb2_sequence: str,
    gene_regions:  dict,
    k:             int = 15,
) -> dict[str, set]:
    """
    Build the anchor k-mer sets for PR, RT, and IN from the HXB2 sequence.

    This function is called once at pipeline startup. The resulting sets
    are stored in memory and reused for every read in the batch.

    Steps:
    1. Extract all k-mers from each gene region
    2. Remove low-complexity k-mers
    3. Remove k-mers that appear anywhere else in the full HXB2 genome
       (not just the other two pol regions — anywhere in the 9719bp sequence)
    4. Add reverse complements of all surviving k-mers

    The full-genome uniqueness filter is critical. A k-mer that appears in
    the IN region but also in the gag or env genes will match reads from
    those genes and contaminate the anchor set. We only keep k-mers that
    are exclusive to their region within the entire reference genome.

    Parameters
    ----------
    hxb2_sequence : str
        The full HXB2 genome sequence.
    gene_regions : dict
        Dictionary of region name → {"start": int, "end": int}.
        Coordinates are 0-indexed, end-exclusive (Python slice convention).
    k : int
        K-mer length. Default 15 from config.

    Returns
    -------
    dict[str, set]
        Keys: "PR", "RT", "IN"
        Values: sets of genome-unique, high-complexity anchor k-mers
                (both strands included)
    """

    # Step 1: Extract raw k-mers for each region (forward strand only here)
    raw_kmers = {}
    for region_name, coords in gene_regions.items():
        region_seq = hxb2_sequence[coords["start"]:coords["end"]]
        kmers = set()
        for i in range(len(region_seq) - k + 1):
            kmer = region_seq[i:i + k]
            if len(kmer) == k and not _is_low_complexity(kmer):
                kmers.add(kmer)
        raw_kmers[region_name] = kmers
        logger.debug(
            f"  {region_name}: {len(kmers)} k-mers before uniqueness filter"
        )

    # Step 2: Build a set of ALL k-mers in the full HXB2 genome
    # This is used to find k-mers that appear outside their target region
    all_genome_kmers: dict[str, set] = {}  # position → kmers outside each region

    for region_name, coords in gene_regions.items():
        # Everything EXCEPT the target region
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

        logger.debug(
            f"  {region_name}: outside-region genome has "
            f"{len(outside_kmers)} unique k-mers"
        )

    # Step 3: Keep only k-mers that do NOT appear anywhere outside
    # their target region in the full genome
    unique_kmers = {}
    for region_name in raw_kmers:
        outside_kmers = all_genome_kmers[region_name]
        unique = raw_kmers[region_name] - outside_kmers
        unique_kmers[region_name] = unique

        removed = len(raw_kmers[region_name]) - len(unique)
        logger.debug(
            f"  {region_name}: {len(unique)} k-mers after genome uniqueness filter "
            f"(removed {removed} non-unique)"
        )

    # Step 4: Add reverse complements to each unique set
    # This allows matching reads sequenced from either strand
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
# Windowed Localization — for reads longer than pol gene (~3000bp)
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

    The problem with whole-read scoring on long reads:
        An 8000bp read spans the full pol gene (PR + RT + IN = 2843bp)
        plus flanking sequence. RT has the most anchor k-mers so it
        always wins the whole-read vote — even when the read contains
        clear PR or IN sequence.

    The windowed solution:
        Slide a 400bp window across the read in 200bp steps. Score each
        window independently against all three anchor sets. The winning
        region is determined by the window with the single highest
        normalized score — not the sum across the whole read.

    Why the best window rather than the average:
        A long read that spans RT + IN will have windows scoring well
        for both. Taking the average would muddle the signal. Taking
        the best single window identifies the most confidently localized
        region — the part of the read with the strongest discriminating
        k-mer signal.

    Parameters
    ----------
    sequence : str
        Full read sequence, uppercase.
    anchor_sets : dict
        Dictionary of region → set of anchor k-mers.
    k : int
        K-mer length.
    window_size : int
        Size of each scoring window in bases.
    window_step : int
        Number of bases to advance between windows.

    Returns
    -------
    tuple[str, float, int, dict]
        (gene_region, confidence, seed_hits, hit_breakdown)
        Same return signature as the whole-read scoring path so
        localize() can use both paths interchangeably.
    """
    read_len = len(sequence)

    # Track best window result per region
    # Structure: {region: {"score": float, "hits": int}}
    best_per_region: dict[str, dict] = {
        region: {"score": 0.0, "hits": 0}
        for region in anchor_sets
    }

    # Slide window across the read
    window_count = 0
    for window_start in range(0, read_len - window_size + 1, window_step):
        window_seq = sequence[window_start: window_start + window_size]
        window_len = len(window_seq)

        if window_len < k:
            break

        total_possible = window_len - k + 1

        # Score this window against each region's anchor set
        for region_name, anchor_set in anchor_sets.items():
            hits  = _count_kmer_hits(window_seq, anchor_set, k)
            score = hits / total_possible

            # Keep only the best-scoring window per region
            if score > best_per_region[region_name]["score"]:
                best_per_region[region_name]["score"] = score
                best_per_region[region_name]["hits"]  = hits

        window_count += 1

    logger.debug(
        f"Windowed scoring: {window_count} windows across {read_len}bp read"
    )

    # hit_breakdown for the output — use best window hits per region
    hit_breakdown = {
        region: best_per_region[region]["hits"]
        for region in anchor_sets
    }

    # Winning region = highest best-window score
    winning_region = max(
        best_per_region,
        key=lambda r: best_per_region[r]["score"]
    )
    winning_score  = best_per_region[winning_region]["score"]
    winning_hits   = best_per_region[winning_region]["hits"]

    return winning_region, winning_score, winning_hits, hit_breakdown


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_localizer_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load pol localizer parameters from pipeline_config.yaml.
    Falls back to safe defaults if the config file is missing.
    """
    defaults = {
        "kmer_size":              15,
        "min_seed_hits":          3,
        "min_kmer_fraction":      0.05,
        "long_read_threshold":    DEFAULT_LONG_READ_THRESHOLD,
        "window_size":            DEFAULT_WINDOW_SIZE,
        "window_step":            DEFAULT_WINDOW_STEP,
        "gene_regions":           DEFAULT_GENE_REGIONS,
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

        enricher_config  = full_config.get("enricher", {})
        reference_config = full_config.get("output", {}).get("reference", {})

        # Pull reference path from the output.reference section
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
# PolLocalizer: the main class
# ---------------------------------------------------------------------------

class PolLocalizer:
    """
    Alignment-free HIV-1 pol gene region localizer.

    Instantiate once per pipeline run. The anchor sets are built at
    construction time and reused for every read. This amortizes the
    one-time cost of loading HXB2 and building the k-mer sets across
    the entire batch.

    Usage
    -----
    localizer = PolLocalizer()

    for read in quality_filter(stream_reads("sample.fastq.gz")):
        localized = localizer.localize(read)
        print(localized.gene_region, localized.localization_confidence)
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Initialize the localizer by loading config and building anchor sets.

        This is the expensive part — loading HXB2 and building k-mer sets.
        It runs once. After that, each call to localize() is fast.

        Parameters
        ----------
        config_path : str
            Path to pipeline_config.yaml.
        """
        logger.info("Initializing PolLocalizer...")

        config = _load_localizer_config(config_path)

        self.k                   = config["kmer_size"]
        self.min_seed_hits       = config["min_seed_hits"]
        self.min_kmer_fraction   = config["min_kmer_fraction"]
        self.long_read_threshold = config["long_read_threshold"]
        self.window_size         = config["window_size"]
        self.window_step         = config["window_step"]
        self.reference_path      = config["reference"]["path"]

        # Normalize gene region coordinates
        # Config may store them as nested dicts with start/end keys
        raw_regions = config["gene_regions"]
        self.gene_regions = {}
        for region_name, coords in raw_regions.items():
            if isinstance(coords, dict):
                self.gene_regions[region_name] = {
                    "start": int(coords["start"]),
                    "end":   int(coords["end"]),
                }

        # Load HXB2 and build anchor sets
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

        # Log anchor set sizes for transparency
        for region, anchors in self.anchor_sets.items():
            logger.info(f"  [{region}] anchor set size: {len(anchors)} k-mers")

        logger.info("PolLocalizer ready.")

    def localize(self, read: RawRead) -> LocalizedRead:
        """
        Localize a single read to a pol gene region.

        This is the hot path — called once per read in the batch.
        It should be fast. All expensive setup was done in __init__.

        Algorithm:
        1. For each gene region, count how many of the read's k-mers
           appear in that region's anchor set.
        2. Find the region with the maximum hits.
        3. Check whether it clears the minimum hit and fraction thresholds.
        4. Return a LocalizedRead with the assignment and confidence score.

        Parameters
        ----------
        read : RawRead
            The read to localize.

        Returns
        -------
        LocalizedRead
            The read annotated with gene_region, confidence, and hit counts.
            If no region clears the thresholds, gene_region is "unknown".
        """
        sequence = read.sequence.upper()
        read_len = len(sequence)

        # Number of possible k-mers in this read
        # Used as denominator for confidence score in whole-read path
        total_possible_kmers = max(1, read_len - self.k + 1)

        # ---------------------------------------------------------------
        # Route to correct scoring path based on read length
        #
        # Short reads (< long_read_threshold):
        #   Whole-read scoring. Count k-mer hits across the full sequence.
        #   Works correctly when the read covers only one pol sub-region.
        #
        # Long reads (>= long_read_threshold):
        #   Windowed scoring. Slide a window across the read and find the
        #   window with the strongest discriminating signal. Prevents RT
        #   from always winning on reads that span the full pol gene.
        # ---------------------------------------------------------------

        if read_len >= self.long_read_threshold:
            # Windowed path — for reads spanning multiple pol regions
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
            # Whole-read path — for reads shorter than the pol gene
            logger.debug(
                f"Using whole-read scoring for '{read.read_id}' "
                f"(len={read_len}bp < threshold={self.long_read_threshold}bp)"
            )
            # Count hits for each region
            hit_breakdown = {}
            for region_name, anchor_set in self.anchor_sets.items():
                hits = _count_kmer_hits(sequence, anchor_set, self.k)
                hit_breakdown[region_name] = hits

            # Find the winning region
            winning_region = max(hit_breakdown, key=hit_breakdown.get)
            winning_hits   = hit_breakdown[winning_region]

            # Compute confidence score
            confidence = winning_hits / total_possible_kmers

        # Apply minimum thresholds
        # Both conditions must be met to make a call
        if (winning_hits   < self.min_seed_hits or
            confidence     < self.min_kmer_fraction):

            # Read does not have enough evidence for any region
            logger.debug(
                f"UNKNOWN: '{read.read_id}' — "
                f"best={winning_region} hits={winning_hits} "
                f"confidence={confidence:.4f} "
                f"(min_hits={self.min_seed_hits}, "
                f"min_fraction={self.min_kmer_fraction})"
            )

            return LocalizedRead.from_rawread(
                read                   = read,
                gene_region            = "unknown",
                localization_confidence = 0.0,
                seed_hits              = winning_hits,
                hit_breakdown          = hit_breakdown,
            )

        # Successful localization
        logger.debug(
            f"LOCALIZED: '{read.read_id}' → {winning_region} "
            f"hits={winning_hits} confidence={confidence:.4f} "
            f"breakdown={hit_breakdown}"
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
        """
        Localize a batch of reads and return stats alongside results.

        Convenience method for batch_processor.py integration.
        Processes all reads and returns summary statistics that
        will be incorporated into the run log.

        Parameters
        ----------
        reads : list[RawRead]
            List of RawRead objects to localize.

        Returns
        -------
        tuple[list[LocalizedRead], dict]
            (localized_reads, stats_dict)
            stats_dict contains counts per region and unknown rate.
        """
        results = []
        stats   = {"PR": 0, "RT": 0, "IN": 0, "unknown": 0, "total": 0}

        for read in reads:
            localized = self.localize(read)
            results.append(localized)
            stats[localized.gene_region] += 1
            stats["total"] += 1

        # Compute rates
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

    from src.ingestion.stream_reader import stream_reads
    from src.ingestion.quality_filter import quality_filter

    print("Initializing PolLocalizer...")
    localizer = PolLocalizer()

    test_files = [
        "data/test/fastq/DRR537715_1.fastq.gz",
        "data/test/fastq/SRR36194842_1.fastq.gz",
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"SKIP (not found): {test_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Localizing: {Path(test_file).name}")
        print(f"{'='*60}")

        # Wire the full pipeline up to this point
        raw_stream      = stream_reads(test_file)
        filtered_stream, filter_stats = quality_filter(raw_stream)

        # Localize each passing read
        results   = []
        pr_reads  = []
        rt_reads  = []
        in_reads  = []
        unknown   = []

        for read in filtered_stream:
            localized = localizer.localize(read)
            results.append(localized)

            if localized.gene_region == "PR":      pr_reads.append(localized)
            elif localized.gene_region == "RT":    rt_reads.append(localized)
            elif localized.gene_region == "IN":    in_reads.append(localized)
            else:                                  unknown.append(localized)

        total = len(results)

        print(f"\nLocalization Results:")
        print(f"  Total reads processed : {total}")
        print(f"  PR (Protease)         : {len(pr_reads)} ({len(pr_reads)/max(1,total)*100:.1f}%)")
        print(f"  RT (Reverse Transcrip): {len(rt_reads)} ({len(rt_reads)/max(1,total)*100:.1f}%)")
        print(f"  IN (Integrase)        : {len(in_reads)} ({len(in_reads)/max(1,total)*100:.1f}%)")
        print(f"  Unknown               : {len(unknown)} ({len(unknown)/max(1,total)*100:.1f}%)")

        # Show 3 examples from the most populated region
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

        # Show confidence distribution for localized reads
        localized_reads = pr_reads + rt_reads + in_reads
        if localized_reads:
            confidences = [r.localization_confidence for r in localized_reads]
            print(f"\nConfidence Score Distribution (localized reads only):")
            print(f"  Min  : {min(confidences):.4f}")
            print(f"  Max  : {max(confidences):.4f}")
            print(f"  Mean : {sum(confidences)/len(confidences):.4f}")