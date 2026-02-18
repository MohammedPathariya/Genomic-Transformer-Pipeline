"""
src/ingestion/quality_filter.py
================================
Quality control gate for the Genomic-Transformer-Pipeline.

Responsibility:
    Receive a stream of RawRead objects from stream_reader.py, apply
    five configurable quality checks, and yield only reads that pass.
    Reads that fail are not silently dropped — every failure is counted,
    categorized by reason, and available for logging.

Position in pipeline:
    stream_reads() → quality_filter() → pol_localizer()

Filters applied (in order):
    1. Enabled check    — if filtering is disabled, pass everything through
    2. Length floor     — discard reads shorter than min_read_length
    3. Length ceiling   — discard reads longer than max_read_length
    4. Quality floor    — discard reads with mean Phred below min_mean_quality
                          (skipped for FASTA reads where quality_is_inferred=True)
    5. N-base fraction  — discard reads with too many ambiguous bases

Configuration:
    All thresholds are loaded from pipeline_config.yaml.
    No magic numbers exist in this file.

Author: Genomic-Transformer-Pipeline
"""

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterator

import yaml

from src.ingestion.stream_reader import RawRead

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config path — relative to project root
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "src/config/pipeline_config.yaml"


# ---------------------------------------------------------------------------
# FilterStats: tracks what happened during a filtering run
# ---------------------------------------------------------------------------

@dataclass
class FilterStats:
    """
    A complete record of what happened during quality filtering.

    Produced by quality_filter() and returned alongside the filtered
    read stream. Used for logging, reporting, and diagnosing data quality
    issues across large batches.

    Fields
    ------
    total_reads : int
        Total number of reads that entered the filter.

    passed : int
        Number of reads that passed all checks.

    failed : int
        Total number of reads that failed at least one check.

    failure_counts : dict
        Breakdown of failures by reason.
        Keys: "too_short", "too_long", "low_quality", "high_n_fraction"
        Values: count of reads failing that specific check.

    skipped_quality_check : int
        Number of reads where the Phred quality check was skipped
        because quality_is_inferred=True (FASTA reads).

    duration_seconds : float
        How long the filtering took in seconds.
        Set by the caller after iteration is complete.

    source_file : str
        The filename being filtered — for traceability in batch logs.
    """

    total_reads:           int   = 0
    passed:                int   = 0
    failed:                int   = 0
    failure_counts:        dict  = field(default_factory=lambda: {
        "too_short":       0,
        "too_long":        0,
        "low_quality":     0,
        "high_n_fraction": 0,
    })
    skipped_quality_check: int   = 0
    duration_seconds:      float = 0.0
    source_file:           str   = ""

    @property
    def pass_rate(self) -> float:
        """Fraction of reads that passed, as a percentage."""
        if self.total_reads == 0:
            return 0.0
        return (self.passed / self.total_reads) * 100

    @property
    def fail_rate(self) -> float:
        """Fraction of reads that failed, as a percentage."""
        if self.total_reads == 0:
            return 0.0
        return (self.failed / self.total_reads) * 100

    def to_dict(self) -> dict:
        """
        Serialize to a plain dictionary for JSON logging.
        Called by batch_processor.py when writing the run log.
        """
        return {
            "total_reads":           self.total_reads,
            "passed":                self.passed,
            "failed":                self.failed,
            "pass_rate_pct":         round(self.pass_rate, 2),
            "fail_rate_pct":         round(self.fail_rate, 2),
            "failure_counts":        self.failure_counts,
            "skipped_quality_check": self.skipped_quality_check,
            "duration_seconds":      round(self.duration_seconds, 4),
            "source_file":           self.source_file,
        }

    def __repr__(self) -> str:
        return (
            f"FilterStats("
            f"total={self.total_reads}, "
            f"passed={self.passed} ({self.pass_rate:.1f}%), "
            f"failed={self.failed} ({self.fail_rate:.1f}%), "
            f"duration={self.duration_seconds:.2f}s"
            f")"
        )


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_filter_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load quality filter parameters from pipeline_config.yaml.

    Falls back to safe defaults if the config file is missing,
    so the filter never crashes due to a missing config.

    Parameters
    ----------
    config_path : str
        Path to pipeline_config.yaml, relative to project root.

    Returns
    -------
    dict
        Quality filter configuration block.
    """
    # Safe defaults — used if config file is missing or malformed
    defaults = {
        "enabled":          True,
        "min_read_length":  500,
        "max_read_length":  50000,
        "min_mean_quality": 20,
        "max_n_fraction":   0.1,
    }

    if not os.path.exists(config_path):
        logger.warning(
            f"Config file not found at '{config_path}'. "
            f"Using default filter thresholds: {defaults}"
        )
        return defaults

    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        filter_config = full_config.get("quality_filter", {})

        # Merge with defaults — config values override defaults,
        # defaults fill in anything missing from the config
        merged = {**defaults, **filter_config}

        logger.info(
            f"Quality filter config loaded from '{config_path}': "
            f"min_len={merged['min_read_length']}, "
            f"max_len={merged['max_read_length']}, "
            f"min_q={merged['min_mean_quality']}, "
            f"max_n={merged['max_n_fraction']}, "
            f"enabled={merged['enabled']}"
        )

        return merged

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config file '{config_path}': {e}")
        logger.warning("Falling back to default filter thresholds.")
        return defaults


# ---------------------------------------------------------------------------
# Core Filter Logic
# ---------------------------------------------------------------------------

def _compute_n_fraction(sequence: str) -> float:
    """
    Calculate the fraction of ambiguous N bases in a sequence.

    Parameters
    ----------
    sequence : str
        Nucleotide string, uppercase.

    Returns
    -------
    float
        Fraction of bases that are N. Range: 0.0 to 1.0.
        Returns 0.0 for empty sequences.

    Example
    -------
    "ATCGNNNATCG" → 3/11 = 0.273
    """
    if not sequence:
        return 0.0
    return sequence.count("N") / len(sequence)


def _apply_filter(
    read:            RawRead,
    min_read_length: int,
    max_read_length: int,
    min_mean_quality: float,
    max_n_fraction:  float,
    stats:           FilterStats,
) -> bool:
    """
    Apply all quality checks to a single read.

    Checks are applied in order of computational cost — cheapest first.
    The moment a read fails any check, we stop checking and return False.
    This avoids wasted computation on reads that are already going to fail.

    Parameters
    ----------
    read             : The RawRead to evaluate.
    min_read_length  : Minimum allowed read length in bases.
    max_read_length  : Maximum allowed read length in bases.
    min_mean_quality : Minimum allowed mean Phred score.
    max_n_fraction   : Maximum allowed fraction of N bases.
    stats            : FilterStats object to update in place.

    Returns
    -------
    bool
        True if the read passes all checks, False if it fails any.
    """

    # --- Check 1: Minimum length ---
    # Cheapest check — just compare an integer. Always run first.
    if read.length < min_read_length:
        stats.failure_counts["too_short"] += 1
        logger.debug(
            f"FAIL too_short: '{read.read_id}' "
            f"({read.length}bp < {min_read_length}bp minimum)"
        )
        return False

    # --- Check 2: Maximum length ---
    if read.length > max_read_length:
        stats.failure_counts["too_long"] += 1
        logger.debug(
            f"FAIL too_long: '{read.read_id}' "
            f"({read.length}bp > {max_read_length}bp maximum)"
        )
        return False

    # --- Check 3: Mean quality ---
    # Skip this check for FASTA reads — their quality is inferred (Q40)
    # and applying a real quality threshold to invented scores is meaningless.
    if read.quality_is_inferred:
        stats.skipped_quality_check += 1
        logger.debug(
            f"SKIP quality_check: '{read.read_id}' "
            f"(quality_is_inferred=True, source={read.source_format})"
        )
    else:
        if read.mean_quality < min_mean_quality:
            stats.failure_counts["low_quality"] += 1
            logger.debug(
                f"FAIL low_quality: '{read.read_id}' "
                f"(Q{read.mean_quality:.1f} < Q{min_mean_quality} minimum)"
            )
            return False

    # --- Check 4: N-base fraction ---
    # Most expensive check — requires iterating through the sequence string.
    # Run last so we only compute it for reads that passed the cheaper checks.
    n_fraction = _compute_n_fraction(read.sequence)
    if n_fraction > max_n_fraction:
        stats.failure_counts["high_n_fraction"] += 1
        logger.debug(
            f"FAIL high_n_fraction: '{read.read_id}' "
            f"({n_fraction:.1%} N bases > {max_n_fraction:.1%} maximum)"
        )
        return False

    # All checks passed
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def quality_filter(
    reads:       Iterator[RawRead],
    config_path: str = DEFAULT_CONFIG_PATH,
) -> tuple[Generator[RawRead, None, None], FilterStats]:
    """
    Filter a stream of RawRead objects by quality criteria.

    This is the single public entry point for this module.
    Called by batch_processor.py and by the test block below.

    The function returns TWO things simultaneously:
        1. A generator that yields passing reads one at a time (streaming)
        2. A FilterStats object that accumulates counts as reads flow through

    The FilterStats object is updated in real time as the generator is
    consumed. To get final counts, exhaust the generator first, then
    read the stats.

    Parameters
    ----------
    reads : Iterator[RawRead]
        Any iterable of RawRead objects — typically from stream_reads().

    config_path : str
        Path to pipeline_config.yaml.
        Defaults to DEFAULT_CONFIG_PATH.

    Returns
    -------
    tuple[Generator[RawRead], FilterStats]
        (filtered_read_stream, stats_object)

    Example Usage
    -------------
    from src.ingestion.stream_reader import stream_reads
    from src.ingestion.quality_filter import quality_filter

    raw_stream = stream_reads("data/test/fastq/DRR537715_1.fastq.gz")
    filtered_stream, stats = quality_filter(raw_stream)

    for read in filtered_stream:
        # do something with read
        pass

    print(stats)  # FilterStats(total=3848, passed=3801, failed=47, ...)
    """
    # Load config
    config = _load_filter_config(config_path)

    # Create the shared stats object
    # Both the generator and the caller share a reference to this same object
    stats = FilterStats()

    # Inner generator function
    # Defined here so it closes over config and stats
    def _filter_generator() -> Generator[RawRead, None, None]:

        start_time = time.time()

        # If filtering is disabled, pass everything through unchanged
        if not config.get("enabled", True):
            logger.warning(
                "Quality filtering is DISABLED in config. "
                "All reads will pass through unfiltered."
            )
            for read in reads:
                stats.total_reads += 1
                stats.passed      += 1
                yield read

            stats.duration_seconds = time.time() - start_time
            return

        # Filtering is enabled — extract thresholds from config
        min_read_length  = config["min_read_length"]
        max_read_length  = config["max_read_length"]
        min_mean_quality = config["min_mean_quality"]
        max_n_fraction   = config["max_n_fraction"]

        # Stream reads through the filter
        for read in reads:
            stats.total_reads += 1

            # Update source_file in stats from the first read we see
            if stats.source_file == "" and read.source_file:
                stats.source_file = read.source_file

            passed = _apply_filter(
                read             = read,
                min_read_length  = min_read_length,
                max_read_length  = max_read_length,
                min_mean_quality = min_mean_quality,
                max_n_fraction   = max_n_fraction,
                stats            = stats,
            )

            if passed:
                stats.passed += 1
                yield read
            else:
                stats.failed += 1

        stats.duration_seconds = time.time() - start_time

    return _filter_generator(), stats


# ---------------------------------------------------------------------------
# Quick Validation: run this file directly to test it on your data
# Usage: python src/ingestion/quality_filter.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import sys
    import json
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Import stream_reader — adjust path if needed
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.ingestion.stream_reader import stream_reads

    test_files = [
        "data/test/fastq/DRR537715_1.fastq.gz",
        "data/test/fastq/SRR36194842_1.fastq.gz",
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"SKIP (not found): {test_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Testing quality filter on: {Path(test_file).name}")
        print(f"{'='*60}")

        # Wire stream_reads into quality_filter
        raw_stream      = stream_reads(test_file)
        filtered_stream, stats = quality_filter(raw_stream)

        # Consume the generator — this is when filtering actually happens
        passing_reads = []
        for read in filtered_stream:
            passing_reads.append(read)

            # Print first 3 passing reads
            if len(passing_reads) <= 3:
                print(f"\nPassing Read #{len(passing_reads)}:")
                print(f"  ID          : {read.read_id}")
                print(f"  Length      : {read.length}bp")
                print(f"  Mean Quality: Q{read.mean_quality:.1f}")
                print(f"  N fraction  : {_compute_n_fraction(read.sequence):.3%}")

        # Print stats summary
        print(f"\nFilter Stats:")
        print(json.dumps(stats.to_dict(), indent=2))