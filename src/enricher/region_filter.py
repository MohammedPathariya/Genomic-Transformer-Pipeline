"""
src/enricher/region_filter.py
==============================
Region-aware length filter for localized HIV-1 pol gene reads.

Responsibility:
    Apply per-region minimum and maximum length thresholds to LocalizedReads
    that have already been assigned to PR, RT, or IN by pol_localizer.

Why this exists — the two-stage filtering problem:
    The global quality_filter runs BEFORE pol_localizer and cannot apply
    region-specific logic because it does not know the read's gene region yet.
    If quality_filter uses min_len=500 to protect RT/IN reads, it rejects
    all PR reads (which are legitimately short — PR is only 297bp).
    If quality_filter uses min_len=150 to allow PR reads, it passes
    reads that are too short to be useful for RT or IN analysis.

    Solution: two-stage filtering.
        Stage 1 (quality_filter):   min_len=150 — catches truly garbage reads
        Stage 2 (region_filter):    per-region minimums after localization
            PR  → min 150bp  (full PR gene is 297bp)
            RT  → min 500bp  (targeted RT amplicons are 800-1500bp)
            IN  → min 500bp  (targeted IN amplicons are 400-867bp)

Position in pipeline:
    quality_filter → pol_localizer → region_filter → codon_framer

Data contract:
    Input:  LocalizedRead (from pol_localizer.py)
    Output: LocalizedRead (same object, filtered) + RejectedRead stats

Author: Genomic-Transformer-Pipeline
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Iterator, Optional, Tuple

import yaml

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "src/config/pipeline_config.yaml"

# Default per-region length thresholds
# These are used if pipeline_config.yaml does not specify region_length_filter
DEFAULT_REGION_THRESHOLDS = {
    "PR": {"min_len": 150,  "max_len": 50000},
    "RT": {"min_len": 500,  "max_len": 50000},
    "IN": {"min_len": 500,  "max_len": 50000},
}


# ---------------------------------------------------------------------------
# RejectionRecord — for tracking why reads were filtered
# ---------------------------------------------------------------------------
@dataclass
class RejectionRecord:
    """
    Records why a LocalizedRead was rejected by the region filter.

    Attributes
    ----------
    read_id : str
        The read that was rejected.
    region : str
        The assigned region (PR, RT, IN, unknown).
    read_length : int
        Actual length of the read in bases.
    reason : str
        Human-readable rejection reason.
        One of: "too_short", "too_long", "unknown_region"
    threshold : int
        The threshold that was violated.
    """
    read_id:     str
    region:      str
    read_length: int
    reason:      str
    threshold:   int


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------
def _load_region_filter_config(
    config_path: str = DEFAULT_CONFIG_PATH,
) -> Tuple[bool, dict]:
    """
    Load region-aware filter thresholds from pipeline_config.yaml.

    Returns
    -------
    tuple[bool, dict]
        (enabled, thresholds)
        enabled:    whether the filter is active
        thresholds: dict mapping region → {min_len, max_len}
    """
    if not os.path.exists(config_path):
        logger.warning(
            f"Config not found at '{config_path}'. "
            f"Using default region length thresholds."
        )
        return True, DEFAULT_REGION_THRESHOLDS.copy()

    try:
        with open(config_path) as f:
            full_config = yaml.safe_load(f)

        enricher_config = full_config.get("enricher", {})
        rf_config       = enricher_config.get("region_length_filter", {})

        enabled = rf_config.get("enabled", True)

        # Build thresholds dict from config, falling back to defaults
        thresholds = {}
        for region, defaults in DEFAULT_REGION_THRESHOLDS.items():
            region_cfg = rf_config.get(region, {})
            thresholds[region] = {
                "min_len": region_cfg.get("min_len", defaults["min_len"]),
                "max_len": region_cfg.get("max_len", defaults["max_len"]),
            }

        logger.info(
            f"Region filter config loaded: enabled={enabled}, "
            f"thresholds={{"
            f"PR:{thresholds['PR']['min_len']}-{thresholds['PR']['max_len']}bp, "
            f"RT:{thresholds['RT']['min_len']}-{thresholds['RT']['max_len']}bp, "
            f"IN:{thresholds['IN']['min_len']}-{thresholds['IN']['max_len']}bp"
            f"}}"
        )
        return enabled, thresholds

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config '{config_path}': {e}")
        return True, DEFAULT_REGION_THRESHOLDS.copy()


# ---------------------------------------------------------------------------
# RegionFilter — the main class
# ---------------------------------------------------------------------------
class RegionFilter:
    """
    Region-aware length filter for LocalizedReads.

    Applies per-region minimum and maximum length thresholds after
    pol_localizer has assigned each read to PR, RT, or IN.

    Usage
    -----
    localizer     = PolLocalizer()
    region_filter = RegionFilter()

    for read in quality_filter(stream_reads("sample.fastq.gz")):
        localized = localizer.localize(read)
        passed, rejection = region_filter.check(localized)
        if passed:
            framed = codon_framer.resolve(localized)
    """

    def __init__(self, config_path: str = DEFAULT_CONFIG_PATH):
        """
        Load config and initialise filter thresholds.

        Parameters
        ----------
        config_path : str
            Path to pipeline_config.yaml.
        """
        logger.info("Initializing RegionFilter...")
        self.enabled, self.thresholds = _load_region_filter_config(config_path)
        logger.info("RegionFilter ready.")

    def check(
        self,
        read,  # LocalizedRead — avoid circular import by not type-hinting
    ) -> Tuple[bool, Optional[RejectionRecord]]:
        """
        Check whether a LocalizedRead passes region-aware length thresholds.

        Parameters
        ----------
        read : LocalizedRead
            A read that has been assigned a gene_region by pol_localizer.

        Returns
        -------
        tuple[bool, Optional[RejectionRecord]]
            (passed, rejection_record)
            If passed=True,  rejection_record is None.
            If passed=False, rejection_record describes why the read failed.
        """
        if not self.enabled:
            return True, None

        region     = read.gene_region
        read_len   = len(read.sequence)

        # Unknown region reads — pass through to let downstream decide
        # The localizer already logged these — region_filter does not re-reject
        if region == "unknown":
            return True, None

        if region not in self.thresholds:
            logger.warning(
                f"RegionFilter: unknown region '{region}' for read "
                f"'{read.read_id}' — passing through."
            )
            return True, None

        thresholds = self.thresholds[region]
        min_len    = thresholds["min_len"]
        max_len    = thresholds["max_len"]

        # Too short
        if read_len < min_len:
            logger.debug(
                f"REJECT (too_short): '{read.read_id}' "
                f"region={region} len={read_len}bp < min={min_len}bp"
            )
            return False, RejectionRecord(
                read_id    = read.read_id,
                region     = region,
                read_length = read_len,
                reason     = "too_short",
                threshold  = min_len,
            )

        # Too long
        if read_len > max_len:
            logger.debug(
                f"REJECT (too_long): '{read.read_id}' "
                f"region={region} len={read_len}bp > max={max_len}bp"
            )
            return False, RejectionRecord(
                read_id    = read.read_id,
                region     = region,
                read_length = read_len,
                reason     = "too_long",
                threshold  = max_len,
            )

        return True, None

    def filter_stream(
        self,
        localized_reads: Iterator,
    ) -> Generator:
        """
        Filter a stream of LocalizedReads, yielding only those that pass.

        This is the primary interface for pipeline integration.
        Tracks and logs rejection statistics at the end of each batch.

        Parameters
        ----------
        localized_reads : Iterator[LocalizedRead]
            Stream of LocalizedReads from pol_localizer.

        Yields
        ------
        LocalizedRead
            Reads that pass region-aware length thresholds.
        """
        stats = {
            "total":    0,
            "passed":   0,
            "rejected": 0,
            "by_reason": {"too_short": 0, "too_long": 0},
            "by_region": {"PR": {"passed": 0, "rejected": 0},
                          "RT": {"passed": 0, "rejected": 0},
                          "IN": {"passed": 0, "rejected": 0}},
        }

        for read in localized_reads:
            stats["total"] += 1
            passed, rejection = self.check(read)

            if passed:
                stats["passed"] += 1
                if read.gene_region in stats["by_region"]:
                    stats["by_region"][read.gene_region]["passed"] += 1
                yield read
            else:
                stats["rejected"] += 1
                stats["by_reason"][rejection.reason] = (
                    stats["by_reason"].get(rejection.reason, 0) + 1
                )
                if rejection.region in stats["by_region"]:
                    stats["by_region"][rejection.region]["rejected"] += 1

        # Log summary
        total    = max(1, stats["total"])
        pass_pct = stats["passed"]   / total * 100
        rej_pct  = stats["rejected"] / total * 100

        logger.info(
            f"RegionFilter: {stats['total']} reads — "
            f"{stats['passed']} passed ({pass_pct:.1f}%), "
            f"{stats['rejected']} rejected ({rej_pct:.1f}%)"
        )
        if stats["rejected"] > 0:
            logger.info(
                f"  Rejection reasons: "
                f"too_short={stats['by_reason']['too_short']}, "
                f"too_long={stats['by_reason'].get('too_long', 0)}"
            )
            for region, counts in stats["by_region"].items():
                if counts["rejected"] > 0:
                    logger.info(
                        f"  {region}: {counts['passed']} passed, "
                        f"{counts['rejected']} rejected"
                    )
                    
    def filter_single(self, read) -> object | None:
        """
        Filter a single LocalizedRead by gene-specific length thresholds.

        This is the single-record equivalent of filter_stream().
        Used when you have one read at a time rather than a stream —
        for example in the Stanford HIVdb validation script where
        each row is processed individually.

        Parameters
        ----------
        read : LocalizedRead
            A single read that has been assigned a gene region
            by pol_localizer.

        Returns
        -------
        LocalizedRead or None
            Returns the read unchanged if it passes the length filter.
            Returns None if it is rejected (too short, too long,
            or unknown region).

        Examples
        --------
        localized = localizer.localize(raw_read)
        passed = region_filter.filter_single(localized)
        if passed is None:
            continue  # rejected
        framed = framer.resolve(passed)
        """
        if read.gene_region == "unknown":
            return None

        thresholds = self.thresholds.get(read.gene_region)
        if thresholds is None:
            return None

        seq_len = len(read.sequence)
        min_len = thresholds.get("min", 0)
        max_len = thresholds.get("max", float("inf"))

        if seq_len < min_len or seq_len > max_len:
            logger.debug(
                f"filter_single rejected '{read.read_id}': "
                f"region={read.gene_region}, "
                f"len={seq_len} outside [{min_len}, {max_len}]"
            )
            return None

        return read
    
    def get_thresholds(self) -> dict:
        """Return the current per-region thresholds (for logging/debugging)."""
        return self.thresholds.copy()


# ---------------------------------------------------------------------------
# Quick Validation
# Usage: python -m src.enricher.region_filter
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    from src.ingestion.stream_reader  import stream_reads
    from src.ingestion.quality_filter import quality_filter
    from src.enricher.pol_localizer   import PolLocalizer

    localizer     = PolLocalizer()
    region_filter = RegionFilter()

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
        print(f"Region filtering: {Path(test_file).name}")
        print(f"{'='*60}")

        raw_stream              = stream_reads(test_file)
        filtered_stream, _      = quality_filter(raw_stream)
        localized               = (localizer.localize(r) for r in filtered_stream)
        passed_stream           = region_filter.filter_stream(localized)

        results    = list(passed_stream)
        regions    = {}
        for r in results:
            regions[r.gene_region] = regions.get(r.gene_region, 0) + 1

        print(f"\n  Reads after region filter: {len(results)}")
        for region, count in sorted(regions.items()):
            print(f"    {region}: {count}")

        if results:
            lengths = [len(r.sequence) for r in results]
            print(f"\n  Length distribution:")
            print(f"    Min:  {min(lengths)}bp")
            print(f"    Max:  {max(lengths)}bp")
            print(f"    Mean: {sum(lengths)//len(lengths)}bp")