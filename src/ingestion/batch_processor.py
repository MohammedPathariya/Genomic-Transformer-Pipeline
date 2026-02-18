"""
src/ingestion/batch_processor.py
=================================
Orchestration layer for the Genomic-Transformer-Pipeline.

Responsibility:
    Manage running the full single-file pipeline (stream_reads →
    quality_filter → write to disk) across many files robustly.
    Handles fault isolation, progress tracking, timing, checkpointing,
    and structured logging for both sequential and parallel execution.

Position in pipeline:
    batch_processor (orchestrates) → stream_reads → quality_filter → disk

Key design principles:
    - One bad file never kills the whole run
    - Every file's result is logged whether success or failure
    - Processed reads are written to disk as they go (resumable)
    - Sequential and parallel execution share identical code paths
    - A lock guards all shared state updates (safe for parallel mode)
    - Run logs are human-inspectable JSON files

Execution modes (set in pipeline_config.yaml):
    "sequential" → one file at a time, MacBook Air M1
    "parallel"   → ProcessPoolExecutor, HPC with multiple cores

Output structure:
    data/processed/
        DRR537715_1.jsonl         ← one JSONL per source file
        SRR36194842_1.jsonl
    logs/
        run_20260218_174522.json  ← one JSON log per batch run

Author: Genomic-Transformer-Pipeline
"""

import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Path setup — ensures src/ is importable regardless of how this is run
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ingestion.quality_filter import quality_filter
from src.ingestion.stream_reader import RawRead, stream_reads

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "src/config/pipeline_config.yaml"

# File extensions the batch processor will pick up when scanning directories
SUPPORTED_EXTENSIONS = (
    ".fastq", ".fastq.gz", ".fq", ".fq.gz",
    ".fasta", ".fa",       ".fna",
    ".fasta.gz", ".fa.gz",
    ".bam",
)


# ---------------------------------------------------------------------------
# FileResult: the outcome of processing one file
# ---------------------------------------------------------------------------

@dataclass
class FileResult:
    """
    Complete record of what happened when processing a single file.

    One FileResult is created per input file, regardless of success or
    failure. These are collected into the run log JSON at the end.

    Fields
    ------
    filename : str
        The source filename (not full path). Used as the key in the log.

    status : str
        "success" or "failed"

    started_at : str
        ISO format timestamp when processing began.

    completed_at : str
        ISO format timestamp when processing finished.

    duration_seconds : float
        Wall clock time for this file.

    reads_processed : int
        Number of reads that passed quality filtering and were written.

    reads_failed : int
        Number of reads that failed quality filtering.

    output_file : Optional[str]
        Path to the JSONL output file. None if processing failed.

    filter_stats : Optional[dict]
        The FilterStats.to_dict() output for this file. None if failed.

    error_type : Optional[str]
        Exception class name if processing failed. None if success.

    error_message : Optional[str]
        Exception message if processing failed. None if success.

    error_traceback : Optional[str]
        Full traceback if processing failed. Stored in log, not console.
    """

    filename:        str
    status:          str   = "pending"
    started_at:      str   = ""
    completed_at:    str   = ""
    duration_seconds: float = 0.0
    reads_processed: int   = 0
    reads_failed:    int   = 0
    output_file:     Optional[str] = None
    filter_stats:    Optional[dict] = None
    error_type:      Optional[str] = None
    error_message:   Optional[str] = None
    error_traceback: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON logging."""
        return {
            "status":           self.status,
            "started_at":       self.started_at,
            "completed_at":     self.completed_at,
            "duration_seconds": round(self.duration_seconds, 4),
            "reads_processed":  self.reads_processed,
            "reads_failed":     self.reads_failed,
            "output_file":      self.output_file,
            "filter_stats":     self.filter_stats,
            "error_type":       self.error_type,
            "error_message":    self.error_message,
            "error_traceback":  self.error_traceback,
        }


# ---------------------------------------------------------------------------
# RunLog: the full record of an entire batch run
# ---------------------------------------------------------------------------

@dataclass
class RunLog:
    """
    Complete structured log of an entire batch processing run.

    Written to disk as a JSON file in the logs/ directory.
    Updated incrementally as each file completes — not just at the end.
    This means a partial log exists even if the run is interrupted.

    Fields
    ------
    run_id : str
        Unique identifier for this run. Format: run_YYYYMMDD_HHMMSS

    log_path : str
        Where this log file is being written on disk.

    started_at : str
        ISO timestamp when the batch run began.

    completed_at : str
        ISO timestamp when the batch run finished. Empty until done.

    execution_mode : str
        "sequential" or "parallel" — from config.

    input_dirs : list[str]
        Directories that were scanned for input files.

    total_files : int
        Total number of files found to process.

    files : dict[str, FileResult]
        Per-file results, keyed by filename.

    _lock : Lock
        Threading lock that guards all writes to shared state.
        Ensures race-condition-free updates in parallel mode.
    """

    run_id:         str
    log_path:       str
    started_at:     str        = ""
    completed_at:   str        = ""
    execution_mode: str        = "sequential"
    input_dirs:     list       = field(default_factory=list)
    total_files:    int        = 0
    files:          dict       = field(default_factory=dict)
    _lock:          Lock       = field(default_factory=Lock, repr=False)

    @property
    def successful(self) -> int:
        return sum(1 for r in self.files.values() if r.status == "success")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.files.values() if r.status == "failed")

    @property
    def total_reads_processed(self) -> int:
        return sum(r.reads_processed for r in self.files.values())

    @property
    def total_reads_failed(self) -> int:
        return sum(r.reads_failed for r in self.files.values())

    def update_file(self, result: FileResult) -> None:
        """
        Thread-safe update of a single file result.

        The lock ensures that in parallel mode, two workers cannot
        simultaneously update the shared files dict and corrupt the log.
        In sequential mode the lock is never contended but exists for
        code path consistency.
        """
        with self._lock:
            self.files[result.filename] = result
            self._write_to_disk()

    def _write_to_disk(self) -> None:
        """
        Write the current state of the run log to disk as JSON.

        Called every time a file result is updated so the log is always
        current even if the run is interrupted mid-batch.

        Called inside the lock — never call this directly from outside.
        """
        os.makedirs(Path(self.log_path).parent, exist_ok=True)

        log_dict = {
            "run_id":         self.run_id,
            "started_at":     self.started_at,
            "completed_at":   self.completed_at,
            "execution_mode": self.execution_mode,
            "input_dirs":     self.input_dirs,
            "summary": {
                "total_files":          self.total_files,
                "successful":           self.successful,
                "failed":               self.failed,
                "total_reads_processed": self.total_reads_processed,
                "total_reads_failed":   self.total_reads_failed,
            },
            "files": {
                fname: result.to_dict()
                for fname, result in self.files.items()
            },
            "checkpoint": {
                "completed_files": [
                    f for f, r in self.files.items()
                    if r.status == "success"
                ],
                "failed_files": [
                    f for f, r in self.files.items()
                    if r.status == "failed"
                ],
                "files_remaining": [
                    f for f in self.files
                    if self.files[f].status == "pending"
                ],
            },
        }

        with open(self.log_path, "w") as f:
            json.dump(log_dict, f, indent=2)

    def finalize(self) -> None:
        """Mark the run as complete and write the final log."""
        with self._lock:
            self.completed_at = datetime.now().isoformat()
            self._write_to_disk()


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_batch_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load batch processor parameters from pipeline_config.yaml.

    Returns safe defaults if the config file is missing.
    """
    defaults = {
        "output_dir":     "data/processed",
        "log_dir":        "logs",
        "execution_mode": "sequential",
        "max_workers":    4,
        "input_dirs": {
            "fastq": "data/test/fastq",
            "fasta": "data/test/fasta",
            "bam":   "data/test/bam",
        },
    }

    if not os.path.exists(config_path):
        logger.warning(
            f"Config not found at '{config_path}'. Using defaults."
        )
        return defaults

    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        ingestion_config = full_config.get("ingestion", {})
        merged = {**defaults, **ingestion_config}
        return merged

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config '{config_path}': {e}")
        return defaults


# ---------------------------------------------------------------------------
# File Discovery
# ---------------------------------------------------------------------------

def _discover_files(input_dirs: dict) -> list[str]:
    """
    Scan input directories and return all supported sequencing files.

    Parameters
    ----------
    input_dirs : dict
        Dictionary of format → directory path.
        Example: {"fastq": "data/test/fastq", "bam": "data/test/bam"}

    Returns
    -------
    list[str]
        Sorted list of full file paths. Sorted for deterministic ordering
        — important for reproducibility and checkpointing.
    """
    found = []

    for fmt, directory in input_dirs.items():
        if not os.path.exists(directory):
            logger.warning(
                f"Input directory not found, skipping: '{directory}'"
            )
            continue

        for filepath in sorted(Path(directory).iterdir()):
            name_lower = filepath.name.lower()
            if any(name_lower.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                found.append(str(filepath))
                logger.debug(f"Discovered [{fmt}]: {filepath.name}")

    logger.info(f"File discovery complete: {len(found)} files found")
    return found


def _load_checkpoint(log_dir: str) -> set[str]:
    """
    Find the most recent run log and return the set of already-completed
    filenames. Used to skip files that succeeded in a previous run.

    Parameters
    ----------
    log_dir : str
        Directory where run logs are stored.

    Returns
    -------
    set[str]
        Set of filenames that completed successfully in the last run.
        Empty set if no previous run log exists.
    """
    log_dir_path = Path(log_dir)

    if not log_dir_path.exists():
        return set()

    # Find the most recent log file
    log_files = sorted(log_dir_path.glob("run_*.json"), reverse=True)

    if not log_files:
        return set()

    most_recent = log_files[0]
    logger.info(f"Found previous run log: {most_recent.name}")

    try:
        with open(most_recent, "r") as f:
            log_data = json.load(f)

        completed = set(log_data.get("checkpoint", {}).get("completed_files", []))
        if completed:
            logger.info(
                f"Checkpoint: {len(completed)} files already completed. "
                f"These will be skipped."
            )
        return completed

    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Could not read checkpoint from {most_recent.name}: {e}")
        return set()


# ---------------------------------------------------------------------------
# RawRead Serialization
# ---------------------------------------------------------------------------

def _rawread_to_dict(read: RawRead) -> dict:
    """
    Serialize a RawRead to a plain dictionary for JSONL output.

    This is what gets written to disk for each passing read.
    The format is one JSON object per line in the .jsonl file.
    """
    return {
        "read_id":             read.read_id,
        "sequence":            read.sequence,
        "quality":             read.quality,
        "quality_is_inferred": read.quality_is_inferred,
        "source_format":       read.source_format,
        "source_file":         read.source_file,
        "raw_header":          read.raw_header,
        "length":              read.length,
        "mean_quality":        round(read.mean_quality, 2),
    }


# ---------------------------------------------------------------------------
# Single File Processor
# ---------------------------------------------------------------------------

def _process_single_file(
    filepath:    str,
    output_dir:  str,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> FileResult:
    """
    Run the full single-file pipeline on one input file.

    This function is the unit of work — it is what gets parallelized
    in parallel mode. Each worker calls this function independently on
    its assigned file. Because it is a standalone function with no
    shared mutable state, it is safe to run in parallel.

    Pipeline executed:
        stream_reads(filepath)
            → quality_filter(reads)
            → write passing reads to JSONL

    Parameters
    ----------
    filepath    : Full path to the input file.
    output_dir  : Directory where the JSONL output will be written.
    config_path : Path to pipeline_config.yaml.

    Returns
    -------
    FileResult
        Complete result record for this file, whether success or failure.
    """
    filename = Path(filepath).name
    result   = FileResult(filename=filename)
    result.started_at = datetime.now().isoformat()
    start_time = time.time()

    # Derive output JSONL path from the source filename
    # DRR537715_1.fastq.gz → data/processed/DRR537715_1.jsonl
    stem = filename
    for ext in (".fastq.gz", ".fq.gz", ".fasta.gz", ".fa.gz",
                ".fastq", ".fq", ".fasta", ".fa", ".fna", ".bam"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    output_path = os.path.join(output_dir, f"{stem}.jsonl")

    try:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Wire the pipeline
        raw_stream      = stream_reads(filepath)
        filtered_stream, stats = quality_filter(raw_stream, config_path)

        # Stream filtered reads to disk one at a time
        # Never loads all reads into memory simultaneously
        with open(output_path, "w") as out_file:
            for read in filtered_stream:
                line = json.dumps(_rawread_to_dict(read))
                out_file.write(line + "\n")

        # Populate result
        result.status          = "success"
        result.reads_processed = stats.passed
        result.reads_failed    = stats.failed
        result.output_file     = output_path
        result.filter_stats    = stats.to_dict()

    except Exception as e:
        # Catch everything — one bad file must never kill the batch
        result.status          = "failed"
        result.error_type      = type(e).__name__
        result.error_message   = str(e)
        result.error_traceback = traceback.format_exc()
        result.output_file     = None

        # Clean up partial output file if it exists
        if os.path.exists(output_path):
            os.remove(output_path)
            logger.debug(f"Cleaned up partial output: {output_path}")

    finally:
        result.completed_at      = datetime.now().isoformat()
        result.duration_seconds  = time.time() - start_time

    return result


# ---------------------------------------------------------------------------
# Console Progress Reporter
# ---------------------------------------------------------------------------

def _report_result(result: FileResult, current: int, total: int) -> None:
    """
    Print a single line to the console summarizing a file result.

    Success: green checkmark with reads and timing
    Failure: red cross with error type

    This is the only console output during a run — the full detail
    is always in the JSON log.
    """
    progress = f"[{current}/{total}]"

    if result.status == "success":
        print(
            f"{progress} ✓ {result.filename} — "
            f"{result.reads_processed} reads passed, "
            f"{result.reads_failed} failed, "
            f"{result.duration_seconds:.2f}s"
        )
    else:
        print(
            f"{progress} ✗ {result.filename} — "
            f"FAILED: {result.error_type}: {result.error_message} "
            f"({result.duration_seconds:.2f}s)"
        )


# ---------------------------------------------------------------------------
# Main Batch Processor
# ---------------------------------------------------------------------------

def run_batch(
    config_path: str  = DEFAULT_CONFIG_PATH,
    resume:      bool = True,
) -> RunLog:
    """
    Run the full ingestion pipeline across all discovered input files.

    This is the main entry point for batch processing. It orchestrates
    file discovery, checkpoint loading, sequential or parallel execution,
    progress reporting, and run log management.

    Parameters
    ----------
    config_path : str
        Path to pipeline_config.yaml.

    resume : bool
        If True, skip files that completed successfully in the most
        recent previous run. Set to False to reprocess everything.

    Returns
    -------
    RunLog
        The complete run log object. Also written to disk as JSON.

    Example Usage
    -------------
    from src.ingestion.batch_processor import run_batch

    run_log = run_batch()
    print(f"Processed {run_log.total_reads_processed} reads")
    print(f"Failed files: {run_log.failed}")
    """
    # Load config
    config         = _load_batch_config(config_path)
    output_dir     = config["output_dir"]
    log_dir        = config["log_dir"]
    execution_mode = config["execution_mode"]
    max_workers    = config["max_workers"]
    input_dirs     = config["input_dirs"]

    # Generate unique run ID from current timestamp
    run_id    = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path  = os.path.join(log_dir, f"{run_id}.json")

    # Initialize the run log
    run_log = RunLog(
        run_id         = run_id,
        log_path       = log_path,
        started_at     = datetime.now().isoformat(),
        execution_mode = execution_mode,
        input_dirs     = list(input_dirs.values()),
    )

    print(f"\n{'='*60}")
    print(f"Batch Run: {run_id}")
    print(f"Mode: {execution_mode}")
    print(f"Log: {log_path}")
    print(f"{'='*60}")

    # Discover all input files
    all_files = _discover_files(input_dirs)

    if not all_files:
        print("No input files found. Check input_dirs in pipeline_config.yaml.")
        run_log.finalize()
        return run_log

    # Load checkpoint — find files already completed in a previous run
    completed_in_previous_run = _load_checkpoint(log_dir) if resume else set()

    # Filter out already-completed files
    files_to_process = [
        f for f in all_files
        if Path(f).name not in completed_in_previous_run
    ]

    skipped_count = len(all_files) - len(files_to_process)
    if skipped_count > 0:
        print(f"Resuming: skipping {skipped_count} already-completed files")

    run_log.total_files = len(all_files)

    # Pre-populate the run log with pending status for all files
    for filepath in files_to_process:
        run_log.files[Path(filepath).name] = FileResult(
            filename = Path(filepath).name,
            status   = "pending",
        )

    print(f"Files to process: {len(files_to_process)}")
    print()

    if not files_to_process:
        print("All files already completed. Use resume=False to reprocess.")
        run_log.finalize()
        return run_log

    total = len(files_to_process)

    # -----------------------------------------------------------------------
    # Sequential execution
    # -----------------------------------------------------------------------
    if execution_mode == "sequential":
        for i, filepath in enumerate(files_to_process, start=1):

            result = _process_single_file(
                filepath    = filepath,
                output_dir  = output_dir,
                config_path = config_path,
            )

            # Update log — writes to disk immediately after each file
            run_log.update_file(result)

            # Console progress
            _report_result(result, i, total)

    # -----------------------------------------------------------------------
    # Parallel execution (HPC mode)
    # -----------------------------------------------------------------------
    elif execution_mode == "parallel":
        print(f"Parallel mode: {max_workers} workers")

        # ProcessPoolExecutor spawns separate Python processes
        # Each process is completely independent — no shared memory
        # This is why _process_single_file must be a standalone function
        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            # Submit all files to the pool at once
            # future_to_file maps each Future back to its filepath
            future_to_file = {
                executor.submit(
                    _process_single_file,
                    filepath,
                    output_dir,
                    config_path,
                ): filepath
                for filepath in files_to_process
            }

            # as_completed yields futures as they finish
            # Order of completion is not guaranteed in parallel mode
            completed_count = 0
            for future in as_completed(future_to_file):
                completed_count += 1

                try:
                    result = future.result()
                except Exception as e:
                    # This catches errors in the executor itself
                    # (not errors inside _process_single_file — those are
                    #  caught inside that function and returned as FileResult)
                    filepath = future_to_file[future]
                    filename = Path(filepath).name
                    result = FileResult(
                        filename      = filename,
                        status        = "failed",
                        error_type    = type(e).__name__,
                        error_message = str(e),
                    )

                # Thread-safe update — the lock inside update_file
                # prevents race conditions when multiple workers finish
                # at nearly the same time
                run_log.update_file(result)
                _report_result(result, completed_count, total)

    else:
        raise ValueError(
            f"Unknown execution_mode: '{execution_mode}'. "
            f"Must be 'sequential' or 'parallel'."
        )

    # Finalize
    run_log.finalize()

    # Print summary
    total_duration = (
        datetime.fromisoformat(run_log.completed_at) -
        datetime.fromisoformat(run_log.started_at)
    ).total_seconds()

    print(f"\n{'='*60}")
    print(f"Batch complete: {run_log.run_id}")
    print(f"  Successful : {run_log.successful}/{run_log.total_files} files")
    print(f"  Failed     : {run_log.failed}/{run_log.total_files} files")
    print(f"  Reads in   : {run_log.total_reads_processed}")
    print(f"  Reads out  : {run_log.total_reads_processed - run_log.total_reads_failed}")
    print(f"  Duration   : {total_duration:.2f}s")
    print(f"  Log saved  : {run_log.log_path}")
    print(f"{'='*60}\n")

    return run_log


# ---------------------------------------------------------------------------
# Quick Validation: run this file directly to test it on your data
# Usage: python -m src.ingestion.batch_processor
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    print("Running batch processor on test files...")
    print("This will process all files in data/test/fastq/")
    print("Output: data/processed/*.jsonl")
    print("Log:    logs/run_*.json")

    run_log = run_batch(resume=False)

    print(f"\nRun log written to: {run_log.log_path}")
    print("Open that file to see the full structured results.")