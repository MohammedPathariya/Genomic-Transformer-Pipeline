"""
src/ingestion/basecaller.py
============================
Thin wrapper around Oxford Nanopore's Dorado basecaller.

Responsibility:
    Accept a POD5 file path, invoke Dorado as a subprocess, and return
    the path to the resulting FASTQ file. This is the only file in the
    pipeline that knows anything about POD5 or Dorado.

Position in pipeline:
    POD5 file → basecaller.py → FASTQ file → stream_reader.py → ...

Why a subprocess and not a Python library:
    Dorado is a standalone binary written in C++/CUDA. Oxford Nanopore
    does not provide a Python API for it. The only way to call it from
    Python is via subprocess. This is the standard approach used by
    every Nanopore pipeline including Nextflow-based ones.

Why this is a separate file and not inside stream_reader.py:
    Single responsibility principle. stream_reader.py reads sequences.
    basecaller.py converts signals to sequences. These are different
    concerns. Dorado also requires a GPU and specific model files —
    keeping it isolated means the rest of the pipeline can run without
    Dorado installed if you already have FASTQ files.

Dorado installation:
    Download from: https://github.com/nanoporetech/dorado/releases
    After downloading, set the dorado_executable path in
    pipeline_config.yaml under the basecalling section.

Dorado models:
    Dorado requires a basecalling model directory. Download with:
        dorado download --model dna_r10.4.1_e8.2_400bps_hac@v4.3.0
    Set the model path in pipeline_config.yaml.

Author: Genomic-Transformer-Pipeline
"""

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# Module-level logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_PATH = "src/config/pipeline_config.yaml"

# Expected output FASTQ extension from Dorado
DORADO_OUTPUT_EXT = ".fastq"


# ---------------------------------------------------------------------------
# BasecallResult: the outcome of a single basecalling run
# ---------------------------------------------------------------------------

@dataclass
class BasecallResult:
    """
    Complete record of a single Dorado basecalling run.

    Fields
    ------
    pod5_file : str
        Path to the input POD5 file.

    fastq_file : Optional[str]
        Path to the output FASTQ file. None if basecalling failed.

    status : str
        "success" or "failed"

    duration_seconds : float
        Wall clock time for the basecalling run.

    started_at : str
        ISO format timestamp when basecalling began.

    completed_at : str
        ISO format timestamp when basecalling finished.

    dorado_version : Optional[str]
        Dorado version string from `dorado --version`. None if not found.

    error_message : Optional[str]
        Error message if basecalling failed. None if success.

    reads_basecalled : Optional[int]
        Number of reads in the output FASTQ. None if failed.
        Computed by counting lines and dividing by 4 (FASTQ structure).
    """

    pod5_file:        str
    fastq_file:       Optional[str]  = None
    status:           str            = "pending"
    duration_seconds: float          = 0.0
    started_at:       str            = ""
    completed_at:     str            = ""
    dorado_version:   Optional[str]  = None
    error_message:    Optional[str]  = None
    reads_basecalled: Optional[int]  = None

    def to_dict(self) -> dict:
        """Serialize to plain dict for JSON logging."""
        return {
            "pod5_file":        self.pod5_file,
            "fastq_file":       self.fastq_file,
            "status":           self.status,
            "duration_seconds": round(self.duration_seconds, 4),
            "started_at":       self.started_at,
            "completed_at":     self.completed_at,
            "dorado_version":   self.dorado_version,
            "error_message":    self.error_message,
            "reads_basecalled": self.reads_basecalled,
        }


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_basecall_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """
    Load basecalling parameters from pipeline_config.yaml.

    Returns safe defaults if the config file is missing.
    """
    defaults = {
        "dorado_executable": "dorado",
        "dorado_model":      "dna_r10.4.1_e8.2_400bps_hac@v4.3.0",
        "output_dir":        "data/basecalled",
        "device":            "cpu",
        "batch_size":        64,
        "emit_fastq":        True,
    }

    if not os.path.exists(config_path):
        logger.warning(
            f"Config not found at '{config_path}'. Using default basecall settings."
        )
        return defaults

    try:
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)

        basecall_config = full_config.get("basecalling", {})
        merged = {**defaults, **basecall_config}

        logger.info(
            f"Basecall config loaded: "
            f"executable='{merged['dorado_executable']}', "
            f"model='{merged['dorado_model']}', "
            f"device='{merged['device']}'"
        )

        return merged

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config '{config_path}': {e}")
        return defaults


# ---------------------------------------------------------------------------
# Dorado Availability Check
# ---------------------------------------------------------------------------

def check_dorado_available(dorado_executable: str = "dorado") -> tuple[bool, Optional[str]]:
    """
    Check whether Dorado is installed and return its version.

    This is called at the start of any basecalling run so failures
    are caught early with a clear error message rather than a
    cryptic subprocess error.

    Parameters
    ----------
    dorado_executable : str
        Path to the Dorado binary, or just "dorado" if it is on PATH.

    Returns
    -------
    tuple[bool, Optional[str]]
        (is_available, version_string)
        is_available is True if Dorado is found and executable.
        version_string is the version number, or None if not found.
    """
    try:
        result = subprocess.run(
            [dorado_executable, "--version"],
            capture_output = True,
            text           = True,
            timeout        = 10,
        )

        if result.returncode == 0:
            # Dorado version is printed to stderr, not stdout
            version = (result.stderr or result.stdout).strip().split("\n")[0]
            logger.info(f"Dorado found: {version}")
            return True, version
        else:
            logger.warning(f"Dorado returned non-zero exit code: {result.returncode}")
            return False, None

    except FileNotFoundError:
        logger.error(
            f"Dorado executable not found: '{dorado_executable}'\n"
            f"Download from: https://github.com/nanoporetech/dorado/releases\n"
            f"Then set 'dorado_executable' in src/config/pipeline_config.yaml"
        )
        return False, None

    except subprocess.TimeoutExpired:
        logger.error("Dorado version check timed out after 10 seconds.")
        return False, None


# ---------------------------------------------------------------------------
# Output Read Counter
# ---------------------------------------------------------------------------

def _count_fastq_reads(fastq_path: str) -> Optional[int]:
    """
    Count the number of reads in a FASTQ file by counting lines
    and dividing by 4 (each FASTQ record is exactly 4 lines).

    Parameters
    ----------
    fastq_path : str
        Path to the FASTQ file.

    Returns
    -------
    Optional[int]
        Number of reads, or None if the file cannot be read.
    """
    try:
        with open(fastq_path, "r") as f:
            line_count = sum(1 for _ in f)
        return line_count // 4
    except Exception as e:
        logger.warning(f"Could not count reads in '{fastq_path}': {e}")
        return None


# ---------------------------------------------------------------------------
# Core Basecalling Function
# ---------------------------------------------------------------------------

def basecall_pod5(
    pod5_path:   str,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir:  Optional[str] = None,
) -> BasecallResult:
    """
    Run Dorado basecalling on a single POD5 file.

    Constructs and executes the Dorado command as a subprocess,
    streams stderr output to the logger in real time so you can
    monitor progress, and returns a BasecallResult with the path
    to the output FASTQ file.

    The Dorado command constructed is:
        dorado basecaller <model> <pod5_path> --emit-fastq > <output.fastq>

    Parameters
    ----------
    pod5_path : str
        Path to the input POD5 file.

    config_path : str
        Path to pipeline_config.yaml.

    output_dir : Optional[str]
        Directory to write the output FASTQ. If None, uses the
        output_dir from pipeline_config.yaml.

    Returns
    -------
    BasecallResult
        Complete result record whether success or failure.

    Example Usage
    -------------
    from src.ingestion.basecaller import basecall_pod5
    from src.ingestion.stream_reads import stream_reads

    result = basecall_pod5("data/raw/patient_001.pod5")

    if result.status == "success":
        for read in stream_reads(result.fastq_file):
            print(read.read_id, read.mean_quality)
    """
    result            = BasecallResult(pod5_file=pod5_path)
    result.started_at = datetime.now().isoformat()
    start_time        = time.time()

    # Validate input file exists
    if not os.path.exists(pod5_path):
        result.status        = "failed"
        result.error_message = f"POD5 file not found: '{pod5_path}'"
        result.completed_at  = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time
        logger.error(result.error_message)
        return result

    # Load config
    config         = _load_basecall_config(config_path)
    dorado_exe     = config["dorado_executable"]
    dorado_model   = config["dorado_model"]
    device         = config["device"]
    batch_size     = config["batch_size"]
    out_dir        = output_dir or config["output_dir"]

    # Check Dorado is available
    available, version = check_dorado_available(dorado_exe)
    result.dorado_version = version

    if not available:
        result.status        = "failed"
        result.error_message = (
            f"Dorado not available at '{dorado_exe}'. "
            f"Install from: https://github.com/nanoporetech/dorado/releases"
        )
        result.completed_at     = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time
        return result

    # Derive output FASTQ path from POD5 filename
    # patient_001.pod5 → data/basecalled/patient_001.fastq
    pod5_stem   = Path(pod5_path).stem
    os.makedirs(out_dir, exist_ok=True)
    fastq_path  = os.path.join(out_dir, f"{pod5_stem}{DORADO_OUTPUT_EXT}")

    # Build the Dorado command
    # dorado basecaller <model> <pod5> --emit-fastq --device <cpu/cuda> > output.fastq
    command = [
        dorado_exe,
        "basecaller",
        dorado_model,
        pod5_path,
        "--emit-fastq",
        "--device", device,
        "--batchsize", str(batch_size),
    ]

    logger.info(f"Starting basecalling: {Path(pod5_path).name}")
    logger.info(f"Command: {' '.join(command)} > {fastq_path}")

    try:
        # Open the output FASTQ file for writing
        # Dorado writes basecalled reads to stdout — we redirect to file
        with open(fastq_path, "w") as fastq_out:

            process = subprocess.Popen(
                command,
                stdout = fastq_out,       # reads go to FASTQ file
                stderr = subprocess.PIPE, # progress/logs go to stderr
                text   = True,
            )

            # Stream stderr in real time so progress is visible in logs
            # This is important for long basecalling runs — without this
            # the process appears frozen until it completes
            for line in process.stderr:
                line = line.strip()
                if line:
                    logger.info(f"[dorado] {line}")

            # Wait for the process to complete
            process.wait()

        # Check exit code
        if process.returncode != 0:
            result.status        = "failed"
            result.error_message = f"Dorado exited with code {process.returncode}"
            logger.error(result.error_message)

            # Clean up empty or partial output file
            if os.path.exists(fastq_path):
                os.remove(fastq_path)
                logger.debug(f"Cleaned up partial output: {fastq_path}")

        else:
            # Success — verify output file exists and has content
            if not os.path.exists(fastq_path) or os.path.getsize(fastq_path) == 0:
                result.status        = "failed"
                result.error_message = (
                    f"Dorado completed but output FASTQ is empty: '{fastq_path}'. "
                    f"Check that the POD5 file contains valid reads and the "
                    f"model is compatible with the flow cell chemistry."
                )
                logger.error(result.error_message)

            else:
                result.status          = "success"
                result.fastq_file      = fastq_path
                result.reads_basecalled = _count_fastq_reads(fastq_path)

                logger.info(
                    f"Basecalling complete: {result.reads_basecalled} reads → {fastq_path}"
                )

    except Exception as e:
        result.status        = "failed"
        result.error_message = str(e)
        logger.error(f"Basecalling failed with exception: {e}")

        # Clean up partial output
        if os.path.exists(fastq_path):
            os.remove(fastq_path)

    finally:
        result.completed_at     = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time

    return result


# ---------------------------------------------------------------------------
# Batch POD5 Basecalling
# ---------------------------------------------------------------------------

def basecall_directory(
    pod5_dir:    str,
    config_path: str = DEFAULT_CONFIG_PATH,
    output_dir:  Optional[str] = None,
) -> list[BasecallResult]:
    """
    Run Dorado basecalling on all POD5 files in a directory.

    Processes files sequentially. On HPC this function can be
    replaced with parallel execution following the same pattern
    as batch_processor.py.

    Parameters
    ----------
    pod5_dir : str
        Directory containing POD5 files.

    config_path : str
        Path to pipeline_config.yaml.

    output_dir : Optional[str]
        Where to write output FASTQ files.

    Returns
    -------
    list[BasecallResult]
        One result per POD5 file found.
    """
    pod5_files = sorted(Path(pod5_dir).glob("*.pod5"))

    if not pod5_files:
        logger.warning(f"No POD5 files found in '{pod5_dir}'")
        return []

    logger.info(f"Found {len(pod5_files)} POD5 files in '{pod5_dir}'")

    results = []
    for i, pod5_path in enumerate(pod5_files, start=1):
        print(f"[{i}/{len(pod5_files)}] Basecalling: {pod5_path.name}")
        result = basecall_pod5(str(pod5_path), config_path, output_dir)
        results.append(result)

        if result.status == "success":
            print(
                f"  ✓ {result.reads_basecalled} reads → "
                f"{Path(result.fastq_file).name} "
                f"({result.duration_seconds:.1f}s)"
            )
        else:
            print(f"  ✗ FAILED: {result.error_message}")

    successful = sum(1 for r in results if r.status == "success")
    print(f"\nBasecalling complete: {successful}/{len(pod5_files)} files succeeded")

    return results


# ---------------------------------------------------------------------------
# Quick Validation
# Usage: python -m src.ingestion.basecaller
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    import json
    logging.basicConfig(
        level  = logging.INFO,
        format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Check if Dorado is available on this machine
    print("Checking Dorado availability...")
    config = _load_basecall_config()
    available, version = check_dorado_available(config["dorado_executable"])

    if available:
        print(f"✓ Dorado is installed: {version}")
        print("  To basecall a POD5 file:")
        print("  from src.ingestion.basecaller import basecall_pod5")
        print("  result = basecall_pod5('path/to/sample.pod5')")
        print("  print(result.fastq_file)  # path to output FASTQ")
    else:
        print("✗ Dorado is not installed or not on PATH.")
        print("  Download from: https://github.com/nanoporetech/dorado/releases")
        print("  After installing, set 'dorado_executable' in:")
        print("  src/config/pipeline_config.yaml → basecalling.dorado_executable")
        print()
        print("  The rest of the pipeline works without Dorado if you already")
        print("  have FASTQ files. Dorado is only needed for raw POD5 inputs.")