#!/usr/bin/env python3
"""
scripts/fasta_to_fastq.py
==========================
Convert Stanford HIVdb ACTG clinical FASTA sequences to FASTQ format
by adding synthetic Phred quality scores.

Why synthetic quality scores are scientifically valid here:
    The ACTG sequences were produced by Sanger sequencing in the 1990s-2000s.
    Sanger sequencing has an error rate of ~0.001% (1 error per 100,000 bases),
    which corresponds to Phred Q50. We assign Q40 (1 error per 10,000 bases)
    as a conservative estimate — slightly lower than true Sanger quality
    to avoid overclaiming precision. This is standard practice when
    converting legacy Sanger sequences for tools that expect FASTQ input.

    For IUPAC ambiguity bases (R, Y, M, K, S, W, B, D, H, V), we assign Q30
    (1 error per 1,000 bases) to reflect genuine biological uncertainty —
    these bases represent mixed populations at that position, not sequencing
    error.

    Dot characters '.' at the start of sequences indicate missing/unsequenced
    regions. These are stripped before conversion.

Input format (ACTG_PR_fasta.txt):
    >PtID 23424 | Alias ACTG320_1 | Week 0
    CCTCAAATCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGG...
    >PtID 23425 | Alias ACTG320_1000 | Week 0
    ...

Output format (FASTQ):
    @PtID_23424|Alias_ACTG320_1|Week_0|Gene_PR|Trial_ACTG320
    CCTCAAATCACTCTTTGGCAACGACCCCTCGTCACAATAAAGATAGGGGGG...
    +
    IIIIIIIIIIIIIIIIIIIIIIII...

Read ID encoding:
    The read ID encodes PtID, Alias, Week, Gene, and Trial — all information
    needed by validation_pipeline.py to look up the matching row in the
    ground truth mutation table.

Usage:
    # Convert all ACTG trials (PR + RT + IN)
    python scripts/fasta_to_fastq.py \\
        --input_dir HIVdb_clinical_data/ \\
        --output_dir data/actg_fastq/ \\
        --trials ACTG5288 ACTG384 ACTG320

    # Convert a single file
    python scripts/fasta_to_fastq.py \\
        --input_file HIVdb_clinical_data/ACTG320_PR_fasta.txt \\
        --output_dir data/actg_fastq/

    # Convert all trials, all genes, compress output
    python scripts/fasta_to_fastq.py \\
        --input_dir HIVdb_clinical_data/ \\
        --output_dir data/actg_fastq/ \\
        --compress

Author: Genomic-Transformer-Pipeline
"""

import argparse
import gzip
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Quality score constants
# ---------------------------------------------------------------------------
# Phred Q40 → ASCII 73 → character 'I'
# Used for all standard unambiguous bases (A, T, G, C)
# Represents Sanger-quality sequencing (error rate ~0.01%)
SANGER_PHRED    = 40
SANGER_ASCII    = chr(SANGER_PHRED + 33)   # 'I'

# Phred Q30 → ASCII 63 → character '?'
# Used for IUPAC ambiguity codes (R, Y, M, K, S, W, B, D, H, V)
# Represents genuine biological uncertainty at that position
IUPAC_PHRED     = 30
IUPAC_ASCII     = chr(IUPAC_PHRED + 33)    # '?'

# Standard unambiguous DNA bases
STANDARD_BASES  = set("ACGTacgt")

# IUPAC ambiguity codes (excluding N which we treat differently)
IUPAC_AMBIGUOUS = set("RYMKSWBDHVrymkswbdhv")

# N bases — unknown base, assign lowest quality Q20
N_PHRED         = 20
N_ASCII         = chr(N_PHRED + 33)        # '5'

# ---------------------------------------------------------------------------
# Gene lengths for basic validation (minimum expected length)
# ---------------------------------------------------------------------------
MIN_GENE_LENGTH = {
    "PR": 150,   # PR = 297bp, accept fragments >= 150bp
    "RT": 300,   # RT = 1680bp amplicons, accept >= 300bp
    "IN": 200,   # IN = 864bp, accept >= 200bp
}


# ---------------------------------------------------------------------------
# FastaRecord dataclass
# ---------------------------------------------------------------------------
@dataclass
class FastaRecord:
    """
    A single parsed FASTA record from an ACTG clinical file.

    Fields
    ------
    ptid     : str — patient ID (e.g. "23424")
    alias    : str — sample alias (e.g. "ACTG320_1")
    week     : str — timepoint (e.g. "0", "48")
    gene     : str — "PR", "RT", or "IN"
    trial    : str — trial name (e.g. "ACTG320")
    sequence : str — cleaned nucleotide sequence (dots stripped, uppercase)
    raw_header: str — original FASTA header line
    """
    ptid:       str
    alias:      str
    week:       str
    gene:       str
    trial:      str
    sequence:   str
    raw_header: str

    @property
    def read_id(self) -> str:
        """
        Encode all metadata into the FASTQ read ID.
        Format: PtID_{ptid}|Alias_{alias}|Week_{week}|Gene_{gene}|Trial_{trial}

        This format is parsed by validation_pipeline.py to reconstruct
        the metadata for ground truth matching.
        """
        return (
            f"PtID_{self.ptid}"
            f"|Alias_{self.alias}"
            f"|Week_{self.week}"
            f"|Gene_{self.gene}"
            f"|Trial_{self.trial}"
        )

    @property
    def quality_string(self) -> str:
        """
        Per-base quality string. Each character corresponds to one base.
        - Standard bases (ACGT) → Q40 ('I')
        - IUPAC ambiguity codes → Q30 ('?')
        - N bases               → Q20 ('5')
        """
        chars = []
        for base in self.sequence:
            if base in STANDARD_BASES:
                chars.append(SANGER_ASCII)
            elif base in IUPAC_AMBIGUOUS:
                chars.append(IUPAC_ASCII)
            else:
                chars.append(N_ASCII)
        return "".join(chars)

    def to_fastq_lines(self) -> list[str]:
        """Return the four FASTQ lines for this record."""
        return [
            f"@{self.read_id}",
            self.sequence,
            "+",
            self.quality_string,
        ]

    def is_valid(self, min_length: int = 50) -> bool:
        """Basic validity check — sequence exists and meets minimum length."""
        return bool(self.sequence) and len(self.sequence) >= min_length


# ---------------------------------------------------------------------------
# Header parser
# ---------------------------------------------------------------------------
def _parse_actg_header(header: str, gene: str, trial: str) -> dict:
    """
    Parse an ACTG FASTA header line into components.

    Supported header formats:
        >PtID 23424 | Alias ACTG320_1 | Week 0
        >PtID 23424 | Alias ACTG320_1 | Week 48
        >23424 ACTG320_1 0          (older format, space separated)

    Parameters
    ----------
    header : str
        The header line without the leading '>'.
    gene : str
        "PR", "RT", or "IN" — from the filename.
    trial : str
        Trial name — from the filename.

    Returns
    -------
    dict with keys: ptid, alias, week
    """
    header = header.lstrip(">").strip()

    # Format 1: PtID 23424 | Alias ACTG320_1 | Week 0
    match = re.match(
        r"PtID\s+(\S+)\s*\|\s*Alias\s+(\S+)\s*\|\s*Week\s+(\S+)",
        header,
        re.IGNORECASE
    )
    if match:
        return {
            "ptid":  match.group(1),
            "alias": match.group(2),
            "week":  match.group(3),
        }

    # Format 2: just space-separated fields
    parts = header.split()
    if len(parts) >= 3:
        return {
            "ptid":  parts[0],
            "alias": parts[1],
            "week":  parts[2],
        }
    if len(parts) == 2:
        return {
            "ptid":  parts[0],
            "alias": parts[1],
            "week":  "unknown",
        }
    if len(parts) == 1:
        return {
            "ptid":  parts[0],
            "alias": parts[0],
            "week":  "unknown",
        }

    # Fallback
    safe = re.sub(r"[^A-Za-z0-9_]", "_", header)[:40]
    return {"ptid": safe, "alias": safe, "week": "unknown"}


def _clean_sequence(raw_sequence: str) -> str:
    """
    Clean a raw FASTA sequence for FASTQ output.

    Operations:
        1. Strip leading/trailing dots (missing sequence regions)
        2. Remove internal dots (missing bases become Ns — debatable,
           but dots are not valid FASTQ bases)
        3. Convert to uppercase
        4. Remove whitespace

    Parameters
    ----------
    raw_sequence : str
        Raw nucleotide sequence from FASTA file.

    Returns
    -------
    str
        Cleaned sequence ready for FASTQ.
    """
    seq = raw_sequence.strip().upper()

    # Remove all whitespace
    seq = re.sub(r"\s+", "", seq)

    # Strip leading and trailing dots (unsequenced flanking regions)
    seq = seq.strip(".")

    # Replace internal dots with N (unknown base)
    # Internal dots occur when the sequencing failed at interior positions
    seq = seq.replace(".", "N")

    return seq


# ---------------------------------------------------------------------------
# FASTA file parser
# ---------------------------------------------------------------------------
def parse_actg_fasta(
    fasta_path: str,
    gene: str,
    trial: str,
    min_length: int = 50,
) -> Iterator[FastaRecord]:
    """
    Parse an ACTG clinical FASTA file into FastaRecord objects.

    Parameters
    ----------
    fasta_path : str
        Path to the FASTA file (e.g. ACTG320_PR_fasta.txt).
    gene : str
        "PR", "RT", or "IN" — inferred from filename by caller.
    trial : str
        Trial name — inferred from filename by caller.
    min_length : int
        Minimum sequence length to accept. Shorter sequences are skipped.

    Yields
    ------
    FastaRecord
    """
    current_header   = None
    current_seq_parts = []
    n_parsed         = 0
    n_skipped        = 0

    def _emit(header, seq_parts):
        """Build and yield a FastaRecord from accumulated lines."""
        nonlocal n_parsed, n_skipped

        if not header:
            return None

        raw_seq = "".join(seq_parts)
        seq     = _clean_sequence(raw_seq)

        if len(seq) < min_length:
            logger.debug(
                f"Skipping short sequence in {trial}/{gene}: "
                f"header='{header[:50]}', len={len(seq)} < {min_length}"
            )
            n_skipped += 1
            return None

        meta = _parse_actg_header(header, gene, trial)
        record = FastaRecord(
            ptid       = meta["ptid"],
            alias      = meta["alias"],
            week       = meta["week"],
            gene       = gene,
            trial      = trial,
            sequence   = seq,
            raw_header = header,
        )
        n_parsed += 1
        return record

    with open(fasta_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n").rstrip("\r")

            if line.startswith(">"):
                # Emit previous record
                record = _emit(current_header, current_seq_parts)
                if record:
                    yield record
                # Start new record
                current_header    = line[1:]  # strip >
                current_seq_parts = []

            elif line:
                current_seq_parts.append(line)

        # Emit final record
        record = _emit(current_header, current_seq_parts)
        if record:
            yield record

    logger.info(
        f"  Parsed {n_parsed} sequences, skipped {n_skipped} "
        f"(too short) from {Path(fasta_path).name}"
    )


# ---------------------------------------------------------------------------
# Filename → gene + trial extractor
# ---------------------------------------------------------------------------
def _infer_gene_and_trial(filename: str) -> tuple[Optional[str], Optional[str]]:
    """
    Infer gene region and trial name from an ACTG filename.

    Examples:
        ACTG320_PR_fasta.txt  → ("PR",  "ACTG320")
        ACTG5288_RT_fasta.txt → ("RT",  "ACTG5288")
        ACTG5175_IN_fasta.txt → ("IN",  "ACTG5175")
        GART_RT_fasta.txt     → ("RT",  "GART")
        HAVANA_PR_fasta.txt   → ("PR",  "HAVANA")
        ACTGA5095_RT_fasta.txt→ ("RT",  "ACTGA5095")
    """
    name = Path(filename).stem  # strip .txt or .fasta

    # Match pattern: {trial}_{gene}_fasta
    match = re.match(
        r"^(ACTG[A-Z]?\d+|GART|HAVANA)_(PR|RT|IN)_fasta$",
        name,
        re.IGNORECASE
    )
    if match:
        return match.group(2).upper(), match.group(1).upper()

    return None, None


# ---------------------------------------------------------------------------
# FASTQ writer
# ---------------------------------------------------------------------------
def write_fastq(
    records: Iterator[FastaRecord],
    output_path: str,
    compress: bool = False,
) -> int:
    """
    Write FastaRecord objects to a FASTQ file.

    Parameters
    ----------
    records : Iterator[FastaRecord]
        Source records to write.
    output_path : str
        Output file path. If compress=True, '.gz' is appended automatically.
    compress : bool
        If True, write gzipped FASTQ.

    Returns
    -------
    int
        Number of records written.
    """
    if compress and not output_path.endswith(".gz"):
        output_path += ".gz"

    os.makedirs(Path(output_path).parent, exist_ok=True)

    n_written = 0
    opener    = gzip.open if compress else open
    mode      = "wt"

    with opener(output_path, mode) as fout:
        for record in records:
            lines = record.to_fastq_lines()
            fout.write("\n".join(lines) + "\n")
            n_written += 1

    return n_written


# ---------------------------------------------------------------------------
# Batch converter — processes a directory of ACTG files
# ---------------------------------------------------------------------------
def convert_directory(
    input_dir:  str,
    output_dir: str,
    trials:     Optional[list[str]] = None,
    genes:      Optional[list[str]] = None,
    compress:   bool = False,
    min_length: int  = 50,
) -> dict:
    """
    Convert all ACTG FASTA files in a directory to FASTQ.

    Parameters
    ----------
    input_dir  : str  — directory containing ACTG *_fasta.txt files
    output_dir : str  — directory to write FASTQ files
    trials     : list — if specified, only convert these trials
                        e.g. ["ACTG5288", "ACTG384"]
    genes      : list — if specified, only convert these genes ["PR","RT","IN"]
    compress   : bool — write .fastq.gz instead of .fastq
    min_length : int  — minimum sequence length to include

    Returns
    -------
    dict — conversion summary statistics
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        "files_processed": 0,
        "files_skipped":   0,
        "total_records":   0,
        "by_gene":         {"PR": 0, "RT": 0, "IN": 0},
        "by_trial":        {},
        "output_files":    [],
    }

    # Find all FASTA files
    fasta_files = sorted(input_path.glob("*_fasta.txt"))

    if not fasta_files:
        logger.error(f"No *_fasta.txt files found in {input_dir}")
        return stats

    logger.info(f"Found {len(fasta_files)} FASTA files in {input_dir}")

    for fasta_file in fasta_files:
        gene, trial = _infer_gene_and_trial(fasta_file.name)

        if gene is None or trial is None:
            logger.debug(f"Skipping unrecognized file: {fasta_file.name}")
            stats["files_skipped"] += 1
            continue

        # Apply trial filter
        if trials and trial not in [t.upper() for t in trials]:
            logger.debug(f"Skipping {trial} (not in trial filter)")
            stats["files_skipped"] += 1
            continue

        # Apply gene filter
        if genes and gene not in [g.upper() for g in genes]:
            logger.debug(f"Skipping {gene} (not in gene filter)")
            stats["files_skipped"] += 1
            continue

        logger.info(f"Converting: {fasta_file.name} → {trial}/{gene}")

        # Build output filename
        ext         = ".fastq.gz" if compress else ".fastq"
        output_file = output_path / f"{trial}_{gene}{ext}"

        # Parse and convert
        records   = parse_actg_fasta(str(fasta_file), gene, trial, min_length)
        n_written = write_fastq(records, str(output_file), compress)

        logger.info(f"  → {output_file.name}: {n_written} reads written")

        # Update stats
        stats["files_processed"] += 1
        stats["total_records"]   += n_written
        stats["by_gene"][gene]   = stats["by_gene"].get(gene, 0) + n_written
        stats["by_trial"][trial] = stats["by_trial"].get(trial, 0) + n_written
        stats["output_files"].append(str(output_file))

    return stats


# ---------------------------------------------------------------------------
# Single file converter
# ---------------------------------------------------------------------------
def convert_file(
    input_file:  str,
    output_dir:  str,
    compress:    bool = False,
    min_length:  int  = 50,
) -> tuple[str, int]:
    """
    Convert a single ACTG FASTA file to FASTQ.

    Returns
    -------
    tuple[str, int]
        (output_file_path, n_records_written)
    """
    fasta_path  = Path(input_file)
    gene, trial = _infer_gene_and_trial(fasta_path.name)

    if gene is None or trial is None:
        # Try to extract from filename more aggressively
        name = fasta_path.stem
        for g in ["PR", "RT", "IN"]:
            if f"_{g}_" in name.upper():
                gene = g
                trial = name.split(f"_{g}_")[0].upper()
                break

    if gene is None:
        raise ValueError(
            f"Cannot infer gene region from filename: {fasta_path.name}\n"
            f"Expected format: {{trial}}_{{gene}}_fasta.txt "
            f"(e.g. ACTG320_PR_fasta.txt)"
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ext         = ".fastq.gz" if compress else ".fastq"
    output_file = output_path / f"{trial}_{gene}{ext}"

    logger.info(f"Converting {fasta_path.name} → {output_file.name}")

    records   = parse_actg_fasta(input_file, gene, trial, min_length)
    n_written = write_fastq(records, str(output_file), compress)

    logger.info(f"Done: {n_written} reads written to {output_file}")
    return str(output_file), n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert ACTG clinical FASTA sequences to FASTQ "
            "with synthetic Sanger-quality Phred scores."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert specific trials
  python scripts/fasta_to_fastq.py \\
      --input_dir HIVdb_clinical_data/ \\
      --output_dir data/actg_fastq/ \\
      --trials ACTG5288 ACTG384 ACTG320

  # Convert single file
  python scripts/fasta_to_fastq.py \\
      --input_file HIVdb_clinical_data/ACTG320_PR_fasta.txt \\
      --output_dir data/actg_fastq/

  # Convert all, compress output, only RT sequences
  python scripts/fasta_to_fastq.py \\
      --input_dir HIVdb_clinical_data/ \\
      --output_dir data/actg_fastq/ \\
      --genes RT \\
      --compress
        """
    )

    # Input — either a single file or a directory
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input_file", "-f",
        type=str,
        help="Path to a single *_fasta.txt file to convert."
    )
    input_group.add_argument(
        "--input_dir", "-d",
        type=str,
        help="Directory containing ACTG *_fasta.txt files."
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="Directory to write output FASTQ files."
    )
    parser.add_argument(
        "--trials",
        nargs="+",
        default=None,
        help="Only convert these trials (e.g. ACTG5288 ACTG384). Default: all."
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        choices=["PR", "RT", "IN"],
        default=None,
        help="Only convert these genes. Default: all."
    )
    parser.add_argument(
        "--compress",
        action="store_true",
        default=False,
        help="Write .fastq.gz instead of .fastq."
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=50,
        help="Minimum sequence length to include. Default: 50."
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Enable debug logging."
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # -------------------------------------------------------------------
    # Single file mode
    # -------------------------------------------------------------------
    if args.input_file:
        output_file, n = convert_file(
            input_file  = args.input_file,
            output_dir  = args.output_dir,
            compress    = args.compress,
            min_length  = args.min_length,
        )
        print(f"\n✓ Converted {n} sequences → {output_file}")
        return

    # -------------------------------------------------------------------
    # Directory mode
    # -------------------------------------------------------------------
    stats = convert_directory(
        input_dir  = args.input_dir,
        output_dir = args.output_dir,
        trials     = args.trials,
        genes      = args.genes,
        compress   = args.compress,
        min_length = args.min_length,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  Files processed : {stats['files_processed']}")
    print(f"  Files skipped   : {stats['files_skipped']}")
    print(f"  Total sequences : {stats['total_records']}")
    print(f"\n  By gene:")
    for gene, n in stats["by_gene"].items():
        if n > 0:
            print(f"    {gene}: {n}")
    print(f"\n  By trial:")
    for trial, n in sorted(stats["by_trial"].items()):
        print(f"    {trial}: {n}")
    print(f"\n  Output files:")
    for f in stats["output_files"]:
        print(f"    {f}")
    print("=" * 60)


if __name__ == "__main__":
    main()