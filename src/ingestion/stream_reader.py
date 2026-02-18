"""
src/ingestion/stream_reader.py
==============================
The front door of the Genomic-Transformer-Pipeline.

Responsibility:
    Read raw sequencing files (FASTQ, FASTA, BAM) and convert every read
    into a standardized RawRead dataclass. This is the ONLY file in the
    entire codebase that knows anything about file formats. Everything
    downstream is format-blind.

Supported formats:
    - FASTQ (.fastq, .fastq.gz)   : Basecalled Nanopore reads from ENA/SRA
    - FASTA (.fasta, .fa, .fna)   : Clean consensus sequences from LANL/GenBank
    - BAM   (.bam)                 : Pre-aligned reads, used for validation only

Data Contract:
    Every read that exits this module is a RawRead object with guaranteed
    fields. No downstream module ever needs to handle raw file formats.

Author: Genomic-Transformer-Pipeline
"""

import gzip
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Optional

import pysam
from Bio import SeqIO

# ---------------------------------------------------------------------------
# Module-level logger
# Every message from this file will be prefixed with "stream_reader"
# so you can trace exactly where in the pipeline a log came from.
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# When a FASTA read has no quality scores, we assign this value to every base.
# Q40 means a 0.01% chance of error — appropriate for curated LANL sequences.
FASTA_DEFAULT_QUALITY: int = 40

# File extensions we recognise for each format.
# Lowercase only — we normalise the extension before checking.
FASTQ_EXTENSIONS: tuple = (".fastq", ".fastq.gz", ".fq", ".fq.gz")
FASTA_EXTENSIONS: tuple = (".fasta", ".fa", ".fna", ".fasta.gz", ".fa.gz")
BAM_EXTENSIONS:   tuple = (".bam",)


# ---------------------------------------------------------------------------
# The Data Contract: RawRead Dataclass
# ---------------------------------------------------------------------------

@dataclass
class RawRead:
    """
    A single sequencing read in standardized format.

    This is the data contract between the ingestion layer and everything
    downstream. Every module after stream_reader.py only ever sees RawRead
    objects — never raw FASTQ, FASTA, or BAM records.

    Fields
    ------
    read_id : str
        Normalized read identifier. Always the first whitespace-delimited
        token from the original header. Consistent and usable as a dict key.
        Example: "DRR537715.1" or "a4761c2a-7c4d-4f5d-be57-e8d82d7c6037"

    sequence : str
        The nucleotide sequence string. Contains only A, T, C, G, N.
        Uppercase always.
        Example: "GTATTGCTAAGGTTAACACAAAG..."

    quality : list[int]
        Per-base Phred quality scores as integers.
        Same length as sequence — one score per base.
        Range: 0 (worst) to 93 (best). Typical Nanopore range: 6–30.
        For FASTA reads: every value is FASTA_DEFAULT_QUALITY (40).
        For BAM reads: decoded from the BAM quality encoding.
        Example: [6, 7, 11, 18, 29, 26, 24, 24]

    quality_is_inferred : bool
        True  → quality scores were ASSIGNED, not measured.
                 This happens for FASTA reads which carry no quality data.
                 Downstream modules should NOT apply quality-based filtering
                 to reads where this is True.
        False → quality scores came directly from the file.

    source_format : str
        The format of the file this read came from.
        One of: "fastq", "fasta", "bam"

    source_file : str
        The filename (not full path) of the original file.
        Used for traceability in logs and reports.
        Example: "DRR537715_1.fastq.gz"

    raw_header : str
        The complete original header line, preserved exactly.
        For FASTQ: everything after the @ symbol.
        For FASTA: everything after the > symbol.
        For BAM:   the query name field.
        Used for debugging and for reconstructing chimeric read metadata.
        Example: "DRR537715.1 seq000001/1"
    """

    read_id:             str
    sequence:            str
    quality:             list
    quality_is_inferred: bool
    source_format:       str
    source_file:         str
    raw_header:          str

    def __post_init__(self):
        """
        Validation that runs automatically after the dataclass is created.
        Catches malformed reads at the point of creation, not later.
        """
        # Sequence and quality must always be the same length.
        # If they differ, the file is malformed or our parser has a bug.
        if len(self.sequence) != len(self.quality):
            raise ValueError(
                f"Read '{self.read_id}': sequence length ({len(self.sequence)}) "
                f"does not match quality length ({len(self.quality)}). "
                f"Source: {self.source_file}"
            )

        # Sequence should only contain valid IUPAC nucleotide characters.
        # N is allowed (ambiguous base). Lowercase is normalised to upper.
        valid_bases = set("ATCGN")
        invalid = set(self.sequence.upper()) - valid_bases
        if invalid:
            logger.warning(
                f"Read '{self.read_id}' contains non-standard bases: {invalid}. "
                f"These will be treated as N by downstream modules."
            )

        # source_format must be one of our three supported types.
        if self.source_format not in ("fastq", "fasta", "bam"):
            raise ValueError(
                f"source_format must be 'fastq', 'fasta', or 'bam'. "
                f"Got: '{self.source_format}'"
            )

    @property
    def length(self) -> int:
        """Convenience property: number of bases in this read."""
        return len(self.sequence)

    @property
    def mean_quality(self) -> float:
        """
        Convenience property: average Phred quality score across all bases.
        Used by quality_filter.py to decide whether to keep or discard a read.
        """
        if not self.quality:
            return 0.0
        return sum(self.quality) / len(self.quality)

    def __repr__(self) -> str:
        """Clean string representation for logging and debugging."""
        return (
            f"RawRead("
            f"id='{self.read_id}', "
            f"len={self.length}, "
            f"mean_q={self.mean_quality:.1f}, "
            f"format='{self.source_format}', "
            f"inferred_q={self.quality_is_inferred}"
            f")"
        )


# ---------------------------------------------------------------------------
# Internal Helper Functions
# (prefixed with _ meaning: used inside this file only, not imported elsewhere)
# ---------------------------------------------------------------------------

def _normalize_read_id(raw_header: str) -> str:
    """
    Extract the normalized read ID from a raw header string.

    Rule: take everything up to the first whitespace character.
    This makes read IDs consistent across all file formats.

    Examples
    --------
    "DRR537715.1 seq000001/1"
        → "DRR537715.1"

    "a4761c2a-7c4d-4f5d-be57-e8d82d7c6037;24e78cbc-0017.../1"
        → "a4761c2a-7c4d-4f5d-be57-e8d82d7c6037;24e78cbc-0017.../1"
        (no whitespace in this header so the whole thing is the ID)

    "B.US.2012.12531 pol gene subtype B"
        → "B.US.2012.12531"
    """
    return raw_header.strip().split()[0] if raw_header.strip() else "unknown"


def _phred_string_to_int_list(phred_string: str) -> list:
    """
    Convert a FASTQ quality string into a list of integer Phred scores.

    FASTQ files encode quality scores as ASCII characters using the formula:
        Phred score = ASCII value of character - 33

    This is called Phred+33 or Sanger encoding, which is the universal
    standard for modern sequencing data.

    Example
    -------
    The character "'" has ASCII value 39.
    39 - 33 = 6  →  Q6  →  ~25% error probability  (poor quality)

    The character "S" has ASCII value 83.
    83 - 33 = 50  →  Q50  →  ~0.001% error probability  (excellent quality)

    Parameters
    ----------
    phred_string : str
        The raw quality string from line 4 of a FASTQ record.

    Returns
    -------
    list[int]
        Integer Phred scores, one per base.
    """
    return [ord(char) - 33 for char in phred_string]


def _detect_format(filepath: str) -> str:
    """
    Determine the file format from the file extension.

    We lowercase the full filename before checking so that
    .FASTQ, .Fastq, .fastq all match correctly.

    Parameters
    ----------
    filepath : str
        Path to the input file.

    Returns
    -------
    str
        One of: "fastq", "fasta", "bam"

    Raises
    ------
    ValueError
        If the extension is not recognised.
    """
    # Convert to lowercase for case-insensitive matching
    name = filepath.lower()

    if any(name.endswith(ext) for ext in FASTQ_EXTENSIONS):
        return "fastq"
    elif any(name.endswith(ext) for ext in FASTA_EXTENSIONS):
        return "fasta"
    elif any(name.endswith(ext) for ext in BAM_EXTENSIONS):
        return "bam"
    else:
        raise ValueError(
            f"Unrecognised file format for: '{filepath}'\n"
            f"Supported extensions:\n"
            f"  FASTQ: {FASTQ_EXTENSIONS}\n"
            f"  FASTA: {FASTA_EXTENSIONS}\n"
            f"  BAM:   {BAM_EXTENSIONS}"
        )


# ---------------------------------------------------------------------------
# Format-Specific Parsers
# Each parser is a Python generator — it yields one RawRead at a time
# rather than loading the entire file into memory.
# This is critical for large files and for streaming applications.
# ---------------------------------------------------------------------------

def _parse_fastq(filepath: str) -> Generator[RawRead, None, None]:
    """
    Parse a FASTQ file and yield RawRead objects one at a time.

    Handles both plain .fastq and gzip-compressed .fastq.gz files.
    Uses BioPython's SeqIO for robust parsing that handles edge cases
    including unusual header formats like the UUID chimeric reads
    we observed in SRR36194842.

    Parameters
    ----------
    filepath : str
        Path to the FASTQ file (plain or gzip-compressed).

    Yields
    ------
    RawRead
        One RawRead per record in the file.
    """
    source_file = Path(filepath).name
    logger.info(f"Parsing FASTQ: {source_file}")

    # BioPython's SeqIO handles gzip transparently when we open the file
    # ourselves and pass the handle. We detect gzip by extension.
    is_gzipped = filepath.lower().endswith(".gz")

    read_count = 0

    try:
        # Open the file — gzip or plain depending on extension
        handle = gzip.open(filepath, "rt") if is_gzipped else open(filepath, "r")

        with handle:
            # SeqIO.parse returns a lazy iterator — reads are loaded one at
            # a time, not all at once. This is the streaming behaviour we want.
            for record in SeqIO.parse(handle, "fastq"):

                # record.id      → everything up to first space in header
                # record.description → the full header line (minus the @)
                # record.seq     → the sequence as a Seq object
                # record.letter_annotations["phred_quality"] → list of ints

                raw_header = record.description  # full header, no @ symbol
                read_id    = _normalize_read_id(raw_header)
                sequence   = str(record.seq).upper()

                # BioPython already decodes the Phred string to integers for us
                quality = record.letter_annotations.get("phred_quality", [])

                # Safety check: if quality is empty something went wrong
                if not quality:
                    logger.warning(
                        f"Read '{read_id}' has no quality scores. Skipping."
                    )
                    continue

                yield RawRead(
                    read_id             = read_id,
                    sequence            = sequence,
                    quality             = quality,
                    quality_is_inferred = False,   # quality came from the file
                    source_format       = "fastq",
                    source_file         = source_file,
                    raw_header          = raw_header,
                )

                read_count += 1

    except FileNotFoundError:
        logger.error(f"FASTQ file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error parsing FASTQ '{filepath}': {e}")
        raise

    logger.info(f"FASTQ parsing complete: {read_count} reads from {source_file}")


def _parse_fasta(filepath: str) -> Generator[RawRead, None, None]:
    """
    Parse a FASTA file and yield RawRead objects one at a time.

    FASTA files have no quality scores. We assign FASTA_DEFAULT_QUALITY (Q40)
    to every base and set quality_is_inferred=True so downstream modules
    know not to apply quality-based filtering to these reads.

    Parameters
    ----------
    filepath : str
        Path to the FASTA file (plain or gzip-compressed).

    Yields
    ------
    RawRead
        One RawRead per record in the file.
    """
    source_file = Path(filepath).name
    logger.info(f"Parsing FASTA: {source_file}")

    is_gzipped = filepath.lower().endswith(".gz")
    read_count = 0

    try:
        handle = gzip.open(filepath, "rt") if is_gzipped else open(filepath, "r")

        with handle:
            for record in SeqIO.parse(handle, "fasta"):

                raw_header = record.description  # full header, no > symbol
                read_id    = _normalize_read_id(raw_header)
                sequence   = str(record.seq).upper()

                # FASTA has no quality scores — assign inferred Q40 for every base
                # One score per base, same length as the sequence
                quality = [FASTA_DEFAULT_QUALITY] * len(sequence)

                yield RawRead(
                    read_id             = read_id,
                    sequence            = sequence,
                    quality             = quality,
                    quality_is_inferred = True,    # quality was assigned, not measured
                    source_format       = "fasta",
                    source_file         = source_file,
                    raw_header          = raw_header,
                )

                read_count += 1

    except FileNotFoundError:
        logger.error(f"FASTA file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error parsing FASTA '{filepath}': {e}")
        raise

    logger.info(f"FASTA parsing complete: {read_count} reads from {source_file}")


def _parse_bam(filepath: str) -> Generator[RawRead, None, None]:
    """
    Parse a BAM file and yield RawRead objects one at a time.

    BAM files are binary and require pysam to read. Each aligned read
    in a BAM file contains the original sequence and quality scores,
    plus alignment information (which we discard — we only want the
    sequence and quality for our alignment-free pipeline).

    Parameters
    ----------
    filepath : str
        Path to the BAM file. An index (.bam.bai) must exist alongside it.

    Yields
    ------
    RawRead
        One RawRead per primary alignment in the file.
        Secondary and supplementary alignments are skipped to avoid
        counting the same read multiple times.
    """
    source_file = Path(filepath).name
    logger.info(f"Parsing BAM: {source_file}")

    read_count   = 0
    skipped_count = 0

    try:
        # pysam.AlignmentFile opens BAM files
        # "rb" means: read, binary (b for BAM vs "r" for SAM text)
        with pysam.AlignmentFile(filepath, "rb") as bam:

            for read in bam.fetch(until_eof=True):

                # Skip secondary alignments (flag 256) and supplementary
                # alignments (flag 2048). These are split or alternative
                # mappings of the same read — we only want the primary record.
                if read.is_secondary or read.is_supplementary:
                    skipped_count += 1
                    continue

                # Skip reads with no sequence (can happen in some BAM files)
                if read.query_sequence is None:
                    logger.warning(
                        f"BAM read '{read.query_name}' has no sequence. Skipping."
                    )
                    skipped_count += 1
                    continue

                raw_header = read.query_name  # BAM has no description, just name
                read_id    = _normalize_read_id(raw_header)
                sequence   = read.query_sequence.upper()

                # BAM quality scores are stored as integers 0-93
                # pysam returns them as a tuple — we convert to list
                quality = list(read.query_qualities) if read.query_qualities else []

                # If quality is missing from the BAM, assign inferred scores
                quality_is_inferred = False
                if not quality:
                    quality = [FASTA_DEFAULT_QUALITY] * len(sequence)
                    quality_is_inferred = True
                    logger.warning(
                        f"BAM read '{read_id}' has no quality scores. "
                        f"Assigning inferred Q{FASTA_DEFAULT_QUALITY}."
                    )

                yield RawRead(
                    read_id             = read_id,
                    sequence            = sequence,
                    quality             = quality,
                    quality_is_inferred = quality_is_inferred,
                    source_format       = "bam",
                    source_file         = source_file,
                    raw_header          = raw_header,
                )

                read_count += 1

    except FileNotFoundError:
        logger.error(f"BAM file not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error parsing BAM '{filepath}': {e}")
        raise

    logger.info(
        f"BAM parsing complete: {read_count} primary reads, "
        f"{skipped_count} secondary/supplementary skipped, "
        f"from {source_file}"
    )


# ---------------------------------------------------------------------------
# Public API
# This is the ONLY function the rest of the pipeline calls.
# Everything above is internal implementation detail.
# ---------------------------------------------------------------------------

def stream_reads(filepath: str) -> Generator[RawRead, None, None]:
    """
    Stream reads from any supported sequencing file as RawRead objects.

    This is the single public entry point for the entire ingestion layer.
    It detects the file format automatically, delegates to the appropriate
    parser, and yields RawRead objects one at a time.

    Downstream modules (quality_filter, pol_localizer, etc.) call this
    function and iterate over the results. They never interact with the
    format-specific parsers directly.

    Parameters
    ----------
    filepath : str
        Path to a FASTQ, FASTA, or BAM file.
        Gzip-compressed FASTQ and FASTA are handled automatically.

    Yields
    ------
    RawRead
        One RawRead per record. Reads are yielded one at a time (streaming),
        so memory usage is bounded regardless of file size.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file extension is not recognised.

    Example Usage
    -------------
    # In quality_filter.py or any downstream module:
    from src.ingestion.stream_reader import stream_reads

    for read in stream_reads("data/test/fastq/DRR537715_1.fastq.gz"):
        print(read.read_id, read.length, read.mean_quality)
    """
    # Validate the file exists before attempting to parse
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Input file not found: '{filepath}'")

    # Detect format from extension
    fmt = _detect_format(filepath)

    logger.info(f"stream_reads: detected format '{fmt}' for '{Path(filepath).name}'")

    # Delegate to the appropriate format-specific parser
    if fmt == "fastq":
        yield from _parse_fastq(filepath)
    elif fmt == "fasta":
        yield from _parse_fasta(filepath)
    elif fmt == "bam":
        yield from _parse_bam(filepath)


# ---------------------------------------------------------------------------
# Quick Validation: run this file directly to test it on your data
# Usage: python src/ingestion/stream_reader.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Set up logging so we can see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    )

    # Test files — adjust paths if running from a different directory
    test_files = [
        "data/test/fastq/DRR537715_1.fastq.gz",
        "data/test/fastq/SRR36194842_1.fastq.gz",
        "data/test/bam/DRR537715.bam",
        "data/test/bam/SRR36194842.bam",
    ]

    for test_file in test_files:
        if not os.path.exists(test_file):
            print(f"SKIP (not found): {test_file}")
            continue

        print(f"\n{'='*60}")
        print(f"Testing: {test_file}")
        print(f"{'='*60}")

        reads_seen   = 0
        total_length = 0
        total_quality = 0

        for read in stream_reads(test_file):

            # Print the first 3 reads in detail
            if reads_seen < 3:
                print(f"\nRead #{reads_seen + 1}:")
                print(f"  ID            : {read.read_id}")
                print(f"  Length        : {read.length} bases")
                print(f"  Mean Quality  : Q{read.mean_quality:.1f}")
                print(f"  Format        : {read.source_format}")
                print(f"  Inferred Q    : {read.quality_is_inferred}")
                print(f"  Raw Header    : {read.raw_header[:60]}...")
                print(f"  Sequence[:30] : {read.sequence[:30]}...")
                print(f"  Quality[:10]  : {read.quality[:10]}")

            reads_seen    += 1
            total_length  += read.length
            total_quality += read.mean_quality

        avg_length  = total_length  / reads_seen if reads_seen else 0
        avg_quality = total_quality / reads_seen if reads_seen else 0

        print(f"\nSummary for {Path(test_file).name}:")
        print(f"  Total reads    : {reads_seen}")
        print(f"  Avg read length: {avg_length:.0f} bases")
        print(f"  Avg quality    : Q{avg_quality:.1f}")