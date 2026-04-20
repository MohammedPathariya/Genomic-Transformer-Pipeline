"""
generate_test_db.py
====================
Synthetic test database generator for the HIV DRM detection pipeline.

Generates controlled, ground-truth-labelled FASTQ test data from the HXB2
reference genome. Every read has a known origin, known reading frame, and
known DRM status — giving the pipeline a deterministic validation surface.

What is generated
-----------------
data/test/synthetic/
├── targeted/                     <- Targeted amplicon profile
│   ├── PR_targeted.fastq.gz      500 PR reads
│   ├── RT_targeted.fastq.gz      500 RT reads
│   ├── IN_targeted.fastq.gz      500 IN reads
│   └── ground_truth.csv          read_id, region, frame, drm_status, mutations
│
├── full_pol/                     <- Full-pol amplicon profile (2500-4000bp)
│   ├── PR_full_pol.fastq.gz
│   ├── RT_full_pol.fastq.gz
│   ├── IN_full_pol.fastq.gz
│   └── ground_truth.csv
│
└── edge_cases/
    ├── edge_cases.fastq.gz
    └── ground_truth.csv

Targeted amplicon model
-----------------------
Targeted amplicon reads use FIXED primer offsets, not random start positions.
This models real clinical sequencing: all reads from the same amplicon start
at the same genomic position. The reading frame is therefore constant across
all reads of the same amplicon, which is what the codon framer expects.

  PR amplicon:  starts at offset 0 within the PR gene (HXB2 pos 2252)
  RT amplicon:  starts at offset 0 within the RT gene (HXB2 pos 2549)
  IN amplicon:  starts at offset 0 within the IN gene (HXB2 pos 4229)

Read lengths vary realistically (800-1500bp for RT/IN, gene-capped for PR)
to simulate variation in read lengths from the same amplicon run without
changing the start position.

This is architecturally important: using random start offsets would mean
every read requires a different reading frame, making statistical frame
selection impossible and preventing generalisation to real data.

Error model
-----------
Simulates Oxford Nanopore R10.4 characteristics:
  - Substitution rate: 2%
  - Insertion rate:    1%
  - Deletion rate:     2%
  Total error ~5%

DRM mutations baked in
-----------------------
PR:  D30N (NFV), L90M (broad PI), V82A (PI)
RT:  M184V (3TC/FTC), K103N (NNRTI), K65R (TDF)
IN:  Q148H (INSTI), N155H (RAL/EVG), G140S (INSTI)

Coordinates: HXB2 K03455.1, 0-indexed.

FIX LOG
-------
v1.0: Initial version — used random start offsets for targeted reads.
      Caused flat frame distribution (all 3 frames equally represented)
      and high stop codon counts because framer had no fixed coordinate.

v2.0: FIX — targeted reads now use fixed primer offset (offset=0 within gene).
      All targeted reads from the same gene start at the same HXB2 position.
      Frame 0 will dominate for all three genes (PR/RT/IN all start in frame 0
      relative to the pol reading frame). This correctly models targeted
      amplicon sequencing and allows the codon framer to work without
      coordinate tracking.

      Also kept:
      - FIX from v1.1: gene coordinates corrected to K03455.1 canonical
        (PR: 2252-2549, RT: 2549-4229, IN: 4229-5096)
      - FIX from v1.1: IN N155H wt_codon corrected to AAT
      - FIX from v1.1: read length floor clamped to gene length for PR

Usage
-----
python generate_test_db.py [--hxb2 PATH] [--reads-per-region N] [--seed N]

Author: HIV DRM Pipeline — Genomic-Transformer-Pipeline
"""

import argparse
import csv
import gzip
import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HXB2 Gene Region Coordinates
# 0-indexed, half-open (Python slice convention)
# Source: K03455.1 — matches pipeline_config.yaml
# ---------------------------------------------------------------------------
GENE_REGIONS: Dict[str, Dict] = {
    "PR": {"start": 2252, "end": 2549, "frame": 0, "length": 297},
    "RT": {"start": 2549, "end": 4229, "frame": 0, "length": 1680},
    "IN": {"start": 4229, "end": 5096, "frame": 0, "length": 867},
}

# ---------------------------------------------------------------------------
# DRM Mutations
# ---------------------------------------------------------------------------
DRM_MUTATIONS: Dict[str, List[Dict]] = {
    "PR": [
        {
            "aa_pos": 30, "wt_aa": "D", "mut_aa": "N",
            "wt_codon": "GAT", "mut_codon": "AAT",
            "drug_class": "PI", "label": "D30N",
        },
        {
            "aa_pos": 90, "wt_aa": "L", "mut_aa": "M",
            "wt_codon": "TTG", "mut_codon": "ATG",
            "drug_class": "PI", "label": "L90M",
        },
        {
            "aa_pos": 82, "wt_aa": "V", "mut_aa": "A",
            "wt_codon": "GTC", "mut_codon": "GCT",
            "drug_class": "PI", "label": "V82A",
        },
    ],
    "RT": [
        {
            "aa_pos": 184, "wt_aa": "M", "mut_aa": "V",
            "wt_codon": "ATG", "mut_codon": "GTG",
            "drug_class": "NRTI", "label": "M184V",
        },
        {
            "aa_pos": 103, "wt_aa": "K", "mut_aa": "N",
            "wt_codon": "AAA", "mut_codon": "AAC",
            "drug_class": "NNRTI", "label": "K103N",
        },
        {
            "aa_pos": 65, "wt_aa": "K", "mut_aa": "R",
            "wt_codon": "AAA", "mut_codon": "AGA",
            "drug_class": "NRTI", "label": "K65R",
        },
    ],
    "IN": [
        {
            "aa_pos": 148, "wt_aa": "Q", "mut_aa": "H",
            "wt_codon": "CAA", "mut_codon": "CAT",
            "drug_class": "INSTI", "label": "Q148H",
        },
        {
            "aa_pos": 155, "wt_aa": "N", "mut_aa": "H",
            "wt_codon": "AAT", "mut_codon": "CAT",
            "drug_class": "INSTI", "label": "N155H",
        },
        {
            "aa_pos": 140, "wt_aa": "G", "mut_aa": "S",
            "wt_codon": "GGA", "mut_codon": "TCA",
            "drug_class": "INSTI", "label": "G140S",
        },
    ],
}

# ---------------------------------------------------------------------------
# Standard genetic code
# ---------------------------------------------------------------------------
CODON_TABLE: Dict[str, str] = {
    'ATA':'I','ATC':'I','ATT':'I','ATG':'M',
    'ACA':'T','ACC':'T','ACG':'T','ACT':'T',
    'AAC':'N','AAT':'N','AAA':'K','AAG':'K',
    'AGC':'S','AGT':'S','AGA':'R','AGG':'R',
    'CTA':'L','CTC':'L','CTG':'L','CTT':'L',
    'CCA':'P','CCC':'P','CCG':'P','CCT':'P',
    'CAC':'H','CAT':'H','CAA':'Q','CAG':'Q',
    'CGA':'R','CGC':'R','CGG':'R','CGT':'R',
    'GTA':'V','GTC':'V','GTG':'V','GTT':'V',
    'GCA':'A','GCC':'A','GCG':'A','GCT':'A',
    'GAC':'D','GAT':'D','GAA':'E','GAG':'E',
    'GGA':'G','GGC':'G','GGG':'G','GGT':'G',
    'TCA':'S','TCC':'S','TCG':'S','TCT':'S',
    'TTC':'F','TTT':'F','TTA':'L','TTG':'L',
    'TAC':'Y','TAT':'Y','TAA':'_','TAG':'_',
    'TGC':'C','TGT':'C','TGA':'_','TGG':'W',
}

# ---------------------------------------------------------------------------
# Nanopore R10.4 error model
# ---------------------------------------------------------------------------
ONT_ERROR_MODEL = {
    "substitution_rate": 0.02,
    "insertion_rate":    0.01,
    "deletion_rate":     0.02,
}

def _phred_char(q: int) -> str:
    return chr(q + 33)

BASES = ["A", "T", "C", "G"]
_COMPLEMENT = str.maketrans("ATCGN", "TAGCN")

def _reverse_complement(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


# ---------------------------------------------------------------------------
# Read dataclass
# ---------------------------------------------------------------------------
@dataclass
class SyntheticRead:
    read_id:     str
    sequence:    str
    quality:     str
    region:      str
    frame:       int
    drm_status:  str
    mutations:   List[str]
    profile:     str
    orientation: str
    hxb2_start:  int
    hxb2_end:    int
    notes:       str = ""


# ---------------------------------------------------------------------------
# HXB2 loader
# ---------------------------------------------------------------------------
def load_hxb2(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"HXB2 reference not found: {path}\n"
            f"Download with:\n"
            f"  curl 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            f"?db=nuccore&id=K03455&rettype=fasta&retmode=text' -o {path}"
        )
    parts = []
    with open(path) as f:
        for line in f:
            if not line.startswith(">"):
                parts.append(line.strip().upper())
    seq = "".join(parts)
    if not seq:
        raise ValueError(f"Empty sequence in {path}")
    logger.info(f"HXB2 loaded: {len(seq)} bases")
    return seq


# ---------------------------------------------------------------------------
# DRM mutation injector
# ---------------------------------------------------------------------------
def _inject_drm(gene_seq: str, region: str, drm: Dict) -> Tuple[str, bool]:
    aa_pos    = drm["aa_pos"]
    nuc_start = (aa_pos - 1) * 3
    nuc_end   = nuc_start + 3

    if nuc_end > len(gene_seq):
        logger.warning(
            f"DRM {drm['label']}: AA position {aa_pos} outside gene. Skipping."
        )
        return gene_seq, False

    actual_codon = gene_seq[nuc_start:nuc_end]
    expected_wt  = drm["wt_codon"]

    if actual_codon != expected_wt:
        logger.warning(
            f"DRM {drm['label']}: codon mismatch at AA {aa_pos}. "
            f"Expected {expected_wt} ({drm['wt_aa']}), "
            f"found {actual_codon} ({CODON_TABLE.get(actual_codon, '?')}). "
            f"Injecting anyway."
        )

    mutant_seq = gene_seq[:nuc_start] + drm["mut_codon"] + gene_seq[nuc_end:]
    return mutant_seq, True


# ---------------------------------------------------------------------------
# Nanopore error simulator
# ---------------------------------------------------------------------------
def _apply_nanopore_errors(
    sequence: str,
    rng: random.Random,
    error_model: Dict = ONT_ERROR_MODEL,
) -> Tuple[str, str]:
    sub_rate = error_model["substitution_rate"]
    ins_rate = error_model["insertion_rate"]
    del_rate = error_model["deletion_rate"]

    output_bases   = []
    output_quality = []

    for base in sequence:
        r = rng.random()

        if r < del_rate:
            continue

        if r < del_rate + ins_rate:
            ins_base = rng.choice(BASES)
            output_bases.append(ins_base)
            output_quality.append(_phred_char(rng.randint(8, 15)))

        if r < del_rate + ins_rate + sub_rate:
            alt_bases = [b for b in BASES if b != base]
            output_bases.append(rng.choice(alt_bases))
            output_quality.append(_phred_char(rng.randint(10, 18)))
        else:
            output_bases.append(base)
            output_quality.append(_phred_char(rng.randint(20, 30)))

    return "".join(output_bases), "".join(output_quality)


# ---------------------------------------------------------------------------
# Read generator — targeted profile
#
# KEY DESIGN DECISION (v2.0):
# Targeted reads use a FIXED primer offset (read_start_offset = 0).
# All reads from the same gene start at the same HXB2 position.
#
# Why this matters:
#   Real targeted amplicon sequencing uses fixed PCR primers. Every read
#   from a PR amplicon starts at the same genomic coordinate. The reading
#   frame is therefore constant across all reads of the same amplicon.
#
#   Using random offsets (v1.0 behaviour) produced flat frame distributions
#   (33/33/33) and high stop codon counts because the framer had no reliable
#   signal — each read needed a different frame. This did not model real
#   clinical data and could not generalise.
#
#   With fixed offsets:
#   - PR reads: all start at HXB2 2252, frame = (2252-2252) % 3 = 0
#   - RT reads: all start at HXB2 2549, frame = (2549-2549) % 3 = 0
#   - IN reads: all start at HXB2 4229, frame = (4229-4229) % 3 = 0
#   All three genes start in frame 0. The framer should see ~100% frame 0
#   on targeted reads and near-zero stop codons in the translated sequence.
#
# Read length still varies (800-1500bp) to simulate realistic ONT length
# distribution from the same amplicon run. Only the START is fixed.
# ---------------------------------------------------------------------------
def generate_targeted_read(
    read_index:   int,
    region:       str,
    hxb2:         str,
    rng:          random.Random,
    drm:          Optional[Dict] = None,
    orientation:  str = "forward",
) -> SyntheticRead:
    """
    Generate a single targeted amplicon read from one pol sub-region.

    Uses a fixed primer offset (offset=0 within gene) to model real
    targeted amplicon sequencing. Read length varies within [min, max]
    bounds to simulate realistic ONT length distribution.
    """
    coords     = GENE_REGIONS[region]
    gene_start = coords["start"]
    gene_end   = coords["end"]
    gene_len   = gene_end - gene_start

    # Fixed primer offset — all reads start at the beginning of the gene.
    # This is the v2.0 fix. Do NOT use rng.randint() here.
    read_start_offset = 0

    # Read length varies to simulate realistic ONT length distribution.
    # Clamped to gene length so we never run off the end.
    min_read_len = min(800, gene_len)
    max_read_len = min(1500, gene_len)
    read_len     = rng.randint(min_read_len, max_read_len)

    genomic_start = gene_start + read_start_offset   # = gene_start
    genomic_end   = genomic_start + read_len

    # Extract subsequence from HXB2 — always starts from gene start
    template = hxb2[genomic_start:genomic_end]

    # Inject DRM if requested
    mutations_applied = []
    if drm is not None:
        nuc_offset_in_gene = (drm["aa_pos"] - 1) * 3
        nuc_offset_in_read = nuc_offset_in_gene - read_start_offset

        if 0 <= nuc_offset_in_read <= len(template) - 3:
            before   = template[:nuc_offset_in_read]
            after    = template[nuc_offset_in_read + 3:]
            template = before + drm["mut_codon"] + after
            mutations_applied.append(drm["label"])

    # Apply Nanopore errors
    errored_seq, quality = _apply_nanopore_errors(template, rng)

    # Reverse complement if requested
    if orientation == "reverse_complement":
        errored_seq = _reverse_complement(errored_seq)
        quality     = quality[::-1]

    drm_status = f"DRM:{','.join(mutations_applied)}" if mutations_applied else "wildtype"

    read_id = (
        f"SYNTH_{region}_targeted_{orientation[:3].upper()}"
        f"_{read_index:04d}"
        f"_{drm_status.replace(':','_').replace(',','_')}"
    )

    return SyntheticRead(
        read_id     = read_id,
        sequence    = errored_seq,
        quality     = quality,
        region      = region,
        frame       = coords["frame"],
        drm_status  = drm_status,
        mutations   = mutations_applied,
        profile     = "targeted",
        orientation = orientation,
        hxb2_start  = genomic_start,
        hxb2_end    = genomic_end,
        notes       = f"fixed_primer_offset=0,read_len={read_len}",
    )


# ---------------------------------------------------------------------------
# Read generator — full-pol profile
# ---------------------------------------------------------------------------
def generate_full_pol_read(
    read_index:    int,
    anchor_region: str,
    hxb2:          str,
    rng:           random.Random,
    drm:           Optional[Dict] = None,
) -> SyntheticRead:
    """
    Generate a full-pol amplicon read (spans all three pol sub-regions).

    These reads cover the full pol gene (PR+RT+IN = 2843bp) plus flanking
    sequence. Used to test the windowed localizer and the future pol_extractor
    module. Not used for codon framer validation.
    """
    pol_start = GENE_REGIONS["PR"]["start"]
    pol_end   = GENE_REGIONS["IN"]["end"]

    left_flank  = rng.randint(0, 600)
    right_flank = rng.randint(0, 600)

    genomic_start = max(0, pol_start - left_flank)
    genomic_end   = min(len(hxb2), pol_end + right_flank)

    template = hxb2[genomic_start:genomic_end]

    mutations_applied = []
    if drm is not None:
        region_start_in_genome = GENE_REGIONS[anchor_region]["start"]
        nuc_offset_in_gene     = (drm["aa_pos"] - 1) * 3
        nuc_offset_in_read     = (region_start_in_genome + nuc_offset_in_gene) - genomic_start

        if 0 <= nuc_offset_in_read <= len(template) - 3:
            before   = template[:nuc_offset_in_read]
            after    = template[nuc_offset_in_read + 3:]
            template = before + drm["mut_codon"] + after
            mutations_applied.append(drm["label"])

    errored_seq, quality = _apply_nanopore_errors(template, rng)

    drm_status = f"DRM:{','.join(mutations_applied)}" if mutations_applied else "wildtype"
    coords     = GENE_REGIONS[anchor_region]

    read_id = (
        f"SYNTH_{anchor_region}_fullpol"
        f"_{read_index:04d}"
        f"_{drm_status.replace(':','_').replace(',','_')}"
    )

    return SyntheticRead(
        read_id     = read_id,
        sequence    = errored_seq,
        quality     = quality,
        region      = anchor_region,
        frame       = coords["frame"],
        drm_status  = drm_status,
        mutations   = mutations_applied,
        profile     = "full_pol",
        orientation = "forward",
        hxb2_start  = genomic_start,
        hxb2_end    = genomic_end,
        notes       = f"full_pol_amplicon spans {genomic_start}-{genomic_end}",
    )


# ---------------------------------------------------------------------------
# Edge case generator
# ---------------------------------------------------------------------------
def generate_edge_cases(hxb2: str, rng: random.Random) -> List[SyntheticRead]:
    """
    Generate edge cases that stress-test each module.

    Categories:
    1. Short reads (300-500bp)
    2. Very short reads (<300bp) — should be filtered
    3. Reverse complement reads
    4. High error rate reads (10%)
    5. Chimeric reads (PR + IN)
    6. Non-pol reads (gag gene) — should be unknown
    7. PR/RT boundary reads
    8. RT/IN boundary reads
    """
    edge_reads = []

    # Category 1: Short but valid reads
    for region in ["PR", "RT", "IN"]:
        coords   = GENE_REGIONS[region]
        gene_seq = hxb2[coords["start"]:coords["end"]]
        gene_len = len(gene_seq)
        for i in range(10):
            read_len = rng.randint(min(300, gene_len), min(500, gene_len))
            template = gene_seq[:read_len]
            errored, quality = _apply_nanopore_errors(template, rng)
            read_id = f"EDGE_short_{region}_{i:03d}"
            edge_reads.append(SyntheticRead(
                read_id=read_id, sequence=errored, quality=quality,
                region=region, frame=coords["frame"],
                drm_status="wildtype", mutations=[],
                profile="edge", orientation="forward",
                hxb2_start=coords["start"],
                hxb2_end=coords["start"] + read_len,
                notes="short_valid",
            ))

    # Category 2: Very short reads (below min_len)
    for i in range(10):
        read_len = rng.randint(100, 299)
        start    = 2252
        template = hxb2[start:start + read_len]
        errored, quality = _apply_nanopore_errors(template, rng)
        read_id = f"EDGE_tooshort_{i:03d}"
        edge_reads.append(SyntheticRead(
            read_id=read_id, sequence=errored, quality=quality,
            region="PR", frame=0,
            drm_status="wildtype", mutations=[],
            profile="edge", orientation="forward",
            hxb2_start=start, hxb2_end=start + read_len,
            notes="below_min_length_should_filter",
        ))

    # Category 3: Reverse complement reads
    for region in ["PR", "RT", "IN"]:
        coords   = GENE_REGIONS[region]
        gene_seq = hxb2[coords["start"]:coords["end"]]
        gene_len = len(gene_seq)
        for i in range(15):
            read_len = rng.randint(min(800, gene_len), min(1200, gene_len))
            template = gene_seq[:read_len]
            errored, quality = _apply_nanopore_errors(template, rng)
            rc_seq  = _reverse_complement(errored)
            rc_qual = quality[::-1]
            read_id = f"EDGE_rc_{region}_{i:03d}"
            edge_reads.append(SyntheticRead(
                read_id=read_id, sequence=rc_seq, quality=rc_qual,
                region=region, frame=coords["frame"],
                drm_status="wildtype", mutations=[],
                profile="edge", orientation="reverse_complement",
                hxb2_start=coords["start"],
                hxb2_end=coords["start"] + read_len,
                notes="reverse_complement_orientation",
            ))

    # Category 4: High error rate reads
    high_error_model = {
        "substitution_rate": 0.04,
        "insertion_rate":    0.03,
        "deletion_rate":     0.03,
    }
    for region in ["PR", "RT", "IN"]:
        coords   = GENE_REGIONS[region]
        gene_seq = hxb2[coords["start"]:coords["end"]]
        gene_len = len(gene_seq)
        for i in range(10):
            read_len = rng.randint(min(800, gene_len), min(1200, gene_len))
            template = gene_seq[:read_len]
            errored, quality = _apply_nanopore_errors(template, rng, high_error_model)
            read_id = f"EDGE_highErr_{region}_{i:03d}"
            edge_reads.append(SyntheticRead(
                read_id=read_id, sequence=errored, quality=quality,
                region=region, frame=coords["frame"],
                drm_status="wildtype", mutations=[],
                profile="edge", orientation="forward",
                hxb2_start=coords["start"],
                hxb2_end=coords["start"] + read_len,
                notes="high_error_rate_10pct",
            ))

    # Category 5: Chimeric reads (PR + IN)
    for i in range(10):
        pr_frag  = hxb2[2252:2252 + 400]
        in_frag  = hxb2[4229:4229 + 400]
        chimeric = pr_frag + in_frag
        errored, quality = _apply_nanopore_errors(chimeric, rng)
        read_id = f"EDGE_chimeric_PR_IN_{i:03d}"
        edge_reads.append(SyntheticRead(
            read_id=read_id, sequence=errored, quality=quality,
            region="chimeric", frame=-1,
            drm_status="wildtype", mutations=[],
            profile="edge", orientation="forward",
            hxb2_start=2252, hxb2_end=4629,
            notes="chimeric_PR_IN_ambiguous",
        ))

    # Category 6: Non-pol reads (gag gene)
    gag_start, gag_end = 790, 2252
    for i in range(15):
        read_len  = rng.randint(800, 1200)
        max_start = gag_end - read_len
        if max_start <= gag_start:
            continue
        start    = rng.randint(gag_start, max_start)
        template = hxb2[start:start + read_len]
        errored, quality = _apply_nanopore_errors(template, rng)
        read_id = f"EDGE_gag_nonpol_{i:03d}"
        edge_reads.append(SyntheticRead(
            read_id=read_id, sequence=errored, quality=quality,
            region="unknown", frame=-1,
            drm_status="wildtype", mutations=[],
            profile="edge", orientation="forward",
            hxb2_start=start, hxb2_end=start + read_len,
            notes="gag_gene_should_be_unknown",
        ))

    # Category 7: PR/RT boundary reads
    pr_rt = GENE_REGIONS["PR"]["end"]
    for i in range(10):
        start    = pr_rt - 400
        end      = pr_rt + 400
        template = hxb2[start:end]
        errored, quality = _apply_nanopore_errors(template, rng)
        read_id = f"EDGE_boundary_PR_RT_{i:03d}"
        edge_reads.append(SyntheticRead(
            read_id=read_id, sequence=errored, quality=quality,
            region="boundary_PR_RT", frame=-1,
            drm_status="wildtype", mutations=[],
            profile="edge", orientation="forward",
            hxb2_start=start, hxb2_end=end,
            notes="straddles_PR_RT_boundary_ambiguous",
        ))

    # Category 8: RT/IN boundary reads
    rt_in = GENE_REGIONS["RT"]["end"]
    for i in range(10):
        start    = rt_in - 400
        end      = rt_in + 400
        template = hxb2[start:end]
        errored, quality = _apply_nanopore_errors(template, rng)
        read_id = f"EDGE_boundary_RT_IN_{i:03d}"
        edge_reads.append(SyntheticRead(
            read_id=read_id, sequence=errored, quality=quality,
            region="boundary_RT_IN", frame=-1,
            drm_status="wildtype", mutations=[],
            profile="edge", orientation="forward",
            hxb2_start=start, hxb2_end=end,
            notes="straddles_RT_IN_boundary_ambiguous",
        ))

    logger.info(f"Generated {len(edge_reads)} edge case reads")
    return edge_reads


# ---------------------------------------------------------------------------
# FASTQ writer
# ---------------------------------------------------------------------------
def write_fastq_gz(reads: List[SyntheticRead], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        for read in reads:
            f.write(f"@{read.read_id}\n")
            f.write(f"{read.sequence}\n")
            f.write("+\n")
            f.write(f"{read.quality}\n")
    logger.info(f"  Wrote {len(reads)} reads → {path}")


# ---------------------------------------------------------------------------
# Ground truth CSV writer
# ---------------------------------------------------------------------------
def write_ground_truth_csv(reads: List[SyntheticRead], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "read_id", "profile", "region", "frame", "orientation",
        "drm_status", "mutations", "hxb2_start", "hxb2_end",
        "read_length", "notes",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in reads:
            writer.writerow({
                "read_id":     r.read_id,
                "profile":     r.profile,
                "region":      r.region,
                "frame":       r.frame,
                "orientation": r.orientation,
                "drm_status":  r.drm_status,
                "mutations":   ";".join(r.mutations) if r.mutations else "",
                "hxb2_start":  r.hxb2_start,
                "hxb2_end":    r.hxb2_end,
                "read_length": len(r.sequence),
                "notes":       r.notes,
            })
    logger.info(f"  Wrote ground truth → {path}")


# ---------------------------------------------------------------------------
# DRM coordinate verifier
# ---------------------------------------------------------------------------
def verify_drm_coordinates(hxb2: str) -> None:
    logger.info("Verifying DRM mutation coordinates against HXB2...")
    all_ok = True
    for region, mutations in DRM_MUTATIONS.items():
        coords   = GENE_REGIONS[region]
        gene_seq = hxb2[coords["start"]:coords["end"]]
        for drm in mutations:
            nuc_start   = (drm["aa_pos"] - 1) * 3
            actual      = gene_seq[nuc_start:nuc_start + 3]
            expected    = drm["wt_codon"]
            actual_aa   = CODON_TABLE.get(actual, "?")
            expected_aa = drm["wt_aa"]
            if actual == expected:
                logger.info(
                    f"  ✓ {region} {drm['label']}: "
                    f"AA{drm['aa_pos']} = {actual} ({actual_aa}) — matches"
                )
            else:
                logger.warning(
                    f"  ✗ {region} {drm['label']}: "
                    f"AA{drm['aa_pos']} expected {expected} ({expected_aa}), "
                    f"found {actual} ({actual_aa})"
                )
                all_ok = False
    if all_ok:
        logger.info("All DRM coordinates verified ✓")
    else:
        logger.warning(
            "Some DRM coordinates do not match HXB2. "
            "Reads will still be generated with the specified mut_codon."
        )


# ---------------------------------------------------------------------------
# Main generation logic
# ---------------------------------------------------------------------------
def generate_database(
    hxb2_path:        str,
    output_dir:       str,
    reads_per_region: int,
    seed:             int,
) -> None:
    rng  = random.Random(seed)
    hxb2 = load_hxb2(hxb2_path)

    verify_drm_coordinates(hxb2)

    logger.info(
        f"\nGenerating synthetic test database (v2.0 — fixed primer offsets):\n"
        f"  Reads per region: {reads_per_region}\n"
        f"  Output dir: {output_dir}\n"
        f"  RNG seed: {seed}\n"
    )

    n_wt      = int(reads_per_region * 0.70)
    n_per_drm = int(reads_per_region * 0.08)
    n_double  = int(reads_per_region * 0.06)

    all_targeted_reads: List[SyntheticRead] = []
    all_full_pol_reads: List[SyntheticRead] = []

    for region in ["PR", "RT", "IN"]:
        drm_list = DRM_MUTATIONS[region]
        t_reads: List[SyntheticRead] = []
        p_reads: List[SyntheticRead] = []

        # Wildtype targeted reads — fixed offset = 0
        for i in range(n_wt):
            orientation = "reverse_complement" if i < n_wt // 10 else "forward"
            t_reads.append(generate_targeted_read(
                read_index=i, region=region, hxb2=hxb2,
                rng=rng, drm=None, orientation=orientation,
            ))

        # Wildtype full-pol reads
        for i in range(n_wt):
            p_reads.append(generate_full_pol_read(
                read_index=i, anchor_region=region, hxb2=hxb2,
                rng=rng, drm=None,
            ))

        # Single DRM targeted reads — fixed offset = 0, DRM codon within read
        drm_idx = n_wt
        for drm in drm_list:
            for j in range(n_per_drm):
                orientation = "reverse_complement" if j < 4 else "forward"
                t_reads.append(generate_targeted_read(
                    read_index=drm_idx + j, region=region, hxb2=hxb2,
                    rng=rng, drm=drm, orientation=orientation,
                ))
                p_reads.append(generate_full_pol_read(
                    read_index=drm_idx + j, anchor_region=region, hxb2=hxb2,
                    rng=rng, drm=drm,
                ))
            drm_idx += n_per_drm

        # Double DRM reads — inject first two DRMs simultaneously
        # Still uses fixed offset = 0
        if len(drm_list) >= 2:
            drm_a, drm_b = drm_list[0], drm_list[1]
            coords   = GENE_REGIONS[region]
            gene_seq = hxb2[coords["start"]:coords["end"]]
            gene_len = len(gene_seq)

            for j in range(n_double):
                min_read_len = min(800, gene_len)
                max_read_len = min(1500, gene_len)
                read_len     = rng.randint(min_read_len, max_read_len)

                # Fixed offset = 0
                template = gene_seq[:read_len]
                muts     = []

                for drm in [drm_a, drm_b]:
                    nuc_off = (drm["aa_pos"] - 1) * 3
                    if 0 <= nuc_off <= len(template) - 3:
                        template = template[:nuc_off] + drm["mut_codon"] + template[nuc_off + 3:]
                        muts.append(drm["label"])

                errored, quality = _apply_nanopore_errors(template, rng)
                drm_status = f"DRM:{','.join(muts)}" if muts else "wildtype"
                read_id    = (
                    f"SYNTH_{region}_targeted_FWD_{drm_idx + j:04d}"
                    f"_{drm_status.replace(':','_').replace(',','_')}"
                )

                t_reads.append(SyntheticRead(
                    read_id     = read_id,
                    sequence    = errored,
                    quality     = quality,
                    region      = region,
                    frame       = coords["frame"],
                    drm_status  = drm_status,
                    mutations   = muts,
                    profile     = "targeted",
                    orientation = "forward",
                    hxb2_start  = coords["start"],
                    hxb2_end    = coords["start"] + read_len,
                    notes       = "double_drm,fixed_primer_offset=0",
                ))

        rng.shuffle(t_reads)
        rng.shuffle(p_reads)

        all_targeted_reads.extend(t_reads)
        all_full_pol_reads.extend(p_reads)

        logger.info(
            f"  {region}: {len(t_reads)} targeted, {len(p_reads)} full-pol reads"
        )

    edge_reads = generate_edge_cases(hxb2, rng)

    # Write targeted
    targeted_dir = os.path.join(output_dir, "targeted")
    for region in ["PR", "RT", "IN"]:
        region_reads = [r for r in all_targeted_reads if r.region == region]
        write_fastq_gz(
            region_reads,
            os.path.join(targeted_dir, f"{region}_targeted.fastq.gz"),
        )
    write_ground_truth_csv(
        all_targeted_reads,
        os.path.join(targeted_dir, "ground_truth.csv"),
    )

    # Write full-pol
    full_pol_dir = os.path.join(output_dir, "full_pol")
    for region in ["PR", "RT", "IN"]:
        region_reads = [r for r in all_full_pol_reads if r.region == region]
        write_fastq_gz(
            region_reads,
            os.path.join(full_pol_dir, f"{region}_full_pol.fastq.gz"),
        )
    write_ground_truth_csv(
        all_full_pol_reads,
        os.path.join(full_pol_dir, "ground_truth.csv"),
    )

    # Write edge cases
    edge_dir = os.path.join(output_dir, "edge_cases")
    write_fastq_gz(edge_reads, os.path.join(edge_dir, "edge_cases.fastq.gz"))
    write_ground_truth_csv(edge_reads, os.path.join(edge_dir, "ground_truth.csv"))

    total        = len(all_targeted_reads) + len(all_full_pol_reads) + len(edge_reads)
    targeted_drm = sum(1 for r in all_targeted_reads if r.drm_status != "wildtype")
    fullpol_drm  = sum(1 for r in all_full_pol_reads  if r.drm_status != "wildtype")

    logger.info(f"""
╔══════════════════════════════════════════════════════════════╗
║     Synthetic Test Database Generation Complete (v2.0)       ║
╠══════════════════════════════════════════════════════════════╣
║  Targeted reads:   {len(all_targeted_reads):>6}  ({targeted_drm} DRM, {len(all_targeted_reads)-targeted_drm} wildtype)
║  Full-pol reads:   {len(all_full_pol_reads):>6}  ({fullpol_drm} DRM, {len(all_full_pol_reads)-fullpol_drm} wildtype)
║  Edge case reads:  {len(edge_reads):>6}
║  Total reads:      {total:>6}
╠══════════════════════════════════════════════════════════════╣
║  Key fix (v2.0): targeted reads use fixed primer offset=0
║  All reads from same gene start at same HXB2 coordinate.
║  Expected frame distribution: ~100% Frame 0 for all genes.
║  Expected stop codons: near-zero in correct frame.
╠══════════════════════════════════════════════════════════════╣
║  DRM mutations: PR(D30N,L90M,V82A) RT(M184V,K103N,K65R)
║                 IN(Q148H,N155H,G140S)
║  Error model: ONT R10.4 (sub=2%, ins=1%, del=2%)
║  RNG seed: {seed}
╚══════════════════════════════════════════════════════════════╝
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic HIV DRM test database (v2.0 — fixed primer offsets)",
    )
    parser.add_argument(
        "--hxb2",
        default="data/public/HXB2_reference.fasta",
    )
    parser.add_argument(
        "--output-dir",
        default="data/test/synthetic",
    )
    parser.add_argument(
        "--reads-per-region",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate_database(
        hxb2_path        = args.hxb2,
        output_dir       = args.output_dir,
        reads_per_region = args.reads_per_region,
        seed             = args.seed,
    )