#!/usr/bin/env python3
"""
scripts/validate_stanford_hivdb.py
====================================
Validate the HIV DRM enricher pipeline against Stanford HIVdb
Genotype-Rx resistant subsets.

Stratified results:
    Table 1 — Subtype B, treatment-experienced  (main headline result)
    Table 2 — Non-B subtypes, treatment-experienced (generalization test)
    Table 3 — Per-key-mutation breakdown across all sequences

Data files expected:
    data/raw/stanford_hivdb/PR_resistant.txt
    data/raw/stanford_hivdb/RT_resistant.txt
    data/raw/stanford_hivdb/IN_resistant.txt

Column structure (Stanford Genotype-Rx format):
    Cols 1-7 : RefID, PtID, IsolateName, Region, Year, Subtype, PIList
    Cols 8-N : P1 ... Pn  (amino acid at each position)
    Col  N+1 : AccessionID
    Col  N+2 : NASeq  (nucleotide sequence — pipeline input)

    Column index formula (0-based): col_idx = position + N_META - 1 = position + 6
    e.g.  P184 → field[190]   P41 → field[47]   P90 → field[96]

Usage:
    python scripts/validate_stanford_hivdb.py --gene PR --max_rows 100 --verbose
    python scripts/validate_stanford_hivdb.py --gene RT
    python scripts/validate_stanford_hivdb.py --gene all
"""

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion.stream_reader  import RawRead
from src.enricher.pol_localizer   import PolLocalizer
from src.enricher.region_filter   import RegionFilter
from src.enricher.codon_framer    import CodonFramer
from src.enricher.feature_builder import FeatureBuilder, DRM_POSITIONS

logging.basicConfig(
    level  = logging.WARNING,
    format = "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("validate_stanford")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/raw/stanford_hivdb")

# Stanford file has 7 metadata columns before P1:
# RefID, PtID, IsolateName, Region, Year, Subtype, PIList/RTIList/INIList
N_META = 7

def col(position: int) -> int:
    """0-based column index for a 1-based amino acid position."""
    return position + N_META - 1

GENE_CONFIG = {
    "PR": {
        "file":        "PR_resistant.txt",
        "n_positions": 99,
        "wt": (
            "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKA"
            "IGTVLVGPTPVNIIGRNLLTQIGCTLNF"
        ),
    },
    "RT": {
        "file":        "RT_resistant.txt",
        "n_positions": 560,
        "wt": (
            "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTK"
            "WLKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDEDFRKYTAFTIPSINNETPG"
            "IRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLR"
            "WGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCK"
        ),
    },
    "IN": {
        "file":        "IN_resistant.txt",
        "n_positions": 288,
        "wt": (
            "FLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGK"
            "IILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTSTTVKAACWWAGIKQEFGIP"
            "YNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKEL"
            "QKQITKIQNFRVYYRDSRNPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASG"
            "RQDED"
        ),
    },
}


def wt_aa(gene: str, position: int) -> str:
    """HXB2 wildtype amino acid at 1-based position."""
    ref = GENE_CONFIG[gene]["wt"]
    idx = position - 1
    return ref[idx] if 0 <= idx < len(ref) else "?"


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------
def parse_ground_truth(
    fields:    list[str],
    gene:      str,
    drm_pos:   list[int],
) -> dict[int, str]:
    """
    Extract non-wildtype amino acid values at DRM positions.

    Returns {position: value} for any position that is not "-" or ".".
    Mixture strings like "MV" are preserved exactly.
    """
    gt: dict[int, str] = {}
    n_pos = GENE_CONFIG[gene]["n_positions"]

    for pos in drm_pos:
        if pos > n_pos:
            continue
        idx = col(pos)
        if idx >= len(fields):
            continue
        val = fields[idx].strip()
        if val in ("", "-", ".", "NA"):
            continue
        gt[pos] = val

    return gt


def gt_has_mutation(gt_val: str, wt: str) -> bool:
    """True if gt_val contains any non-wildtype, non-special amino acid."""
    return any(
        aa not in (".", "-", "*", "~", "#") and aa != wt
        for aa in gt_val
    )


# ---------------------------------------------------------------------------
# Stratum result container
# ---------------------------------------------------------------------------
@dataclass
class StratumResult:
    """Accumulates TP/FN/FP counts for one stratum (B or non-B)."""
    label: str

    n_seqs:       int = 0   # sequences with NASeq present
    n_passed:     int = 0   # sequences that cleared the pipeline
    n_wrong_gene: int = 0   # localizer assigned wrong gene

    tp: dict = field(default_factory=dict)  # {position: count}
    fn: dict = field(default_factory=dict)
    fp: dict = field(default_factory=dict)

    exact_match:   int = 0
    partial_match: int = 0
    fp_reads:      int = 0
    both_wt:       int = 0

    pos_possible: int = 0
    pos_covered:  int = 0

    subtypes_seen: dict = field(default_factory=dict)


def compute_f1(s: StratumResult) -> tuple[float, float, float]:
    tp = sum(s.tp.values())
    fn = sum(s.fn.values())
    fp = sum(s.fp.values())
    recall    = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return recall, precision, f1


# ---------------------------------------------------------------------------
# Single-sequence pipeline runner
# ---------------------------------------------------------------------------
def run_pipeline(
    sequence:   str,
    gene:       str,
    read_id:    str,
    localizer:  PolLocalizer,
    reg_filter: RegionFilter,
    framer:     CodonFramer,
    builder:    FeatureBuilder,
):
    """
    Feed one nucleotide sequence through the enricher pipeline.

    Skips stream_reader and quality_filter — Stanford Sanger data
    is already clean. Starts directly at pol_localizer.

    Returns FeatureVector or None if the read is rejected.
    """
    sequence = sequence.strip().upper()
    if not sequence:
        return None

    raw = RawRead(
        read_id             = read_id,
        sequence            = sequence,
        quality             = [40] * len(sequence),  # synthetic Q40 Sanger quality
        quality_is_inferred = True,
        source_format       = "fasta",               # closest valid format type
        source_file         = f"stanford_{gene}_resistant.txt",
        raw_header          = f">{read_id}",
    )

    # pol_localizer
    localized = localizer.localize(raw)

    # region_filter — single-record method
    passed = reg_filter.filter_single(localized)
    if passed is None:
        return None

    if passed.gene_region == "unknown":
        return None

    # codon_framer
    framed = framer.resolve(passed)

    # feature_builder
    fv = builder.extract(framed, read_start_position=1)

    return fv


# ---------------------------------------------------------------------------
# Validation loop for one gene
# ---------------------------------------------------------------------------
def validate_gene(
    gene:     str,
    max_rows: Optional[int] = None,
    verbose:  bool = False,
) -> tuple[StratumResult, StratumResult]:
    """
    Validate one gene. Returns (stratum_B, stratum_nonB).
    """
    s_b   = StratumResult(label="Subtype B (treatment-experienced)")
    s_non = StratumResult(label="Non-B subtypes (treatment-experienced)")

    filepath = DATA_DIR / GENE_CONFIG[gene]["file"]
    if not filepath.exists():
        print(f"  ERROR: {filepath} not found.")
        print(f"         Download with the curl commands first.")
        return s_b, s_non

    print(f"  Initializing pipeline components...")
    localizer  = PolLocalizer()
    reg_filter = RegionFilter()
    framer     = CodonFramer()
    builder    = FeatureBuilder()
    drm_pos    = DRM_POSITIONS.get(gene, [])
    n_done     = 0

    print(f"  Reading {filepath.name}...")

    with open(filepath, "r", encoding="utf-8", errors="replace") as fh:

        header     = fh.readline().rstrip("\n").split("\t")
        naseq_col  = len(header) - 1  # last column
        sub_col    = 5                 # 0-based index of Subtype column
        # drug_col = 6               # PIList/RTIList/INIList (not used in loop)

        for line in fh:
            if max_rows and n_done >= max_rows:
                break

            fields = line.rstrip("\n").split("\t")
            if len(fields) < naseq_col + 1:
                continue

            naseq   = fields[naseq_col].strip()
            subtype = fields[sub_col].strip()

            # Skip rows without nucleotide sequence
            if not naseq or naseq[0].upper() not in "ACGTN":
                continue

            # Assign to stratum
            s = s_b if subtype == "B" else s_non
            s.n_seqs += 1
            s.subtypes_seen[subtype] = s.subtypes_seen.get(subtype, 0) + 1

            ptid    = fields[1].strip()
            isname  = fields[2].strip()
            read_id = f"Stanford_{gene}_{ptid}_{isname}"

            # Build ground truth from P columns
            gt = parse_ground_truth(fields, gene, drm_pos)

            # Run pipeline
            fv = run_pipeline(
                sequence   = naseq,
                gene       = gene,
                read_id    = read_id,
                localizer  = localizer,
                reg_filter = reg_filter,
                framer     = framer,
                builder    = builder,
            )

            if fv is None:
                continue

            s.n_passed += 1
            if fv.gene_region != gene:
                s.n_wrong_gene += 1

            s.pos_possible += len(drm_pos)
            s.pos_covered  += len(fv.positions_extracted)

            # -----------------------------------------------------------
            # TP / FN / FP comparison at each covered DRM position
            # -----------------------------------------------------------
            pipeline_mut = set(fv.drm_candidates.keys())

            gt_mut = {
                pos for pos, val in gt.items()
                if gt_has_mutation(val, wt_aa(gene, pos))
            }

            tp_set = set()
            fn_set = set()
            fp_set = set()

            for pos in fv.positions_extracted:
                if pos not in drm_pos:
                    continue
                pipe_called = pos in pipeline_mut
                gt_called   = pos in gt_mut

                if gt_called and pipe_called:
                    tp_set.add(pos)
                    s.tp[pos] = s.tp.get(pos, 0) + 1
                elif gt_called and not pipe_called:
                    fn_set.add(pos)
                    s.fn[pos] = s.fn.get(pos, 0) + 1
                elif not gt_called and pipe_called:
                    fp_set.add(pos)
                    s.fp[pos] = s.fp.get(pos, 0) + 1

            # Per-read bucket
            if not gt_mut and not pipeline_mut:
                s.both_wt += 1
            elif tp_set and not fn_set and not fp_set:
                s.exact_match += 1
            elif tp_set:
                s.partial_match += 1
            elif fp_set:
                s.fp_reads += 1
            else:
                s.partial_match += 1

            if verbose and (gt_mut or pipeline_mut):
                gt_fmt  = [
                    f"{wt_aa(gene,p)}{p}{gt.get(p,'?')}"
                    for p in sorted(gt_mut)
                ]
                pip_fmt = fv.mutation_list()
                status = (
                    "EXACT"   if tp_set and not fn_set and not fp_set else
                    "PARTIAL" if tp_set else
                    "FP"      if fp_set else "FN"
                )
                print(
                    f"    [{status}] {read_id[:45]} | "
                    f"pipeline={pip_fmt} | truth={gt_fmt}"
                )

            n_done += 1

    return s_b, s_non


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------
SEP = "=" * 67


def print_stratum(s: StratumResult, gene: str) -> None:
    if s.n_seqs == 0:
        print(f"\n  {s.label}: 0 sequences — skipped.\n")
        return

    recall, precision, f1 = compute_f1(s)
    n    = max(1, s.n_passed)
    cov  = s.pos_covered / max(1, s.pos_possible)
    tp   = sum(s.tp.values())
    fn   = sum(s.fn.values())
    fp   = sum(s.fp.values())

    print(f"\n  ── {s.label} ──")
    print(f"    Sequences (with NASeq) : {s.n_seqs}")
    print(f"    Passed pipeline        : {s.n_passed}")
    print(f"    Rejected               : {s.n_seqs - s.n_passed}")
    print(f"    Wrong gene assigned    : {s.n_wrong_gene}")
    print(f"    DRM position coverage  : {cov:.1%}")
    print()
    print(f"    Per-read breakdown (n={s.n_passed}):")
    print(f"      Both wildtype        : {s.both_wt:>5}  ({s.both_wt/n:.1%})")
    print(f"      Exact match          : {s.exact_match:>5}  ({s.exact_match/n:.1%})")
    print(f"      Partial match        : {s.partial_match:>5}  ({s.partial_match/n:.1%})")
    print(f"      False positive reads : {s.fp_reads:>5}  ({s.fp_reads/n:.1%})")
    print()
    print(f"    Mutation-level results:")
    print(f"      TP={tp}  FN={fn}  FP={fp}")
    print(f"      Recall    : {recall:.1%}")
    print(f"      Precision : {precision:.1%}")
    print(f"      F1        : {f1:.3f}")

    # Top mutations table
    all_pos = sorted(
        set(s.tp) | set(s.fn) | set(s.fp),
        key=lambda p: -(s.tp.get(p, 0))
    )
    if all_pos:
        print(f"\n    Top mutations (TP ranked):")
        print(f"      {'Mut':<8} {'TP':>5} {'FN':>5} {'FP':>5}"
              f" {'Recall':>8} {'Prec':>8}")
        print(f"      {'-'*8} {'-'*5} {'-'*5} {'-'*5}"
              f" {'-'*8} {'-'*8}")
        for pos in all_pos[:20]:
            tp_ = s.tp.get(pos, 0)
            fn_ = s.fn.get(pos, 0)
            fp_ = s.fp.get(pos, 0)
            wt  = wt_aa(gene, pos)
            rec = tp_ / max(1, tp_ + fn_)
            pre = tp_ / max(1, tp_ + fp_)
            print(f"      {wt}{pos}?{'':<4} {tp_:>5} {fn_:>5} {fp_:>5}"
                  f" {rec:>7.1%} {pre:>8.1%}")

    # Non-B subtype breakdown
    if "Non-B" in s.label and s.subtypes_seen:
        print(f"\n    Non-B subtype distribution:")
        for sub, cnt in sorted(
            s.subtypes_seen.items(), key=lambda x: -x[1]
        )[:10]:
            print(f"      {sub:<18} {cnt:>5}")


def print_gene_report(
    gene:  str,
    s_b:   StratumResult,
    s_non: StratumResult,
) -> None:
    print(f"\n{SEP}")
    print(f"  VALIDATION REPORT — {gene}")
    print(SEP)
    print_stratum(s_b,   gene)
    print_stratum(s_non, gene)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate HIV DRM pipeline vs Stanford HIVdb data."
    )
    parser.add_argument(
        "--gene", choices=["PR", "RT", "IN", "all"], default="all",
    )
    parser.add_argument(
        "--max_rows", type=int, default=None,
        help="Max rows per gene (default: all)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-read comparison lines",
    )
    args = parser.parse_args()

    genes = ["PR", "RT", "IN"] if args.gene == "all" else [args.gene]

    print(f"\n{SEP}")
    print(f"  HIV DRM Pipeline — Stanford HIVdb Validation")
    print(f"  Data dir  : {DATA_DIR}")
    print(f"  Genes     : {', '.join(genes)}")
    if args.max_rows:
        print(f"  Max rows  : {args.max_rows} per gene")
    print(SEP)

    all_results = []

    for gene in genes:
        print(f"\n{'─'*67}")
        print(f"  Processing {gene}...")
        print(f"{'─'*67}")

        s_b, s_non = validate_gene(
            gene     = gene,
            max_rows = args.max_rows,
            verbose  = args.verbose,
        )
        print_gene_report(gene, s_b, s_non)
        all_results.append((gene, s_b, s_non))

    # Cross-gene summary
    if len(all_results) > 1:
        print(f"\n{SEP}")
        print(f"  CROSS-GENE SUMMARY")
        print(SEP)
        print(
            f"\n  {'Gene':<4} {'Stratum':<35} {'N':>5} "
            f"{'Cov':>6} {'Recall':>7} {'Prec':>7} {'F1':>6}"
        )
        print(
            f"  {'-'*4} {'-'*35} {'-'*5} "
            f"{'-'*6} {'-'*7} {'-'*7} {'-'*6}"
        )
        for gene, s_b, s_non in all_results:
            for s in (s_b, s_non):
                if s.n_passed == 0:
                    continue
                rec, pre, f1 = compute_f1(s)
                cov = s.pos_covered / max(1, s.pos_possible)
                print(
                    f"  {gene:<4} {s.label:<35} {s.n_passed:>5} "
                    f"{cov:>5.1%} {rec:>6.1%} {pre:>7.1%} {f1:>6.3f}"
                )
        print()


if __name__ == "__main__":
    main()