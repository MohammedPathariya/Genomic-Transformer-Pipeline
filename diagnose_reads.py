"""
diagnose_reads.py
=================
Pre-fix diagnostic tool to understand the true nature of reads
before attempting to fix CodonFramer.

Run BEFORE touching any pipeline code:
    python diagnose_reads.py

This will tell you:
1. Actual read length distribution (are these really IN amplicons?)
2. Stop codon density per frame for a sample of reads
3. Whether the correct frame is identifiable at all from stop codons alone
4. Localization confidence distribution (is the localizer trustworthy?)

If stop codon density is uniformly high across all frames, the problem
is upstream (wrong data, not a framing bug). If one frame is clearly
lower, the framer's scoring logic is broken but fixable.
"""

import gzip
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CODON_TABLE: dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

# HXB2 pol gene boundaries (0-indexed)
PR_START, PR_END = 2253, 2550   # 297 bp
RT_START, RT_END = 2550, 4229   # 1679 bp
IN_START, IN_END = 4229, 5096   # 867 bp
POL_START = PR_START            # 2253
POL_END   = IN_END              # 5096  → pol = 2843 bp total


def translate(seq: str, frame: int) -> str:
    s = seq[frame:]
    return "".join(
        CODON_TABLE.get(s[i:i+3], "X")
        for i in range(0, len(s) - 2, 3)
    )


def stop_density(aa: str) -> float:
    if not aa:
        return 1.0
    return aa.count("*") / len(aa)


def stream_fastq(path: str):
    """Yield (read_id, sequence, quality_str) tuples from a FASTQ/FASTQ.gz."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        while True:
            header = fh.readline().strip()
            if not header:
                break
            seq  = fh.readline().strip()
            _    = fh.readline().strip()  # +
            qual = fh.readline().strip()
            yield header[1:].split()[0], seq.upper(), qual


def analyze_file(fastq_path: str, sample_n: int = 200) -> None:
    """
    Run all diagnostics on a FASTQ file.

    Parameters
    ----------
    fastq_path : str
        Path to FASTQ or FASTQ.gz file.
    sample_n : int
        Number of reads to sample for stop codon analysis.
    """
    print(f"\n{'='*70}")
    print(f"DIAGNOSING: {Path(fastq_path).name}")
    print(f"{'='*70}")

    lengths = []
    stop_profiles = []   # list of (read_id, len, density_f0, density_f1, density_f2, best_frame)
    gc_contents  = []

    for i, (read_id, seq, qual) in enumerate(stream_fastq(fastq_path)):
        lengths.append(len(seq))

        # GC content
        gc = (seq.count("G") + seq.count("C")) / max(len(seq), 1)
        gc_contents.append(gc)

        # Stop codon analysis on first sample_n reads
        if i < sample_n:
            densities = {}
            for frame in range(3):
                aa = translate(seq, frame)
                densities[frame] = stop_density(aa)

            best_frame = min(densities, key=densities.get)
            # Margin: difference between best and worst
            margin = max(densities.values()) - min(densities.values())

            stop_profiles.append({
                "read_id":    read_id,
                "length":     len(seq),
                "f0_density": densities[0],
                "f1_density": densities[1],
                "f2_density": densities[2],
                "best_frame": best_frame,
                "margin":     margin,
                "best_density": densities[best_frame],
            })

    total = len(lengths)
    print(f"\n── READ LENGTH DISTRIBUTION ({total} reads) ──")
    print(f"  Min    : {min(lengths):,} bp")
    print(f"  Max    : {max(lengths):,} bp")
    print(f"  Mean   : {sum(lengths)//len(lengths):,} bp")
    print(f"  Median : {sorted(lengths)[len(lengths)//2]:,} bp")

    # Bucket into meaningful ranges
    buckets = Counter()
    for l in lengths:
        if l < 500:        buckets["<500bp"] += 1
        elif l < 1000:     buckets["500-999bp"] += 1
        elif l < 2000:     buckets["1000-1999bp"] += 1
        elif l < 3000:     buckets["2000-2999bp (IN amplicon)"] += 1
        elif l < 5000:     buckets["3000-4999bp (full pol)"] += 1
        elif l < 10000:    buckets["5000-9999bp (full pol+flanking)"] += 1
        else:              buckets["≥10000bp (multi-gene)"] += 1

    print(f"\n  Length buckets:")
    for bucket, count in sorted(buckets.items()):
        print(f"    {bucket:40s}: {count:5d} ({count/total*100:.1f}%)")

    print(f"\n── GC CONTENT ──")
    print(f"  Mean GC: {sum(gc_contents)/len(gc_contents)*100:.1f}%")
    print(f"  (HIV-1 pol is ~40% GC — significantly higher suggests non-HIV)")

    if not stop_profiles:
        print("\n  [No reads sampled for stop codon analysis]")
        return

    print(f"\n── STOP CODON DENSITY ANALYSIS (first {len(stop_profiles)} reads) ──")
    print(f"  This is the KEY diagnostic. In the CORRECT frame, a real HIV pol")
    print(f"  read should have <5% stop codon density. Wrong frames: 15-30%.")
    print(f"  If ALL frames show <10%, the reads may NOT be standard HIV pol.")

    f0_dens = [p["f0_density"] for p in stop_profiles]
    f1_dens = [p["f1_density"] for p in stop_profiles]
    f2_dens = [p["f2_density"] for p in stop_profiles]
    margins  = [p["margin"] for p in stop_profiles]
    best_dens = [p["best_density"] for p in stop_profiles]

    for frame_name, dens in [("Frame 0", f0_dens), ("Frame 1", f1_dens), ("Frame 2", f2_dens)]:
        mean_d = sum(dens) / len(dens)
        print(f"\n  {frame_name}:")
        print(f"    Mean stop density : {mean_d*100:.2f}%")
        print(f"    Min stop density  : {min(dens)*100:.2f}%")
        print(f"    Max stop density  : {max(dens)*100:.2f}%")

    print(f"\n  Best-frame stop density (the winning frame per read):")
    mean_best = sum(best_dens) / len(best_dens)
    print(f"    Mean : {mean_best*100:.2f}%")
    print(f"    Min  : {min(best_dens)*100:.2f}%")
    print(f"    Max  : {max(best_dens)*100:.2f}%")

    mean_margin = sum(margins) / len(margins)
    print(f"\n  Frame discrimination margin (max_density - min_density per read):")
    print(f"    Mean : {mean_margin*100:.2f}%")
    print(f"    Min  : {min(margins)*100:.2f}%")
    print(f"    Max  : {max(margins)*100:.2f}%")
    print(f"\n  INTERPRETATION:")
    if mean_margin < 0.05:
        print(f"  ⚠️  CRITICAL: Mean margin < 5% — frames are indistinguishable.")
        print(f"     This indicates reads are NOT standard protein-coding HIV pol.")
        print(f"     Possible causes: reverse-complement reads, non-HIV contamination,")
        print(f"     or reads spanning non-coding regions.")
    elif mean_margin < 0.10:
        print(f"  ⚠️  WARNING: Mean margin < 10% — frame discrimination is weak.")
        print(f"     Nanopore error rate is likely masking the stop codon signal.")
        print(f"     Confidence scores will be unreliable for many reads.")
    else:
        print(f"  ✓  Frame discrimination looks viable. The framer's scoring")
        print(f"     logic may be the primary issue, not the data itself.")

    # Show the 5 reads with highest discrimination (the "best case")
    top5 = sorted(stop_profiles, key=lambda p: p["margin"], reverse=True)[:5]
    print(f"\n  TOP 5 reads with best frame discrimination:")
    for p in top5:
        print(
            f"    {p['read_id']:25s} len={p['length']:5d}bp  "
            f"F0={p['f0_density']*100:.1f}%  "
            f"F1={p['f1_density']*100:.1f}%  "
            f"F2={p['f2_density']*100:.1f}%  "
            f"margin={p['margin']*100:.1f}%  "
            f"→ frame {p['best_frame']}"
        )

    # Show the 5 worst (the "problem reads")
    bot5 = sorted(stop_profiles, key=lambda p: p["margin"])[:5]
    print(f"\n  BOTTOM 5 reads with worst frame discrimination:")
    for p in bot5:
        print(
            f"    {p['read_id']:25s} len={p['length']:5d}bp  "
            f"F0={p['f0_density']*100:.1f}%  "
            f"F1={p['f1_density']*100:.1f}%  "
            f"F2={p['f2_density']*100:.1f}%  "
            f"margin={p['margin']*100:.1f}%  "
            f"→ frame {p['best_frame']}"
        )

    # Frame bias check — if one frame wins consistently, it's the real frame
    frame_wins = Counter(p["best_frame"] for p in stop_profiles)
    print(f"\n  Best-frame vote (by lowest stop density):")
    for frame in [0, 1, 2]:
        count = frame_wins.get(frame, 0)
        print(f"    Frame {frame}: {count} reads ({count/len(stop_profiles)*100:.1f}%)")

    print(f"\n  INTERPRETATION:")
    dominant_frame = frame_wins.most_common(1)[0]
    dominance = dominant_frame[1] / len(stop_profiles)
    if dominance > 0.70:
        print(f"  ✓  Frame {dominant_frame[0]} wins {dominance*100:.0f}% of the time.")
        print(f"     There IS a real reading frame signal in the data.")
        print(f"     The CodonFramer logic needs to be fixed, not the data.")
    else:
        print(f"  ⚠️  No dominant frame — distribution is roughly uniform.")
        print(f"     This is consistent with mixed-orientation reads or")
        print(f"     reads not originating from protein-coding HIV pol sequence.")
        print(f"     CHECK: Are reverse-complement reads being handled?")

    print()


def main():
    test_files = [
        "data/test/fastq/DRR537715_1.fastq.gz",
        "data/test/fastq/SRR36194842_1.fastq.gz",
    ]

    found_any = False
    for f in test_files:
        if os.path.exists(f):
            analyze_file(f, sample_n=300)
            found_any = True
        else:
            print(f"SKIP (not found): {f}")

    if not found_any:
        print("\nNo test files found. Update the paths in main() and retry.")

    print("\n" + "="*70)
    print("NEXT STEPS BASED ON OUTPUT:")
    print("="*70)
    print("""
1. If mean margin < 5% across all reads:
   → The reads are likely in mixed orientations (fwd + rev complement).
   → Fix: Add RC detection in CodonFramer — try all 6 frames (3 fwd + 3 RC).
   → This is the most common cause of uniform frame scores.

2. If best-frame stop density is still >5% (even in the winning frame):
   → The localizer is passing non-pol reads through.
   → Fix: Tighten PolLocalizer thresholds OR add a post-localization
     stop-codon-based reject gate before the framer.

3. If one frame wins 70%+ of reads in the stop-density vote:
   → The CodonFramer's scoring logic is the bug, not the data.
   → Fix: Replace combined score with pure stop-density margin as the
     primary discriminator. The alignment score is adding noise.

4. If read lengths are >3000bp and localizer says everything is IN:
   → These are full-pol or full-genome reads. The localizer's IN anchor
     set is winning because IN is at the 3' end and the windowed
     scoring is biased toward it.
   → Fix: Implement subsequence extraction rather than whole-read typing.
""")


if __name__ == "__main__":
    main()