# Source Code Structure Reference

This document explains every file in the `src/` directory — what it does, why it exists, and how it connects to the rest of the pipeline. Use this as your reference whenever you are lost about where a piece of logic belongs or what a module is responsible for.

---

## The Pipeline Flow

Every file in this codebase corresponds to one step in a linear data flow:

```
[POD5] → basecaller → stream_reader → quality_filter → pol_localizer → codon_framer →
feature_builder → dna_encoder → projection → reasoning_head →
drm_head → confidence → aggregator → report_generator
```

The data contract between every step from `stream_reader` onward is the standardized `RawRead` dataclass defined in `stream_reader.py`. Nothing downstream ever asks what format a read came from — it only ever sees a `RawRead`.

The `basecaller.py` step is optional — it only runs when the input is a raw POD5 file from a MinION sequencer. If you already have FASTQ files (from ENA, SRA, or a previous basecalling run), the pipeline starts at `stream_reader.py` directly.

---

## Directory Overview

```
src/
├── ingestion/       # Part 1 — The front door. Signal conversion, format parsing, quality control.
├── enricher/        # Part 1 — Alignment-free bioinformatics. No neural networks.
├── inference/       # Part 3 — The AI core. DNA encoder + reasoning head.
├── classification/  # Part 3 — Resistance calling and uncertainty quantification.
├── output/          # Part 3 — Aggregation and clinical report generation.
├── training/        # Part 3 — Offline training loop. Not used during inference.
└── config/          # Global configuration. All parameters live here.
```

---

## `src/ingestion/` — The Front Door

This is Part 1 of the pipeline. Nothing enters the system without passing through here. This layer knows about file formats and signal conversion. Everything downstream is format-blind.

### `basecaller.py`

**What it does:** A thin wrapper around Oxford Nanopore's Dorado basecaller. Accepts a POD5 file path, invokes Dorado as a subprocess, and returns the path to the resulting FASTQ file. This is the only file in the pipeline that knows anything about POD5 format or Dorado.

**When it runs:** Only when the input is a raw POD5 file directly from a MinION/PromethION sequencer. If you already have FASTQ files, this module is skipped entirely. The rest of the pipeline is completely unaffected by whether this step ran.

**Why a subprocess and not a Python library:** Dorado is a standalone C++/CUDA binary. Oxford Nanopore does not provide a Python API for it. Every production Nanopore pipeline (including Nextflow-based ones) calls it via subprocess. This is the standard approach.

**Why this is separate from stream_reader.py:** Single responsibility. Basecalling converts electrical signal to nucleotide sequences — that is a signal processing concern. Reading sequences from files is a format parsing concern. Keeping them separate means the pipeline can run without Dorado installed if FASTQ files are already available.

**Key outputs:**
- A `BasecallResult` dataclass with the path to the output FASTQ file
- Timing, read count, Dorado version, and error information for logging

**Configuration (in pipeline_config.yaml):**
```yaml
basecalling:
  dorado_executable: "dorado"
  dorado_model:      "dna_r10.4.1_e8.2_400bps_hac@v4.3.0"
  device:            "cpu"        # "metal" on M1/M2 Mac, "cuda:0" on HPC
  batch_size:        64
  output_dir:        "data/basecalled"
```

**Dorado installation:** Download from https://github.com/nanoporetech/dorado/releases

### `stream_reader.py`

**What it does:** Opens FASTQ, FASTA, and BAM files and reads them record by record. Converts every read into a `RawRead` dataclass regardless of source format. This is the only file in the entire codebase that knows anything about file format internals.

**Key design decisions:**
- `read_id` is normalized to the first whitespace-delimited token of the header
- `quality` is always a `list[int]` of Phred scores — never `None`
- FASTA reads get an inferred quality of Q40 (high confidence, since LANL sequences are curated)
- `quality_is_inferred = True` flags any read whose quality was assigned rather than measured
- `raw_header` preserves the full original header for traceability

**Input sources:**
- FASTQ from Dorado (via `basecaller.py`) for real clinical samples
- FASTQ from ENA/SRA for publicly available Nanopore datasets
- FASTA from LANL/Stanford HIVdb for clean reference sequences
- BAM for legacy pipeline validation

**Output:** A generator that yields `RawRead` objects one at a time (streaming, not loading all into memory).

**The `RawRead` Dataclass:**
```python
@dataclass
class RawRead:
    read_id:             str        # normalized: first token only
    sequence:            str        # nucleotide string "ATCG..."
    quality:             list[int]  # Phred scores as integers
    quality_is_inferred: bool       # True if quality was assigned, not measured
    source_format:       str        # "fastq", "fasta", or "bam"
    source_file:         str        # original filename for traceability
    raw_header:          str        # full original header line, preserved
```

### `quality_filter.py`

**What it does:** Receives a stream of `RawRead` objects and drops reads that are too low quality to be useful. Applies four checks: minimum read length, maximum read length, minimum average Phred score, and maximum fraction of ambiguous N bases. Reads that pass are forwarded to the enricher. Reads that fail are logged and discarded with their failure reason.

**Key design decision:** Reads where `quality_is_inferred = True` (FASTA sources) skip the Phred score check — you cannot filter on quality that was invented. Only the length and N-fraction checks apply to FASTA reads.

**Output:** A generator of passing `RawRead` objects plus a `FilterStats` object tracking total/passed/failed counts and failure breakdowns.

### `batch_processor.py`

**What it does:** Orchestrates running the full single-file pipeline (`stream_reads → quality_filter → write to disk`) across many files simultaneously. Handles fault isolation (one bad file never kills the batch), progress tracking, structured JSON run logging, and checkpoint-based resumability.

**Execution modes:**
- `"sequential"` — one file at a time, safe on MacBook Air M1
- `"parallel"` — `ProcessPoolExecutor` with configurable workers, designed for HPC

**Key outputs:**
- Per-file JSONL files in `data/processed/`
- A structured JSON run log in `logs/` with full timing, read counts, filter stats, and checkpoint state

**Race condition protection:** A threading `Lock` guards all writes to the shared `RunLog` object. In sequential mode the lock is never contended. In parallel mode it ensures two workers cannot corrupt the log simultaneously.

---

## `src/enricher/` — The Bioinformatics Brain

This is the alignment-free localization layer. **No neural networks exist in this directory.** Pure bioinformatics logic, independently testable without any GPU or model weights. This layer can be validated in complete isolation before the inference engine is built.

The hard separation between this layer and `src/inference/` is the most important architectural boundary in the system. See `docs/ARCHITECTURE.md` for the full explanation.

### `pol_localizer.py`

**What it does:** Takes a `RawRead` and determines whether it plausibly originates from the HIV-1 *pol* gene, and if so, which region — Protease (PR), Reverse Transcriptase (RT), or Integrase (IN). Uses k-mer seed matching against a small set of highly conserved anchor sequences from each region.

**What it does NOT do:** Full reference alignment. There is no minimap2, no BWA, no coordinate system. This is a fast lookup, not an aligner.

**Why this matters:** This is Novel Component 1 of the research contribution. Every existing clinical tool requires completed alignment before calling a single mutation. This module replaces that 10–20 minute step with a sub-second k-mer matching operation.

**Output:** An annotated read with a `gene_region` field (`"PR"`, `"RT"`, `"IN"`, or `"unknown"`) and a localization confidence score.

### `codon_framer.py`

**What it does:** Takes a pol-localized read and determines the correct reading frame. DNA is read in triplets (codons) but there are three possible starting positions — frame 0, frame 1, frame 2. The wrong frame produces nonsense amino acids. This file evaluates all three frames and scores each against known *pol* codon distributions, selecting the most biologically consistent one.

**Why this matters:** Without correct reading frame assignment, codon-level mutation detection is impossible. This module is what allows downstream components to reason about amino acid changes (e.g. K103N) rather than raw nucleotide differences.

### `feature_builder.py`

**What it does:** Takes the localized, frame-resolved read and assembles the final enricher output payload. This payload is the handoff package between the bioinformatics layer and the inference engine.

**What the payload contains:**
- Cleaned nucleotide sequence
- Identified gene region (PR / RT / IN)
- Resolved reading frame (0, 1, or 2)
- K-mer frequency vectors
- Per-position quality profile
- Original `RawRead` for full traceability

---

## `src/inference/` — The AI Core

This is the BioReason-inspired engine. The most computationally expensive layer. This corresponds to Novel Component 2 of the research contribution — treating DRM detection as a sequence reasoning problem rather than a lookup problem.

### `dna_encoder.py`

**What it does:** A wrapper around a frozen pretrained DNA foundation model — either Evo2 or the Nucleotide Transformer (NT). Takes the enriched sequence from `feature_builder.py` and converts it into high-dimensional contextual embeddings that capture long-range sequence dependencies across the full *pol* gene.

**Critical design point:** The weights of this model are **frozen**. We never train this component. It is used purely as a feature extractor. Only the projection layer and reasoning head are trained.

**Why frozen:** Pretraining a DNA foundation model from scratch requires billions of nucleotides and weeks of compute. Evo2 and NT were trained on the entire known genome database. We inherit that knowledge for free and only train the task-specific layers on top.

### `projection.py`

**What it does:** A single learnable linear layer that maps DNA embeddings from the encoder's dimensional space into the dimensional space expected by the reasoning head. This is the architectural bridge between the frozen DNA world and the trainable reasoning world.

**This is directly adapted from the BioReason paper**, which uses an identical projection layer to connect Evo2/NT embeddings to a Qwen3 LLM backbone.

### `reasoning_head.py`

**What it does:** A lightweight Transformer that attends over the projected DNA embeddings and produces a resistance-relevant sequence representation. This is where the model learns the "genomic grammar" of resistance — understanding not just individual mutations but how co-occurring mutations across the pol gene relate to drug resistance pathways.

**What makes this novel:** Current tools treat DRM detection as a dictionary lookup — align a read, find the codon, check it against a table. This module treats it as a reasoning problem, learning contextual patterns that a lookup table fundamentally cannot capture.

---

## `src/classification/` — The Decision Layer

### `drm_head.py`

**What it does:** Takes the output of the reasoning head and produces resistance probability scores. It is a **multi-label classifier** — one output per drug class. The drug classes are: NRTI (Nucleoside RT Inhibitors), NNRTI (Non-Nucleoside RT Inhibitors), PI (Protease Inhibitors), and INSTI (Integrase Strand Transfer Inhibitors).

**Why multi-label:** A single patient read can show resistance to multiple drug classes simultaneously. This is common in treatment-experienced patients who have accumulated mutations under selection pressure from multiple drugs. A multi-class (single-label) classifier would force a choice and lose this information.

**Output:** A probability vector, not a binary call. `[NRTI: 0.82, NNRTI: 0.91, PI: 0.12, INSTI: 0.05]` for a single read.

### `confidence.py`

**What it does:** Quantifies uncertainty in the classifier's predictions by analyzing the spread of predictions across all reads from the same patient sample. Computes per-drug-class confidence intervals.

**Why this matters:** This is Novel Component 3 of the research contribution. Current tools report mutations as present or absent — binary. This module enables outputs like "K103N detected at 23% frequency [CI: 18%–29%], exceeding the 20% NNRTI clinical threshold." That is a clinically meaningful statement that the legacy pipeline cannot produce.

---

## `src/output/` — The Reporting Layer

### `aggregator.py`

**What it does:** A single read is not a clinical result. This file pools read-level predictions across all reads from one patient sample. Applies drug-class-specific frequency thresholds (e.g. 20% for NNRTIs, 1% for certain PIs) based on Stanford HIVdb clinical evidence. Produces a sample-level resistance summary.

**Key design point:** Drug-class thresholds are loaded from `config/pipeline_config.yaml`, not hardcoded. Changing a clinical threshold never requires touching source code.

### `report_generator.py`

**What it does:** Takes the aggregated sample-level result and renders two output formats. A structured JSON file for programmatic downstream use, and a human-readable clinical report with resistance levels (Sensitive / Intermediate / Resistant) per drug class, variant frequency estimates, confidence intervals, and quality control metrics.

---

## `src/training/` — The Offline Training Loop

These files are only used during model training. They are not part of the inference pipeline.

### `dataset.py`

**What it does:** Manages the training corpus assembled in Part 2 of the project. Handles loading from the dataset registry, applying train/validation/test splits, enforcing subtype balance across batches, and feeding data to the trainer. Interfaces directly with the Part 1 ingestion pipeline to process raw source files on demand.

### `trainer.py`

**What it does:** The training loop. Handles forward passes through the full model (encoder → projection → reasoning head → classification head), loss computation, backpropagation through the trainable layers only, learning rate scheduling, checkpoint saving, and evaluation metric tracking.

**What is trained:** Only `projection.py`, `reasoning_head.py`, and `drm_head.py`. The DNA encoder in `dna_encoder.py` remains frozen throughout.

---

## `src/config/pipeline_config.yaml`

**What it does:** Holds every configurable parameter in the entire pipeline. No magic numbers exist anywhere in the source code — every threshold, path, model name, and hyperparameter is defined here.

**Sections:**
- `basecalling` — Dorado executable path, model, device, batch size
- `ingestion` — input directories, output directory, execution mode, worker count
- `quality_filter` — length thresholds, Phred minimum, N-fraction maximum
- `enricher` — k-mer size, seed hit thresholds, HXB2 gene region coordinates
- `inference` — DNA encoder model name, max sequence length, batch size
- `classification` — drug-class resistance frequency thresholds
- `output` — report formats, output directories, reference genome info

**Rule:** If you find yourself writing a number directly in source code, it belongs in this file instead.

---

## Architectural Boundaries

**The most important boundary** is between `src/enricher/` and `src/inference/`. These two layers communicate only through the feature payload produced by `feature_builder.py`. Neither knows anything about the other's internals.

**The second important boundary** is between `src/ingestion/basecaller.py` and `src/ingestion/stream_reader.py`. Basecalling is a signal processing concern. Format reading is a parsing concern. They are decoupled so that the pipeline works with or without Dorado installed.

These boundaries give us three concrete engineering benefits:
- The enricher can be tested and validated with zero ML dependencies
- The DNA encoder can be swapped (Evo2 → NT → custom) without touching the enricher
- On edge hardware, the enricher runs on CPU while the GPU handles inference in parallel

For the full research context and novelty framing of these boundaries, see `docs/ARCHITECTURE.md`.