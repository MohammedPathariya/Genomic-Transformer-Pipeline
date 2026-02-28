# Genomic-Transformer-Pipeline

An AI-driven diagnostic pipeline for real-time HIV Drug Resistance Mutation (DRM) detection from Oxford Nanopore sequencing data. Built at Professor Weihua Guan's Lab, Indiana University Bloomington.

The core thesis: existing clinical tools require a completed reference-genome alignment before they can call a single resistance mutation — a process that takes 10–20 minutes, fails on non-subtype-B sequences, and produces binary calls with no uncertainty. This pipeline replaces that workflow with an alignment-free Transformer-based reasoning engine that is robust to Nanopore's 3–10% error rate, generalizes across HIV-1 subtypes, and produces probabilistic resistance calls with clinical confidence intervals.

---

## What This Pipeline Does — In Plain English

A blood sample is taken from an HIV-positive patient. The lab extracts the virus and runs it through an Oxford Nanopore MinION sequencer, which reads the genetic code of the virus and saves it as a raw signal file (POD5). That signal file enters one end of this pipeline. A clinical report — telling the doctor which HIV drugs the virus is still susceptible to and which it has become resistant to — comes out the other end.

Here is what happens in between, step by step.

**Step 1 — Signal to Sequence (basecaller.py)**
The raw electrical signal from the sequencer is converted into DNA letters (ATCG) using Oxford Nanopore's Dorado software. The output is a FASTQ file — a standard format containing the DNA sequence of each viral read along with a confidence score for every base call. This step only runs when starting from raw device output. If you already have FASTQ files from a public database, this step is skipped.

**Step 2 — Reading the Files (stream_reader.py)**
The pipeline accepts FASTQ files from the sequencer, FASTA files from databases like Stanford HIVdb and LANL, and BAM files from legacy pipelines. Regardless of the source format, every read is converted into the same standardized internal format — a RawRead object — so that everything downstream never needs to know where the data came from.

**Step 3 — Throwing Out Bad Reads (quality_filter.py)**
Not every read is worth analyzing. Some are too short to contain meaningful information. Some are too noisy. Some have too many ambiguous bases. This step filters those out, keeping only reads that meet minimum quality thresholds. Every discarded read is logged with its reason so nothing is silently lost.

**Step 4 — Managing Large Batches (batch_processor.py)**
In a real clinical setting, dozens of patient samples may need processing simultaneously. This step manages running the pipeline across many files at once — handling failures gracefully, tracking progress, writing structured logs, and supporting resumability if the run is interrupted halfway through.

**Step 5 — Finding the Right Gene (pol_localizer.py)**
The HIV genome contains several genes. Drug resistance mutations only occur in one of them — the pol gene, which has three sub-regions: Protease (PR), Reverse Transcriptase (RT), and Integrase (IN). Each drug class targets one of these regions. This step identifies which region each read comes from using a fast k-mer matching approach — no alignment to a reference genome required. This is the first of three novel technical contributions.

**Step 6 — Reading the Code Correctly (codon_framer.py)**
DNA encodes proteins in triplets called codons. A resistance mutation like K103N means the amino acid at position 103 changed from lysine to asparagine. To detect these changes you must read the DNA in the right triplets — there are three possible starting positions, and the wrong one produces nonsense. This step figures out the correct reading frame for each read.

**Step 7 — Packaging for the AI (feature_builder.py)**
The AI model cannot read DNA strings directly. This step converts each localized, frame-resolved read into numerical features — k-mer frequency vectors, quality profiles, and the cleaned pol subsequence — that the model can reason about.

**Step 8 — Understanding the Sequence (dna_encoder.py)**
A large pretrained DNA foundation model (Evo2 or the Nucleotide Transformer) reads the pol subsequence and produces a rich numerical representation of its meaning — capturing long-range dependencies across the full gene. This model's weights are frozen; we borrow its knowledge without retraining it. This is the second novel contribution: treating DRM detection as a reasoning problem rather than a lookup.

**Step 9 — Making the Resistance Call (reasoning_head.py + drm_head.py)**
A lightweight Transformer reasoning head attends over the DNA representation and learns the patterns that distinguish resistant sequences from susceptible ones — not just individual mutations but co-occurring mutation combinations that a dictionary lookup cannot capture. A classification head then produces a resistance probability for each drug class: NRTI, NNRTI, PI, and INSTI.

**Step 10 — Accounting for Viral Diversity (confidence.py + aggregator.py)**
HIV exists in a patient as a swarm of related variants, not a single sequence. A resistance mutation present in 23% of viral copies may be clinically significant for one drug class but not another. This step pools predictions across all reads from one patient and computes per-drug-class resistance calls with variant frequency estimates and confidence intervals. This is the third novel contribution: quasispecies-aware probabilistic output.

**Step 11 — The Clinical Report (report_generator.py)**
The final output is a human-readable PDF report and a machine-readable JSON file. The report shows resistance levels (Susceptible / Intermediate / Resistant) per drug class, the variant frequencies that drove each call, and quality control metrics. A clinician reads the report and decides which drugs to use for that patient.

---

## The Three Novel Contributions

| Contribution | What Existing Tools Do | What This Pipeline Does |
|---|---|---|
| Alignment-Free Localization | Requires Minimap2/BWA alignment to HXB2 — 10-20 min latency | K-mer seed matching in sub-second, no reference genome needed |
| Sequence Reasoning | Dictionary lookup: codon → resistance score, no context | Transformer reasoning over full pol gene context |
| Quasispecies Output | Binary call: mutation present or absent | Frequency estimate + confidence interval per drug class |

---

## Repository Structure

```
Genomic-Transformer-Pipeline/
├── src/
│   ├── ingestion/              # File reading, signal conversion, quality control
│   │   ├── basecaller.py       # Dorado wrapper: POD5 → FASTQ
│   │   ├── stream_reader.py    # Universal FASTQ/FASTA/BAM parser
│   │   ├── quality_filter.py   # Read quality filtering
│   │   └── batch_processor.py  # Multi-file orchestration and logging
│   ├── enricher/               # Alignment-free bioinformatics (no GPU needed)
│   │   ├── pol_localizer.py    # K-mer based PR/RT/IN localization
│   │   ├── codon_framer.py     # Reading frame resolution
│   │   └── feature_builder.py  # Feature payload assembly
│   ├── inference/              # AI core (GPU required)
│   │   ├── dna_encoder.py      # Frozen DNA foundation model
│   │   ├── projection.py       # Learnable embedding bridge
│   │   └── reasoning_head.py   # Transformer resistance reasoning
│   ├── classification/         # Resistance calling
│   │   ├── drm_head.py         # Multi-label drug class classifier
│   │   └── confidence.py       # Uncertainty quantification
│   ├── output/                 # Report generation
│   │   ├── aggregator.py       # Read-level → sample-level pooling
│   │   └── report_generator.py # JSON + clinical PDF output
│   ├── training/               # Offline training (not used during inference)
│   │   ├── dataset.py          # Dataset registry and loaders
│   │   └── trainer.py          # Training loop
│   └── config/
│       └── pipeline_config.yaml  # All parameters, thresholds, and paths
│
├── data/                       # Local only — not tracked by git
│   ├── raw/pod5/               # Raw POD5 files from sequencer
│   ├── basecalled/             # FASTQ output from Dorado
│   ├── test/                   # Test FASTQ/BAM files
│   ├── processed/              # JSONL output from ingestion pipeline
│   └── public/                 # HXB2 reference genome
│
├── docs/
│   ├── ARCHITECTURE.md         # Full system architecture and Mermaid diagrams
│   └── references.md           # Research bibliography
│
├── logs/                       # Runtime batch processing logs (not tracked)
├── results/                    # Clinical report outputs (not tracked)
├── experiments/                # Exploratory notebooks (local only)
└── requirements.txt            # Python dependencies
```

---

## Build Phases

The project is organized into three sequential phases:

**Part 1 — Ingestion Pipeline** *(in progress)*
Data engineering layer. Converts raw sequencing files into clean, biologically annotated records ready for model training. No machine learning dependencies — runs entirely on CPU and can be validated before any model is built.

**Part 2 — Dataset Construction** *(upcoming)*
Collect sequences from Stanford HIVdb, LANL HIV Database, ENA, and NCBI GenBank. Run through Part 1, attach resistance labels, balance across HIV-1 subtypes (A, B, C, D minimum), and produce the training corpus.

**Part 3 — Model and Output** *(upcoming)*
Train the projection layer, reasoning head, and classification head on the Part 2 corpus. Wire up the clinical report generator. Optimize for edge deployment on NVIDIA Jetson AGX Orin via TensorRT.

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.10+ |
| ML Frameworks | PyTorch, HuggingFace Transformers |
| DNA Foundation Models | Nucleotide Transformer (InstaDeepAI), Evo2 (Arc Institute) |
| Bioinformatics | BioPython, pysam, minimap2, Dorado |
| Data | NumPy, Pandas, PyYAML |
| Hardware Target | NVIDIA Jetson AGX Orin (edge), HPC cluster (training) |
| Optimization | TensorRT, INT8 quantization, model pruning |

---

## Development Setup

**Clone the repository:**
```bash
git clone https://github.com/MohammedPathariya/Genomic-Transformer-Pipeline.git
cd Genomic-Transformer-Pipeline
```

**Create and activate the environment:**
```bash
conda create -n genomic-env python=3.10
conda activate genomic-env
pip install -r requirements.txt
```

**Download test data:**
See `data/README.md` for instructions on downloading the test FASTQ files and HXB2 reference genome.

**Run the ingestion pipeline on test data:**
```bash
python -m src.ingestion.batch_processor
```

**Check the output:**
```bash
ls data/processed/       # JSONL files — one per input file
cat logs/run_*.json      # Structured run log with timing and stats
```

---

## Data Sources

| Source | URL | What It Provides |
|---|---|---|
| Stanford HIVdb | hivdb.stanford.edu | Gold standard resistance labels |
| LANL HIV Database | hiv.lanl.gov | Curated subtype-annotated sequences |
| European Nucleotide Archive | ebi.ac.uk/ena | Nanopore FASTQ datasets |
| NCBI GenBank | ncbi.nlm.nih.gov | HIV-1 pol sequences across subtypes |
| IAS-USA | iasusa.org | Annual clinical mutation list |

Raw sequencing data and clinical outputs are never committed to this repository.

---

## References

See `docs/references.md` for the full research bibliography including the BioReason paper, Nucleotide Transformer, Evo2, and Stanford HIVdb documentation.