# Project References & Foundational Literature
This document tracks the core research papers and technical documentation used to develop the Genomic-Transformer-Pipeline.

## 1. Core Lab & Project Literature
### BioReason: Incentivizing Multimodal Biological Reasoning within a DNA-LLM Model
Citation: Fallahpour, A., et al. (2025). arXiv:2505.23579 [cs.LG].

Role: Provides the architectural foundation for the specific DNA-based Large Language Models used in this lab's context.

Summary: Introduces a framework that integrates a DNA foundation model with an LLM to enable multi-step, interpretable biological reasoning.

Key Takeaway: The model uses supervised fine-tuning and reinforcement learning to explain decisions step-by-step. This is the blueprint for making HIV mutation detection "explainable" rather than just a black-box prediction.

Relevance: Direct link to the bioreason-edge repository currently under study.

### Update and Latest Advances in Antiretroviral Therapy
Citation: Menéndez-Arias, L., et al. (2022). Trends in Pharmacological Sciences, 43(1), 16-29.

Role: Clinical background on drug resistance.

Summary: A comprehensive review of current HIV-1 treatment strategies and the mechanisms of drug resistance.

Key Takeaway: Highlights the ongoing need for rapid, accurate resistance testing to guide clinical decisions.

Relevance: Provides the clinical "ground truth" for why the mutations we are detecting (DRMs) matter for patient outcomes.

## 2. Technical Databases & Standards
### Stanford HIV Drug Resistance Database (HIVdb)
URL: https://hivdb.stanford.edu/

Utility: Source for ground-truth mutation scoring and penalty rules.

### Oxford Nanopore Technologies (ONT) Documentation
URL: https://nanoporetech.com/

Utility: Technical specifications for FASTQ/Fast5/POD5 data structures and basecalling protocols.

Resource Library: ONT Resource Library

## 3. Comparative Methodology (External)
Planned Additions:

DNABERT: Pre-trained bidirectional encoder representations from DNA.

HyenaDNA: Long-context genomic language modeling.

Caduceus: Bi-directional Mamba for genomic understanding.