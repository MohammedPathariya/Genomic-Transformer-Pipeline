# Stanford HIVdb Genotype-Rx Dataset

Source: Stanford HIV Drug Resistance Database
URL: https://hivdb.stanford.edu/_wrapper/download/GenoRxDatasets/
Downloaded: 2026-04-18
Subset: First 5000 isolates per gene (header + 5000 rows)

Files:
  PR.txt  - Protease isolates with PI treatment history
  RT.txt  - RT isolates with NRTI/NNRTI treatment history
  IN.txt  - Integrase isolates with INSTI treatment history

Citation: Rhee et al., Stanford HIV Drug Resistance Database
https://hivdb.stanford.edu

Ground truth: P1...Pn columns contain amino acid at each position.
  '-' = wildtype (same as HXB2 consensus)
  letter = mutation at that position
  two letters = mixture (both amino acids present)

Note: NASeq column contains nucleotide sequence where available.
AccessionID column links to GenBank for full sequence retrieval.
