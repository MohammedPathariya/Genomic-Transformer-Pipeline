# Stanford HIVdb Genotype-Rx Dataset — Expanded 15K

Source: Stanford HIV Drug Resistance Database
URL: https://hivdb.stanford.edu/_wrapper/download/GenoRxDatasets/
Downloaded: 2026-04-21
Subset: First 15,000 isolates for PR and RT, full database for IN

Files:
  PR_resistant.txt  — header + 15,000 PR isolates
  RT_resistant.txt  — header + 15,000 RT isolates
  IN_resistant.txt  — full IN database (~25,916 rows)

Purpose: Expanded validation dataset for comparison against the
3,000-sequence baseline run (data/raw/stanford_hivdb/).
Treatment-experienced patients only (PIList/RTIList/INIList != None).
NASeq column contains nucleotide sequence used as pipeline input.

Citation: Rhee et al., Stanford HIV Drug Resistance Database
https://hivdb.stanford.edu
