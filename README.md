# On-the-limitations-of-molecular-representations
This repository provides code for assessing learned and classical molecular representations.

## Abstract

Molecular representations such as fingerprints and learned embeddings play a central role in cheminformatics 
and small-molecule machine learning. They serve as the primary input for a diverse array of
tasks, including clustering, dataset splitting, and virtual screening, yet their evaluation is often limited to
downstream model performance on narrow benchmark tasks. We carried out a systematic investigation
of the quality of molecular representations, finding that none of the state-of-the-art molecular embedding
models capture graph structure, 3D shape, or ADMET endpoint similarity better than fingerprints or
physicochemical descriptors. However, classical fingerprints also revealed their own material limitations.
For example, we show that depending on the representation, between 225M and ∼4B molecules have
non-unique representations in the 10.8B Enamine REAL database, with implications for virtual screening.
Our findings provide an insight into the current limitations of molecular representations, outlining future
research directions for the development of more robust encodings of small molecules.

## Requirements 

This repository contains evaluations for 14 molecular representations, including four fingerprints, a physicochemical 
descriptor vector and 9 molecular embedding models. We provide YAML files required to recreate conda environments
for each molecular embedding model.

## Project structure

<details>
<summary>📂 Click to expand project structure</summary>

```text
├── ADMET similarity
│   ├── ADMET similarity.ipynb
│   └── ROGI correlation with performance.ipynb
├── Butina clustering
│   └── Butina clustering.ipynb
├── Degeneracy analyses
│   ├── ChEMBL
│   │   └── ChEMBL analysis grouped.ipynb
│   └── Enamine
│       ├── combine_run_histograms.py
│       ├── fingerprint_bitpacked.py
│       ├── hash_fingerprints.py
│       ├── preprocess_new.py
│       └── sort_bucket_and_hist.py
├── MCES
│   ├── censor_mask.npy
│   ├── COATI invalid.npy
│   ├── MCES preprocessing.ipynb
│   └── MolBERT valid.npy
├── Molecular embeddings
│   ├── censor_mask.npy
│   ├── ChemBERTa.ipynb
│   ├── CheMeleon.ipynb
│   ├── COATI invalid.npy
│   ├── COATI.ipynb
│   ├── COATI MCES embeddings.npy
│   ├── Environment YAML files
│   │   ├── chemberta_environment.yml
│   │   ├── chemeleon_environment.yml
│   │   ├── coati_environment.yml
│   │   ├── minimol_environment.yml
│   │   ├── molbert_environment.yml
│   │   ├── molclr_environment.yml
│   │   ├── mole_environment.yml
│   │   ├── molformer_environment.yml
│   │   └── unimol_environment.yml
│   ├── logs
│   │   └── unimol_tools_2026-09-28-14-17-43.log
│   ├── Minimol.ipynb
│   ├── models
│   │   ├── barlow_closed.pkl
│   │   ├── gcn_finetune.py
│   │   ├── gcn_molclr.py
│   │   ├── ginet_finetune.py
│   │   ├── ginet_molclr.py
│   │   └── __pycache__
│   ├── MolBERT invalid.npy
│   ├── MolBERT.ipynb
│   ├── MolCLR.ipynb
│   ├── MolE_GuacaMol_27113.ckpt
│   ├── MolE.ipynb
│   ├── MolFormer.ipynb
│   └── UniMol.ipynb
├── README.md
└── Shape similarity
    └── Shape and color scoring.ipynb
```
<details>

## Citing this work



