# On the limitations of molecular representations
This repository provides code for assessing learned molecular embeddings and classical molecular representations, as 
described in the following paper: https://arxiv.org/abs/2611.08420v2

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
descriptor vector and 9 molecular embedding models. Most molecular embedding models require different versions of Python
and popular machine-learning packages such as PyTorch and NumPy. We therefore suggest creating a separate conda environment
for each molecular embedding model. We provide YAML files required to recreate conda the environments used for this paper.
More details are available in the *project structure* section below.

## Project structure

There are seven primary folders for this project:
- **ADMET similarity**: contains ADMET endpoint similarity evaluations for the Biogen dataset (Extended Data Figures 3 and 4).
- **Butina clustering**: contains Butina cluster data leakage evaluations for the ECFP4 representation (Figure 5).
- **Datasets**: contains the MCES, ZINC30K and Biogen datasets used for the analyses in this paper. \
  Note that you will need to download ChEMBL37 SQLite to recreate the degeneracy analyses. The Q4 Enamine REAL database is no
  longer available on the Enamine website, but can be made available upon request.
- **Degeneracy analyses**: contains degeneracy analysis code for Enamine REAL and ChEMBL 37 (Figures 1 and 3).
- **MCES**: contains preprocessing and evaluations (Extended Data Figure 1).
- **Molecular embeddings**: contains YAML files and notebooks for generating molecular embeddings for ZINC, Biogen, and the MCES dataset.
- **Shape similarity**: contains code for evaluating correlations between cosine and Tanimoto similarities against maximal conformational overlap
  (Extended Data Figure 2).

The full project tree is given below:
<details>
<summary>📂 Click to expand project structure</summary>

```text
├── ADMET similarity
│   ├── ADMET similarity.ipynb
│   ├── Biogen embeddings
│   │   ├── ChemBERTa Biogen embeddings.npy
│   │   ├── CheMeleon Biogen embeddings.npy
│   │   ├── COATI Biogen embeddings.npy
│   │   ├── COATI Biogen invalid.npy
│   │   ├── MiniMol Biogen embeddings.npy
│   │   ├── MolBERT Biogen embeddings.npy
│   │   ├── MolBERT Biogen invalid.npy
│   │   ├── MolCLR Biogen embeddings.npy
│   │   ├── MolE Biogen embeddings.npy
│   │   ├── MolFormer Biogen embeddings.npy
│   │   └── UniMol Biogen embeddings.npy
│   └── ROGI correlation with performance.ipynb
├── Butina clustering
│   ├── Butina_clustering.ipynb
│   └── butina_network_outputs
│       ├── all_valid_mces_pairs_master.csv
│       ├── butina_network_0p40.pdf
│       ├── butina_network_0p40.png
│       ├── butina_network_0p40.svg
│       ├── mces_pairs_threshold_0p40.csv
│       └── tanimoto_sample.npy
├── Datasets
│   ├── biogen.csv
│   ├── mces_distances.npz
│   └── ZINC 30K.csv
├── Degeneracy analyses
│   ├── ChEMBL
│   │   ├── ChEMBL analysis grouped.ipynb
│   │   └── morgan_groups_valid_max_fold.csv
│   └── Enamine
│       ├── combine_run_histograms.py
│       ├── fingerprint_bitpacked.py
│       ├── hash_fingerprints.py
│       ├── preprocess_new.py
│       └── sort_bucket_and_hist.py
├── .gitattributes
├── MCES
│   ├── censor_mask.npy
│   ├── COATI invalid.npy
│   ├── MCES embeddings
│   │   ├── ChemBERTa MCES embeddings.npy
│   │   ├── CheMeleon MCES embeddings.npy
│   │   ├── COATI MCES embeddings.npy
│   │   ├── MiniMol MCES embeddings.npy
│   │   ├── MolBERT MCES embeddings.npy
│   │   ├── MolCLR MCES embeddings.npy
│   │   ├── MolE MCES embeddings.npy
│   │   ├── MolFormer MCES embeddings.npy
│   │   └── UniMol MCES embeddings.npy
│   ├── MCES hexbins.ipynb
│   ├── MCES preprocessing.ipynb
│   └── MolBERT valid.npy
├── Molecular embeddings
│   ├── Embedding notebooks
│   │   ├── censor_mask.npy
│   │   ├── ChemBERTa.ipynb
│   │   ├── CheMeleon.ipynb
│   │   ├── COATI.ipynb
│   │   ├── MiniMol.ipynb
│   │   ├── MolBERT checkpoint
│   │   ├── MolBERT.ipynb
│   │   ├── MolCLR.ipynb
│   │   ├── MolCLR_models
│   │   ├── MolE checkpoint
│   │   ├── MolE.ipynb
│   │   ├── MolFormer.ipynb
│   │   └── UniMol.ipynb
│   └── Environment YAML files
│       ├── ChemBERTa_environment.yml
│       ├── CheMeleon_environment.yml
│       ├── COATI_environment.yml
│       ├── MiniMol_environment.yml
│       ├── MolBERT_environment.yml
│       ├── MolCLR_environment.yml
│       ├── MolE_environment.yml
│       ├── MolFormer_environment.yml
│       └── UniMol_environment.yml
└── Shape similarity
    ├── Shape and color scoring.ipynb
    ├── shape_color_similarity_equal_width_results
    │   ├── all_equal_width_interval_definitions.csv
    │   ├── all_equal_width_interval_diagnostics.csv
    │   ├── all_model_shape_color_spearman_correlations.csv
    │   ├── all_sampled_pairs_shape_color_scores.csv
    │   ├── Embedding__shape_color_correlation_checkpoint.csv
    │   ├── Fingerprint__shape_color_correlation_checkpoint.csv
    │   ├── interval_diagnostics
    │   └── sampled_pairs
    └── ZINC30K
        ├── Embeddings
        └── Fingerprints
```
</details>

## Citing this work

If you wish to cite this work, please use the following reference:

```bibtex
@article{Blake2026,
  title   = {On the limitations of molecular representations},
  author  = {Blake, Acer and Imrie, Fergus and M. Deane, Charlotte},
  journal = {ArXiV},
  year    = {2026},
  doi     = {https://doi.org/10.48550/arXiv.2611.08420}
}
```


