# On-the-limitations-of-molecular-representations
This repository provides code for assessing learned and classical molecular representations.

## Abstract

Molecular representations such as fingerprints and learned embeddings play a central role in chemin-
formatics and small-molecule machine learning. They serve as the primary input for a diverse array of
tasks, including clustering, dataset splitting, and virtual screening, yet their evaluation is often limited to
downstream model performance on narrow benchmark tasks. We carried out a systematic investigation
of the quality of molecular representations, finding that none of the state-of-the-art molecular embedding
models capture graph structure, 3D shape, or ADMET endpoint similarity better than fingerprints or
physicochemical descriptors. However, classical fingerprints also revealed their own material limitations.
For example, we show that depending on the representation, between 225M and ∼4B molecules have
non-unique representations in the 10.8B Enamine REAL database, with implications for virtual screening.
Our findings provide an insight into the current limitations of molecular representations, outlining future
research directions for the development of more robust encodings of small molecules.
