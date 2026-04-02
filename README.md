# Antibody Property Prediction

## What

This project benchmarks two protein language models -- ESM-2 (Meta, general protein pretraining, 650M parameters) and AbLang2 (Oxford, antibody-specific pretraining on OAS) -- on two antibody property prediction tasks. It also includes a neurosymbolic CDR constraint loss that encodes the biological prior that CDR (complementarity-determining region) mutations should produce larger functional effects than framework region mutations.

The central question: does domain-specific pretraining (AbLang2) outperform general protein pretraining (ESM-2) for antibody-specific tasks? And does injecting a symbolic biological constraint improve predictions, and does it interact differently with each model's learned representations?

## Why

Antibodies are composed of structurally distinct regions: the framework (conserved structural scaffold) and CDRs (hypervariable loops directly involved in antigen binding). ESM-2 was trained on 250M diverse proteins from all kingdoms of life and has no specific exposure to antibody biology. AbLang2 was trained exclusively on antibody sequences via masked language modeling, making it antibody-aware by construction. Whether that specificity translates to better downstream performance on mutation effect and binding affinity prediction is an open empirical question.

The CDR constraint loss adds a further test: can explicitly encoding a biological prior (CDR mutations matter more than FR mutations) improve prediction quality? And does this interact differently with ESM-2, which preliminary EDA suggests encodes the *inverse* of this prior geometrically?

## Tasks

**Task 1: Mutation effect prediction (AbAgym)**
- 5,318 single-point mutations across 5 antibodies
- Datasets: Ang2_2017_G6, EGFR_2013_Cetuximab, lysozyme_2019_D441, VEGF_2017b_G6, HER2_2021_trastuzumab
- Label: MinMax-normalized DMS score [0, 1]
- Metric: Spearman correlation, per-dataset and aggregate

**Task 2: Binding affinity prediction (SAbDab)**
- 491 antibody-antigen pairs
- Label: pKd = -log10(Kd)
- Metric: Pearson correlation + RMSE

## Project Structure

```
antibody-property-prediction/
notebooks/          -- thin orchestration layers; all logic lives in src/
    01_data_pipeline.ipynb
    02_model_exploration.ipynb
    03_embedding_generation.ipynb
    04_embedding_eda.ipynb
    05_training.ipynb
    06_analysis.ipynb
src/
    config.py       -- paths, device, constants
    data/           -- CSV loaders and PyTorch Dataset classes
    embeddings/     -- ESM-2, AbLang2, delta computation
    models/         -- MLP prediction head
    training/       -- training loop, losses (including CDR constraint)
    visualization/  -- figure generation
```

## Setup

```bash
git clone https://github.com/Aaron1776/antibody-property-prediction.git
cd antibody-property-prediction
git checkout implementation
pip install -r requirements.txt
```

ANARCI requires HMMER to be installed before the pip install. On Colab:
```bash
apt-get install -y hmmer
pip install anarci
```

See CONTRIBUTING.md for full Colab session setup and Drive configuration.

## How to Run

The notebooks are designed to be run in sequence in Google Colab:

1. `01_data_pipeline.ipynb` -- download and process raw data, produce CSVs on Drive
2. `02_model_exploration.ipynb` -- inspect model APIs, verify residue indexing
3. `03_embedding_generation.ipynb` -- generate and cache all embedding tensors
4. `04_embedding_eda.ipynb` -- exploratory analysis of embedding geometry
5. `05_training.ipynb` -- run experiment matrix, train all models
6. `06_analysis.ipynb` -- produce final figures and tables

All logic (embedding extraction, dataset loading, training loops) lives in `src/`. Notebooks import from `src/` and orchestrate calls. No modeling code in notebook cells.

## Data Sources

- AbAgym: github.com/3BioCompBio/AbAgym
- SAbDab: Zenodo DOI 10.5281/zenodo.13120765 (Apache 2.0)
