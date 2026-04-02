# Project Progress

## Project Overview

Deep Learning final project (course 5922, 30% of grade). Two collaborators: Oscar (biology + ML background) and Lucas (CS, neurosymbolic AI). Comparing ESM-2 (general protein language model, Meta, 650M params) vs AbLang2 (antibody-specific, Oxford, trained on OAS). Two tasks: mutation effect prediction (AbAgym) and binding affinity prediction (SAbDab). Neurosymbolic component: CDR constraint loss applied to both models.

Note: this document is the living record of decisions and findings. It is the best available expectation of project state, but claims about notebook outputs and cached files should be verified by re-executing through `src/` modules. If re-execution contradicts a claim here, update this document to reflect what is actually true.

---

## Central Hypotheses

1. Does domain-specific pretraining (AbLang2) outperform general pretraining (ESM-2) for antibody property prediction?
2. Does encoding the CDR prior via a constraint loss improve performance?
3. Does the constraint interact differently with each model's learned representations? (Motivated by the finding that ESM-2 encodes the *inverse* of the CDR prior.)

---

## Experiment Matrix

All experiments run for both ESM-2 and AbLang2.

| # | Experiment | MLP Input | Owner | Notes |
|---|---|---|---|---|
| 1 | ESM-2 vs AbLang2 | (framework, applies across all) | Both | |
| 2 | Delta Residue only | mutant_residue_emb[mut_pos] - wt_residue_emb[mut_pos] | Lucas | Single token at mutation site |
| 3 | Delta Residue + Full Wild | concat(delta_residue, mean_pool(wt_sequence)) | Oscar | Residue delta + sequence context |
| 4 | Delta Sequence only | mean_pool(mutant_seq) - mean_pool(wt_seq) | Oscar | Already cached from prior work |
| 5 | Delta Residue + dim reduction | reduce(delta_residue) via PCA or learned | Lucas | |
| 6 | Delta Residue + max/mean pool | TBD -- needs clarification | TBD | Possibly concat(delta_residue, max_pool(seq), mean_pool(seq)) |
| 7 | CDR Constraint | Best-performing embedding strategy + constraint loss | Both | Lambda sweep [0, 0.1, 0.5, 1.0] |

Experiment 4 uses existing cached sequence-level delta embeddings.
Experiments 2, 3, 5, 6 require residue-level embeddings (to be generated in NB03).

---

## Datasets

### Task 1: AbAgym -- Antibody Mutation Effect Prediction

- Source: github.com/3BioCompBio/AbAgym
- Repo structure: PDB files in `PDB_files.zip` (extracts to `DMS_big_table_PDB_files/`), DMS data in `AbAgym_data_non-redundant.csv.zip`. No per-antibody CSVs. Full CSV also available (`AbAgym_data_full.csv.zip`) but non-redundant used -- column names already match our schema (`chains`, `mut_names`) and row counts for our 5 datasets are identical.
- 5 antibody-side DMS datasets, 5,318 mutations total:
  - Ang2_2017_G6: 981 mutations (79% CDR, 21% FR, G6 scaffold)
  - EGFR_2013_Cetuximab: 1,071 mutations (65% CDR, 35% FR)
  - lysozyme_2019_D441: 2,094 mutations (34% CDR, 66% FR -- primary FR source)
  - VEGF_2017b_G6: 988 mutations (79% CDR, 21% FR, same G6 scaffold as Ang2)
  - HER2_2021_trastuzumab: 184 mutations (100% CDR H3, 0% FR -- zero constraint gradient, bimodal score distribution, report separately)
- Chain breakdown: 3,067 heavy (H), 2,251 light (L). HER2 all heavy.
- Label: MinMax_normalized_DMS_score (per-dataset [0,1] normalization, retains effect magnitude)
- Metric: Spearman correlation, per-dataset and aggregate

### Task 2: SAbDab -- Binding Affinity Prediction

- Source: Zenodo DOI 10.5281/zenodo.13120765 (Apache 2.0)
- 491 antibody-antigen pairs (493 original, dropped 2 full-IgG: 6d6u, 6d6t)
- 97 sequences had artifacts stripped (His-tags, TEV sites, linkers)
- Label: pKd = -log10(Kd), mean=8.23, range 3.70-12.40
- Metric: Pearson correlation + RMSE
- No CDR/FR mapping -- constraint loss applies to Task 1 only

Design decision: embed antibody sequences only, no antigen embeddings. Rationale: antigen is fixed within each AbAgym dataset; SAbDab N=491 too small for cross-attention; AbLang2 cannot embed arbitrary antigens. Acknowledged as a limitation.

---

## Models

### ESM-2 (650M)

- General protein language model from Meta
- Trained on ~250M diverse protein sequences
- Install: `pip install fair-esm`, model: `esm2_t33_650M_UR50D`
- Per-residue hidden dim: 1280
- Repr layer: derived at runtime via `model.num_layers` (= 33, do not hardcode)
- Token structure: [BOS, res_1, ..., res_n, EOS, PAD...]
  - Special token indices via alphabet: `alphabet.cls_idx` (BOS), `alphabet.eos_idx`, `alphabet.padding_idx`
  - Exclude all three from mean pool
- Heavy and light chains embedded in separate forward passes
- Sequence-level: concat(mean_pool(H), mean_pool(L)) = 2560-dim
- Residue-level: token at position seq_idx + 1 (BOS offset) = 1280-dim

### AbLang2-paired

- Antibody-specific model from Oxford
- Trained on Observed Antibody Space (OAS) via masked language modeling
- Install: `pip install ablang2`
- Load: `ablang2.pretrained(model_to_use='ablang2-paired', random_init=False, ncpu=1, device=str(DEVICE))`
- Input: `VH|VL` pipe-separated string
- Tokenization: `ablang.tokenizer(seqs, pad=True, w_extra_tkns=False, device=str(device))`
- Representations: `ablang.AbRep(tokens).last_hidden_states`
  - Always returns final layer. No repr layer selection API.
- Per-residue hidden dim: 480 (NOT 768 -- empirically confirmed. The original spec assumed 768. This was wrong.)
- Token structure (w_extra_tkns=False): [H_1, ..., H_n, SEP(25), L_1, ..., L_m, PAD(21)...]
  - No BOS or EOS tokens
  - SEP token: `ablang.tokenizer.sep_token` (= 25)
  - PAD token: `ablang.AbRep.padding_tkn` (= 21)
  - Read these programmatically, do not hardcode
- Chain boundary: separator at position len(heavy_seq) exactly
  - Located via cumulative sum over separator mask, vectorized across batch
  - H mask: sep_cumsum == 0 and not PAD
  - L mask: sep_cumsum >= 1 and not SEP and not PAD
- Both chains in a single forward pass
- Sequence-level: concat(mean_pool(H), mean_pool(L)) = 960-dim
- Residue-level: token at seq_idx (heavy) or len(heavy) + 1 + seq_idx (light, +1 for separator) = 480-dim

---

## Biological Context

Antibodies have structurally distinct regions annotated via IMGT numbering:
- Framework Regions (FR): conserved structural scaffold
- CDRs (IMGT boundaries, same for H and L chains):
  - CDR1: positions 27-38
  - CDR2: positions 56-65
  - CDR3: positions 105-117 (most important for binding specificity)
- Region labels in data: CDR_H1, CDR_H2, CDR_H3, CDR_L1, CDR_L2, CDR_L3, FR

AbAgym uses PDB numbering, NOT IMGT. The mapping pipeline is:
PDB position -> ANARCI (sequence-index alignment) -> IMGT position -> CDR/FR
Verified: 4zfg chain H, PDB 100A -> IMGT 113 -> CDR_H3.

The per-mutation region label is pre-computed and stored in abagym_antibody.csv as the `region` column. No runtime mapping needed.

---

## Neurosymbolic Constraint (Experiment 7)

Encodes the biological prior: CDR mutations should have higher predicted effect magnitude than framework mutations.

```
constraint_loss = mean(ReLU(|FR_predicted| - |CDR_predicted|))
total_loss = task_loss + lambda * constraint_loss
```

- Applied to Task 1 (AbAgym) only -- no CDR mapping for SAbDab
- Applied to BOTH ESM-2 and AbLang2
- Fires only when model predicts FR effect > CDR effect
- Lambda = 0 must reproduce unconstrained baseline exactly (sanity check)
- Lambda sweep: [0, 0.1, 0.5, 1.0]
- Lysozyme dominates FR coverage (1,390 FR mutations) -- primary constraint gradient source
- HER2 contributes zero constraint gradient (all CDR H3, no FR)
- Batching: ensure each batch mixes datasets to avoid HER2-only batches

Why the constraint is especially interesting for ESM-2:
EDA showed ESM-2 encodes the INVERSE of the CDR prior -- FR mutations have significantly larger delta norms than CDR mutations (Mann-Whitney p=2.78e-122). ESM-2 is more perturbed by FR mutations because FR positions are highly conserved and unusual to mutate from a sequence statistics perspective. The constraint directly opposes ESM-2's learned geometric structure. Whether this helps or hurts ESM-2 is a key open question.

---

## Key EDA Findings

### ESM-2 embedding structure (Notebook 2 EDA)

- Raw mutant embeddings cluster extremely tightly around wildtype within each antibody (cosine similarity mean=0.9999, std=0.0001)
- Spearman(cosine_sim, DMS_score): r = -0.081 to -0.153 across datasets
- HER2 weakest signal (r=-0.125, p=0.09)

### ESM-2 delta embedding structure (Notebook 2.5 EDA)

- CoV of delta norms 177x-301x higher than raw embedding norms across all 5 datasets -- strong discriminability gain
- Spearman(delta_norm, DMS_score): r = 0.079 to 0.151 (equivalent to cosine sim signal -- Spearman r differs by < 0.002)
- ESM-2 inverse CDR prior: FR delta norms > CDR delta norms (mean 0.108 vs 0.084, Mann-Whitney p=2.78e-122)
- Cross shape in PCA caused by chain identity: heavy chain mutations perturb first 1280 dims, light chain perturb last 1280. Orthogonal by construction. HER2 has no cross (all heavy chain).
- PC1+PC2 explain only 26% of variance. No DMS score gradient in 2D. Meaningful structure distributed across many dimensions.

### AbLang2 delta EDA

NOT YET RUN. Notebook 3.5 was defined but not executed. Key open question: does AbLang2's delta space show the same inverse CDR prior as ESM-2 (FR > CDR norms), or does antibody-specific pretraining produce a different geometric structure? Does AbLang2 show orthogonal H/L subspaces like ESM-2, or does cross-chain attention mix them?

---

## Data File Schemas

### abagym_antibody.csv (14 columns, 5,318 rows)

Columns: DMS_name, PDB_file, chains, site, wildtype, mutation, mut_names, DMS_score, MinMax_normalized_DMS_score, Rank_quartile_normalized_DMS_score, closest_interface_atom_distance, mutant_heavy_seq, mutant_light_seq, region

Key columns:
- chains: 'H' or 'L' -- which chain carries this mutation
- site: PDB position label as string (e.g. '100', '100A', '28') -- never cast to int
- wildtype: single-letter AA at that position in wildtype
- mutation: single-letter AA being substituted in
- mutant_heavy_seq: full heavy chain with exactly one substitution
- mutant_light_seq: full light chain (unchanged if mutation on H chain)
- region: CDR/FR label. One of: CDR_H1, CDR_H2, CDR_H3, CDR_L1, CDR_L2, CDR_L3, FR

### abagym_sequences.csv (5 rows)

Columns: dms_name, heavy_seq, light_seq, mapping_H, mapping_L
- mapping_H/mapping_L: JSON strings. Load with json.loads().
  Keys: PDB position label (string). Values: dict with keys {imgt_pos: int, imgt_ins: str, region: str, seq_idx: int}
- seq_idx is the 0-based index into the amino acid sequence string

### sabdab_affinity.csv (491 rows)

Columns: Antibody_ID, heavy_seq, light_seq, Antigen_ID, Antigen, Y, pKd
- heavy_seq, light_seq: clean amino acid strings (artifacts stripped)
- Y: raw Kd in molar units
- pKd = -log10(Y): training label

---

## Completed Work

### Notebook 1: Data Pipeline

Status: IN PROGRESS -- running locally. ANARCI mappings verified. SAbDab download, mutant reconstruction, and CSV save pending.

#### AbAgym data source

The AbAgym GitHub repo does not contain per-antibody CSV files. The actual structure is:

| File | Rows | Notes |
|---|---|---|
| `AbAgym_data_full.csv.zip` | 572,719 | All 68 experiments, all mutations. Columns: `chain`, `mut_name` (singular) |
| `AbAgym_data_non-redundant.csv.zip` | 323,752 | Redundant experiments removed at dataset level. Columns: `chains`, `mut_names` (plural) |
| `AbAgym_data_full_interface.csv` | 37,259 | Full dataset, interface-adjacent residues only |
| `AbAgym_data_non-redundant_interface.csv` | 36,541 | Non-redundant, interface residues only |
| `AbAgym_metadata.csv` | 68 | One row per experiment, includes antigen, PDB ID, DOI |
| `PDB_files.zip` | 68 PDB files | Extracts to `DMS_big_table_PDB_files/`. One processed PDB per experiment, chains renamed to H and L. |

Decision: use `AbAgym_data_non-redundant.csv.zip`. Rationale: (1) column names already match our schema (`chains`, `mut_names`) so no renaming needed; (2) row counts for our 5 datasets are identical in both zip files (5318 total), so the non-redundant filter removed nothing from our subset.

Interface-adjacent residues: residues within a small distance threshold of the antibody-antigen binding interface, measured by `closest_interface_atom_distance` (in Angstroms). The interface CSVs filter to these only. We use the full (non-interface-filtered) CSV to preserve the full CDR/FR signal range -- in particular, framework mutations far from the interface are the primary source of the CDR constraint gradient.

#### Site column dtype

The `site` column must be read with `dtype={'site': str}`. Default numeric parsing silently converts insertion code sites (e.g. `'100A'`, `'30A'`, `'52A'`) to NaN. Confirmed insertion code sites in Ang2_2017_G6: `100A`, `100B`, `30A`, `52A`.

#### ANARCI mapping pipeline

PDB numbering is arbitrary -- residue labels are assigned by the depositor and differ across structures. Insertion codes (e.g. `100A`) are a patch to insert extra residues without renumbering existing ones. Two antibodies can both have a residue called `28` that are in completely different structural contexts.

ANARCI resolves this by aligning each chain sequence to a curated antibody reference and assigning standardized IMGT position numbers. Once IMGT positions are assigned, CDR/FR assignment is a fixed lookup (CDR1: 27-38, CDR2: 56-65, CDR3: 105-117 for both H and L).

Implementation: `anarci.anarci([(seq_id, sequence)], scheme='imgt')`. Returns `(numbering_list, seqstart, seqend)` where `seqstart` is the index into the full chain sequence where the variable domain begins. This matters because some PDB structures include constant domain residues before the variable domain. `seq_idx = seqstart + i` gives the 0-based index into the full amino acid string, which is what the embedding models need.

Verified chain residue counts (full PDB chain, including any constant domain residues):

| DMS name | H residues | L residues |
|---|---|---|
| Ang2_2017_G6 | 215 | 213 |
| EGFR_2013_Cetuximab | 220 | 211 |
| lysozyme_2019_D441 | 218 | 214 |
| VEGF_2017b_G6 | 211 | 213 |
| HER2_2021_trastuzumab | 220 | 214 |

Spot-check (canonical verification): Ang2_2017_G6 chain H, PDB site `100A` → IMGT 113 → CDR_H3, seq_idx=104. IMGT 113 falls within CDR_H3 range (105-117). Consistent with prior notebook output.

#### SAbDab pipeline

Raw Zenodo CSV: 493 rows, 5 columns (`Antibody_ID`, `Antibody`, `Antigen_ID`, `Antigen`, `Y`). `Y` is raw Kd in molar units (median ~6.2e-9 M, i.e. ~6.2 nM). `parse_sabdab_raw` derives `pKd = -log10(Y)` and splits the `Antibody` column (Python list literal) into `heavy_seq` and `light_seq`. Two full-IgG entries dropped (6d6u, 6d6t -- anomalously long sequences). 97 sequences had artifacts stripped (N/C-terminal His-tags, TEV sites, GGGGS linkers, FLAG, StrepII, Factor Xa, thrombin sites). Final: 491 rows.

pKd summary: mean=8.23, std=1.50, min=3.70, max=12.40. Range corresponds to Kd from ~200 µM (weak binding) to ~4 pM (very strong binding).

#### Mutant sequence reconstruction

For each of the 5318 mutations: locate `seq_idx` from the ANARCI mapping for the relevant chain, substitute one amino acid at that index, verify exactly 1 position differs from wildtype. Spot-check on 5 random samples (random_state=42): all showed exactly 1 diff.

- lysozyme_2019_D441 H:F64E → 1 diff
- lysozyme_2019_D441 L:C88V → 1 diff
- lysozyme_2019_D441 L:I29T → 1 diff
- EGFR_2013_Cetuximab L:I55K → 1 diff
- Ang2_2017_G6 H:G54K → 1 diff

#### CDR/FR region distribution

Region labels assigned by looking up each mutation site in the ANARCI mapping. Full distribution:

| Region | Count | % of 5318 |
|---|---|---|
| FR | 2188 | 41.1% |
| CDR_H3 | 851 | 16.0% |
| CDR_L3 | 646 | 12.1% |
| CDR_H2 | 558 | 10.5% |
| CDR_L1 | 433 | 8.1% |
| CDR_H1 | 415 | 7.8% |
| CDR_L2 | 227 | 4.3% |
| **Total CDR** | **3130** | **58.9%** |

FR mutations are dominated by lysozyme_2019_D441 (2094 total mutations, ~66% FR ≈ 1382 FR mutations). HER2_2021_trastuzumab contributes 184 mutations, all in CDR_H3 -- confirmed 0 FR contribution, consistent with prior expectation. All 7 expected region labels present, no unexpected values.

Status: COMPLETE. All assertions passed. CSVs saved to data/ and committed to git on implementation branch.

### Notebook 2: ESM-2 Sequence-Level Embeddings

Status: NOT YET RUN IN NEW WORKSPACE. Prior Colab outputs exist at old Drive path (DL_Final_Project/embeddings/) but are not accessible via the new DRIVE_ROOT (DL_Final_Project/Antibody_Project/). Must re-run after NB01 completes.

Produced:
- esm2_abagym.pt (5318, 2560)
- esm2_abagym_index.json
- esm2_abagym_wildtype.pt (5, 2560)
- esm2_abagym_wildtype_index.json
- esm2_sabdab.pt (491, 2560)
- esm2_sabdab_index.json

### Notebook 2.5: ESM-2 Delta Embeddings + EDA

Status: NOT YET RUN IN NEW WORKSPACE. Same Drive path caveat as NB02.

Produced:
- esm2_abagym_delta.pt (5318, 2560)
- esm2_abagym_delta_index.json
- EDA figures in results/figures/ (esm2_delta_*.png)

### Notebook 3: AbLang2 Sequence-Level Embeddings + Deltas

Status: NOT YET RUN IN NEW WORKSPACE. Same Drive path caveat as NB02.

Produced:
- ablang2_abagym.pt (5318, 960)
- ablang2_abagym_index.json
- ablang2_abagym_wildtype.pt (5, 960)
- ablang2_abagym_wildtype_index.json
- ablang2_abagym_delta.pt (5318, 960)
- ablang2_abagym_delta_index.json
- ablang2_sabdab.pt (491, 960)
- ablang2_sabdab_index.json

---

## Still To Do

- Notebook 3.5 equivalent (AbLang2 delta EDA) -- part of 04_embedding_eda.ipynb
- Residue-level embedding extraction for both models -- part of 03_embedding_generation.ipynb
  - esm2_abagym_residue_mutsite.pt (5318, 1280)
  - esm2_abagym_residue_wtsite.pt (5318, 1280)
  - ablang2_abagym_residue_mutsite.pt (5318, 480)
  - ablang2_abagym_residue_wtsite.pt (5318, 480)
- All training experiments (2-7) -- 05_training.ipynb
- Analysis and figures -- 06_analysis.ipynb

---

## Open Questions

- Experiment 6 (delta residue + max/mean pool): exact formulation TBD
- AbLang2 delta EDA: does it show the same inverse CDR prior as ESM-2? Does cross-chain attention mix the H/L subspaces?
- Whether the CDR constraint helps or hurts ESM-2 (inverse prior finding makes this genuinely uncertain)
