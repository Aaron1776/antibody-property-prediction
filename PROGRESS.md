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
| 6 | CDR Constraint | Best-performing embedding strategy + constraint loss | Both | Lambda sweep [0, 0.1, 0.5, 1.0] |

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
constraint_loss = ReLU(mean(|FR_predicted|) - mean(|CDR_predicted|))
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

---

### Notebook 02: Model Exploration (02_model_exploration.ipynb)


Purpose: verify all ESM-2 and AbLang2 API details before writing embedding generation code. Nothing is saved to disk from this notebook -- findings are documented here.

#### ESM-2 vocabulary

The full token vocabulary was inspected directly via `alphabet.tok_to_idx` (33 entries, indices 0-32):
- 0: `<cls>` (BOS / Beginning of Sequence) -- always at position 0, shifts all residues right by 1
- 1: `<pad>` -- fills shorter sequences in a batch; ignored during attention
- 2: `<eos>` -- end of sequence marker
- 3: `<unk>` -- unknown residue; if this appears in a forward pass the embedding at that position is uninformative
- 4-23: the 20 standard amino acids
- 24-28: ambiguous/non-standard AAs: X (any), B (Asp/Asn), U (selenocysteine), Z (Glu/Gln), O (pyrrolysine)
- 29-30: MSA gap characters (`.`, `-`) -- irrelevant for single-sequence input
- 31: `<null_1>` -- reserved placeholder, no defined function
- 32: `<mask>` -- masked language model pretraining token; never appears in inference

None of indices 24-31 are expected in AbAgym or SAbDab sequences. UNK (3) is the token to watch for -- its presence indicates a non-standard residue that was not handled upstream.

#### ESM-2 forward pass shapes

Single H+L forward pass on Ang2_2017_G6 (H=215 residues, L=213 residues):
- Heavy repr: `(1, 217, 1280)` -- 215 + 2 (BOS + EOS) = 217 tokens, 1280-dim per position
- Light repr: `(1, 215, 1280)` -- 213 + 2 = 215 tokens

H and L are embedded in separate forward calls; their mean-pooled outputs are concatenated to form the 2560-dim sequence-level representation.

#### ESM-2 residue indexing

Verified on the first mutation in Ang2_2017_G6: H:P100A (PDB site `100A`).
- `seq_idx = 103`: 0-based index into the 215-residue heavy chain string, returned by `get_mutation_site_index` via the ANARCI mapping
- `token_pos = 104 = seq_idx + 1`: BOS at position 0 shifts all residues right by one
- AA at seq_idx 103: `'P'` -- matches wildtype, confirming the ANARCI-derived index is correct
- Residue embedding shape: `(1280,)`, norm = 10.13
- BOS embedding norm = 10.03 -- differs from residue embedding (similar L2 magnitude is normal for ESM-2; the 1280-dimensional direction encodes residue identity, not the norm)

#### ESM-2 throughput (MPS)

Profiled across sequence lengths [50, 100, 150, 200, 250] and batch sizes [1, 8, 32, 64]. VRAM not available on MPS (Apple unified memory has no equivalent to `cuda.max_memory_allocated`).

Key result: batching helps substantially up to bs=32; gains plateau or reverse above 32 at longer sequences, consistent with MPS memory bandwidth saturation. At the length most representative of our data (200 residues), bs=32 gives **0.054 sec/sample**.

Estimated cost for 5318 residue-level embeddings at bs=32, length ~200: ~5 minutes on MPS. Colab CUDA (T4/A100) will be significantly faster. Recommendation: use bs=32 as default for NB03 ESM-2 embedding generation.

#### AbLang2 load

AbLang2-paired (Oxford) downloads as a `.tar` archive on first use. `ablang.AbRep` is the representation module; `ablang.tokenizer` handles encoding. No repr layer selection -- always returns final layer hidden states.

Confirmed: SEP=25, PAD=21, hidden_dim=480, parameters=44,348,304 (~44M, ~15x smaller than ESM-2). Combined memory footprint with ESM-2 resident: ~2.5 GB + ~0.17 GB -- both models fit simultaneously on MPS.

#### AbLang2 vocabulary

Full vocabulary confirmed via `ablang.tokenizer.token_to_aa` (26 entries, indices 0-25): indices 1-20 are the 20 standard amino acids. Special tokens: start=0 (`<`), end=22 (`>`), pad=21 (`-`), sep=25 (`|`), mask=23 (`*`), unknown=24 (`X`).

With `w_extra_tkns=False` (always used), start and end tokens are suppressed from output. If `w_extra_tkns=True` were used, start would appear at position 0 and shift all residues right by 1, replicating ESM-2's BOS layout. We always use `False` to avoid this offset.

#### AbLang2 forward pass shapes

Single H+L forward pass on Ang2_2017_G6 (H=215, L=213): output shape `(1, 429, 480)`. seq_len = 215 + 1 (SEP) + 213 = 429 exactly. No +2 for BOS/EOS. Both chains processed in one pass, allowing cross-chain attention -- architectural contrast with ESM-2's separate per-chain passes.

#### AbLang2 chain boundary detection

`get_chain_masks` locates the SEP token and constructs boolean masks for heavy and light residues. Confirmed on Ang2_2017_G6: SEP at position 215 = `len(heavy_seq)`, heavy mask covers 215 tokens, light mask covers 213 tokens. No overlap or leakage. Assertions passed.

#### AbLang2 residue indexing

Verified on same mutation as ESM-2: Ang2_2017_G6 H:P100A.
- `seq_idx = 103`, `token_pos = 103` (heavy chain, no BOS offset)
- Token id 13 = `'P'` -- matches wildtype, correct residue targeted
- Not SEP (25), not PAD (21) -- clean residue token
- Residue embedding: shape `(480,)`, norm = 7.25

Critical comparison: same mutation, same seq_idx (103), but ESM-2 gives token_pos=104 (seq_idx+1 for BOS) while AbLang2 gives token_pos=103 (seq_idx directly). This difference is handled in `embed_sequences_residue` for each model separately.

#### AbLang2 throughput (MPS)

Profiled across same lengths and batch sizes as ESM-2. AbLang2 is ~10x faster than ESM-2 at bs=1 and ~2x faster at bs=32 for length=200, consistent with 15x parameter count difference.

First-call anomaly: length=50, bs=1 reported 0.524 sec -- MPS JIT compilation on first forward pass, not representative. Subsequent calls show steady-state performance (~0.02-0.05 sec/sample).

Batching saturates earlier than ESM-2: at length=200, bs=8 is fastest (0.019 sec/sample); larger batches do not improve throughput. At length=250, bs=32: 0.025 sec/sample. AbLang2 processes paired sequences (~429 tokens for Ang2) in practice, so real-world cost is above the profiled range. Estimated cost for 5318 embeddings: ~2-3 minutes on MPS.

Recommendation: use bs=32 for NB03 AbLang2 embedding generation (consistent with ESM-2 recommendation). Discard first-batch timing on Colab.

Status: COMPLETE (local MPS run). All API details verified for both models.

---

### Notebook 03: Embedding Generation (03_embedding_generation.ipynb)


This notebook consolidates all embedding generation that was previously split across NB02, NB02.5, and NB03 in the prior workspace. All prior outputs at `DL_Final_Project/embeddings/` are superseded by this run at `DL_Final_Project/Antibody_Project/embeddings/`. The notebook is environment-agnostic: runs unchanged on Colab (mounts Drive, clones repo, installs dependencies) or locally (auto-resolves DRIVE_ROOT via Google Drive Desktop glob, skips installs).

#### Design: wt_expanded_sequences

Residue-level embedding requires a wildtype embedding at every mutation site. The naive approach would be to embed only 5 wildtype sequences and then index into them at extraction time. Instead, we construct `wt_expanded_sequences` -- a list of 5318 wildtype `(heavy, light)` tuples, one per mutation row, where each entry is the wildtype sequence for that mutation's antibody. This allows `embed_sequences_residue` to receive aligned inputs (mutant sequence at row i, wildtype sequence at row i, same site_index at row i) and extract both mutsite and wtsite in a single unified pass with no special indexing logic inside the extraction function.

#### ESM-2 embeddings (complete)

All tensors generated on MPS, bs=32. Two progress bars per pooled call (separate H and L passes). Single bar for residue extraction (only the mutated chain is embedded per mutation).

| File | Shape | Time | Throughput |
|---|---|---|---|
| esm2_abagym.pt | (5318, 2560) | 6:02 + 6:32 | 14.65 / 13.54 seq/s |
| esm2_abagym_wildtype.pt | (5, 2560) | ~1 sec | -- |
| esm2_sabdab.pt | (491, 2560) | ~38 sec | 12.93 seq/s |
| esm2_abagym_residue_mutsite.pt | (5318, 1280) | 7:07 | 12.44 seq/s |
| esm2_abagym_residue_wtsite.pt | (5318, 1280) | 7:48 | 11.35 seq/s |
| esm2_abagym_delta.pt | (5318, 2560) | instant | tensor subtraction |
| esm2_abagym_residue_delta.pt | (5318, 1280) | instant | tensor subtraction |

Wildtype index saved: `esm2_abagym_wildtype_index.json` maps row → antibody name (required by `compute_delta_sequence`). SAbDab index saved: `esm2_sabdab_index.json` maps row → `Antibody_ID`.

Residue extraction is slower than pooled embedding per sequence (7+ min vs 6 min for same N) because each sequence in the residue pass is embedded individually by the mutated chain only, which are variable-length and cannot benefit from length-sorted batching as effectively as the pooled pass.

#### AbLang2 embeddings (complete)

All tensors generated on MPS, bs=32. Single progress bar per call -- AbLang2 processes both chains jointly in one pass, unlike ESM-2's separate H/L passes. AbLang2 is approximately 3x faster than ESM-2 at bs=32, consistent with the 15x parameter count difference (44M vs 651M params).

| File | Shape | Time | Throughput |
|---|---|---|---|
| ablang2_abagym.pt | (5318, 960) | 2:02 | 43.44 seq/s |
| ablang2_abagym_wildtype.pt | (5, 960) | ~1 sec | -- |
| ablang2_sabdab.pt | (491, 960) | 14 sec | 33.22 seq/s |
| ablang2_abagym_residue_mutsite.pt | (5318, 480) | 2:04 | 42.75 seq/s |
| ablang2_abagym_residue_wtsite.pt | (5318, 480) | 2:21 | 37.66 seq/s |
| ablang2_abagym_delta.pt | (5318, 960) | instant | tensor subtraction |
| ablang2_abagym_residue_delta.pt | (5318, 480) | instant | tensor subtraction |

Note on residue extraction: AbLang2 always embeds both chains together (`VH|VL`) even for residue-level extraction, because the model has no single-chain forward pass. The token at the mutation position is extracted from the joint representation. This means AbLang2 residue embeddings encode cross-chain context that ESM-2 residue embeddings do not -- ESM-2 embeds only the mutated chain for residue extraction. Whether this cross-chain information is beneficial for mutation effect prediction is an open empirical question.

#### Verification

All 14 tensors passed shape, NaN, and Inf checks. Spot check on row 0 (Ang2_2017_G6 H:P100A):
- ESM-2 sequence delta norm: 0.0720
- AbLang2 sequence delta norm: 0.0728

Both models produce similar-magnitude deltas for the same mutation at the sequence level. Whether this similarity holds across the full distribution -- and whether it extends to the residue-level deltas -- is examined in NB04.

Status: COMPLETE. All 14 tensors saved to Drive and verified.

---

## Notebook 04: Embedding EDA (04_embedding_eda.ipynb)

Self-contained EDA covering both models (ESM-2 and AbLang2) at both embedding levels
(sequence and residue). No model inference -- loads all 6 tensors from Drive cache.

### Analyses in NB04

1. Discriminability gain (CoV raw vs delta) -- both models, sequence level
2. Delta norm distributions by dataset -- both models, sequence level
3. CDR vs FR delta norms with Mann-Whitney test -- both models, both levels
4. Spearman(delta norm, DMS score) per dataset -- both models, both levels
5. PCA structure (colored by chain, CDR/FR, dataset) -- both models, both levels
6. Cross-model comparison: Spearman between ESM-2 and AbLang2 delta norm vectors
7. Full Spearman summary table: 2 models x 2 levels x 5 datasets + aggregate

### src/ changes made for NB04

`src/visualization/plots.py` -- 4 NB04 stubs implemented plus 1 new function:
- `plot_delta_norm_by_dataset`: violin per dataset, 5 panels
- `plot_delta_norm_cdr_vs_fr`: binary CDR/FR violin + 7-region detail panel; Mann-Whitney annotated
- `plot_delta_pca`: 2D PCA scatter; handles continuous (dms_score) and categorical (chain, region, dataset, cdr_fr) coloring
- `plot_delta_variance_ratio`: raw CoV vs delta CoV bar chart (log scale) + ratio panel
- `plot_model_comparison_norms` (new): scatter of ESM-2 vs AbLang2 delta norms, colored CDR/FR, Spearman annotated

### Results

---

#### Finding 1: Sequence delta norm global statistics

The L2 norm of each row of the delta tensor (||mutant_embedding - wildtype_embedding||_2)
measures how much the model's representation shifted in response to a single amino acid
substitution. This is the core scalar quantity used throughout the EDA.

| Metric | ESM-2 | AbLang2 |
|---|---|---|
| min | 0.0254 | 0.0437 |
| median | 0.0872 | 0.1084 |
| max | 0.3167 | 0.6065 |

AbLang2 has a higher median (+24%) and a wider range. Notably, AbLang2's minimum is
0.0437 vs ESM-2's 0.0254 -- AbLang2 has no near-zero deltas, suggesting it responds
with nonzero magnitude to every mutation in the dataset, even conservative ones.
AbLang2's maximum (0.6065) is approximately 2x ESM-2's (0.3167), indicating that some
mutations produce substantially larger perturbations in the antibody-specific space.

---

#### Finding 2: Discriminability gain from delta computation (CoV ratio)

Raw sequence embeddings cluster tightly around wildtype (high cosine similarity, low
CoV of L2 norms). Computing delta = mutant - wildtype amplifies the mutation-specific
signal. The discriminability gain is measured as CoV(delta norms) / CoV(raw norms),
where CoV = std / mean. A higher ratio means the delta computation extracted more
variance relative to what was present in the raw embeddings.

| Dataset | ESM-2 ratio | AbLang2 ratio |
|---|---|---|
| Ang2_2017_G6 | 239x | 96x |
| EGFR_2013_Cetuximab | 223x | 105x |
| HER2_2021_trastuzumab | 302x | 91x |
| VEGF_2017b_G6 | 225x | 178x |
| lysozyme_2019_D441 | 178x | 117x |
| Range | 178x–302x | 91x–178x |

ESM-2's gains (178x–302x) are roughly 1.7–3x larger than AbLang2's (91x–178x). This
does not mean ESM-2's deltas are more informative -- it means AbLang2's raw embeddings
are already less compressed around wildtype. AbLang2 was trained on OAS antibody
sequences and has encountered far more antibody-specific sequence diversity, so the
baseline spread of its raw embeddings is higher. The delta computation still provides
large absolute gains for both models.

Key finding -- HER2 inversion between models: ESM-2 has its highest ratio on HER2
(302x) while AbLang2 has its lowest (91x). All 184 HER2 mutations are in CDR H3.
The high ESM-2 ratio is driven by an extremely low raw CoV denominator: ESM-2's raw
mean-pooled embeddings for the 184 HER2 CDR H3 mutants have nearly identical L2 norms.
CDR H3 is severely under-represented in UniRef50 -- antibodies are a small fraction of
the proteome, and 50% identity clustering further collapses CDR H3 sequence diversity.
ESM-2 therefore assigns relatively uniform contextual embeddings at CDR H3 positions
regardless of amino acid identity; the flanking conserved IMGT framework context
dominates the mean-pooled representation, not the single variable position. ESM-2 is
insensitive to CDR H3 amino acid variation in raw embedding space -- this is reflected
also in the delta space, where CDR H3 produces the smallest delta norms of any loop
(~0.075, tied for lowest with CDR_L2 in Finding 4). The raw embeddings do not vary
because ESM-2 does not distinguish amino acids at CDR H3 positions well, compressing
the raw CoV and inflating the ratio.

From AbLang2's perspective, trained on OAS where CDR H3 is the most hypervariable
region in the antibody repertoire, CDR H3 amino acid identity is highly informative.
Its raw embeddings at CDR H3 positions vary meaningfully with amino acid identity,
giving a higher raw CoV and therefore a lower ratio (91x). This HER2 inversion is a
fingerprint of domain specificity: ESM-2 is insensitive to CDR H3 variation because
it has barely encountered it; AbLang2 is sensitive to it because it has trained on it
extensively.

---

#### Finding 3: Delta norm distributions by dataset (violin plots)

**ESM-2:** Dataset-level norm distributions show meaningful variation in median and
spread. HER2 has the lowest median (~0.065) and tightest distribution. Lysozyme has
the highest median (~0.10) and widest spread. The ordering of medians -- lysozyme >
others > HER2 -- is consistent with the inverse CDR prior: lysozyme is 66% FR
mutations (higher norms) while HER2 is 100% CDR H3 (lower norms for ESM-2). All
five distributions are right-skewed.

**AbLang2:** In contrast to ESM-2, all five datasets have nearly identical medians
(~0.09–0.10), suggesting AbLang2 responds with similar average magnitude regardless
of dataset CDR/FR composition. Most datasets show strong right skew with long upper
tails (outlier mutations reaching 0.33–0.60). HER2 is the exception: its distribution
is bell-shaped and approximately symmetric with a tight spread (~0.07–0.15) and no
upper tail. This is the opposite of what AbLang2 shows on other datasets and the
opposite of what ESM-2 shows on HER2. The symmetric HER2 shape in AbLang2 reflects
the model responding uniformly to CDR H3 mutations -- they are all within the expected
range of variation, producing neither very large nor very small deltas.

---

#### Finding 4: CDR vs FR delta norms -- the inverse CDR prior (sequence level)

The CDR prior states that CDR mutations should have larger functional effects than
framework mutations, since CDRs form the binding interface while FR is structural
scaffold. A model that encodes this prior would show CDR delta norms > FR delta norms.

**Both models show the inverse: FR > CDR.** Antibody-specific pretraining attenuates
but does not eliminate this inversion.

| | ESM-2 | AbLang2 |
|---|---|---|
| CDR median | 0.0802 | 0.1030 |
| FR median | 0.1022 | 0.1180 |
| FR/CDR ratio | 1.275x | 1.145x |
| Direction | FR > CDR | FR > CDR |
| Mann-Whitney p | ~0 (float underflow) | ~0 (float underflow) |

The p-values display as 0.00e+00 due to scipy float underflow at extreme significance
(N=5318, effect clear throughout). Both results are unambiguously significant.

**Mechanistic interpretation:**
ESM-2 (1.275x effect): Trained on ~250M diverse protein sequences, ESM-2 has seen
FR regions as highly conserved structural elements. Any mutation to a conserved position
is statistically unusual and produces a large representational shift. CDR mutations,
particularly in CDR H3, are less unusual to a general protein model because loop
diversity exists throughout the proteome. The result is FR mutations produce larger
delta norms than CDR mutations -- an inversion of the biological CDR prior.

AbLang2 (1.145x effect): Trained on OAS antibody sequences, AbLang2 has seen extensive
CDR diversity and knows that CDRs are expected to vary. This partially corrects the
inversion -- the FR/CDR ratio drops from 1.275x to 1.145x. However, FR positions are
still more conserved even within the antibody repertoire, so the inversion persists.
The residual inverse prior in AbLang2 may reflect genuinely unusual FR mutations
(structural disruptions) rather than a systematic training artifact.

The driver of the FR > CDR result differs between models: in ESM-2 the bulk CDR and
FR distributions are clearly separated. In AbLang2 the bulk distributions overlap more,
and it is primarily FR's extreme upper tail (extending to ~0.5) that pulls FR median
above CDR, while CDR distributions top out around 0.28–0.35.

**7-loop breakdown -- ESM-2 (approximate medians):**

| Region | Approx. median |
|---|---|
| CDR_H1 | ~0.104 |
| FR | ~0.102 |
| CDR_L1 | ~0.090 |
| CDR_H2 | ~0.082 |
| CDR_L3 | ~0.078 |
| CDR_L2 | ~0.075 |
| CDR_H3 | ~0.075 |

CDR_H1 is an outlier: its median (~0.104) is essentially equal to FR (~0.102). CDR_H1
adopts a small number of canonical conformations (canonical loop structures), making it
more structurally constrained than CDR_H3. ESM-2 treats CDR_H1 mutations as nearly as
unusual as FR mutations. CDR_H3 and CDR_L2 are the lowest (~0.075), furthest below FR.
CDR_H3 is the most hypervariable loop; ESM-2 has seen loop diversity at H3-equivalent
positions throughout the proteome. The inverse CDR prior is not uniform across loops:
it is strongest for CDR_H3/CDR_L2 and absent for CDR_H1.

**7-loop breakdown -- AbLang2 (approximate medians):**

| Region | Approx. median |
|---|---|
| FR | ~0.118 |
| CDR_L3 | ~0.108 |
| CDR_L1 | ~0.107 |
| CDR_L2 | ~0.105 |
| CDR_H1 | ~0.104 |
| CDR_H3 | ~0.100 |
| CDR_H2 | ~0.098 |

Three key differences from ESM-2's loop breakdown:

1. CDR_H3 is no longer the lowest loop. In ESM-2, CDR_H3 was at 0.075 (tied for
   lowest). In AbLang2 it is mid-range (~0.100). OAS training has seen extensive CDR_H3
   variation in antibodies, so those mutations are expected and produce moderate norms.

2. CDR loops are far more homogeneous. AbLang2's CDR medians span 0.098–0.108 (range
   0.010) vs ESM-2's 0.075–0.104 (range 0.029). The antibody-specific model treats all
   CDR loops similarly -- diversity is expected throughout the paratope.

3. Light chain CDRs (L1, L2, L3, medians 0.105–0.108) have slightly higher medians
   than heavy chain CDRs (H1, H2, H3, medians 0.098–0.104). This asymmetry may reflect
   AbLang2's joint VH|VL encoding via cross-chain attention, which has no analog in
   ESM-2's separate per-chain forward passes.

**Implication for Experiment 7 (CDR constraint loss):**
The constraint penalizes predictions where |FR effect| > |CDR effect|, directly opposing
the inverse prior encoded in both models. The constraint fires more forcefully against
ESM-2 (stronger inversion, 1.275x) than AbLang2 (weaker inversion, 1.145x). Whether
this helps or hurts each model's predictive performance is an open empirical question,
but the geometry suggests the constraint is a harder correction to impose on ESM-2.

---

#### Finding 5: Spearman(delta norm, DMS score) -- sequence level

The L2 norm of the sequence-level delta is a scalar unsupervised predictor of mutation
effect. Spearman correlation is computed per dataset and in aggregate. This measures the
signal available from embedding geometry alone, before any trained model is applied.

| Dataset | ESM-2 r | ESM-2 p | AbLang2 r | AbLang2 p |
|---|---|---|---|---|
| Ang2_2017_G6 | 0.1086 | 6.57e-04 | 0.3174 | 2.16e-24 |
| EGFR_2013_Cetuximab | 0.0788 | 9.88e-03 | 0.1484 | 1.08e-06 |
| HER2_2021_trastuzumab | 0.1270 | 8.59e-02 | 0.1234 | 9.50e-02 |
| VEGF_2017b_G6 | 0.1510 | 1.86e-06 | 0.3753 | 2.15e-34 |
| lysozyme_2019_D441 | 0.1371 | 2.95e-10 | 0.2116 | 1.25e-22 |
| ALL (N=5318) | 0.0834 | 1.13e-09 | 0.2157 | 4.99e-57 |

AbLang2 substantially outperforms ESM-2 on every dataset except HER2. The aggregate
Spearman (ALL) is 0.2157 for AbLang2 vs 0.0834 for ESM-2 -- AbLang2 is 2.6x higher.
Per-dataset gains: Ang2 (3x), VEGF (2.5x), EGFR (1.9x), lysozyme (1.5x).

HER2 is the sole exception: both models yield nearly identical, marginally non-significant
correlations (ESM-2 r=0.1270, p=0.086; AbLang2 r=0.1234, p=0.095). Neither achieves
p<0.05. This is expected given HER2's bimodal DMS score distribution and its composition
of 100% CDR H3 mutations. HER2 should be reported separately in the paper and excluded
from aggregate comparisons where possible, or explicitly noted as an outlier.

Ang2 and VEGF are the strongest datasets for AbLang2 (r=0.32 and r=0.38). Both are
G6-scaffold antibodies with 79% CDR mutations. The high correlation may reflect AbLang2
encoding G6-specific CDR variation better, having seen similar scaffolds in OAS.

ESM-2 values are consistent with prior NB02.5 findings (r=0.079–0.151 reported there).

Important caveat: these Spearman values are on the norm only (1 scalar per mutation).
The trained MLP operates on the full 2560-dim or 960-dim delta vector and is expected to
achieve substantially higher correlations by learning directional structure beyond magnitude.
The norm correlation is a floor estimate of what supervised training can achieve.

---

#### Finding 6: PCA structure of sequence-level delta embeddings

PCA was run on the full delta tensor (5318 x 2560 for ESM-2, 5318 x 960 for AbLang2).
The top 2 principal components were examined with three colorings: chain (H/L), CDR/FR,
and dataset identity.

**ESM-2 PCA:**
PC1 (18.3%) and PC2 (7.8%) together explain 26.1% of variance. The scatter shows a
perfect orthogonal cross: H mutations (heavy chain) lie entirely on the PC1 axis (PC2
approximately 0), and L mutations (light chain) lie entirely on the PC2 axis (PC1
approximately 0). No mixing between chains in 2D.

Mechanism: ESM-2 embeds H and L chains in separate forward passes. The 2560-dim
sequence delta is concat(H_delta, L_delta). For an H-chain mutation, L_delta = 0
because the L chain sequence is unchanged. The perturbation lives entirely in the first
1280 dims. For an L-chain mutation, H_delta = 0 and the perturbation lives in the last
1280 dims. PCA identifies these two orthogonal subspaces as PC1 and PC2 respectively.
The cross shape is a direct geometric consequence of separate per-chain embedding, not
an emergent property of the model's representations.

CDR/FR coloring on the cross: both CDR and FR mutations appear on both arms, mixed
throughout. FR mutations tend toward more extreme positions along each arm (larger
distance from origin), CDR mutations cluster closer to center. CDR/FR is a secondary
magnitude gradient within each arm, not a primary axis of variation in 2D.

Dataset coloring: no dataset-specific clustering along either arm. All five datasets
fully overlap within each arm. HER2 (N=184, all H chain) appears only on the PC1 arm
and concentrates near the center (smaller |PC1|), consistent with its lower delta norms.
Lysozyme appears throughout both arms, reflecting its mix of H and L chain mutations.

**AbLang2 PCA:**
PC1 (28.3%) and PC2 (14.6%) together explain 42.9% of variance -- substantially higher
than ESM-2's 26.1%. The scatter shows no cross shape. H and L mutations are fully mixed
throughout with no axis along which either chain dominates.

Mechanism: AbLang2 processes the full VH|VL sequence in a single forward pass through a
shared transformer. Cross-chain attention means a mutation on chain H perturbs both H and
L token representations via attention, making the delta nonzero in both halves of the
960-dim concatenated output. The orthogonal subspace structure requires chain-specific
independence, which cross-chain attention breaks.

The higher explained variance (42.9% vs 26.1%) follows from this: in ESM-2 each PC
captures only one chain's variance (H or L), so two PCs are needed to cover both chains
and each only reaches ~8–18%. In AbLang2 both chains contribute to every PC, allowing
more total variance to be captured per component.

CDR/FR coloring: the dense bulk cluster (PC1 < 0.1) mixes CDR and FR. The outlier scatter
at high PC1 (>0.1, up to ~0.5) is almost entirely FR. AbLang2's PC1 partly encodes
mutation magnitude -- FR mutations have larger norms and project further along PC1 -- but
CDR/FR is not a clean separation in 2D; it emerges as a density gradient.

Dataset coloring: the extreme outlier scatter (PC1 > 0.3) is almost entirely lysozyme,
the FR-heaviest dataset (66% FR, 1382 FR mutations of 2094 total). This connects directly:
FR → high norm → high PC1 → lysozyme (most FR mutations) dominates the extremes. In the
bulk, all five datasets mix without clustering. HER2 is nearly invisible (N=184, tight
near-origin cluster). No antibody-specific organization in the bulk of AbLang2's delta
space, consistent with the embedding being mutation-level rather than antibody-level.

**Summary of PCA findings:**
The two models organize their sequence-level delta spaces in fundamentally different
geometric forms. ESM-2 produces chain-specific orthogonal axes dominated by chain
identity; CDR/FR and dataset are secondary gradients. AbLang2 produces a mixed,
isotropic representation with no dedicated chain axes; mutation magnitude (driven by
CDR/FR composition) is the primary source of structured variance in 2D. Both observations
are direct consequences of architectural choices: separate per-chain passes (ESM-2) vs
joint VH|VL forward pass with cross-chain attention (AbLang2).

---

#### Finding 7: Residue-level delta norm global statistics

Residue-level deltas are extracted at the single mutation site token, unlike
sequence-level deltas which mean-pool across the full chain. Norms are therefore
much larger in absolute terms -- no dilution from averaging over ~200 residues.

| Metric | ESM-2 residue | AbLang2 residue | ESM-2 sequence | AbLang2 sequence |
|---|---|---|---|---|
| min | 0.7874 | 3.2136 | 0.0254 | 0.0437 |
| median | 3.5169 | 5.7217 | 0.0872 | 0.1084 |
| max | 5.7526 | 7.9464 | 0.3167 | 0.6065 |
| max/min ratio | 7.3x | 2.5x | 12.5x | 13.9x |

Residue medians are ~40x (ESM-2) and ~53x (AbLang2) larger than sequence medians.
AbLang2 residue norms have a strikingly tight range: max/min = 2.5x, compared to 7.3x
for ESM-2 residue and 13.9x for AbLang2 sequence. AbLang2's minimum residue norm (3.21)
is itself large -- there are effectively no near-zero residue deltas. This likely reflects
cross-chain attention: even conservative mutations at one position propagate through the
joint representation, establishing a minimum nonzero perturbation floor.

---

#### Finding 8: CDR vs FR delta norms -- residue level (major finding)

AbLang2 reverses direction at the residue level. ESM-2 maintains FR > CDR throughout.

| Model | Sequence direction | Sequence ratio | Residue direction | Residue ratio | Residue p |
|---|---|---|---|---|---|
| ESM-2 | FR > CDR | 1.275x | FR > CDR | 1.046x | 5.85e-22 |
| AbLang2 | FR > CDR | 1.145x | CDR > FR | 1.039x | 1.47e-26 |

AbLang2 sequence level: CDR median 0.1030, FR median 0.1180 (FR > CDR, 1.145x).
AbLang2 residue level: CDR median 5.8089, FR median 5.5917 (CDR > FR, 1.039x).
The direction flips -- AbLang2 encodes the correct biological CDR prior at the
single-residue level, even though it inverts it at the sequence level.

ESM-2 sequence level: CDR median 0.0802, FR median 0.1022 (FR > CDR, 1.275x).
ESM-2 residue level: CDR median 3.4586, FR median 3.6193 (FR > CDR, 1.046x).
No reversal -- ESM-2's inverse prior persists at both levels, attenuated but directionally
consistent.

Mechanistic interpretation of the AbLang2 reversal:
The sequence-level delta is the mean of all per-residue token changes across the chain.
A FR mutation in AbLang2 propagates broadly through the joint VH|VL representation via
cross-chain attention, making the chain-averaged delta large. At the single-token level,
the question is different: how much does AbLang2's embedding change at the mutated
position itself? CDR positions in AbLang2 (trained on OAS) encode rich per-position
amino acid diversity -- AbLang2 has strong position-specific representations for CDR
residues that change substantially when the amino acid identity changes. FR positions
are more conserved in OAS, so per-position token embeddings are less sensitive to
substitution even though FR mutations spread more broadly. In short: CDR positions have
higher single-token sensitivity in AbLang2, while FR mutations have higher global
propagation. These two effects produce opposite CDR/FR orderings depending on which
level is measured.

ESM-2 lacks cross-chain attention and cannot produce the propagation effect that inflates
FR sequence-level deltas. Its per-token sensitivity is determined by sequence-statistical
conservation: FR positions are conserved across diverse proteins, making any substitution
unusual at the token level. This same effect explains the inverse prior at both levels.

7-loop breakdown -- AbLang2 residue (approximate medians):
CDR_H3 ~6.00 (highest), CDR_L2/L3 ~5.80, CDR_H1 ~5.60, FR ~5.60, CDR_L1 ~5.55,
CDR_H2 ~5.40 (lowest CDR).
CDR_H3 is the highest loop at the residue level, opposite to sequence level where it
was mid-range. AbLang2 encodes the strongest single-position sensitivity for CDR_H3,
the most hypervariable CDR in the repertoire.

7-loop breakdown -- ESM-2 residue (approximate medians):
CDR_H2 ~3.60, FR ~3.60, CDR_H3 ~3.45, CDR_H1 ~3.40, CDR_L1/L2/L3 ~3.35.
CDR_H2 and FR are essentially tied at residue level (both ~3.60). All loops compressed
into a narrow range (3.35-3.60). FR has a long lower tail (down to ~1.0) -- some FR
mutations produce very small per-token perturbations, unlike at sequence level.

Implication for Experiment 7 (CDR constraint) -- updated:
At the sequence level, both models oppose the CDR prior (FR > CDR). At the residue
level, AbLang2 already encodes the correct prior (CDR > FR), while ESM-2 does not.
This means the CDR constraint loss operates against ESM-2's geometry at both levels,
but against AbLang2's geometry only at the sequence level. For experiments using
residue-level embeddings (Exp 2, 3, 5, 6), AbLang2's token representations are
already geometrically aligned with the CDR prior before any constraint is applied.

---

#### Finding 9: PCA structure of residue-level delta embeddings

PCA was run on the full residue delta tensor (5318 x 1280 for ESM-2, 5318 x 480 for
AbLang2). Three colorings were used: chain (H/L), CDR/FR, and dataset identity.

**ESM-2 residue PCA:**
PC1 (7.2%) + PC2 (6.6%) = 13.8% variance explained. Despite having no explicit H||L
concatenation at the residue level (each row is a 1280-dim token embedding from a single
forward pass, not concat(H, L)), ESM-2 residue PCA shows a clear cross structure with
well-defined arms along PC1 (horizontal) and PC2 (vertical). The mechanism differs from
the sequence-level cross: H and L mutations were processed in separate single-chain
forward passes, so their 1280-dim residue embeddings were contextualized by entirely
different surrounding sequences. PCA identifies the H-context embedding subspace and the
L-context embedding subspace as the two dominant directions, producing a cross even
without explicit concatenation.

Critically, H and L chains are fully mixed within both arms -- unlike the sequence-level
cross where each arm was nearly pure H or L. The chain-identity signal is present in
the residue embeddings (it drives the cross shape) but is not strong enough to cleanly
separate the two groups in 2D projection. CDR and FR are also mixed throughout both arms.

A distinct outlier cluster of ~50-100 points is visible at approximately (-2.0, 0.4),
clearly separated from the main mass. Dataset coloring shows all 5 antibody systems
distributed throughout both the cross arms and the outlier cluster with no dataset-
specific concentration. The cross geometry is therefore a structural property of ESM-2's
per-chain forward passes, not an artifact of mixing datasets with different mutation
compositions. The outlier cluster's origin is not resolved by any of the three colorings.

**AbLang2 residue PCA:**
PC1 (6.4%) + PC2 (6.1%) = 12.5% variance explained -- lower than ESM-2 residue (13.8%)
and dramatically lower than AbLang2 sequence-level (42.9%). The scatter is a diffuse
ellipse elongated along PC1 with no substructure: no cross, no clusters, no separation
by chain, CDR/FR, or dataset. All 5 antibody systems are distributed uniformly throughout
the ellipse. Dataset coloring confirms the flat structure is a property of the model's
joint VH|VL forward pass, not of dataset mixing.

One subtle asymmetry in the CDR/FR figure: CDR points extend slightly further along
positive PC1 and into the upper PC2 region relative to FR points. This is consistent
with the CDR > FR residue norm finding from Finding 8 -- larger per-token delta
magnitudes scatter further from the origin in PC space. It does not indicate separation
or clustering.

**Why residue PCA explains so little variance at both levels:**
Each residue delta row is a 1280-dim (ESM-2) or 480-dim (AbLang2) token embedding change
that encodes position-specific, amino acid-specific information across many independent
dimensions. The residue-level signal is genuinely high-dimensional -- it cannot be
compressed into 2 PCs without substantial loss. At the sequence level, mean pooling over
~200 residues averages out position-specific variation, concentrating the remaining
mutation-level signal into fewer dimensions and explaining why sequence PCA captures far
more variance (42.9% AbLang2, 26.1% ESM-2) than residue PCA (12.5% AbLang2, 13.8%
ESM-2) in two components.

This high dimensionality does not mean the residue vectors are uninformative. A trained
MLP with nonlinear projections can extract signal that 2D PCA cannot surface. AbLang2
residue maintains r=0.105 aggregate Spearman despite flat PCA, confirming that the
signal exists in directions not captured by the top 2 PCs.

**Comparison summary:**

| Model   | PC1+PC2 | Structure                     | Chain separation | CDR/FR sep | Dataset sep |
|---------|---------|-------------------------------|-----------------|------------|-------------|
| AbLang2 | 12.5%   | Diffuse ellipse               | None            | None       | None        |
| ESM-2   | 13.8%   | Clear cross + outlier cluster | None (arms mixed)| None      | None        |

---

#### Finding 10: Cross-model norm agreement

Spearman correlation was computed between ESM-2 and AbLang2 delta norm vectors
across all 5318 mutations. This measures whether the two models rank the same mutations
as large vs small perturbations -- i.e., how much shared information their embedding
geometries encode about mutation magnitude.

| Level    | r      | p           | Shared variance (r²) |
|----------|--------|-------------|----------------------|
| Sequence | 0.3617 | 4.66e-164   | 13%                  |
| Residue  | 0.0522 | 1.41e-04    | <1%                  |

**Sequence level (r=0.3617):** Moderate positive agreement -- both models partially
rank the same mutations as large. However, r²=0.13 means 87% of variance is model-
specific; the two models are substantially independent. This is consistent with AbLang2
having 2.6x higher DMS Spearman (0.216 vs 0.083): AbLang2 captures mutation-relevant
signal that ESM-2 does not, even though they share a partial common axis.

**Residue level (r=0.0522):** Near-zero cross-model agreement. The p-value (1.41e-04)
reaches significance only because N=5318; the effect size is negligible. The models
assign essentially uncorrelated per-token delta magnitudes to the same mutations. This
makes sense given the level-specific DMS Spearman results: ESM-2 residue norms are
uninformative (r=0.006, not significant), while AbLang2 residue norms carry moderate
signal (r=0.105). ESM-2 assigns residue norm rankings that are effectively random with
respect to both DMS scores and AbLang2's rankings.

**Scatter plot observations:**
At the sequence level, the mass of points sits below the y=x diagonal -- AbLang2 norms
are generally smaller than ESM-2 norms for most mutations. However, FR mutations show
a prominent upward plume: many FR points reach AbLang2 sequence norms of 0.3-0.6 while
their ESM-2 norms remain below 0.2. CDR points cluster tightly near or below the
diagonal. The FR-specific amplification in AbLang2 is mechanistically interpretable:
FR mutations propagate through cross-chain attention to affect the full VH|VL
representation, inflating AbLang2 sequence norms for FR mutations relative to ESM-2's
chain-isolated encoding.

At the residue level, the cloud sits above the y=x diagonal throughout -- AbLang2
residue norms are systematically larger than ESM-2 residue norms (consistent with
Finding 7: AbLang2 residue medians are ~1.63x larger). The cloud is roughly circular
with no positive slope, confirming near-zero rank agreement.

**Implication for experiment design:**
The near-zero cross-model agreement at the residue level means ESM-2 and AbLang2
residue delta vectors are largely complementary -- they encode different per-token
information about the mutation. Concatenating both models' residue embeddings as MLP
input (e.g., [esm2_res_delta || ablang2_res_delta] = 1280+480 = 1760 dims) may capture
more signal than either alone, since the two sources are nearly independent. At the
sequence level, the moderate agreement (r=0.36) implies that combining the two models'
sequence deltas will show diminishing returns relative to the residue level combination.
This is relevant to the design of any multi-model experiment beyond the current matrix.

---

## Open Questions

- NB04 open questions -- all answered:
  - [ANSWERED] Does AbLang2 show the same inverse CDR prior as ESM-2?
    Sequence level: YES, FR > CDR in both (ESM-2 1.275x, AbLang2 1.145x).
    Residue level: NO -- AbLang2 reverses to CDR > FR (1.039x). ESM-2 stays FR > CDR (1.046x).
  - [ANSWERED] Does cross-chain attention mix the H/L subspaces in AbLang2 PCA? YES --
    the orthogonal cross is completely absent. PC1+PC2 = 42.9% vs ESM-2's 26.1%.
  - [ANSWERED] Is residue-level Spearman stronger or weaker than sequence-level?
    Weaker for both models. ESM-2 residue aggregate r=0.0057 (p=0.677, not significant).
    AbLang2 residue aggregate r=0.1052, retaining ~half its sequence-level signal.
  - [ANSWERED] Do ESM-2 and AbLang2 agree on which mutations are large vs small?
    Partially at sequence level (r=0.3617, 13% shared variance), near-zero at residue
    level (r=0.0522). Models are largely complementary at residue level.

- CDR constraint geometry is now more nuanced:
  - ESM-2: inverse prior at both levels -- constraint opposes model geometry throughout
  - AbLang2: inverse prior at sequence level only -- residue-level geometry already
    encodes correct CDR > FR prior. Residue-based experiments (Exp 2, 3, 5) with AbLang2
    may not need constraint correction; the constraint will fire less and have less impact.

- Open training questions:
  - Does global context (Exp 3: delta_residue + wildtype) improve over local delta alone (Exp 2)?
  - Does PCA reduction (Exp 5) improve over raw delta residue (Exp 2)?
  - Which strategy best for ESM-2 vs AbLang2 -- do they favor different inputs?
  - Does the CDR constraint help or hurt each model (and does direction differ by model)?
  - Optimal lambda for constraint sweep [0, 0.1, 0.5, 1.0]?

---

## Notebook 05: MLP Training

### Infrastructure (complete as of this writing)

Two separate notebooks to avoid merge conflicts (Oscar: 05_oscar.ipynb, Lucas: 05_lucas.ipynb).
Identical setup/data-loading cells with shared random_state=42 for reproducible, identical splits.

#### Splits (verified)

| Split | N | % |
|---|---|---|
| Train | 4256 | 80.0% |
| Val | 531 | 10.0% |
| Test | 531 | 10.0% |

Per-antibody counts confirmed identical across both notebooks:
Train: lysozyme=1676, EGFR=857, VEGF=790, Ang2=785, HER2=148
Val:   lysozyme=209, EGFR=107, VEGF=99, Ang2=98, HER2=18
Test:  lysozyme=209, EGFR=107, VEGF=99, Ang2=98, HER2=18

HER2 val/test N=18 -- Spearman on 18 samples is noisy; report but note unreliability.

#### Training architecture (src/training/trainer.py)

- MLP: [input_dim -> 256 -> 128 -> 1], ReLU + Dropout(0.1) after each hidden layer
- Optimizer: Adam, lr=1e-3
- Loss: MSE
- Early stopping: patience=10 on aggregate val Spearman (all 5 datasets pooled)
- Evaluation: Spearman per dataset + aggregate (all 5) + aggregate excluding HER2
- tqdm: outer epoch bar (train_mse, val_rho, best, patience) + inner batch bar (current batch mse)
- W&B: per-epoch log + summary with best-epoch metrics
- CDR constraint: lambda_cdr=0.0 by default; set >0 for Exp 6

#### Experiment assignments

Oscar (05_oscar.ipynb): Exp 4 (DELTA_SEQUENCE) and Exp 3 (DELTA_RESIDUE_PLUS_WILD)
Lucas (05_lucas.ipynb): Exp 2 (DELTA_RESIDUE) and Exp 5 (DELTA_RESIDUE_REDUCED via PCA)
Exp 6 (CDR constraint): both, on best-performing strategy from Exp 2-5

#### Bugs fixed during NB05 setup

1. `src/data/datasets.py` DELTA_RESIDUE_PLUS_WILD: wildtype index loaded from JSON as
   `{row_int: dms_name}` but was being looked up as `wt_sequence_index[dms_name]`.
   Fix: invert at load time to `{dms_name: row_int}`. COMMITTED (f5ff08a).

2. Both notebooks: `load_abagym_antibody(DATA_DIR / 'abagym_antibody.csv')` should be
   `load_abagym_antibody(DATA_DIR)`. The function appends the filename internally.

#### Sanity checks (verified)

All input dims confirmed correct:
- ESM-2 DELTA_SEQUENCE: 2560
- AbLang2 DELTA_SEQUENCE: 960
- ESM-2 DELTA_RESIDUE_PLUS_WILD: 3840 (1280 + 2560)
- AbLang2 DELTA_RESIDUE_PLUS_WILD: 1440 (480 + 960)
- ESM-2 DELTA_RESIDUE: 1280
- AbLang2 DELTA_RESIDUE: 480
- ESM-2 DELTA_RESIDUE_REDUCED (PCA 64): 64, explained variance = 0.810
- AbLang2 DELTA_RESIDUE_REDUCED: not yet verified (training cell only)

#### Results

---

### Experiment 4: Delta Sequence (Oscar)

**Input:** `mean_pool(mutant) - mean_pool(wt)` | ESM-2=2560-dim, AbLang2=960-dim

**Val results (best epoch):**

| Model | Best epoch | Val Spearman (all) | Ang2 | EGFR | HER2 | VEGF | lysozyme |
|---|---|---|---|---|---|---|---|
| ESM-2 | 24 | 0.6381 | 0.7034 | 0.7259 | 0.4370 | 0.4921 | 0.6049 |
| AbLang2 | 35 | 0.6552 | 0.7980 | 0.6540 | 0.3963 | 0.6651 | 0.5413 |

**Test results:**

| Model | Spearman (all) | Spearman (excl HER2) | HER2 | Ang2 | EGFR | VEGF | lysozyme |
|---|---|---|---|---|---|---|---|
| ESM-2 | 0.6122 | 0.6026 | 0.8262* | 0.6470 | 0.6604 | 0.5416 | 0.5632 |
| AbLang2 | 0.6024 | 0.6034 | 0.5418 | 0.7303 | 0.5265 | 0.7451 | 0.5258 |

*ESM-2 HER2 test=0.826 vs val=0.437 -- high variance from N=18, not reliable.

**Key findings:**

Both models far exceed the EDA norm baseline (ESM-2: 0.083→0.638 val, AbLang2: 0.216→0.655 val).
The full directional delta vector carries substantially more signal than norm magnitude alone.

Excluding HER2, the models are essentially tied on test (ESM-2 0.603, AbLang2 0.603).
The dataset-level split is sharp and consistent with EDA predictions:
- ESM-2 leads on EGFR (+0.134) and lysozyme (+0.037) -- the FR-heavy datasets
- AbLang2 leads on Ang2 (+0.083) and VEGF (+0.204) -- the CDR-heavy G6-scaffold datasets

AbLang2's 2.6x EDA advantage in norm Spearman nearly vanishes once the MLP is trained
on the full vector. ESM-2's delta space contains rich directional signal that the norm
does not capture.

---

### Experiment 3: Delta Residue + Full Wildtype (Oscar)

**Input:** `concat(delta_residue[mut_pos], mean_pool(wt_sequence))` | ESM-2=3840-dim, AbLang2=1440-dim

**Val results (best epoch):**

| Model | Best epoch | Val Spearman (all) | Ang2 | EGFR | HER2 | VEGF | lysozyme |
|---|---|---|---|---|---|---|---|
| ESM-2 | 78 | 0.7006 | 0.7899 | 0.6116 | 0.6388 | 0.7437 | 0.6328 |
| AbLang2 | 50 | 0.6621 | 0.7807 | 0.6047 | 0.3942 | 0.6182 | 0.6280 |

**Test results:**

| Model | Spearman (all) | Spearman (excl HER2) | HER2 | Ang2 | EGFR | VEGF | lysozyme |
|---|---|---|---|---|---|---|---|
| ESM-2 | 0.6963 | 0.6946 | 0.7068 | 0.8294 | 0.6621 | 0.7815 | 0.5737 |
| AbLang2 | 0.6640 | 0.6693 | 0.5189 | 0.7610 | 0.5909 | 0.6802 | 0.6259 |

**Key findings:**

Exp 3 outperforms Exp 4 for both models (ESM-2: 0.603→0.695, AbLang2: 0.603→0.669, excl HER2).
Adding wildtype context adds substantial value -- the wildtype embedding provides global scaffold
identity that the per-position delta alone does not encode.

ESM-2 now clearly leads AbLang2 (0.695 vs 0.669 excl HER2). The gain is larger for ESM-2
(+0.092 vs +0.066) because global antibody context compensates for ESM-2's per-chain blind spot --
AbLang2 already encodes cross-chain context via joint VH|VL forward passes, so the explicit
wildtype embedding adds less marginal information.

ESM-2 best epoch = 78 (vs 24 for Exp 4) -- the 3840-dim input requires more epochs to converge.
ESM-2 HER2 val = 0.639 (vs 0.437 in Exp 4) -- more stable, consistent with test 0.707.

**Current best:** ESM-2 Exp 3, test Spearman excl HER2 = 0.6946.

**Framing note:** Foundation models are frozen throughout all experiments. No fine-tuning
occurs. All performance differences reflect what the MLP regression head can extract
from pre-computed, fixed embeddings. "ESM-2 performs better" means the MLP trained on
ESM-2 embeddings performs better -- the foundation model weights do not change.

**Information-scaling hypothesis:** The MLP trained on ESM-2 embeddings scales more
steeply with input richness than the MLP trained on AbLang2 embeddings. AbLang2 encodes
global antibody context internally (joint VH|VL forward pass), so its embeddings already
contain cross-chain and scaffold context -- the wildtype embedding adds less marginal
information to the AbLang2 MLP. ESM-2 embeds chains separately with no cross-chain
attention, so the MLP lacks global scaffold context unless it is provided explicitly.
Adding the wildtype embedding to the ESM-2 MLP fills this gap, producing a larger gain.

Testable prediction: Exp 2 (single-token delta residue, no global context) should
flip back to AbLang2 MLP leading, since the input is even more local than Exp 4.

---

### Oscar Experiment Summary (Exp 3 vs Exp 4)

| Experiment | Model | Spearman (all) | Spearman (excl HER2) |
|---|---|---|---|
| Exp4: Delta Sequence | ESM-2 | 0.6122 | 0.6026 |
| Exp4: Delta Sequence | AbLang2 | 0.6024 | 0.6034 |
| Exp3: Delta Res + WT | ESM-2 | **0.6963** | **0.6946** |
| Exp3: Delta Res + WT | AbLang2 | 0.6640 | 0.6693 |

---

### Experiment 2: Delta Residue Only (Lucas)

**Input:** `mutant_residue_emb[mut_pos] - wt_residue_emb[mut_pos]` | ESM-2=1280-dim, AbLang2=480-dim

**Val results (best epoch):**

| Model | Best epoch | Val Spearman (all) |
|---|---|---|
| ESM-2 | 10 | 0.5715 |
| AbLang2 | 19 | 0.6663 |

**Test results:**

| Model | Spearman (all) | Spearman (excl HER2) | HER2 | Ang2 | EGFR | VEGF | lysozyme |
|---|---|---|---|---|---|---|---|
| ESM-2 | 0.5834 | 0.5799 | 0.7649 | 0.6381 | 0.6287 | 0.4920 | 0.5977 |
| AbLang2 | 0.6198 | 0.6153 | 0.6466 | 0.7605 | 0.5932 | 0.5838 | 0.5419 |

**Key findings:**

AbLang2 leads ESM-2 clearly (0.615 vs 0.580 excl HER2, +0.035). The information-scaling
hypothesis prediction holds: with only a single-token residue delta and no global context,
AbLang2's domain-specific pretraining advantages show through.

ESM-2 converges at epoch 10 -- the fastest of any experiment -- indicating the signal
available in a single ESM-2 residue delta is limited and quickly exhausted. Consistent
with EDA finding that ESM-2 residue delta norms are nearly uninformative (aggregate r=0.006).

AbLang2 Exp 2 (0.615) exceeds AbLang2 Exp 4 (0.603): the residue-level delta is more
informative for AbLang2 than the sequence-level delta. Consistent with EDA showing
AbLang2 encodes the correct CDR prior at the residue level (CDR > FR, r=0.105). The
sequence delta averages over the full chain and dilutes this position-specific signal.

ESM-2 goes the other direction: Exp 2 (0.580) < Exp 4 (0.603). Residue-level is worse
for ESM-2, consistent with its per-token embeddings being less sensitive to amino acid
identity at individual positions.

**Confirmed pattern across Oscar + Lucas experiments:**

| Exp | Input | ESM-2 (excl HER2) | AbLang2 (excl HER2) | Leader |
|---|---|---|---|---|
| 2 | Residue delta only | 0.580 | 0.615 | AbLang2 +0.035 |
| 4 | Sequence delta only | 0.603 | 0.603 | tied |
| 3 | Residue delta + wildtype | **0.695** | 0.669 | ESM-2 +0.026 |

As input richness increases, the ESM-2 MLP goes from behind to tied to ahead. This is
the central empirical finding of the project so far.

---

### Experiment 5: Delta Residue + PCA(64) (Lucas)

**Input:** `PCA(delta_residue[mut_pos], n=64)` | ESM-2=64-dim, AbLang2=64-dim
PCA fit on train split only (4256 samples). ESM-2 explained variance=0.810, AbLang2=0.798.

**Val results (best epoch):**

| Model | Best epoch | Val Spearman (all) |
|---|---|---|
| ESM-2 | 13 | 0.5263 |
| AbLang2 | 20 | 0.5887 |

**Test results:**

| Model | Spearman (all) | Spearman (excl HER2) | HER2 | Ang2 | EGFR | VEGF | lysozyme |
|---|---|---|---|---|---|---|---|
| ESM-2 | 0.4999 | 0.5004 | 0.4878 | 0.5152 | 0.5449 | 0.4136 | 0.5492 |
| AbLang2 | 0.5628 | 0.5598 | 0.5750 | 0.7333 | 0.6122 | 0.3905 | 0.5285 |

**Key findings:**

PCA compression hurts both models vs raw delta residue (Exp 2):
- ESM-2: 0.580→0.500 (-0.080)
- AbLang2: 0.615→0.560 (-0.055)

Retaining 80% of variance is not sufficient. The signal for mutation effect prediction
is distributed across dimensions that PCA does not prioritize -- PCA selects for
high-variance directions, which are not the same as high-predictive-signal directions.
ESM-2 takes the larger hit (-0.080 vs -0.055), consistent with its useful signal being
more spread across the full 1280-dim residue space.

AbLang2 still leads ESM-2 (0.560 vs 0.500 excl HER2), and the gap (+0.060) is the
largest of any experiment -- PCA amplifies the disadvantage of ESM-2's more distributed
signal structure.

---

### Full Experiment Summary (Exp 2-5, test Spearman excl HER2)

| Exp | Input | ESM-2 | AbLang2 | Leader |
|---|---|---|---|---|
| 5 | Residue delta + PCA(64) | 0.500 | 0.560 | AbLang2 +0.060 |
| 2 | Residue delta only | 0.580 | 0.615 | AbLang2 +0.035 |
| 4 | Sequence delta only | 0.603 | 0.603 | tied |
| 3 | Residue delta + wildtype | **0.695** | **0.669** | ESM-2 +0.026 |

As input richness increases, the ESM-2 MLP goes from behind to tied to ahead.
This is the central empirical finding of the project.

**Best strategy:** Exp 3 (delta residue + wildtype) for both models.
**Exp 6 decision:** CDR constraint lambda sweep on Exp 3 strategy for both ESM-2 and AbLang2.

---

### Experiment 6: CDR Constraint Lambda Sweep (Both)

**Strategy:** Exp 3 (delta residue + wildtype), best-performing from Exp 2-5.
**Lambdas:** [0, 0.1, 0.5, 1.0] x 2 models = 8 runs. Lambda=0 reproduces Exp 3 baseline exactly.

**Test results (excl HER2):**

| Model | λ=0.0 | λ=0.1 | λ=0.5 | λ=1.0 |
|---|---|---|---|---|
| ESM-2 | 0.6946 | 0.6927 | 0.7030 | 0.6219 |
| AbLang2 | 0.6693 | 0.6622 | 0.6614 | 0.6459 |

**Seed robustness check (ESM-2 λ=0.5 vs λ=0.0):**

| Seed | λ=0.0 | λ=0.5 | Delta |
|---|---|---|---|
| 42 | 0.6946 | 0.7030 | +0.0084 |
| 0  | 0.6910 | 0.6767 | -0.0143 |
| 1  | 0.6396 | 0.6387 | -0.0009 |

**Key findings:**

ESM-2 (null result): The λ=0.5 peak (+0.008) does not hold across seeds. Deltas are
+0.008, -0.014, -0.001 across seeds 42, 0, 1. Effect is within noise. The constraint
has no reliable impact on ESM-2 despite EDA showing ESM-2 encodes the inverse CDR prior.
The batch-mean constraint may be too weak to correct an entrenched geometric bias.

AbLang2 (negative, consistent): Monotonic decline at every λ > 0. Drop grows with λ
(0.669 → 0.662 → 0.661 → 0.646). Consistent with EDA finding that AbLang2 residue-level
embeddings already encode the correct CDR prior (CDR > FR at token level, Finding 8).
The constraint is redundant and adds gradient noise that hurts task performance.

**The neurosymbolic finding:**
The constraint interacts with each model's geometry in the predicted direction:
irrelevant to ESM-2 (no reliable correction of its inverse prior) and harmful to
AbLang2 (redundant with its already-correct geometry). The negative AbLang2 result
is indirect evidence that AbLang2's representations encode biologically meaningful
CDR/FR structure internally.

**Best overall result:** ESM-2 Exp 3 λ=0.0, test Spearman excl HER2 = 0.6946.

---

### Experiment 6 Extension: Pairwise Ranking Constraint

**Motivation:** Two weaknesses in the batch-mean formulation: (1) one gradient term per
batch (group mean comparison), (2) no margin -- satisfied as soon as CDR mean exceeds
FR mean by any epsilon. The pairwise formulation addresses both:

```
loss = mean_{i in FR, j in CDR} ReLU(|pred_i| - |pred_j| + margin)
```

Implemented in `src/training/losses.py` as `pairwise_cdr_constraint_loss`. TrainConfig
gains `constraint_type` ('batch_mean' | 'pairwise') and `constraint_margin` fields.

**Margin=0.1 (first run):** Catastrophic collapse at λ≥0.5. The margin ensures the loss
fires on nearly every pair even when the model is roughly ordered, overwhelming task
gradient. ESM-2 dropped to 0.439, AbLang2 to 0.442 excl HER2 at λ=0.5.

**Margin=0.0 (final run):** Restores self-regulation -- pairs where |CDR| >= |FR|
contribute zero gradient. Still more aggressive than batch-mean.

**Test results (excl HER2), margin=0.0:**

| Formulation | λ=0.0 | λ=0.1 | λ=0.5 | λ=1.0 |
|---|---|---|---|---|
| ESM-2 batch-mean | 0.6946 | 0.6927 | **0.7030** | 0.6219 |
| ESM-2 pairwise margin=0.0 | 0.6946 | 0.6932 | 0.5266 | 0.5219 |
| ESM-2 pairwise margin=0.1 | 0.6946 | 0.6105 | 0.4386 | 0.3920 |
| AbLang2 batch-mean | 0.6693 | 0.6622 | 0.6614 | 0.6459 |
| AbLang2 pairwise margin=0.0 | 0.6693 | 0.6393 | 0.5669 | 0.4742 |
| AbLang2 pairwise margin=0.1 | 0.6693 | 0.6262 | 0.4424 | 0.3739 |

**Key findings:**

Monotonic degradation as constraint strength increases across all three formulations
and both models: batch-mean < pairwise margin=0.0 < pairwise margin=0.1 at every λ > 0.
This is not an artifact of any single implementation choice.

The batch-mean formulation was operating at the only point where the constraint is
weak enough not to dominate task learning. The pairwise extension confirms this: richer
gradient signal does not help, it accelerates the collapse.

HER2 decorrelation: at high lambda, excl_her2 collapses while HER2 Spearman stays
elevated or rises. HER2 mutations are CDR mutations (trastuzumab binds via CDR loops),
so the constraint pushes all HER2 predictions up together. Relative ranking within HER2
is preserved by residual task signal on 18 points. This decorrelation is additional
evidence that the constraint has taken over from task learning at high lambda.

**Extended conclusion:**
The CDR constraint in any formulation does not improve on the unconstrained baseline.
Stronger constraint formulations produce worse outcomes, confirming the batch-mean null
result for ESM-2 and monotonic decline for AbLang2 are not artifacts of constraint
weakness. The neurosymbolic finding stands.

---

---

### NB07: Analysis and Figures

Three figures generated from confirmed test results. No model loading required --
results serialized from NB06 via experiment_results.json.

**Figure 1: Embedding strategy comparison**
Grouped bar chart (4 strategies x 2 models, excl HER2). ESM-2 starts below AbLang2
at Delta Residue (0.613 vs 0.635) and finishes above at Delta Res + Wildtype
(0.695 vs 0.669). AbLang2 is stable across strategies; ESM-2 gains more from
additional context (+0.093 vs +0.034).

**Figure 2: CDR constraint sweep**
Line plots for all three formulations (batch-mean, pairwise margin=0, pairwise
margin=0.1) x 2 models. Monotonic degradation with constraint strength holds
across all formulations and both models. Batch-mean is the most robust.

**Figure 3: Per-dataset Spearman (best model, Exp 3 lambda=0)**

| Dataset | ESM-2 | AbLang2 |
|---|---|---|
| Ang2 2017 G6 | 0.829 | 0.761 |
| VEGF 2017b G6 | 0.782 | 0.680 |
| HER2 2021 trastuzumab | 0.707 (N=18) | 0.519 (N=18) |
| EGFR 2013 Cetuximab | 0.662 | 0.591 |
| lysozyme 2019 D441 | 0.574 | 0.626 |

ESM-2 leads on 4 of 5 datasets (excl HER2). AbLang2 leads on lysozyme 2019 D441
(0.626 vs 0.574).

---

## Still To Do

Project complete. SAbDab (Task 2) not pursued -- was contingent on the CDR
constraint showing a positive result to inform an biologically-grounded affinity
regression head. Constraint showed no benefit; SAbDab dropped.

### Later
- NB06: Analysis and figures
- train_sabdab / evaluate_sabdab implementation (Task 2)
- SAbDab binding affinity experiments

---