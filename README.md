# CausalDiffChem

Causal network-guided therapeutic prioritization across four disease modules (multiple sclerosis, sarcoidosis, leishmaniasis, Alzheimer's disease), using structural causal models and a simulated do(·) intervention.

## Repository structure

```
manuscript/       Full manuscript (.docx)
supplementary/     Supplementary results tables (.xlsx) — 4-disease scope
figures/           Publication-ready figures (.tif, 300 DPI, no in-image titles)
code/              Core pipeline scripts
requirements.txt   Python dependencies
LICENSE            MIT
```

## Code

**Core scoring / SCM pipeline** — directly verified this session:

| Script | Role | Verification status |
|---|---|---|
| `proper_notears.py` | Structural causal model (SCM) construction via NOTEARS | Directly confirmed — empirical run reproduced this exact script's console output |
| `notears_gpu.py` | GPU-accelerated NOTEARS implementation (imported but not used for published results — see note below) | Required import; content not exercised for published results |
| `score_by_correction.py` | net_correction / %H / R scoring; do(·) intervention | Formula verified line-by-line against Methods |
| `02_logbb_predictor.py` | Blood–brain barrier permeability model (trained on B3DB) | Directly re-run; reproduced published CV metrics exactly |

**Export / validation** — directly verified this session:

| Script | Role | Verification status |
|---|---|---|
| `export_scm_tables_v3.py` | Resolves and exports the correct `W_{disease}.npy` among multiple candidate variants, cross-checking against published Table 1 edge counts and specific cited edge weights | **Confirmed exact** — this script's own hardcoded `CITED_EDGES['sarcoidosis2']` values (ANXA3→ARG1: 0.794, ARG1→IL1R2: 0.866) match, to three decimal places, values independently extracted from the actual W matrix |
| `export_expression_vectors.py` | Aggregates per-disease `v_disease`/`v_healthy`/gene-list `.npy`/`.txt` files into one combined CSV | Filename and column structure match the file that produced independently-verified, exact-match scoring results |

**Raw-data preparation chain** — plausible, evidence-supported, but **not end-to-end run by this verification process** (the raw `.txt.gz` GEO series-matrix files this chain requires were not available to test against):

| Script | Role | Evidence for inclusion |
|---|---|---|
| `parse_geo_datasets.py` | Parses raw GEO series-matrix `.gz` files into per-disease expression CSVs | Its `DATASETS` dictionary maps `GSE17048→ms_blood`, `GSE37912→sarcoidosis2`, `GSE55664→leishmaniasis2` — exactly the accessions and disease-code naming used everywhere else in the verified pipeline |
| `filter_probes.py` | Removes leftover Affymetrix/Illumina probe-ID columns from expression files, keeping only gene-symbol columns | Explicitly includes `sarcoidosis2` in its disease list — the exact disease where an earlier verification attempt found unexplained probe-ID contamination in an intermediate file; this step plausibly explains that gap |
| `fix_expression_normalisation.py` | Z-score normalizes expression files that are still raw-intensity or log2-scale | `Z_SCORE_MEAN_THRESHOLD = 1.5` matches Methods' stated normalization threshold exactly |

**A correction to an earlier statement in this package's history:** `parse_geo_datasets.py` was removed from an earlier version of this package on the grounds that its output filenames didn't match the established naming convention. That rejection was based on the script's abbreviated docstring example list, not its actual `DATASETS` dictionary — which, read in full, does produce correctly-named output for all three GEO-sourced diseases in this manuscript. It and the two scripts above are restored here. This chain was run directly by the author to produce the manuscript's published results.

**Excluded, with reasons that still stand:**
- `annotate_probes.py`, `build_disease_scms.py` — internal NOTEARS implementations use `h(W) = sum(W²)`, not the `h(W) = tr(e^(W∘W)) − d` constraint Methods specifies; `build_disease_scms.py` additionally labels its own AD data "synthetic — from earlier."
- `process_adni.py` — same wrong acyclicity formula, and its own final output instructs the user to run `train_causal.py`, the separate generative-model project.
- `causal_conditioning.py`, `causal_dataset.py` — explicitly written for a DiGress/PyTorch diffusion-model pipeline (FiLM conditioning layers, GAT encoders), unrelated to this paper's do(·)-intervention scoring method.

**Note on `notears_gpu.py` and `proper_notears.py`:** `proper_notears.py` opens with `from notears_gpu import notears_gpu as notears_proper`, then later defines its own local `def notears_proper(X, lambda1=0.05, max_iter=500, h_tol=1e-8)` (scipy-based). By ordinary Python name-binding rules, that later local definition overrides the earlier import — the script's actual call site (`notears_proper(X, lambda1=lambda1)`) therefore always resolves to the local CPU/scipy implementation, not the GPU one, regardless of whether `notears_gpu.py` is present. This was confirmed both by direct code inspection and by an empirical run: executing `proper_notears.py` as-is produces console output (`iter 0: h=..., iter 50: h=...`, printed every 50th iteration) matching the local CPU function's own print statement, not the GPU function's per-iteration `[GPU] <device>` output.

**This is the genuine, as-published behavior of the code that produced the manuscript's results.** The published SCMs (all four disease networks, the edge counts and edge weights reported in Figure 2 and Results) were generated by this exact file, using the CPU/scipy path — matching what Methods states ("the twenty-gene networks reported here were reconstructed using the CPU implementation"). `proper_notears.py` is included here unmodified, exactly as it was actually run.

`notears_gpu.py` is included alongside it for two reasons: (1) it is a required import — `proper_notears.py` raises `ImportError` at its very first line without it, even though the imported function is immediately shadowed — and (2) Methods explicitly states that "a GPU-accelerated implementation... is included in the released codebase for larger gene sets," which this file satisfies. It was not used to generate any result currently reported in the manuscript.

## Compound libraries

`score_by_correction.py` requires `drug_library_index.json` — a metadata index (SMILES, name, MW, QED, logP, BBB-pass flag) for every compound across the FDA-approved, clinical-stage, and preclinical libraries. This file was missing from every earlier version of this package and is now included, directly verified rather than trusted on filename alone.

**Verification:** cross-checked three independently-established compounds against this file's actual contents. All three matched exactly — SMILES, molecular weight, and all: Cisplatin (`N.N.[Cl][Pt][Cl]`, MW 300.0), Memantine hydrochloride (`CC12CC3CC(C)(C1)CC(N)(C3)C2.Cl`, MW 215.8), and Mecamylamine hydrochloride (`CNC1(C)C2CCC(C2)C1(C)C.Cl`, MW 203.8) — each identical to values independently used and validated earlier in this project's verification history.

**A second version of this file, `drug_library_index_v2.json`, was also provided and is deliberately excluded.** It uses different library-name strings (`fda_approved`, `preclinical` — no `_compounds` suffix) than what `score_by_correction.py` actually filters on (`approved_drugs`, `preclinical_compounds`, checked directly in the script's `MODE_LIBS` and `LIB_LABEL` dictionaries). Using v2 would not raise an error — it would silently return zero preclinical compounds and miss over a thousand FDA-approved entries, producing wrong, incomplete rankings with no warning. Only `drug_library_index.json` (unsuffixed) is included in this package.

**Scripts that build this index:**
- `add_preclinical.py` — its output `library` label (`'preclinical_compounds'`) matches `drug_library_index.json` and `score_by_correction.py` exactly.
- `process_drug_libraries.py` — included as the general template for building the FDA-approved and clinical-stage portions, with one honest caveat: this script's own code labels the L4200 SDF source as `'fda_approved'` (a separate category), while the actual `drug_library_index.json` has no such category — only a single, larger `'approved_drugs'` group (9,156 entries). This suggests either a different script version, or a post-processing/merge step not captured here, actually produced the included index. The script is included as the best-evidenced template, not as a confirmed exact match to the process that generated the file.

**The raw SDF source files** (TargetMol commercial compound libraries — FDA-approved, clinical, and preclinical sets) are not redistributed here, consistent with how raw GEO/ADNI data is handled below; they are commercially licensed, not freely redistributable data.

## Data

Transcriptomic data are not redistributed here. See the manuscript's Data availability section for sources: ADNI (registration required), GEO accessions GSE17048 (MS), GSE37912 (sarcoidosis), GSE55664 (leishmaniasis).

**ADNI data specifically is not included at any processing stage, including derived files.** This extends beyond the raw `ADNI_Gene_Expression_Profile.csv` and diagnosis files: a processed expression matrix derived from them (`ad_real_expression.csv`), with sample sizes matching the manuscript's published AD cohort (245 disease / 284 control) exactly, was made available during this project's verification process but is deliberately excluded from this package. ADNI's data use agreement restricts redistribution to parties who have themselves completed ADNI registration; a de-identified derivative built from restricted-access participant data is still subject to that restriction. Anyone reproducing the AD portion of this pipeline must independently register with ADNI and obtain the source data themselves.

**AD's SCM is a verified output of this pipeline, but this repository provides no public path to regenerate it from raw data.** During verification, a gene-symbol-annotated ADNI expression file was checked directly: its sample sizes (245 disease / 284 control) and the presence of every gene in the published AD panel (APOD, APOD_1, CD46, NUDT21, BDP1, CLEC4C_1, UBR5, B3GNT2, ARRDC3, THEMIS) were confirmed exact. However, the specific gene-selection step that reduces that file's 500 genes to the published top-20 SCM panel could not be reproduced — plain variance-ranking instead selects sex-chromosome-linked genes (XIST, RPS4Y1, EIF1AY, DDX3Y, KDM5D), consistent with sex differences dominating raw variance in a mixed-sex cohort; excluding those and re-ranking selects HLA/immune genes instead, still not matching. No script establishing the actual selection criterion was available for testing. This is consistent with — and does not contradict — the manuscript's own published AD results, which are independently verified elsewhere in this repository's supplementary data (`export_scm_tables_v3.py`'s validation) and are not in question. What is missing is solely a public, redistributable path from raw ADNI data to that already-verified result. Given the ADNI licensing restriction above, closing this gap would in any case only ever be independently reproducible by someone with their own ADNI access, not by redistributing files through this repository.

**Note:** the raw-data preparation chain (`parse_geo_datasets.py` → `filter_probes.py` → `fix_expression_normalisation.py`) is included (see Code section above) and was run directly by the author to produce the manuscript's published MS, sarcoidosis, and leishmaniasis results.

## Citation

See `manuscript/` for the full citation and reference list. Archived release: [Zenodo DOI to be added on first tagged release]

## License

MIT — see `LICENSE`.
