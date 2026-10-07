# Leakage-free re-scoring (October 2026)

Corrected evaluation for the SPMB 2026 abstract. Why it exists: [AUDIT.md](AUDIT.md).

**No retraining.** The June encoder (`encoder.pt`) and its saved night-one embeddings (`features2_enc/`) are
re-scored. The only new computation is night-two embeddings for the 28 evaluation subjects that have a second
night (Sleep-EDF subject 13 has only one night), made with the same frozen encoder.

## Protocol

| Item | Setting |
|---|---|
| Subjects | 73 Sleep Cassette subjects; 44 train / 29 eval, subject-disjoint (sorted IDs, seed 42, train fraction 0.6) |
| Enrolment / probe | per subject and stage, chronological order; template = first part, probes = last part |
| Guard | ≥ 10 same-stage epochs (≥ 5 min) between template and probe epochs |
| Minimum | ≥ 10 epochs per part; cells with fewer are skipped (per-cell counts in `results_leakfree.json`) |
| Template | mean of enrolment embeddings (night one) |
| Probe | mean of 10 consecutive same-stage probe epochs ("10 accumulated epochs", not wall-clock time) |
| Sharing | the same templates and probe segments for within-stage, cross-stage and cross-night cells |
| Impostors | zero-effort: each probe vs every other evaluation subject's template |
| Score | cosine similarity |
| EER aggregation | pooled per (enrol stage, probe stage) cell, then averaged (5 diagonal / 20 off-diagonal cells); per-subject EER for paired tests |
| Wake | ±30 min around sleep plus wake after sleep onset (primary); untrimmed reported as sensitivity |
| CIs | subject-level cluster bootstrap, B = 1000, with replacement; duplicate-identity pairs excluded |
| Tests | Wilcoxon signed-rank (paired, per subject), Friedman, Holm correction |
| Multi-stage | equal enrolment length (20 epochs = 10 min), identical probes, subjects with enough epochs in every stage (n = 13) |
| Re-enrolment | offline template update; update epochs precede all night-two probes; probe set fixed across update sizes |
| Band-stop | 4th-order zero-phase Butterworth (delta 0.5–4, theta 4–8, alpha 8–12, sigma 12–16, beta 16–30 Hz) on enrolment and probe epochs; encoder not retrained; single-epoch probes |

## Files

```
code/rerun_leakfree.py        steps: embed, main, band_pkl (band), numbers
code/make_numbers_extra.py    extra macros (pairwise stage tests, subject counts) -> numbers_extra.tex
code/make_abstract_fig.py     Figure 1: two 5x5 EER matrices (night-1 / night-2 probes), grayscale
colab/leakfree_rescoring_colab.ipynb   CPU-only notebook (Drive mounted read-only)

results/results_leakfree.json             all numbers, both wake settings ("trim", "untrimmed")
results/band_ablation_leakfree.json       per-band EER increase, CI, Wilcoxon, Holm
results/matrix_night1_clean_*.csv         5x5 EER matrices, corrected protocol
results/matrix_crossnight_clean_*.csv     night-1 template vs night-2 probes
results/matrix_night1_LEAKY_June_protocol_*.csv   old protocol run inside the new code (for comparison only)
results/scores_for_DET_{trim,untrimmed}.npz       genuine/impostor scores (within, cross-stage, cross-night, both) for ROC/DET
results/numbers.tex, numbers_extra.tex    LaTeX macros used by the abstract
results/main.log, embed.log, band.log, args_*.json, runtime.json, embed_check_runtime.json
results/fig_matrices.pdf / .png
results/norm_stats.npz                    training-subject z-normalisation statistics
```

Personal storage paths in the logs and JSON files were replaced by `<SOURCE_ROOT>` (June project folder) and
`<OUT>` (output folder); nothing else was edited. Night-two embeddings (`emb_n2/`, 35 MB) are not committed;
`embed` regenerates them.

## How to run

`ROOT` must contain the June artifacts: `encoder.pt`, `features2_enc/`, `features2/`, `epochs2/`, `crossnight.py`
(same as the copy in the repository root), `emb_ablation.pkl` (from `band_ablation2.py`). `ROOT` is only read.

```bash
python code/rerun_leakfree.py embed    --root ROOT --out OUT                 # consistency check + night-2 embeddings
python code/rerun_leakfree.py main     --root ROOT --out OUT --emb_n1 ROOT/features2_enc --emb_n2 OUT/emb_n2
python code/rerun_leakfree.py band_pkl --root ROOT --out OUT
python code/rerun_leakfree.py numbers  --out OUT
python code/make_numbers_extra.py OUT trim
python code/make_abstract_fig.py --out OUT
```

`embed` stops if re-embedding two evaluation subjects does not reproduce the saved June embeddings
(the reported run gave minimum cosine 0.9999994). The reported run (2026-10-04, Google Colab CPU) used the
previous version of `rerun_leakfree.py`; the current version only adds copying `crossnight.py` into `OUT`
before importing it, so no `__pycache__` is written into `ROOT`. The scoring code is identical.

## Main results (wake trimmed)

| Quantity | Value |
|---|---|
| Within-stage EER | 4.85% [2.74, 7.76] |
| Cross-stage EER | 12.10% [8.27, 15.42]; difference +7.25 [4.82, 9.39], Wilcoxon p = 6.1e-8, d_z = 1.23 |
| Same-stage cross-night EER | 19.36% [13.78, 24.69]; within-night on the same 28 subjects 4.91%, p = 7.1e-4 |
| Cross-stage + cross-night EER | 24.76% [19.49, 30.95] |
| Within-stage by stage | W 12.11, N1 3.65, N2 4.51, N3 2.11 (n = 19), REM 1.90; W worse than each sleep stage (Holm p ≤ 0.009) |
| FRR at FAR 1% (within) | 20.44% |
| HTER, matched / mismatched stage threshold | 6.20% / 7.43% |
| Multi-stage, probe stage absent from enrolment | four-other-stage template 10.09% vs one other stage 13.74% (Holm p = 3.7e-4) |
| Offline template update with night-two data | 19.36% → 6.67% |
| Band-stop | all five bands increase EER (Holm p < 0.001); theta largest, +4.72 points |

Untrimmed values are in `results_leakfree.json` under `"untrimmed"` (within 4.44%, cross 12.48%).
