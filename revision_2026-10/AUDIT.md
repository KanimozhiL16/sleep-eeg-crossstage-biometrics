# Audit: enrolment/probe leakage in the submitted SPMB 2026 paper (3 October 2026)

**Trigger.** An SPMB 2026 reviewer inspected this repository (commit `c5370b6`) and reported that same-stage probes
were drawn from the pool averaged into their enrolment template. Checked against the code: **confirmed.**

## Where it happens
Template = mean of the FULL stage pool; same-stage probes drawn from that same pool.

| File | Function | Used for (submitted paper) |
|---|---|---|
| sleep_pipeline_v2.py | `_crossstage` | within/cross matrix (cosine, LDA, encoder 0.21% / 9.58%), fusion 1/5/10; called by encoder_train.py |
| stats_tests.py | `per_subject_eer` | Wilcoxon p = 1.9e-9, d = 1.14, gap 12.4 points, encoder vs cosine |
| export_enc_matrix.py | main loop | Fig. 2 heatmap |
| sleep_pipeline_v3.py | `crossstage`, `stage_ttd` | time-to-decision |
| sleep_pipeline_v4.py | `scores` | multi-stage (single-match and all-five 4.39%), threshold transfer, FRR at FAR 1%, DET |
| band_ablation2.py | `wover` (via `_crossstage`) | band-ablation baseline |

The docstring of `sleep_pipeline_v3.crossstage` says "disjoint epochs implicitly via mean vs samples"; that is not disjoint.

## Affected (recomputed)
Every within-stage (diagonal) number; the gap test (p, d, CI); time-to-decision; FRR at FAR 1%; HTER threshold
transfer; band ablation; multi-stage all-five. The 9.58% → 4.39% multi-stage comparison was also not like-for-like
(different enrolment amounts and probe sets).

## Not affected by this leak
- Off-diagonal cross-stage cells (template stage A, probes stage B ≠ A). They still change slightly, because
  templates now use only the enrolment part.
- Cross-night scoring (night-one template, night-two probes; crossnight.py).
- Re-enrolment (reenroll.py) split update and probe epochs disjointly, but randomly rather than chronologically,
  and the probe set changed across update fractions. Fixed in the re-scoring.

## Other problems found in the same audit
1. **Bootstrap duplicate identities** (encoder_train.py, sleep_pipeline_v2.validate, band_ablation2.py). Subjects
   resampled with replacement were given unique keys, so copies of the same person were scored as impostors.
   The reported within-stage CI [2.09, 5.39] did not contain its own 0.21% point estimate.
2. **Cross-night "95% CI"** was a percentile range over 80% subject subsamples without replacement, not a
   bootstrap CI. Replaced by a cluster bootstrap with duplicate handling.
3. **Wake definition.** The cassette recordings include long daytime wake periods; the paper described W as quiet
   wakefulness. Wake is now trimmed to ±30 min around sleep (primary), with the untrimmed analysis as sensitivity.
4. **Encoder description.** The network is a plain four-block 1-D CNN, not an EEGNet-family model.
5. **Mixed aggregations.** Pooled-cell (5.24%) and per-subject (6.06%) values were reported without labels. Both
   are now labelled.

## Resolution
`code/rerun_leakfree.py`: chronological split with a ≥ 10-epoch guard, shared templates and probes, equal-length
multi-stage control, chronological fixed-probe re-enrolment, cluster bootstrap, Holm correction. It also runs the
old protocol side by side: 0.22% / 9.61% (untrimmed wake), close to the submitted 0.21% / 9.58%.
Corrected values: within 4.85%, cross-stage 12.10%, cross-night 19.36%, both 24.76% (wake trimmed).
The June scripts and results are kept unchanged in the repository root, `results/` and `paper_assets/`, marked SUPERSEDED.
