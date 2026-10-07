# Sleep-Stage and Night Dependence of EEG Biometric Verification on Sleep-EDF

Code and results for:

> **Sleep-Stage and Night Dependence of EEG Biometric Verification on Sleep-EDF**
> L. Kanimozhi and S. Shridevi, Vellore Institute of Technology, Chennai.
> *IEEE Signal Processing in Medicine and Biology Symposium (SPMB) 2026*, poster abstract.
> (Submitted paper title: "EEG Biometric Identity Is Sleep-Stage and Night Dependent: Per-Stage and Cross-Night Verification on Sleep-EDF".)

---

## Correction (October 2026) — read this first

A reviewer of the submitted paper inspected this repository (commit `c5370b6`) and found that, **for same-stage
trials, probe epochs were drawn from the same pool that was averaged into the enrolment template.** We confirmed
this. It inflated every within-stage number and the multi-stage enrolment result. The earlier README statement
"templates and probes never share epochs" was wrong.

All results were recomputed under a corrected protocol in [`revision_2026-10/`](revision_2026-10/README.md).
No model was retrained: the June encoder and its saved embeddings were re-scored, and night-two recordings were
embedded with the same frozen encoder.

**Corrected protocol (applied to every condition):**
- Per subject and stage, epochs are kept in chronological order: the first part forms the template, the last part
  supplies probes, and the two are separated by at least 10 epochs of the same stage (at least 5 min).
- The same templates and probe segments are used for within-stage, cross-stage and cross-night trials (paired comparisons).
- A probe is the mean of 10 consecutive same-stage epochs; impostors are zero-effort.
- Wake is restricted to 30 min around sleep (primary); the untrimmed analysis is reported as a sensitivity check.
- 95% CIs: subject-level cluster bootstrap (1000 resamples, with replacement; copies of one subject are never
  scored as impostors). Multiple comparisons: Holm.

**Corrected results** (29 evaluation subjects, 28 with a second night; pooled-cell EER, wake trimmed):

| Condition | Submitted (invalid) | Corrected EER, % [95% CI] |
|---|---|---|
| Within stage, within night | 0.21 | 4.85 [2.74, 7.76] |
| Across stages | 9.58 | 12.10 [8.27, 15.42] |
| Same stage, across nights | 17.28 | 19.36 [13.78, 24.69] |
| Across stages and nights | 23.87 | 24.76 [19.49, 30.95] |
| Random identity baseline | ~49.6 | 51.6 |

- Cross-stage minus within-stage: +7.25 points [4.82, 9.39]; per-subject Wilcoxon p = 6.1e-8, d_z = 1.23.
- Cross-night vs within-night on the same 28 subjects: 19.36% vs 4.91% (pooled-cell EER); per-subject Wilcoxon p = 7.1e-4.
- Running the old (leaky) protocol inside the new code gives 0.22% / 9.61% (untrimmed wake, as in June), close to
  the submitted 0.21% / 9.58%, which isolates the size of the error.
- Withdrawn claims: the "46-fold" increase, "near-perfect" within-stage identity, wake as the most stable stage
  across nights, the large multi-stage enrolment gain (9.6% → 4.4%), and the conclusion that stage-blind systems
  "silently fail" or that stage-aware calibration is required.

The original scripts and results are **kept unchanged** as the record of the submitted paper. Affected scripts
carry a `SUPERSEDED` comment banner at the top; `results/` and `paper_assets/` contain a `SUPERSEDED.md` note.
The exact state inspected by the reviewers is commit `c5370b6`. The corrected protocol, code, and results are in `revision_2026-10/` on the current main branch.

---

## Repository structure

```
revision_2026-10/                 CORRECTED protocol, code and results (use these)
  README.md                       protocol, how to run, file list
  AUDIT.md                        what was wrong, where, and what changed
  code/rerun_leakfree.py          embed / main / band_pkl / numbers steps
  code/make_numbers_extra.py      extra LaTeX macros for the abstract
  code/make_abstract_fig.py       Figure 1 of the abstract
  colab/leakfree_rescoring_colab.ipynb   CPU-only Colab notebook
  results/                        JSON, CSV matrices, logs, DET scores, numbers*.tex, figure

Original (June 2026, submitted paper; SUPERSEDED where marked)
  encoder_train.py                1D-CNN ArcFace encoder training (training itself unaffected)
  sleep_pipeline.py               early pipeline
  sleep_pipeline_v2.py            per-stage / cross-stage protocol (leaky)
  sleep_pipeline_v3.py            time-to-decision (leaky)
  sleep_pipeline_v4.py            multi-stage enrolment, threshold transfer, DET (leaky)
  crossnight.py                   night-2 preprocessing and cross-night scoring
  reenroll.py                     re-enrolment (random split)
  band_ablation2.py               band-stop ablation (leaky baseline)
  stats_tests.py                  paired tests (leaky per-subject EER)
  export_enc_matrix.py            old Fig. 2 matrix
  make_paper_assets.py, demographics.py, medication.py
  results/, paper_assets/         June outputs
```

---

## Data

- **Database:** Sleep-EDF Expanded, Sleep Cassette subset (PhysioNet): https://physionet.org/content/sleep-edfx/
- **Derivations:** Fpz–Cz and Pz–Oz EEG, 100 Hz; 30-s epochs with expert hypnograms (W, N1, N2, N3 [R&K 3–4 merged], REM).
- **Split:** 73 subjects; 44 training / 29 evaluation, subject-disjoint (sorted IDs, shuffle seed 42, train fraction 0.6).
- Raw EDF files are not redistributed; download them from PhysioNet.
- The June encoder checkpoint (`encoder.pt`) and saved embeddings, which `revision_2026-10/` re-scores, are not
  included in this repository.

## Encoder

Four 1-D convolution blocks (kernel 7, stride 2, 32–128 filters, batch norm, ELU, dropout 0.3), global average
pooling, linear layer to a 128-D unit-norm embedding; 204,384 parameters. ArcFace loss (s = 30, m = 0.30), Adam
(lr 1e-3, weight decay 1e-4), batch 256, 40 epochs, final-epoch model; trained on training subjects only.

## Requirements

```
python >= 3.9
numpy, scipy, scikit-learn      (scoring)
torch, mne                      (embedding night-two recordings)
matplotlib                      (figures)
```

## Reproducing the corrected results

See [`revision_2026-10/README.md`](revision_2026-10/README.md). In short, with `ROOT` = the June project folder
(read-only) and `OUT` = a new folder:

```bash
python revision_2026-10/code/rerun_leakfree.py embed    --root ROOT --out OUT
python revision_2026-10/code/rerun_leakfree.py main     --root ROOT --out OUT --emb_n1 ROOT/features2_enc --emb_n2 OUT/emb_n2
python revision_2026-10/code/rerun_leakfree.py band_pkl --root ROOT --out OUT
python revision_2026-10/code/rerun_leakfree.py numbers  --out OUT
```

## Hardware

Encoder training (June 2026): NVIDIA A100 GPUs through the NVIDIA Academic Grant Program (awarded to S. Shridevi).
Re-scoring (October 2026): Google Colab, CPU only; 3.3 ms to embed one 30-s epoch.

## Citation

```bibtex
@inproceedings{kanimozhi2026sleepeeg,
  title     = {Sleep-Stage and Night Dependence of {EEG} Biometric Verification on {Sleep-EDF}},
  author    = {Kanimozhi, L. and Shridevi, S.},
  booktitle = {IEEE Signal Processing in Medicine and Biology Symposium (SPMB)},
  year      = {2026}
}
```

## Acknowledgements

We thank PhysioNet for the Sleep-EDF Expanded database and the SPMB 2026 reviewers, one of whom identified the
enrolment/probe error. This work used NVIDIA A100 GPUs provided through the NVIDIA Academic Grant Program
(awarded to S. Shridevi). AI tools (Anthropic Claude) were used to assist with code review; the authors verified
all code and results.

## License

Released for academic and research use. Please cite the paper above if you use this code or build on it.
