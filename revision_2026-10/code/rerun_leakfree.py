#!/usr/bin/env python3
"""
rerun_leakfree.py  --  leakage-free re-evaluation for the IEEE SPMB 2026 abstract (p018)
=====================================================================================
Written 2026-10-03 in response to SPMB Reviewer #1, who found (correctly) that in the
June-2026 code the same-stage probe epochs were drawn from the SAME pool that was averaged
into the enrolment template (sleep_pipeline_v2._crossstage, stats_tests.per_subject_eer,
export_enc_matrix, sleep_pipeline_v3.crossstage, sleep_pipeline_v4.scores).
That inflates every within-stage number and the multi-stage "all5" result.

THIS SCRIPT DOES NOT OVERWRITE ANY EARLIER RESULT. Everything goes to --out (new folder).

Protocol (applied identically to every condition)
-------------------------------------------------
* Subjects: same TRAIN/EVAL split as encoder_train.py (sorted ids, shuffle seed 42,
  train_frac 0.6 -> 44 TRAIN / 29 EVAL). Only EVAL subjects are scored.
* Per subject and stage, the stage's epochs are kept in CHRONOLOGICAL order and split:
      enrolment = first part, probe = last part, separated by GUARD same-stage epochs
  (a gap of G same-stage epochs guarantees >= G*30 s wall-clock separation), so no probe
  epoch is in its template and no probe is temporally adjacent to an enrolment epoch.
* The SAME enrolment template (stage A, night 1) is used for within-stage, cross-stage and
  cross-night cells, and the SAME probe set (second part of stage B) is used for night 1
  and night 2 -> within / cross-stage / cross-night are directly paired comparisons.
* Probe fusion: N CONSECUTIVE (chronological, non-overlapping) same-stage probe epochs are
  averaged -> "N accumulated same-stage epochs" (= N*30 s of that stage, NOT wall-clock).
* Impostors: zero-effort, a subject's probes vs every other EVAL subject's template.
* Aggregation (both reported, always labelled):
    pooled-cell EER : EER over all trials in one (enrol A, probe B) cell; "within" = mean of
                      the 5 diagonal cells, "cross" = mean of the 20 off-diagonal cells.
    per-subject EER : EER of one subject's trials pooled over the relevant cells; used for
                      paired Wilcoxon tests.
* CIs: subject-level cluster bootstrap WITH replacement (B draws). A subject drawn k times
  contributes its genuine trials with weight k; impostor pairs between copies of the same
  person are EXCLUDED (the June code counted them as impostors, which biased CIs upward).
  Paired differences use the same draws for both conditions.
* Multiple comparisons: Holm correction.

Sub-commands (run in this order from the June project root, which holds encoder.pt,
features2_enc/, epochs2/, crossnight.py and emb_ablation.pkl; --root is only READ)
-------------------------------------------------------------------------------
  python rerun_leakfree.py embed    --root ROOT --out OUT   # inference only; night-2 EVAL embeddings (CPU is enough)
  python rerun_leakfree.py main     --root ROOT --out OUT --emb_n1 ROOT/features2_enc --emb_n2 OUT/emb_n2
  python rerun_leakfree.py band_pkl --root ROOT --out OUT   # re-scores June band-stop embeddings (emb_ablation.pkl)
  python rerun_leakfree.py numbers  --out OUT               # writes numbers.tex for the abstract
  (`band` re-embeds band-stopped epochs from scratch instead of using the pkl; not used for the reported run.)
The reported run (2026-10-04) used Google Colab, CPU only. See revision_2026-10/README.md.
Each step logs to <out>/<step>.log (tee) and writes JSON/CSV; nothing is overwritten.

Requires: numpy, scipy, scikit-learn (main); torch + mne (embed, band).
"""
import argparse, csv, datetime, glob, json, os, sys, time
import numpy as np

STAGES = ["W", "N1", "N2", "N3", "REM"]
BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
         "sigma": (12.0, 16.0), "beta": (16.0, 30.0)}


# ----------------------------------------------------------------------------- utils
class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
    def __call__(self, m):
        line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {m}"
        print(line, flush=True); self.f.write(line + "\n"); self.f.flush()


def d(*p): return os.path.join(*p)
def L2(X): return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-12)


def jdump(obj, path):
    if os.path.exists(path):  # never overwrite: keep the old one as evidence
        os.replace(path, path + f".prev_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    json.dump(obj, open(path, "w"), indent=2, default=float)


def split_subjects(root, train_frac=0.6, seed=42):
    src = "features2" if glob.glob(d(root, "features2", "*.npz")) else "epochs2"
    subs = sorted(os.path.basename(f)[:6] for f in glob.glob(d(root, src, "*.npz")))
    alt = "epochs2" if src == "features2" else "features2"
    subs_alt = sorted(os.path.basename(f)[:6] for f in glob.glob(d(root, alt, "*.npz")))
    if subs_alt and subs_alt != subs:
        raise SystemExit(f"Subject lists differ between features2/ and epochs2/ -> split would not match encoder training")
    rng = np.random.default_rng(seed); rng.shuffle(subs)
    n_tr = int(round(len(subs) * train_frac))
    tr, ev = subs[:n_tr], subs[n_tr:]
    assert set(tr).isdisjoint(ev)
    return tr, ev


def wake_trim_mask(y, trim_min):
    """Keep all sleep epochs and wake within trim_min minutes (2 epochs/min) of the first/last
    sleep epoch, plus all intra-sleep wake. Positions are in the amplitude-cleaned sequence,
    so the trim is approximate where artefact epochs were rejected (stated in the paper)."""
    if trim_min is None or trim_min < 0: return np.ones(len(y), bool)
    sleep_idx = np.where(y != "W")[0]
    if len(sleep_idx) == 0: return np.zeros(len(y), bool)
    k = int(round(trim_min * 2))
    lo, hi = max(0, sleep_idx[0] - k), min(len(y) - 1, sleep_idx[-1] + k)
    m = np.zeros(len(y), bool); m[lo:hi + 1] = True
    return m


def load_npz_dir(folder, subs, trim_min=None):
    out = {}
    for s in subs:
        fp = d(folder, f"{s}.npz")
        if not os.path.exists(fp): continue
        z = np.load(fp, allow_pickle=True); X, y = z["X"], np.asarray(z["y"]).astype(str)
        m = wake_trim_mask(y, trim_min)
        out[s] = (X[m], y[m])
    return out


def pools_of(cache):
    """subject -> stage -> chronologically ordered array (order of the cleaned recording)."""
    return {s: {st: X[y == st] for st in STAGES} for s, (X, y) in cache.items()}


def chrono_split(arr, guard):
    """enrol = first part, probe = last part, >= guard same-stage epochs between them."""
    n = len(arr)
    if n == 0: return arr[:0], arr[:0]
    n_e = max(0, (n - guard) // 2)
    return arr[:n_e], arr[n_e + guard:]


def fuse_consecutive(P, fuse):
    nt = len(P) // fuse
    if nt == 0: return None
    return P[:nt * fuse].reshape(nt, fuse, -1).mean(1)


def fuse_random(P, fuse, rng):  # ONLY used to reproduce the June (leaky) protocol
    nt = max(1, len(P) // fuse)
    idx = rng.permutation(len(P))[:nt * fuse].reshape(nt, fuse)
    return P[idx].mean(1)


# --------------------------------------------------------------------- scoring core
class Cell:
    """All trials of one (template set, probe set) cell, stored flat for fast weighted EER."""
    def __init__(self, templates, probes, U):
        # templates: s -> vector; probes: s -> (n, D) fused probe matrix; U: subject order
        self.U = U; ui = {s: i for i, s in enumerate(U)}
        tid = [s for s in U if s in templates]
        pid = [s for s in U if s in probes and probes[s] is not None and len(probes[s])]
        self.n_t, self.n_p = len(tid), len(pid)
        if len(tid) < 3 or not pid:
            self.empty = True; return
        self.empty = False
        T = L2(np.stack([templates[s] for s in tid]))
        sc, po, to = [], [], []
        for s in pid:
            S = L2(probes[s]) @ T.T                         # (n_probe, n_templ)
            sc.append(S.ravel())
            po.append(np.full(S.size, ui[s]))
            to.append(np.tile([ui[t] for t in tid], S.shape[0]))
        self.s = np.concatenate(sc); self.po = np.concatenate(po); self.to = np.concatenate(to)
        self.gen = self.po == self.to
        has_own = np.isin(self.po, [ui[t] for t in tid])
        self.s, self.po, self.to, self.gen = (self.s[has_own], self.po[has_own],
                                              self.to[has_own], self.gen[has_own])
        o = np.argsort(-self.s, kind="stable")
        self.s, self.po, self.to, self.gen = self.s[o], self.po[o], self.to[o], self.gen[o]

    def weights(self, c=None):
        if c is None:
            return self.gen.astype(float), (~self.gen).astype(float)
        wg = np.where(self.gen, c[self.po], 0.0)
        wi = np.where(self.gen, 0.0, c[self.po] * c[self.to])
        return wg, wi

    def eer(self, c=None, subj_mask=None):
        if self.empty: return np.nan
        wg, wi = self.weights(c)
        if subj_mask is not None:  # restrict to probes owned by these subjects
            keep = subj_mask[self.po]; wg = wg * keep; wi = wi * keep
        return weighted_eer(wg, wi)

    def threshold_at_eer(self, subj_mask=None):
        wg, wi = self.weights()
        if subj_mask is not None:
            keep = subj_mask[self.po] & subj_mask[self.to]; wg = wg * keep; wi = wi * keep
        P, N = wg.sum(), wi.sum()
        if P == 0 or N == 0: return np.nan
        far = np.cumsum(wi) / N; frr = 1 - np.cumsum(wg) / P
        i = int(np.argmin(np.abs(far - frr))); return float(self.s[i])

    def far_frr_at(self, tau, subj_mask=None):
        wg, wi = self.weights()
        if subj_mask is not None:
            keep = subj_mask[self.po] & subj_mask[self.to]; wg = wg * keep; wi = wi * keep
        acc = self.s >= tau
        far = (wi * acc).sum() / max(wi.sum(), 1e-12) * 100
        frr = (wg * ~acc).sum() / max(wg.sum(), 1e-12) * 100
        return far, frr


def weighted_eer(wg, wi):
    """Scores are pre-sorted descending; threshold sweeps down. Same rule as the June code
    (mean of FAR and FRR at the point where they are closest), now with trial weights."""
    P, N = wg.sum(), wi.sum()
    if P <= 0 or N <= 0: return np.nan
    far = np.cumsum(wi) / N; frr = 1 - np.cumsum(wg) / P
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2 * 100)


def plain_eer(gen, imp):
    s = np.concatenate([gen, imp]); o = np.argsort(-s, kind="stable")
    l = np.concatenate([np.ones(len(gen)), np.zeros(len(imp))])[o]
    return weighted_eer(l, 1 - l)


def per_subject_eer(cells, U):
    """subject -> EER over that subject's own trials pooled across the given cells."""
    out = {}
    for i, s in enumerate(U):
        g, im = [], []
        for c in cells:
            if c.empty: continue
            m = c.po == i
            g.append(c.s[m & c.gen]); im.append(c.s[m & ~c.gen])
        g = np.concatenate(g) if g else np.array([]); im = np.concatenate(im) if im else np.array([])
        if len(g) and len(im): out[s] = plain_eer(g, im)
    return out


def matrix_eer(M, c=None):
    E = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            if M[i][j] is not None: E[i, j] = M[i][j].eer(c)
    return E


def diag_off(E):
    return float(np.nanmean(np.diag(E))), float(np.nanmean(E[~np.eye(5, dtype=bool)]))


def bootstrap(fn, U, B, seed):
    """fn(counts) -> array of statistics. Subject-level cluster bootstrap with replacement."""
    rng = np.random.default_rng(seed); n = len(U); out = []
    for _ in range(B):
        c = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        out.append(fn(c))
    return np.array(out, dtype=float)


def ci(a):
    a = np.asarray(a, float); a = a[~np.isnan(a)]
    if len(a) == 0: return [np.nan, np.nan]
    return [round(float(np.percentile(a, 2.5)), 2), round(float(np.percentile(a, 97.5)), 2)]


def holm(pvals):
    keys = list(pvals); p = np.array([pvals[k] for k in keys], float)
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return {k: float(adj[i]) for i, k in enumerate(keys)}


def wilcoxon_paired(a, b, alternative="greater"):
    """a, b: dict subject->value. Tests a > b (default). Returns n, medians, p, mean diff."""
    from scipy.stats import wilcoxon
    ks = sorted(set(a) & set(b)); x = np.array([a[k] for k in ks]); y = np.array([b[k] for k in ks])
    if len(ks) < 5: return {"n": len(ks), "p": np.nan}
    diff = x - y
    try:
        p = float(wilcoxon(x, y, alternative=alternative, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
    except ValueError:
        p = np.nan
    sd = diff.std(ddof=1)
    return {"n": len(ks), "mean_a": round(float(x.mean()), 2), "mean_b": round(float(y.mean()), 2),
            "median_a": round(float(np.median(x)), 2), "median_b": round(float(np.median(y)), 2),
            "mean_diff": round(float(diff.mean()), 2),
            "cohens_dz": (round(float(diff.mean() / sd), 2) if sd > 0 else None), "p": p}


# --------------------------------------------------------------- protocol builders
def build_templates(pools, U, stage, guard, min_ep, K=None, leaky=False):
    T = {}
    for s in U:
        X = pools[s][stage]
        E = X if leaky else chrono_split(X, guard)[0]
        if K is not None: E = E[:K]
        if len(E) >= (min_ep if K is None else K): T[s] = E.mean(0)
    return T


def build_probes(pools, U, stage, guard, min_ep, fuse, leaky=False, rng=None, part="probe"):
    P = {}
    for s in U:
        X = pools[s][stage]
        if leaky:
            if len(X) >= min_ep: P[s] = fuse_random(X, fuse, rng)
            continue
        Pr = chrono_split(X, guard)[1] if part == "probe" else X
        if len(Pr) >= max(min_ep, fuse):
            f = fuse_consecutive(Pr, fuse)
            if f is not None: P[s] = f
    return P


def stage_matrix(pools_enrol, pools_probe, U, guard, min_ep, fuse, leaky=False, seed=42):
    rng = np.random.default_rng(seed)
    M = [[None] * 5 for _ in range(5)]
    for i, A in enumerate(STAGES):
        T = build_templates(pools_enrol, U, A, guard, min_ep, leaky=leaky)
        for j, Bst in enumerate(STAGES):
            P = build_probes(pools_probe, U, Bst, guard, min_ep, fuse, leaky=leaky, rng=rng)
            M[i][j] = Cell(T, P, U)
    return M


def counts_table(M):
    return [[(None if M[i][j] is None or M[i][j].empty else [M[i][j].n_t, M[i][j].n_p])
             for j in range(5)] for i in range(5)]


# ------------------------------------------------------------------------- MAIN step
def step_main(a):
    log = Tee(d(a.out, "main.log")); os.makedirs(a.out, exist_ok=True)
    log(f"MAIN start  root={a.root} out={a.out} guard={a.guard} min_ep={a.min_ep} fuse={a.fuse} "
        f"B={a.boot} wake_trim_min={a.wake_trim_min} K={a.K}")
    tr, ev = split_subjects(a.root, a.train_frac, a.seed)
    log(f"split: {len(tr)} TRAIN / {len(ev)} EVAL (subject-disjoint, asserted)")
    n1_dir = a.emb_n1 or d(a.out, "emb_n1"); n2_dir = a.emb_n2 or d(a.out, "emb_n2")
    if not glob.glob(d(n1_dir, "*.npz")):
        n1_dir = d(a.root, "features2_enc"); log(f"emb_n1 not found -> falling back to {n1_dir}")
    R = {"protocol": {"guard_epochs": a.guard, "min_epochs": a.min_ep, "fuse": a.fuse, "bootstrap_B": a.boot,
                      "wake_trim_min": a.wake_trim_min, "K_multistage": a.K, "seed": a.seed,
                      "n_train": len(tr), "n_eval": len(ev), "emb_n1": n1_dir, "emb_n2": n2_dir}}

    for trim_label, trim in [("trim", a.wake_trim_min), ("untrimmed", None)]:
        log(f"===== wake handling: {trim_label} =====")
        c1 = load_npz_dir(n1_dir, ev, trim); c2 = load_npz_dir(n2_dir, ev, trim)
        p1 = pools_of(c1); p2 = pools_of(c2); U = sorted(p1)
        R[trim_label] = res = {"n_eval_night1": len(p1), "n_eval_night2": len(p2)}
        res["epochs_per_stage_night1"] = {st: int(sum(len(p1[s][st]) for s in p1)) for st in STAGES}
        log(f"night-1 EVAL subjects {len(p1)}, night-2 {len(p2)}; epochs/stage {res['epochs_per_stage_night1']}")

        # ---- (1) within / cross-stage, clean vs leaky (encoder, fuse N)
        M = stage_matrix(p1, p1, U, a.guard, a.min_ep, a.fuse)
        E = matrix_eer(M); w, x = diag_off(E)
        ML = stage_matrix(p1, p1, U, a.guard, a.min_ep, a.fuse, leaky=True, seed=a.seed)
        EL = matrix_eer(ML); wl, xl = diag_off(EL)
        res["night1_matrix_clean"] = np.round(E, 2).tolist(); res["night1_matrix_leaky_June"] = np.round(EL, 2).tolist()
        res["night1_cell_counts_[n_templates,n_probe_subjects]"] = counts_table(M)
        log(f"[clean] within {w:.2f}  cross {x:.2f}   |  [June leaky protocol] within {wl:.2f} cross {xl:.2f}")

        def f_wc(c):
            Eb = matrix_eer(M, c); wb, xb = diag_off(Eb); return [wb, xb, xb - wb]
        bt = bootstrap(f_wc, U, a.boot, a.seed)
        res["within_stage"] = {"pooled_cell_EER": round(w, 2), "CI95": ci(bt[:, 0])}
        res["cross_stage"] = {"pooled_cell_EER": round(x, 2), "CI95": ci(bt[:, 1])}
        res["cross_minus_within"] = {"diff": round(x - w, 2), "CI95": ci(bt[:, 2])}
        res["leaky_vs_clean"] = {"within_leaky": round(wl, 2), "within_clean": round(w, 2),
                                 "cross_leaky": round(xl, 2), "cross_clean": round(x, 2)}
        diag_cells = [M[i][i] for i in range(5)]; off_cells = [M[i][j] for i in range(5) for j in range(5) if i != j]
        ps_w = per_subject_eer(diag_cells, U); ps_x = per_subject_eer(off_cells, U)
        res["per_subject_within_vs_cross"] = wilcoxon_paired(ps_x, ps_w, "greater")
        res["per_subject_within_vs_cross"]["aggregation"] = "per-subject EER, mean over subjects"
        log(f"per-subject: {res['per_subject_within_vs_cross']}")

        # per-stage within (diag) with CIs + Friedman across stages on per-subject EER
        res["within_by_stage"] = {}
        bdiag = bootstrap(lambda c: [M[i][i].eer(c) for i in range(5)], U, a.boot, a.seed + 1)
        for i, st in enumerate(STAGES):
            res["within_by_stage"][st] = {"EER": round(float(E[i, i]), 2), "CI95": ci(bdiag[:, i]),
                                          "n_subjects": M[i][i].n_p}
        ps_stage = {st: per_subject_eer([M[i][i]], U) for i, st in enumerate(STAGES)}
        res["within_stage_differences"] = stage_tests(ps_stage)

        # ---- (2) scorer ablation (cosine 24-D, LDA, encoder) under the clean protocol
        res["scorer_ablation"] = scorer_ablation(a, tr, ev, trim, log)
        res["scorer_ablation"]["encoder"] = {"within": round(w, 2), "cross": round(x, 2)}

        # ---- (3) probe-fusion depth / time-to-decision (accumulated same-stage epochs)
        res["fusion_depth"] = {}
        for N in a.fuse_list:
            MN = stage_matrix(p1, p1, U, a.guard, max(a.min_ep, N), N)
            wn, xn = diag_off(matrix_eer(MN))
            res["fusion_depth"][str(N)] = {"accumulated_minutes_of_stage": N * 0.5,
                                           "within": round(wn, 2), "cross": round(xn, 2)}
        log(f"fusion depth: {res['fusion_depth']}")

        # ---- (4) operating point + threshold transfer (2-fold subject-disjoint calibration)
        res["operating_point"] = operating_point(M, U, a.seed)
        log(f"operating point: {res['operating_point']}")

        # ---- (5) cross-night (same night-1 templates, night-2 probe half)
        both = sorted(set(p1) & set(p2))
        if len(both) >= 5:
            Mn = stage_matrix({s: p1[s] for s in both}, {s: p2[s] for s in both}, both,
                              a.guard, a.min_ep, a.fuse)
            Mw = stage_matrix({s: p1[s] for s in both}, {s: p1[s] for s in both}, both,
                              a.guard, a.min_ep, a.fuse)   # within-night reference on SAME subjects
            En = matrix_eer(Mn); Ew = matrix_eer(Mw)
            sn, xn = diag_off(En); sw, xw = diag_off(Ew)

            def f_cn(c):
                a1, b1 = diag_off(matrix_eer(Mn, c)); a0, b0 = diag_off(matrix_eer(Mw, c))
                return [a1, b1, a1 - a0, b1 - a0, b0]
            bn = bootstrap(f_cn, both, a.boot, a.seed + 2)
            res["cross_night"] = {
                "n_subjects": len(both),
                "within_night_same_subjects": {"within": round(sw, 2), "cross_stage": round(xw, 2), "cross_stage_CI95": ci(bn[:, 4])},
                "cross_night_same_stage": {"EER": round(sn, 2), "CI95": ci(bn[:, 0])},
                "cross_night_cross_stage": {"EER": round(xn, 2), "CI95": ci(bn[:, 1])},
                "diff_cross_night_minus_within": {"diff": round(sn - sw, 2), "CI95": ci(bn[:, 2])},
                "matrix": np.round(En, 2).tolist(),
                "cell_counts": counts_table(Mn)}
            bstage = bootstrap(lambda c: [Mn[i][i].eer(c) for i in range(5)], both, a.boot, a.seed + 3)
            res["cross_night"]["same_stage_by_stage"] = {
                st: {"EER": round(float(En[i, i]), 2), "CI95": ci(bstage[:, i]), "n_subjects": Mn[i][i].n_p}
                for i, st in enumerate(STAGES)}
            ps_n = {st: per_subject_eer([Mn[i][i]], both) for i, st in enumerate(STAGES)}
            res["cross_night"]["stage_differences"] = stage_tests(ps_n)
            ps_cn = per_subject_eer([Mn[i][i] for i in range(5)], both)
            ps_wn = per_subject_eer([Mw[i][i] for i in range(5)], both)
            res["cross_night"]["per_subject_cross_night_vs_within"] = wilcoxon_paired(ps_cn, ps_wn, "greater")
            # random-identity sanity baseline: shuffle template identities
            res["cross_night"]["random_identity_EER"] = random_baseline(Mn, a.seed)
            log(f"cross-night: {json.dumps({k: res['cross_night'][k] for k in ['cross_night_same_stage','cross_night_cross_stage','diff_cross_night_minus_within']})}")

            # ---- (6) template update with night-2 data (offline, chronological, fixed probes)
            res["reenrolment"] = reenrol(p1, p2, both, a, log)
            # scores for DET curves
            save_scores(a.out, trim_label, M, Mn)
        else:
            log("night-2 embeddings missing -> cross-night / re-enrolment skipped (run `embed` first)")

        res["random_identity_EER_night1"] = random_baseline(M, a.seed)

        # ---- (7) multi-stage enrolment with EQUAL enrolment duration + identical probes
        res["multistage"] = multistage(p1, U, a, log)

    jdump(R, d(a.out, "results_leakfree.json"))
    write_csvs(a.out, R)
    log("MAIN done -> results_leakfree.json (+ CSVs). Next: `band`, then `numbers`.")


def stage_tests(ps_stage):
    """Friedman across stages (subjects with all 5) + pairwise Wilcoxon two-sided, Holm."""
    from scipy.stats import friedmanchisquare, wilcoxon
    out = {}
    common = sorted(set.intersection(*[set(v) for v in ps_stage.values()])) if all(ps_stage.values()) else []
    if len(common) >= 5:
        arr = [[ps_stage[st][s] for s in common] for st in STAGES]
        try: out["friedman"] = {"n": len(common), "p": float(friedmanchisquare(*arr).pvalue)}
        except ValueError: out["friedman"] = {"n": len(common), "p": None}
    pv, info = {}, {}
    for i in range(5):
        for j in range(i + 1, 5):
            A, Bs = STAGES[i], STAGES[j]
            ks = sorted(set(ps_stage[A]) & set(ps_stage[Bs]))
            if len(ks) < 5: continue
            x = np.array([ps_stage[A][k] for k in ks]); y = np.array([ps_stage[Bs][k] for k in ks])
            try: p = float(wilcoxon(x, y).pvalue) if np.any(x != y) else 1.0
            except ValueError: p = 1.0
            pv[f"{A}-{Bs}"] = p
            info[f"{A}-{Bs}"] = {"n": len(ks), "mean_diff": round(float((x - y).mean()), 2), "p_raw": p}
    for k, v in holm(pv).items(): info[k]["p_holm"] = v
    out["pairwise_wilcoxon_two_sided"] = info
    return out


def random_baseline(M, seed):
    """Shuffle template-identity labels in every cell; expected EER ~ 50%."""
    rng = np.random.default_rng(seed); vals = []
    for row in M:
        for c in row:
            if c is None or c.empty: continue
            perm = rng.permutation(len(c.U))
            gen = c.po == perm[c.to]
            sc = c.s; wg = gen.astype(float); wi = (~gen).astype(float)
            vals.append(weighted_eer(wg, wi))
    return round(float(np.nanmean(vals)), 2)


def operating_point(M, U, seed):
    """FRR at FAR=1% (pooled within-stage) and threshold transfer with 2-fold subject-disjoint
    calibration: tau calibrated (EER point) on stage A within-stage trials of fold-1 subjects,
    applied to stage B within-stage trials of fold-2 subjects (and vice versa)."""
    rng = np.random.default_rng(seed); n = len(U); perm = rng.permutation(n)
    folds = [np.zeros(n, bool), np.zeros(n, bool)]
    folds[0][perm[: n // 2]] = True; folds[1][perm[n // 2:]] = True
    # FRR @ FAR = 1 %
    g = np.concatenate([M[i][i].s[M[i][i].gen] for i in range(5) if not M[i][i].empty])
    im = np.concatenate([M[i][i].s[~M[i][i].gen] for i in range(5) if not M[i][i].empty])
    tau1 = np.quantile(im, 0.99); frr1 = float((g < tau1).mean() * 100)
    hter = np.full((5, 5), np.nan)
    for i in range(5):
        for j in range(5):
            if M[i][i].empty or M[j][j].empty: continue
            hs = []
            for cal, tst in [(0, 1), (1, 0)]:
                tau = M[i][i].threshold_at_eer(folds[cal])
                far, frr = M[j][j].far_frr_at(tau, folds[tst]); hs.append((far + frr) / 2)
            hter[i, j] = np.mean(hs)
    return {"FRR_at_FAR1pct_within_pooled": round(frr1, 2),
            "HTER_matched_stage_threshold": round(float(np.nanmean(np.diag(hter))), 2),
            "HTER_mismatched_stage_threshold": round(float(np.nanmean(hter[~np.eye(5, dtype=bool)])), 2),
            "HTER_matrix_cal_row_test_col": np.round(hter, 2).tolist(),
            "note": "2-fold subject-disjoint calibration/test; thresholds at the calibration-fold EER point"}


def reenrol(p1, p2, both, a, log):
    """Offline template-update experiment (NOT a validated refresh policy).
    Template = night-1 enrolment part (+ the FIRST k epochs of night-2 stage S, chronological);
    probes = FIXED: the night-2 probe part of stage S (chronologically after every update epoch,
    separated by >= guard epochs), identical for every k. Also reports 'replace' (night-2 update
    epochs only) to separate refresh from augmentation."""
    out = {"definition": "night-2 update epochs are chronologically EARLIER than every night-2 probe; probe set fixed across update sizes"}
    fracs = [0.0, 0.25, 0.5, 1.0]
    for mode in ["augment", "replace"]:
        curve = {}
        for f in fracs:
            if mode == "replace" and f == 0: continue
            cells = []; minutes = []
            for S in STAGES:
                T, P = {}, {}
                for s in both:
                    e1 = chrono_split(p1[s][S], a.guard)[0]
                    upd_pool, prb = chrono_split(p2[s][S], a.guard)
                    if len(e1) < a.min_ep or len(prb) < max(a.min_ep, a.fuse): continue
                    k = int(round(f * len(upd_pool))); upd = upd_pool[:k]
                    if mode == "replace":
                        if k < a.min_ep: continue
                        T[s] = upd.mean(0)
                    else:
                        T[s] = np.concatenate([e1, upd]).mean(0) if k else e1.mean(0)
                    P[s] = fuse_consecutive(prb, a.fuse); minutes.append(k * 0.5)
                cells.append(Cell(T, P, both))
            vals = [c.eer() for c in cells]
            bt = bootstrap(lambda c: [np.nanmean([cc.eer(c) for cc in cells])], both, a.boot, a.seed + 7)
            curve[str(f)] = {"same_stage_EER": round(float(np.nanmean(vals)), 2), "CI95": ci(bt[:, 0]),
                             "mean_minutes_night2_added_per_stage": round(float(np.mean(minutes)) if minutes else 0, 1)}
        out[mode] = curve
    log(f"re-enrolment: {json.dumps(out)}")
    return out


def multistage(p1, U, a, log):
    """Equal enrolment duration K epochs for every strategy; identical probe set (probe part of B).
    single_other : template from ONE other stage (K epochs), averaged over the 4 choices
    multi_other  : K/4 epochs from each of the 4 OTHER stages (probe stage excluded)
    multi_all5   : K/5 epochs from each of the 5 stages (enrolment parts only; probe part never used)
    single_match : K epochs from the probe stage's own enrolment part (equal-duration within-stage)
    Only subjects valid for ALL strategies at a given probe stage are scored (paired)."""
    K = a.K; out = {"K_epochs": K, "K_minutes": K * 0.5}
    enrol = {s: {st: chrono_split(p1[s][st], a.guard)[0] for st in STAGES} for s in U}
    per = {k: {} for k in ["single_match", "single_other", "multi_other", "multi_all5"]}
    ps = {k: {} for k in per}
    for Bst in STAGES:
        others = [st for st in STAGES if st != Bst]
        P = {}
        for s in U:
            pr = chrono_split(p1[s][Bst], a.guard)[1]
            if len(pr) >= max(a.min_ep, a.fuse): P[s] = fuse_consecutive(pr, a.fuse)
        k4, k5 = K // 4, K // 5
        valid = [s for s in U if s in P and len(enrol[s][Bst]) >= K
                 and all(len(enrol[s][o]) >= K for o in others)]
        if len(valid) < 5:
            log(f"multistage: probe {Bst}: only {len(valid)} subjects valid for all strategies -> skipped"); continue
        Pv = {s: P[s] for s in valid}
        T_match = {s: enrol[s][Bst][:K].mean(0) for s in valid}
        T_mo = {s: np.concatenate([enrol[s][o][:k4] for o in others]).mean(0) for s in valid}
        T_m5 = {s: np.concatenate([enrol[s][o][:k5] for o in STAGES]).mean(0) for s in valid}
        cm, cmo, cm5 = Cell(T_match, Pv, valid), Cell(T_mo, Pv, valid), Cell(T_m5, Pv, valid)
        cso = [Cell({s: enrol[s][o][:K].mean(0) for s in valid}, Pv, valid) for o in others]
        per["single_match"][Bst] = cm.eer(); per["multi_other"][Bst] = cmo.eer(); per["multi_all5"][Bst] = cm5.eer()
        per["single_other"][Bst] = float(np.nanmean([c.eer() for c in cso]))
        for name, cells in [("single_match", [cm]), ("multi_other", [cmo]), ("multi_all5", [cm5]), ("single_other", cso)]:
            for s, v in per_subject_eer(cells, valid).items():
                ps[name].setdefault(s, []).append(v)
        out.setdefault("n_subjects_by_probe_stage", {})[Bst] = len(valid)
    out["pooled_cell_EER_by_probe_stage"] = {k: {st: round(v, 2) for st, v in d_.items()} for k, d_ in per.items()}
    out["mean_over_probe_stages"] = {k: (round(float(np.nanmean(list(d_.values()))), 2) if d_ else None) for k, d_ in per.items()}
    psm = {k: {s: float(np.mean(v)) for s, v in d_.items()} for k, d_ in ps.items()}
    tests = {"multi_other_vs_single_other": wilcoxon_paired(psm["single_other"], psm["multi_other"], "greater"),
             "multi_all5_vs_single_other": wilcoxon_paired(psm["single_other"], psm["multi_all5"], "greater"),
             "multi_all5_vs_single_match": wilcoxon_paired(psm["multi_all5"], psm["single_match"], "greater")}
    for k, v in holm({k: (v["p"] if v.get("p") == v.get("p") else 1.0) for k, v in tests.items()}).items():
        tests[k]["p_holm"] = v
    out["paired_tests_per_subject"] = tests
    log(f"multistage (K={K}): {out['mean_over_probe_stages']}")
    return out


def scorer_ablation(a, tr, ev, trim, log):
    out = {}
    if not glob.glob(d(a.root, "features2", "*.npz")):
        log("features2/ not found -> scorer ablation skipped"); return out
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.preprocessing import StandardScaler
    ctr = load_npz_dir(d(a.root, "features2"), tr, None)   # LDA fit: TRAIN subjects only
    cev = load_npz_dir(d(a.root, "features2"), ev, trim)
    Xtr = np.concatenate([ctr[s][0] for s in sorted(ctr)]); ytr = np.concatenate([[s] * len(ctr[s][0]) for s in sorted(ctr)])
    sc = StandardScaler().fit(Xtr); lda = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), ytr)
    for name, proj in [("cosine_24D", lambda X: X), ("LDA_subject_disjoint", lambda X: lda.transform(sc.transform(X)))]:
        cc = {s: (proj(X), y) for s, (X, y) in cev.items()}
        p = pools_of(cc); U = sorted(p)
        for N in [1, a.fuse]:
            w, x = diag_off(matrix_eer(stage_matrix(p, p, U, a.guard, max(a.min_ep, N), N)))
            out[f"{name}_fuse{N}"] = {"within": round(w, 2), "cross": round(x, 2)}
    log(f"scorer ablation: {out}")
    return out


def save_scores(out, tag, M, Mn):
    def gather(cells):
        g = np.concatenate([c.s[c.gen] for c in cells if not c.empty]); i = np.concatenate([c.s[~c.gen] for c in cells if not c.empty])
        return g, i
    gw, iw = gather([M[i][i] for i in range(5)]); gx, ix = gather([M[i][j] for i in range(5) for j in range(5) if i != j])
    gn, inn = gather([Mn[i][i] for i in range(5)]); gnx, inx = gather([Mn[i][j] for i in range(5) for j in range(5) if i != j])
    np.savez_compressed(d(out, f"scores_for_DET_{tag}.npz"), within_gen=gw, within_imp=iw, cross_gen=gx, cross_imp=ix,
                        night_gen=gn, night_imp=inn, nightcross_gen=gnx, nightcross_imp=inx)


def write_csvs(out, R):
    for tag in ["trim", "untrimmed"]:
        if tag not in R: continue
        for key, fname in [("night1_matrix_clean", "matrix_night1_clean"), ("night1_matrix_leaky_June", "matrix_night1_LEAKY_June_protocol")]:
            write_matrix(d(out, f"{fname}_{tag}.csv"), R[tag][key])
        if "cross_night" in R[tag]:
            write_matrix(d(out, f"matrix_crossnight_clean_{tag}.csv"), R[tag]["cross_night"]["matrix"])


def write_matrix(path, M):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["enrol\\probe"] + STAGES)
        for i, st in enumerate(STAGES): w.writerow([st] + ["NA" if v is None or v != v else f"{v:.2f}" for v in M[i]])


# ----------------------------------------------------------------- torch helpers
def make_encoder(C, emb):
    import torch.nn as nn, torch.nn.functional as F
    class Enc(nn.Module):  # IDENTICAL to encoder_train.py / crossnight.py
        def __init__(self, C, emb):
            super().__init__()
            def blk(i, o, k=7, s=2): return nn.Sequential(nn.Conv1d(i, o, k, s, k // 2), nn.BatchNorm1d(o), nn.ELU(), nn.Dropout(0.3))
            self.net = nn.Sequential(blk(C, 32), blk(32, 64), blk(64, 128), blk(128, 128), nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(128, emb)
        def forward(self, x):
            z = self.net(x).squeeze(-1); return F.normalize(self.fc(z), dim=1)
    return Enc(C, emb)


def train_norm_stats(root, tr, out, log):
    """Per-channel mean/sd over TRAIN night-1 epochs2 (same definition as encoder_train.py),
    accumulated in float64 to avoid holding ~3 GB in memory. Cached to <out>/norm_stats.npz."""
    fp = d(out, "norm_stats.npz")
    if os.path.exists(fp):
        z = np.load(fp); return z["mu"], z["sd"]
    s1 = s2 = None; n = 0
    for s in tr:
        X = np.load(d(root, "epochs2", f"{s}.npz"), allow_pickle=True)["X"].astype(np.float64)
        a1 = X.sum(axis=(0, 2)); a2 = (X ** 2).sum(axis=(0, 2))
        s1 = a1 if s1 is None else s1 + a1; s2 = a2 if s2 is None else s2 + a2; n += X.shape[0] * X.shape[2]
    mu = s1 / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0)) + 1e-6
    mu = mu[None, :, None].astype(np.float32); sd = sd[None, :, None].astype(np.float32)
    np.savez(fp, mu=mu, sd=sd); log(f"TRAIN z-norm stats computed over {len(tr)} subjects -> {fp}")
    return mu, sd


def embed_array(enc, X, mu, sd, dev, bs=512):
    import torch
    Z = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            Z.append(enc(torch.tensor((X[i:i + bs] - mu) / sd).to(dev)).cpu().numpy())
    return np.concatenate(Z) if Z else np.zeros((0, 128), np.float32)


def step_embed(a):
    """Night-2 embeddings with the EXISTING encoder.pt (no training).
    1. TRAIN-subject z-norm stats from epochs2/ (same definition as encoder_train.py).
    2. Consistency check: re-embed 2 EVAL subjects' night-1 epochs and compare with the saved
       features2_enc/ -> proves encoder.pt + normalisation are the ones used in June.
    3. Download night 2 (PhysioNet, via MNE) for EVAL subjects into --n2_cache (NOT into --root),
       preprocess with crossnight.epochs2 (identical to June), embed -> <out>/emb_n2/.
    --root is only READ. Works on CPU (inference only)."""
    import torch
    log = Tee(d(a.out, "embed.log")); os.makedirs(a.out, exist_ok=True)
    tr, ev = split_subjects(a.root, a.train_frac, a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; log(f"EMBED start device={dev} (inference only, no training)")
    saved = sorted(os.path.basename(f)[:6] for f in glob.glob(d(a.root, "features2_enc", "*.npz")))
    if saved and saved != sorted(ev):
        raise SystemExit(f"EVAL split {sorted(ev)} != features2_enc subjects {saved}: split mismatch, stop.")
    log(f"EVAL split matches features2_enc/ ({len(saved)} subjects)")
    mu, sd = train_norm_stats(a.root, tr, a.out, log)
    x0 = np.load(d(a.root, "epochs2", f"{ev[0]}.npz"), allow_pickle=True)["X"]
    enc = make_encoder(x0.shape[1], a.emb).to(dev)
    enc.load_state_dict(torch.load(d(a.root, "encoder.pt"), map_location=dev)); enc.eval()
    n_par = sum(p.numel() for p in enc.parameters())
    # ---- consistency check against the June embeddings
    chk = {}
    for s in ev[:2]:
        X = np.load(d(a.root, "epochs2", f"{s}.npz"), allow_pickle=True)["X"].astype(np.float32)
        Z = embed_array(enc, X, mu, sd, dev); Zo = np.load(d(a.root, "features2_enc", f"{s}.npz"), allow_pickle=True)["X"]
        cos = float(np.min(np.sum(L2(Z) * L2(Zo), axis=1))) if Z.shape == Zo.shape else None
        chk[s] = {"shape_new": list(Z.shape), "shape_saved": list(Zo.shape), "min_cosine_new_vs_saved": cos}
    log(f"consistency check (expect shapes equal and min cosine > 0.999): {chk}")
    ok = all(v["min_cosine_new_vs_saved"] is not None and v["min_cosine_new_vs_saved"] > 0.999 for v in chk.values())
    if not ok:
        jdump({"consistency": chk}, d(a.out, "embed_consistency_FAILED.json"))
        raise SystemExit("Re-embedding does not reproduce features2_enc -> do not continue; send embed.log to Claude.")
    # ---- runtime (Reviewer 4)
    xb = np.repeat(x0[:1].astype(np.float32), 1000, axis=0)
    t0 = time.perf_counter(); embed_array(enc, xb, mu, sd, dev); dt = (time.perf_counter() - t0)
    import platform
    rt = {"encoder_parameters": int(n_par), "model_size_MB_fp32": round(n_par * 4 / 2 ** 20, 2),
          "ms_per_30s_epoch_batched": round(dt, 3), "device": dev,
          "hardware": (torch.cuda.get_device_name(0) if dev == "cuda" else platform.processor() or platform.machine())}
    jdump({"consistency": chk, "runtime": rt}, d(a.out, "embed_check_runtime.json")); jdump(rt, d(a.out, "runtime.json"))
    log(f"runtime: {rt}")
    # ---- night 2
    import importlib.util, mne, shutil
    sys.dont_write_bytecode = True                       # never write __pycache__ next to the (read-only) source
    cn_copy = d(a.out, "crossnight_june_copy.py"); shutil.copyfile(d(a.root, "crossnight.py"), cn_copy)
    sp = importlib.util.spec_from_file_location("cn", cn_copy)
    cn = importlib.util.module_from_spec(sp); sp.loader.exec_module(cn)   # IDENTICAL preprocessing to June
    cache_dir = a.n2_cache or d(a.out, "data_n2"); os.makedirs(cache_dir, exist_ok=True)
    sub_ids = sorted({int(s[3:5]) for s in ev})
    paths = mne.datasets.sleep_physionet.age.fetch_data(subjects=sub_ids, recording=[2], path=cache_dir, on_missing="warn")
    o2 = d(a.out, "emb_n2"); os.makedirs(o2, exist_ok=True); n_ok = 0
    for psg, hyp in paths:
        num = os.path.basename(psg)[3:5]; key = next((s for s in ev if s[3:5] == num), None)
        if key is None: continue
        X, y, sf = cn.epochs2(psg, hyp)
        if X is None: continue
        np.savez_compressed(d(o2, f"{key}.npz"), X=embed_array(enc, X, mu, sd, dev), y=y); n_ok += 1
        log(f"  night-2 {key}: {len(y)} epochs embedded")
    log(f"night-2 embeddings for {n_ok} EVAL subjects -> {o2} (June run had 28)")


def band_scoring(cache_by_band, a, log, source):
    """Clean-protocol re-scoring of band-stop embeddings: dict band -> {s: (Z, y)} incl. 'baseline'."""
    fuse = 1
    def mat(c):
        c = {s: (Z[wake_trim_mask(np.asarray(y).astype(str), a.wake_trim_min)], np.asarray(y).astype(str)[wake_trim_mask(np.asarray(y).astype(str), a.wake_trim_min)]) for s, (Z, y) in c.items()}
        p = pools_of(c); U = sorted(p); return stage_matrix(p, p, U, a.guard, a.min_ep, fuse), U
    M0, U = mat(cache_by_band["baseline"])
    w0 = diag_off(matrix_eer(M0))[0]; ps0 = per_subject_eer([M0[i][i] for i in range(5)], U)
    out = {"source": source, "fuse": fuse, "baseline_within_EER": round(w0, 2),
           "filter": "Butterworth order 4, zero-phase (sosfiltfilt), band-stop; applied to enrolment AND probe epochs",
           "encoder_retrained": False, "bands": {}}
    pv, ps_all = {}, {}
    for band in [b for b in cache_by_band if b != "baseline"]:
        Mb, Ub = mat(cache_by_band[band]); assert Ub == U
        bt = bootstrap(lambda c: [diag_off(matrix_eer(Mb, c))[0] - diag_off(matrix_eer(M0, c))[0]], U, a.boot, a.seed + 11)
        wb = diag_off(matrix_eer(Mb))[0]; psb = per_subject_eer([Mb[i][i] for i in range(5)], U)
        t = wilcoxon_paired(psb, ps0, "greater"); pv[band] = t["p"] if t["p"] == t["p"] else 1.0
        ps_all[band] = {s: psb[s] - ps0[s] for s in psb if s in ps0}
        out["bands"][band] = {"within_EER": round(wb, 2), "dEER": round(wb - w0, 2), "dEER_CI95": ci(bt[:, 0]),
                              "per_subject_wilcoxon_vs_baseline": t}
        log(f"band {band}: dEER {wb - w0:+.2f}  CI {ci(bt[:, 0])}  p={t['p']}")
    for band, p in holm(pv).items(): out["bands"][band]["p_holm"] = p
    out["between_bands"] = stage_tests_generic(ps_all)
    jdump(out, d(a.out, "band_ablation_leakfree.json")); log("BAND done -> band_ablation_leakfree.json")


def step_band_pkl(a):
    """Re-score the band-stop embeddings ALREADY computed in June (emb_ablation.pkl, made by
    band_ablation2.py with encoder.pt; delta band = 0.5-4 Hz there). No new embedding, no GPU."""
    import pickle
    log = Tee(d(a.out, "band.log")); os.makedirs(a.out, exist_ok=True)
    pk = a.pkl or d(a.root, "emb_ablation.pkl")
    emb = pickle.load(open(pk, "rb")); log(f"loaded {pk}: keys {list(emb)}; subjects {len(emb['baseline'])}")
    band_scoring(emb, a, log, f"re-scored from {os.path.basename(pk)} (June band-stop embeddings; delta = 0.5-4 Hz)")


def step_band(a):
    """Recompute band-stop embeddings from epochs2/ (only needed if emb_ablation.pkl is unavailable)."""
    import torch
    from scipy.signal import butter, sosfiltfilt
    log = Tee(d(a.out, "band.log")); os.makedirs(a.out, exist_ok=True)
    tr, ev = split_subjects(a.root, a.train_frac, a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mu, sd = train_norm_stats(a.root, tr, a.out, log)
    raw = {s: np.load(d(a.root, "epochs2", f"{s}.npz"), allow_pickle=True) for s in ev}
    sf = float(raw[ev[0]]["sf"]) if "sf" in raw[ev[0]] else 100.0
    enc = make_encoder(raw[ev[0]]["X"].shape[1], a.emb).to(dev)
    enc.load_state_dict(torch.load(d(a.root, "encoder.pt"), map_location=dev)); enc.eval()
    cache = {}
    for name in ["baseline"] + list(BANDS):
        c = {}
        for s in ev:
            X = raw[s]["X"].astype(np.float32)
            if name != "baseline":
                sos = butter(4, list(BANDS[name]), btype="bandstop", fs=sf, output="sos")
                X = sosfiltfilt(sos, X, axis=-1).astype(np.float32)
            c[s] = (embed_array(enc, X, mu, sd, dev), np.asarray(raw[s]["y"]).astype(str))
        cache[name] = c; log(f"embedded {name}")
    band_scoring(cache, a, log, "recomputed from epochs2/ with encoder.pt")


def stage_tests_generic(ps):
    from scipy.stats import friedmanchisquare
    keys = list(ps); common = sorted(set.intersection(*[set(v) for v in ps.values()]))
    if len(common) < 5: return {"n": len(common)}
    try: p = float(friedmanchisquare(*[[ps[k][s] for s in common] for k in keys]).pvalue)
    except ValueError: p = None
    return {"friedman_across_bands_on_per_subject_dEER": {"n": len(common), "p": p}}


# ---------------------------------------------------------------- numbers.tex
def step_numbers(a):
    R = json.load(open(d(a.out, "results_leakfree.json"))); r = R["trim" if a.use == "trim" else "untrimmed"]
    B = json.load(open(d(a.out, "band_ablation_leakfree.json"))) if os.path.exists(d(a.out, "band_ablation_leakfree.json")) else None
    def f(v): return "--" if v is None or v != v else f"{v:.2f}"
    def c(v): return f"[{f(v[0])}, {f(v[1])}]"
    def pfmt(p):
        if p is None or p != p: return "--"
        if p < 1e-3:
            m, e = f"{p:.1e}".split("e"); return f"{m}\\times 10^{{{int(e)}}}"
        return f"{p:.3f}"
    L = ["% AUTO-GENERATED by rerun_leakfree.py numbers -- do not edit by hand",
         f"% source: {d(a.out, 'results_leakfree.json')}  wake handling: {a.use}"]
    def nc(name, val): L.append(f"\\newcommand{{\\{name}}}{{{val}}}")
    nc("NEval", R["protocol"]["n_eval"]); nc("NTrain", R["protocol"]["n_train"]); nc("Guard", R["protocol"]["guard_epochs"])
    nc("GuardMin", f"{R['protocol']['guard_epochs'] * 0.5:g}"); nc("Fuse", R["protocol"]["fuse"]); nc("Boot", R["protocol"]["bootstrap_B"])
    nc("Within", f(r["within_stage"]["pooled_cell_EER"])); nc("WithinCI", c(r["within_stage"]["CI95"]))
    nc("Cross", f(r["cross_stage"]["pooled_cell_EER"])); nc("CrossCI", c(r["cross_stage"]["CI95"]))
    nc("GapDiff", f(r["cross_minus_within"]["diff"])); nc("GapCI", c(r["cross_minus_within"]["CI95"]))
    t = r["per_subject_within_vs_cross"]; nc("GapP", pfmt(t.get("p"))); nc("GapDz", f(t.get("cohens_dz")))
    nc("WithinLeaky", f(r["leaky_vs_clean"]["within_leaky"]))
    nc("RandNight", f(r["random_identity_EER_night1"]))
    sa = r.get("scorer_ablation", {})
    for k, nm in [("cosine_24D", "Cos"), ("LDA_subject_disjoint", "Lda")]:
        v = sa.get(f"{k}_fuse{R['protocol']['fuse']}", {}); nc(f"{nm}Within", f(v.get("within"))); nc(f"{nm}Cross", f(v.get("cross")))
    fd = r.get("fusion_depth", {})
    for N, nm in [("1", "FuseOne"), ("5", "FuseFive"), ("10", "FuseTen")]:
        nc(f"{nm}Within", f(fd.get(N, {}).get("within"))); nc(f"{nm}Cross", f(fd.get(N, {}).get("cross")))
    op = r["operating_point"]; nc("FRRone", f(op["FRR_at_FAR1pct_within_pooled"]))
    nc("HTERmatch", f(op["HTER_matched_stage_threshold"])); nc("HTERmismatch", f(op["HTER_mismatched_stage_threshold"]))
    cn = r.get("cross_night", {})
    if cn:
        nc("NBoth", cn["n_subjects"]); nc("NightSame", f(cn["cross_night_same_stage"]["EER"])); nc("NightSameCI", c(cn["cross_night_same_stage"]["CI95"]))
        nc("NightCross", f(cn["cross_night_cross_stage"]["EER"])); nc("NightCrossCI", c(cn["cross_night_cross_stage"]["CI95"]))
        nc("NightWithinRef", f(cn["within_night_same_subjects"]["within"]))
        nc("NightP", pfmt(cn["per_subject_cross_night_vs_within"].get("p"))); nc("RandCN", f(cn["random_identity_EER"]))
        for st in STAGES:
            v = cn["same_stage_by_stage"][st]; nm = {"W": "W", "N1": "NOne", "N2": "NTwo", "N3": "NThree", "REM": "REM"}[st]
            nc(f"Night{nm}", f(v["EER"])); nc(f"Night{nm}CI", c(v["CI95"]))
        fr = cn["stage_differences"].get("friedman", {}); nc("NightFriedmanP", pfmt(fr.get("p")))
    for st in STAGES:
        nm = {"W": "W", "N1": "NOne", "N2": "NTwo", "N3": "NThree", "REM": "REM"}[st]
        nc(f"Within{nm}", f(r["within_by_stage"][st]["EER"]))
    re_ = r.get("reenrolment", {})
    if re_:
        nc("ReZero", f(re_["augment"]["0.0"]["same_stage_EER"])); nc("ReHalf", f(re_["augment"]["0.5"]["same_stage_EER"]))
        nc("ReFull", f(re_["augment"]["1.0"]["same_stage_EER"])); nc("ReFullCI", c(re_["augment"]["1.0"]["CI95"]))
        nc("ReReplace", f(re_["replace"]["1.0"]["same_stage_EER"]))
    ms = r["multistage"]; m = ms["mean_over_probe_stages"]; nc("MsK", ms["K_epochs"]); nc("MsMin", f"{ms['K_minutes']:g}")
    nc("MsSingleOther", f(m.get("single_other"))); nc("MsMultiOther", f(m.get("multi_other")))
    nc("MsAllFive", f(m.get("multi_all5"))); nc("MsMatch", f(m.get("single_match")))
    nc("MsP", pfmt(ms["paired_tests_per_subject"]["multi_other_vs_single_other"].get("p_holm")))
    if B:
        nc("BandBase", f(B["baseline_within_EER"]))
        for bnd in BANDS:
            v = B["bands"][bnd]; nc(f"Band{bnd.capitalize()}", f(v["dEER"])); nc(f"Band{bnd.capitalize()}CI", c(v["dEER_CI95"]))
            nc(f"Band{bnd.capitalize()}P", pfmt(v.get("p_holm")))
    rt = os.path.join(a.out, "runtime.json")
    if os.path.exists(rt):
        R2 = json.load(open(rt)); nc("Params", f"{R2['encoder_parameters']:,}".replace(",", "{,}")); nc("MsPerEpoch", f"{R2['ms_per_30s_epoch_batched']:.2f}")
    path = d(a.out, "numbers.tex")
    if os.path.exists(path): os.replace(path, path + f".prev_{datetime.datetime.now():%Y%m%d_%H%M%S}")
    open(path, "w").write("\n".join(L) + "\n"); print(f"wrote {path} ({len(L)} lines)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["embed", "main", "band", "band_pkl", "numbers"])
    ap.add_argument("--root", default="./"); ap.add_argument("--out", default="./rerun_leakfree_20261003")
    ap.add_argument("--train_frac", type=float, default=0.6); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--guard", type=int, default=10, help="same-stage epochs between enrolment and probe parts (10 = >=5 min)")
    ap.add_argument("--min_ep", type=int, default=10); ap.add_argument("--fuse", type=int, default=10)
    ap.add_argument("--fuse_list", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    ap.add_argument("--K", type=int, default=20, help="equal enrolment length (epochs) for the multi-stage comparison")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--wake_trim_min", type=float, default=30.0, help="keep wake within this many minutes of sleep (primary); untrimmed also reported")
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--emb_n1", default=None); ap.add_argument("--emb_n2", default=None)
    ap.add_argument("--n2_cache", default=None, help="where night-2 EDFs are downloaded (default <out>/data_n2; never --root)")
    ap.add_argument("--pkl", default=None, help="path to emb_ablation.pkl (default <root>/emb_ablation.pkl)")
    ap.add_argument("--use", choices=["trim", "untrimmed"], default="trim", help="which wake handling feeds numbers.tex")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    json.dump(vars(a), open(d(a.out, f"args_{a.step}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"), "w"), indent=2)
    {"embed": step_embed, "main": step_main, "band": step_band, "band_pkl": step_band_pkl, "numbers": step_numbers}[a.step](a)


if __name__ == "__main__":
    sys.exit(main())
