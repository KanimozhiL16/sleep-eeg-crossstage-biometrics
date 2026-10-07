#!/usr/bin/env python3
# v3 (2026-10-07): TrueType (fonttype 42) Times-family fonts for IEEE PDF checks. v2 (2026-10-04): cell values rounded to whole % so neighbouring numbers do not touch; exact values are in the CSVs.
"""make_abstract_fig_v2_20261004.py -- Figure 1 of the SPMB 2026 abstract from results_leakfree.json.
Two 5x5 EER heatmaps (night-1 probes | night-2 probes), shared grey scale, values printed in
every cell so the figure reads in grayscale.  Usage:
    python make_abstract_fig_v2_20261004.py --out ./rerun_leakfree_20261003 [--use trim]
writes <out>/fig_matrices.pdf (+ .png preview). Never overwrites: an existing file is renamed.
"""
import argparse, json, os, datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGES = ["W", "N1", "N2", "N3", "REM"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
    ap.add_argument("--use", choices=["trim", "untrimmed"], default="trim"); a = ap.parse_args()
    R = json.load(open(os.path.join(a.out, "results_leakfree.json")))[a.use]
    M1 = np.array(R["night1_matrix_clean"], float)
    M2 = np.array(R["cross_night"]["matrix"], float) if "cross_night" in R else np.full((5, 5), np.nan)
    vmax = float(np.nanmax([np.nanmax(M1), np.nanmax(M2)]))
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "Nimbus Roman", "TeX Gyre Termes", "DejaVu Serif"],
                         "font.size": 8})
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), constrained_layout=True)
    for ax, M, title in [(axes[0], M1, "Probe: same night"), (axes[1], M2, "Probe: second night")]:
        im = ax.imshow(M, cmap="Greys", vmin=0, vmax=vmax)
        for i in range(5):
            for j in range(5):
                v = M[i, j]; txt = "NA" if np.isnan(v) else f"{v:.0f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=6.5,
                        color="white" if (not np.isnan(v) and v > 0.55 * vmax) else "black",
                        fontweight="bold" if i == j else "normal")
        ax.set_xticks(range(5)); ax.set_yticks(range(5))
        ax.set_xticklabels(STAGES, fontsize=6.5); ax.set_yticklabels(STAGES, fontsize=6.5)
        ax.set_title(title, fontsize=7.5, pad=3); ax.set_xlabel("Probe stage", fontsize=7)
        ax.tick_params(length=0)
    axes[0].set_ylabel("Enrolment stage (night 1)", fontsize=7)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02); cb.set_label("EER (%)", fontsize=7); cb.ax.tick_params(labelsize=6)
    for ext in ["pdf", "png"]:
        p = os.path.join(a.out, f"fig_matrices.{ext}")
        if os.path.exists(p): os.replace(p, p + f".prev_{datetime.datetime.now():%Y%m%d_%H%M%S}")
        fig.savefig(p, dpi=300)
    print("wrote", os.path.join(a.out, "fig_matrices.pdf"))


if __name__ == "__main__":
    main()
