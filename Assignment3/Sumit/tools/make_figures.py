"""Figures of the large-test results (PNG, into docs/figures/), from the run folders via analyze.py:
    ~/.venvs/kaggle/bin/python tools/make_figures.py
depth.png        accuracy and perplexity damage vs layers removed (Qwen3-0.6B)
model_17b.png    the methods on Qwen3-1.7B (4 layers removed)
ablations.png    which layer is kept x training text, and 400 vs 1000 steps (Qwen3-0.6B, 4 layers removed)"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "figures")
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402

# same palette as the other figures (validated: categorical slots 1-3, drop / original in neutrals)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURF, DARK = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb", "#3d3c39"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False, "axes.axisbelow": True})
# method key -> (label, colour, marker); the same method at every depth
SER = [("drop", "drop", MUTED, "s"), ("dropkd_icl", "drop + KD (control)", AQUA, "^"),
       ("aam_anneal0.7", "AAM", BLUE, "o"), ("prism_shrink_icl_energy0.7", "PRISM-Shrink", ORANGE, "D")]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


def acc(r):
    est, lo, hi, _ = A.boot(A.cells(r, A.LANGS))
    return est, est - lo, hi - est


def by(R, method, k):
    return next((r for r in R.values() if r["method"] == method and r["k"] == k), None)


R = A.load(ROOT)
R6, R17 = R["Qwen3-0.6B-Base"], R.get("Qwen3-1.7B-Base", {})
O6 = R6["original|0|—"]

# 1. depth: two panels (accuracy, perplexity damage), one x axis
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
for m, lab, c, mk in SER:
    ks = [k for k in (2, 4, 6, 8) if by(R6, m, k)]
    ys = [acc(by(R6, m, k)) for k in ks]
    a1.errorbar(ks, [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]], color=c, marker=mk, ms=7,
                lw=2, capsize=3, label=lab)
    a2.plot(ks, [A.ppl_inc(by(R6, m, k), O6) for k in ks], color=c, marker=mk, ms=7, lw=2, label=lab)
    dy = {"AAM": 7, "drop + KD (control)": -7}.get(lab, 0)  # AAM and control end at the same point
    a2.annotate(lab, (ks[-1], A.ppl_inc(by(R6, m, ks[-1]), O6)), xytext=(7, dy), textcoords="offset points",
                va="center", color=INK2, fontsize=8.5)
o = acc(O6)[0]
a1.axhline(o, color=DARK, lw=1, ls="--")
a1.text(9.3, o + 0.4, f"original {o:.1f}", ha="right", va="bottom", color=INK2, fontsize=8.5)
a1.axhline(26, color=MUTED, lw=1, ls=":")
a1.text(9.3, 26.3, "≈ chance", ha="right", va="bottom", color=MUTED, fontsize=8.5)
a1.set(xlabel="layers removed (of 28)", ylabel="few-shot avg accuracy % (95 % CI)", xticks=[2, 4, 6, 8], xlim=(1.5, 9.4))
a1.set_title("Accuracy: AAM is best at every depth", loc="left")
a1.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
a2.set_yscale("log")
a2.set(xlabel="layers removed (of 28)", ylabel="mC4 perplexity increase % (log)", xticks=[2, 4, 6, 8], xlim=(1.5, 10.2))
a2.set_title("Perplexity damage: PRISM-Shrink is lowest from 4 layers", loc="left")
save(fig, "depth.png")

# 2. Qwen3-1.7B: three small panels, one per measure
if "original|0|—" in R17:
    O17 = R17["original|0|—"]
    S17 = [("drop", "drop", MUTED), ("dropkd_plain", "control", AQUA), ("aam_anneal0.7", "AAM", BLUE),
           ("prism_shrink_energy0.7", "PRISM-Shrink", ORANGE)]
    rs = [(lab, c, by(R17, m, 4)) for m, lab, c in S17 if by(R17, m, 4)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.9))
    meas = [("Few-shot avg accuracy %\n(higher is better)", lambda r: acc(r)[0], acc(O17)[0]),
            ("mC4 perplexity increase %\n(lower is better)", lambda r: A.ppl_inc(r, O17), None),
            ("In-context copying loss (NLL)\n(lower is better)", lambda r: r["eval"]["copy_nll"], O17["eval"]["copy_nll"])]
    for ax, (t, fn, ref) in zip(axs, meas):
        vals = [fn(r) for _, _, r in rs]
        ax.bar(range(len(rs)), vals, color=[c for _, c, _ in rs], width=0.62, edgecolor=SURF, linewidth=2)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=9)
        if ref is not None:
            ax.axhline(ref, color=DARK, lw=1, ls="--")
            ax.text(-0.45, ref, "original", ha="left", va="bottom", color=INK2, fontsize=8.5)
        ax.set_xticks(range(len(rs)), [lab for lab, _, _ in rs])
        ax.set_title(t, loc="left")
        ax.grid(axis="x", visible=False)
    save(fig, "model_17b.png")

# 3. ablations at 4 layers removed: kept layer x training text, and training steps
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
pts = lambda ax, x, r, c, mk, lab=None: ax.errorbar([x], [acc(r)[0]], yerr=[[acc(r)[1]], [acc(r)[2]]], color=c, marker=mk,
                                                   ms=8, capsize=3, lw=2, label=lab)
ax = axs[0]
for i, (kept, plain, icl, c) in enumerate([("first layer kept\n(control)", "dropkd_plain", "dropkd_icl", AQUA),
                                           ("better layer kept\n(AAM)", "aam_anneal0.7", "aam_icl", BLUE)]):
    pts(ax, i - 0.12, by(R6, plain, 4), c, "o", "plain text" if i == 0 else None)
    pts(ax, i + 0.12, by(R6, icl, 4), c, "s", "ICL-packed text" if i == 0 else None)
    for x, m in ((i - 0.12, plain), (i + 0.12, icl)):
        ax.text(x + 0.05, acc(by(R6, m, 4))[0], f"{acc(by(R6, m, 4))[0]:.1f}", va="center", color=INK2, fontsize=8.5)
ax.set_xticks([0, 1], ["first layer kept\n(control)", "better layer kept\n(AAM)"])
ax.set_xlim(-0.5, 1.6)
ax.set_ylabel("few-shot avg accuracy % (95 % CI)")
ax.set_title("Accuracy: the kept layer matters", loc="left")
leg = ax.legend(loc="upper left")
for h in leg.legend_handles:
    h.set_color(INK2)
ax = axs[1]
for j, text in enumerate(("plain", "ICL-packed")):
    ms = [("dropkd_plain", "aam_anneal0.7"), ("dropkd_icl", "aam_icl")][j]
    xs = [i + (j - 0.5) * 0.36 for i in range(2)]
    vs = [by(R6, m, 4)["eval"]["copy_nll"] for m in ms]
    ax.bar(xs, vs, width=0.34, color=[AQUA, BLUE], alpha=1.0 if j == 0 else 0.55, edgecolor=SURF, linewidth=2,
           hatch="//" if j else None)
    for x, v in zip(xs, vs):
        ax.text(x, v, f"{v:.2f}\n{text}", ha="center", va="bottom", color=INK2, fontsize=8.5)
ax.set_xticks([0, 1], ["first layer kept\n(control)", "better layer kept\n(AAM)"])
ax.set_ylim(0, 0.72)
ax.set_title("Copying loss: ICL-packed text helps", loc="left")
ax.grid(axis="x", visible=False)
ax = axs[2]
for i, (m, c) in enumerate([("aam_anneal0.7", BLUE), ("prism_shrink_energy0.7", ORANGE)]):
    for j, sfx in enumerate(("", "_steps1000")):
        r = by(R6, m + sfx, 4)
        if r:
            x = i + (j - 0.5) * 0.3
            pts(ax, x, r, c, "o" if not sfx else "s")
            ax.text(x + (0.05 if sfx else -0.05), acc(r)[0], f"{acc(r)[0]:.1f}\n{'1000' if sfx else '400'} steps",
                    ha="left" if sfx else "right", va="center", color=INK2, fontsize=8.5)
ax.set_xticks([0, 1], ["AAM", "PRISM-Shrink\n(plain text)"])
ax.set_xlim(-0.75, 1.6)
ax.set_title("Accuracy: 400 vs 1000 training steps", loc="left")
save(fig, "ablations.png")
