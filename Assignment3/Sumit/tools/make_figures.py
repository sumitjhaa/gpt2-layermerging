"""Figures of the results (PNG, 300 dpi, into docs/figures/), from the run folders via analyze.py:
    ~/.venvs/kaggle/bin/python tools/make_figures.py
Every figure is drawn at its printed size in the ACL report (full width 6.3 in, one column 3.03 in) in Times, 8 pt,
so the report includes them unscaled and their text matches the captions.
depth.png                 accuracy and perplexity damage vs layers removed (Qwen3-0.6B)
model_17b.png             the methods on Qwen3-1.7B (4 layers removed)
ablations.png             which layer is kept x training text, copying, 400 vs 1000 steps (Qwen3-0.6B, 4 removed)
benchmarks_vs_chance.png  each benchmark against random guessing, and the skill kept above chance
layer_profile.png         what each layer does (layer analysis, experiment 00)
alpha_schedule.png        the fade schedule shared by every trained method
diff_vs_control.png       forest plot: method minus the matched control, with 95 % CIs (is the gain real?)
tradeoff.png              accuracy vs perplexity damage, one path per method across depths
per_language.png          heatmap of the mC4 perplexity increase per language and method
seeds.png                 strip plot of the three seeds"""
import csv
import math
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "docs", "figures")
sys.path.insert(0, HERE)
import analyze as A  # noqa: E402
import make_metrics as MM  # noqa: E402

FULL, COL = 6.3, 3.03  # ACL A4: text width 16 cm, column width 7.7 cm
# validated palette (categorical slots 1-3; drop / original in neutrals); text in ink tokens, never series colours
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF, DARK = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#ffffff", "#3d3c39"
SMALL = 7
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "legend.frameon": False,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "lines.linewidth": 1.4, "lines.markersize": 4.5, "errorbar.capsize": 2})
BAR = dict(edgecolor=SURF, linewidth=0.8)
# method key -> (label, colour, marker); the same method at every depth
SER = [("drop", "drop", MUTED, "s"), ("dropkd_icl", "control", AQUA, "^"),
       ("aam_anneal0.7", "AAM", BLUE, "o"), ("prism_shrink_icl_energy0.7", "PRISM-Shrink", ORANGE, "D")]


def save(fig, name):
    fig.tight_layout(pad=0.3, w_pad=1.2)
    fig.savefig(os.path.join(OUT, name), dpi=300)
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
O17 = R17.get("original|0|—")

# 1. depth: accuracy and perplexity damage vs layers removed
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.55))
for m, lab, c, mk in SER:
    ks = [k for k in (2, 4, 6, 8) if by(R6, m, k)]
    ys = [acc(by(R6, m, k)) for k in ks]
    a1.errorbar(ks, [y[0] for y in ys], yerr=[[y[1] for y in ys], [y[2] for y in ys]], color=c, marker=mk, label=lab)
    a2.plot(ks, [A.ppl_inc(by(R6, m, k), O6) for k in ks], color=c, marker=mk, label=lab)
    dy = {"AAM": 6, "control": -6}.get(lab, 0)  # AAM and the control end at the same point
    a2.annotate(lab, (ks[-1], A.ppl_inc(by(R6, m, ks[-1]), O6)), xytext=(5, dy), textcoords="offset points",
                va="center", color=INK2, fontsize=SMALL)
o = acc(O6)[0]
a1.axhline(o, color=DARK, lw=0.8, ls="--")
a1.text(9.3, o + 0.5, f"original {o:.1f}", ha="right", va="bottom", color=INK2, fontsize=SMALL)
a1.axhline(21.7, color=MUTED, lw=0.8, ls=":")
a1.text(9.3, 22.0, "random guessing 21.7", ha="right", va="bottom", color=MUTED, fontsize=SMALL)
a1.set(xlabel="layers removed (of 28)", ylabel="few-shot avg accuracy % (95% CI)", xticks=[2, 4, 6, 8], xlim=(1.5, 9.4))
a1.set_title("(a) Accuracy", loc="left")
a1.legend(loc="center left", bbox_to_anchor=(0.0, 0.62))
a2.set_yscale("log")
a2.set(xlabel="layers removed (of 28)", ylabel="mC4 perplexity increase % (log)", xticks=[2, 4, 6, 8], xlim=(1.5, 10.6))
a2.set_title("(b) Perplexity damage", loc="left")
save(fig, "depth.png")

# 2. Qwen3-1.7B: one small panel per measure
if O17:
    S17 = [("drop", "drop", MUTED), ("dropkd_plain", "control", AQUA), ("aam_anneal0.7", "AAM", BLUE),
           ("prism_shrink_energy0.7", "PRISM-\nShrink", ORANGE)]
    rs = [(lab, c, by(R17, m, 4)) for m, lab, c in S17 if by(R17, m, 4)]
    fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.3))
    meas = [("(a) Accuracy % (higher is better)", lambda r: acc(r)[0], acc(O17)[0]),
            ("(b) mC4 perplexity increase %", lambda r: A.ppl_inc(r, O17), None),
            ("(c) Copying loss (lower is better)", lambda r: r["eval"]["copy_nll"], O17["eval"]["copy_nll"])]
    for ax, (t, fn, ref) in zip(axs, meas):
        vals = [fn(r) for _, _, r in rs]
        ax.bar(range(len(rs)), vals, color=[c for _, c, _ in rs], width=0.65, **BAR)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:.1f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=SMALL)
        if ref is not None:
            ax.axhline(ref, color=DARK, lw=0.8, ls="--")
            ax.text(-0.45, ref, "original", ha="left", va="bottom", color=INK2, fontsize=SMALL)
        ax.set_xticks(range(len(rs)), [lab for lab, _, _ in rs])
        ax.set_title(t, loc="left")
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.12)
    save(fig, "model_17b.png")

# 3. ablations at 4 layers removed: kept layer x training text, copying, training steps
fig, axs = plt.subplots(1, 3, figsize=(FULL, 2.45))


def pts(ax, x, r, c, mk, lab=None):
    a, lo, hi = acc(r)
    ax.errorbar([x], [a], yerr=[[lo], [hi]], color=c, marker=mk, label=lab)


KEPT = ["first layer\n(control)", "better layer\n(AAM)"]
ax = axs[0]
for i, (plain, icl, c) in enumerate([("dropkd_plain", "dropkd_icl", AQUA), ("aam_anneal0.7", "aam_icl", BLUE)]):
    pts(ax, i - 0.13, by(R6, plain, 4), c, "o", "plain text" if i == 0 else None)
    pts(ax, i + 0.13, by(R6, icl, 4), c, "s", "ICL-packed text" if i == 0 else None)
    for x, m in ((i - 0.13, plain), (i + 0.13, icl)):
        ax.text(x + 0.09, acc(by(R6, m, 4))[0], f"{acc(by(R6, m, 4))[0]:.1f}", va="center", color=INK2, fontsize=SMALL)
ax.set_xticks([0, 1], KEPT)
ax.set_xlim(-0.5, 1.6)
ax.set_ylabel("few-shot avg accuracy % (95% CI)")
ax.set_title("(a) Kept layer × text", loc="left")
leg = ax.legend(loc="upper left")
for h in leg.legend_handles:
    h.set_color(INK2)
ax = axs[1]
for j, text in enumerate(("plain", "ICL")):
    ms = [("dropkd_plain", "aam_anneal0.7"), ("dropkd_icl", "aam_icl")][j]
    xs = [i + (j - 0.5) * 0.38 for i in range(2)]
    vs = [by(R6, m, 4)["eval"]["copy_nll"] for m in ms]
    ax.bar(xs, vs, width=0.36, color=[AQUA, BLUE], alpha=1.0 if j == 0 else 0.55, hatch="//" if j else None, **BAR)
    for x, v in zip(xs, vs):
        ax.text(x, v, f"{v:.2f}\n{text}", ha="center", va="bottom", color=INK2, fontsize=SMALL)
ax.set_xticks([0, 1], KEPT)
ax.set_ylim(0, 0.75)
ax.set_ylabel("copying loss (lower is better)")
ax.set_title("(b) Copying loss", loc="left")
ax.grid(axis="x", visible=False)
ax = axs[2]
for i, (m, c) in enumerate([("aam_anneal0.7", BLUE), ("prism_shrink_energy0.7", ORANGE)]):
    for j, sfx in enumerate(("", "_steps1000")):
        r = by(R6, m + sfx, 4)
        if r:
            x = i + (j - 0.5) * 0.3
            pts(ax, x, r, c, "s" if sfx else "o")
            ax.text(x + (0.06 if sfx else -0.06), acc(r)[0], f"{acc(r)[0]:.1f}\n{'1000' if sfx else '400'} steps",
                    ha="left" if sfx else "right", va="center", color=INK2, fontsize=SMALL)
ax.set_xticks([0, 1], ["AAM", "PRISM-Shrink\n(plain text)"])
ax.set_xlim(-0.85, 1.75)
ax.set_title("(c) 400 vs 1000 steps", loc="left")
save(fig, "ablations.png")

# 4. each benchmark against random guessing, and the skill kept above chance (4 layers removed)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(FULL, 2.6), gridspec_kw={"width_ratios": [1.65, 1]})
S4 = [("original", 0, "original", DARK), ("drop", 4, "drop", MUTED), ("dropkd_icl", 4, "control", AQUA),
      ("aam_anneal0.7", 4, "AAM", BLUE), ("prism_shrink_icl_energy0.7", 4, "PRISM-Shrink", ORANGE)]
w = 0.16
for j, (m, k, lab, c) in enumerate(S4):
    a = MM.acc_by({"eval": {"hits": A.cells(by(R6, m, k), A.LANGS)}}, 0)
    a1.bar([i + (j - 2) * w for i in range(len(A.TASKS))], [a[t] for t in A.TASKS], width=w, color=c, label=lab, **BAR)
for i, t in enumerate(A.TASKS):
    a1.plot([i - 2.6 * w, i + 2.6 * w], [A.CHANCE[t]] * 2, color=INK, lw=1.0, ls="--",
            label="random guessing" if i == 0 else None)
a1.set_xticks(range(len(A.TASKS)), ["MMMLU", "INCLUDE", "MGSM", "Belebele", "XNLI"])
a1.set_ylabel("accuracy % (mean of 6 languages)")
a1.set_title("(a) Each benchmark, Qwen3-0.6B, 4 layers removed", loc="left")
a1.legend(ncol=3, loc="upper left", columnspacing=1.0, handlelength=1.4)
a1.set_ylim(0, 82)
a1.grid(axis="x", visible=False)
labs = ["drop", "control", "PRISM-\nShrink", "AAM"]
cols = [MUTED, AQUA, ORANGE, BLUE]
v6 = [MM.skill_kept(by(R6, m, 4), O6) for m in ("drop", "dropkd_icl", "prism_shrink_icl_energy0.7", "aam_anneal0.7")]
xs = range(len(labs))
a2.bar([x - 0.2 for x in xs], v6, width=0.38, color=cols, **BAR)
for x, v in zip(xs, v6):
    a2.text(x - 0.2, v + 0.8, f"{v:.0f}", ha="center", va="bottom", color=INK2, fontsize=SMALL)
if O17:
    v17 = [MM.skill_kept(by(R17, m, 4), O17) for m in ("drop", "dropkd_plain", "prism_shrink_energy0.7", "aam_anneal0.7")]
    a2.bar([x + 0.2 for x in xs], v17, width=0.38, color=cols, alpha=0.5, hatch="//", **BAR)
    for x, v in zip(xs, v17):
        a2.text(x + 0.2, v + 0.8, f"{v:.0f}", ha="center", va="bottom", color=INK2, fontsize=SMALL)
    a2.bar([0], [0], color=INK2, label="0.6B (solid)")
    a2.bar([0], [0], color=INK2, alpha=0.5, hatch="//", label="1.7B (hatched)")
    a2.legend(loc="upper left")
a2.set_xticks(list(xs), labs)
a2.set_ylim(0, 58)
a2.set_ylabel("% of above-chance skill kept")
a2.set_title("(b) Skill kept above guessing", loc="left")
a2.grid(axis="x", visible=False)
save(fig, "benchmarks_vs_chance.png")

# 5. layer profile (experiment 00): what each layer does
la = list(csv.DictReader(open(os.path.join(ROOT, "analysis", "layer_analysis", "layer_analysis_Qwen3-0.6B-Base.csv"))))
num = lambda x: float(x) if x not in ("", None) else math.nan
lay = [int(r["layer"]) for r in la]
panels = [("(a) Angular change in → out", [("angular change", BLUE, "")]),
          ("(b) Write size / ‖h‖ (log)", [("attn write / |h|", BLUE, "attention"), ("mlp write / |h|", ORANGE, "MLP")]),
          ("(c) Cosine with next update", [("update cos w/ next", BLUE, "")]),
          ("(d) Cross-lingual retrieval", [("cross-lingual retrieval", BLUE, "")]),
          ("(e) Language identity", [("language identity", BLUE, "")]),
          ("(f) Logit-lens accuracy", [("logit-lens acc (mean)", BLUE, "")])]
fig, axs = plt.subplots(2, 3, figsize=(FULL, 3.6))
for ax, (title, series) in zip(axs.flat, panels):
    for col, c, lab in series:
        ys = [num(r[col]) for r in la]
        if col == "angular change":
            ys[-1] = math.nan  # no value for the last layer (stored as 0)
        ax.plot(lay, ys, color=c, marker="o", ms=2.5, lw=1.1, label=lab or None)
    ax.axvspan(9.5, 17.5, color=YELLOW, alpha=0.15, lw=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("layer")
    if "log" in title:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([0.2, 0.5, 1, 2, 5])
        ax.legend(loc="upper right")
axs.flat[0].text(13.5, axs.flat[0].get_ylim()[1] * 0.95, "merged\n(10–17)", ha="center", va="top", color=INK2,
                 fontsize=SMALL)
save(fig, "layer_profile.png")

# 6. the fade schedule shared by every trained method
steps = list(range(400))
T = 280
al = [0.5 * (1 + math.cos(math.pi * (s + 1) / T)) if s + 1 < T else 0.0 for s in steps]
fig, ax = plt.subplots(figsize=(COL, 1.75))
ax.plot(steps, al, color=BLUE)
ax.axvline(T, color=MUTED, lw=0.8, ls="--")
ax.text(135, 0.62, "hand-over: the removed part\nfades out", ha="center", color=INK2, fontsize=SMALL)
ax.text(340, 0.22, "merged layer\nalone (α = 0)", ha="center", color=INK2, fontsize=SMALL)
ax.set_xlabel("distillation step")
ax.set_ylabel("α (weight of the original)")
save(fig, "alpha_schedule.png")

# 7. forest plot: method minus the matched control on the same questions (paired bootstrap, 95 % CI)
rows = [("0.6B, 2 layers", R6, 2, "", "dropkd_icl", "prism_shrink_icl_energy0.7"),
        ("0.6B, 4 layers", R6, 4, "", "dropkd_icl", "prism_shrink_icl_energy0.7"),
        ("0.6B, 4 layers, seed 1", R6, 4, "_seed1", "dropkd_icl", "prism_shrink_icl_energy0.7"),
        ("0.6B, 4 layers, seed 2", R6, 4, "_seed2", "dropkd_icl", "prism_shrink_icl_energy0.7"),
        ("0.6B, 6 layers", R6, 6, "", "dropkd_icl", "prism_shrink_icl_energy0.7"),
        ("0.6B, 8 layers", R6, 8, "", "dropkd_icl", "prism_shrink_icl_energy0.7")]
if O17:
    rows.append(("1.7B, 4 layers", R17, 4, "", "dropkd_plain", "prism_shrink_energy0.7"))
fig, ax = plt.subplots(figsize=(COL, 2.75))
for i, (lab, RR, k, sfx, ctl, shr) in enumerate(rows):
    y = len(rows) - 1 - i
    for m, c, mk, dy in (("aam_anneal0.7", BLUE, "o", 0.16), (shr, ORANGE, "D", -0.16)):
        e, lo, hi, p = A.boot(A.cells(by(RR, m + sfx, k), A.LANGS), A.cells(by(RR, ctl + sfx, k), A.LANGS))
        ax.errorbar([e], [y + dy], xerr=[[e - lo], [hi - e]], color=c, marker=mk, ms=3.5, lw=1.1, capsize=1.5,
                    label=("AAM" if m.startswith("aam") else "PRISM-Shrink") if i == 0 else None)
ax.axvline(0, color=INK, lw=0.8)
ax.set_yticks(range(len(rows)), [r[0] for r in rows][::-1])
ax.set_xlabel("accuracy vs control (points, 95% CI)")
ax.grid(axis="y", visible=False)
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
save(fig, "diff_vs_control.png")

# 8. trade-off: accuracy vs perplexity damage, one path per method across depths (0.6B)
fig, ax = plt.subplots(figsize=(COL, 2.6))
for m, lab, c, mk in SER:
    ks = [k for k in (2, 4, 6, 8) if by(R6, m, k)]
    xs = [A.ppl_inc(by(R6, m, k), O6) for k in ks]
    ys = [acc(by(R6, m, k))[0] for k in ks]
    ax.plot(xs, ys, color=c, marker=mk, ms=3.5, lw=1.0, label=lab)
    for k, x, y in zip(ks, xs, ys):
        if m in ("drop", "aam_anneal0.7"):
            ax.annotate(str(k), (x, y), xytext=(3, 3), textcoords="offset points", color=INK2, fontsize=6.5)
ax.axhline(21.7, color=MUTED, lw=0.8, ls=":")
ax.text(4.2, 22.0, "random guessing", color=MUTED, fontsize=SMALL, va="bottom")
ax.set_xscale("log")
ax.set_xlabel("mC4 perplexity increase % (log; left is better)")
ax.set_ylabel("few-shot avg accuracy %")
ax.legend(loc="upper right")
save(fig, "tradeoff.png")

# 9. heatmap: mC4 perplexity increase per language (0.6B, 4 layers removed), one hue, darker = worse
HM = [("drop", "drop"), ("dropkd_icl", "control"), ("aam_anneal0.7", "AAM"), ("prism_shrink_icl_energy0.7", "PRISM-Shrink")]
vals = [[100 * (by(R6, m, 4)["eval"]["mc4"][l] / O6["eval"]["mc4"][l] - 1) for l in A.LANGS] for m, _ in HM]
from matplotlib.colors import LinearSegmentedColormap, LogNorm  # noqa: E402
cmap = LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#9ec5f4", "#3987e5", "#184f95", "#0d2d57"])
fig, ax = plt.subplots(figsize=(COL, 1.75))
im = ax.imshow(vals, cmap=cmap, norm=LogNorm(vmin=8, vmax=200), aspect="auto")
for i, row in enumerate(vals):
    for j, v in enumerate(row):
        ax.text(j, i, f"+{v:.0f}", ha="center", va="center", fontsize=6.5, color="white" if v > 40 else INK)
ax.set_xticks(range(len(A.LANGS)), A.LANGS)
ax.set_yticks(range(len(HM)), [lab for _, lab in HM])
ax.grid(False)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
cb.set_label("mC4 ppl increase %", fontsize=7)
cb.ax.tick_params(labelsize=6.5)
cb.outline.set_visible(False)
save(fig, "per_language.png")

# 10. strip plot: the three seeds (0.6B, 4 layers removed)
fig, ax = plt.subplots(figsize=(COL, 1.9))
for i, (m, lab, c) in enumerate([("dropkd_icl", "control", AQUA), ("prism_shrink_icl_energy0.7", "PRISM-Shrink", ORANGE),
                                 ("aam_anneal0.7", "AAM", BLUE)]):
    ys = [acc(by(R6, m + s, 4))[0] for s in ("", "_seed1", "_seed2")]
    ax.scatter(ys, [i - 0.13, i, i + 0.13], color=c, s=18, zorder=3, edgecolor=SURF, linewidth=0.6)  # offset: equal scores stay visible
    mu = sum(ys) / 3
    ax.plot([mu, mu], [i - 0.25, i + 0.25], color=INK, lw=1.0)
    ax.text(max(ys) + 0.15, i, f"mean {mu:.1f}", va="center", color=INK2, fontsize=SMALL)
ax.set_yticks(range(3), ["control", "PRISM-Shrink", "AAM"])
ax.set_xlim(23, 27.2)
ax.set_ylim(-0.6, 2.6)
ax.set_xlabel("few-shot avg accuracy % (seeds 0, 1, 2)")
ax.grid(axis="y", visible=False)
save(fig, "seeds.png")
