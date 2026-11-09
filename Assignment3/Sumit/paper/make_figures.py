"""Figures for LAYER_MERGING_EXPLAINED.md (PNG, into ../notes_images/). Reads the archived runs only.
    ~/.venvs/kaggle/bin/python paper/make_figures.py"""
import csv
import json
import math
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "notes_images")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
import analyze  # noqa: E402

# reference palette (light): categorical slots 1-3 + neutrals for reference rows; text in ink tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
YELLOW, VIOLET = "#eda100", "#6250d6"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
DARKGRAY = "#3d3c39"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=150)
    plt.close(fig)
    print("wrote", name)


def P(rel):  # archived runs (gitignored archive/) are read when present
    return os.path.join(ROOT, rel) if os.path.exists(os.path.join(ROOT, rel)) else os.path.join(ROOT, "archive", rel)


def rows(path):
    return list(csv.DictReader(open(path)))


f = lambda x: float(x) if x not in ("", None) else math.nan

# 1. layer profile (notebook 00)
la = rows(P("experimented/00_layer_analysis/results/layer_analysis_Qwen3-0.6B-Base.csv"))
lay = [int(r["layer"]) for r in la]
panels = [("How much each layer turns the state\n(angular change, input→output)", [("angular change", BLUE)]),
          ("How big each layer's write is\n(relative to the residual stream ‖h‖)", [("attn write / |h|", BLUE), ("mlp write / |h|", ORANGE)]),
          ("Same direction as the next layer?\n(cosine of consecutive updates)", [("update cos w/ next", BLUE)]),
          ("Language-neutral 'concept space'\n(cross-lingual retrieval accuracy)", [("cross-lingual retrieval", BLUE)]),
          ("Language identity in the state\n(share of variance explained by language)", [("language identity", BLUE)]),
          ("Can the model already answer here?\n(logit-lens next-token accuracy)", [("logit-lens acc (mean)", BLUE)])]
fig, axs = plt.subplots(2, 3, figsize=(13, 6.6))
for ax, (title, series) in zip(axs.flat, panels):
    for col, c in series:
        ys = [f(r[col]) for r in la]
        ax.plot(lay, ys, color=c, lw=2, marker="o", ms=3.5, label=col.split(" /")[0])
    ax.axvspan(9.5, 17.5, color=YELLOW, alpha=0.12, lw=0)
    ax.set_title(title, loc="left")
    ax.set_xlabel("layer")
    if "write" in title:
        ax.set_yscale("log")
        ax.legend(loc="upper right")
axs.flat[0].text(13.5, axs.flat[0].get_ylim()[1] * 0.92, "merged here\n(layers 10–17)", ha="center", va="top", color=INK2, fontsize=8.5)
save(fig, "fig01_layer_profile.png")

# 2. pair scores: why angular distance and CoPlan pick different pairs
pa = rows(P("experimented/00_layer_analysis/results/pair_analysis_Qwen3-0.6B-Base.csv"))
log15 = open(os.path.join(ROOT, "experimented/15_polyshrink_full/run.log")).read()
uerr = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\d+-\d+) ([\d.]+)%", log15.split("parallel-union error per pair:")[1].split("\n")[0])}
pc = {r["pair"]: f(r["composition_risk"]) for r in rows(os.path.join(ROOT, "paper_runs/p20a_data_refs/results/pair_circuits_Qwen3-0.6B-Base.csv"))}
pairs = [r["pair"] for r in pa]
ang4, cop4 = {"10-11", "12-13", "14-15", "16-17"}, {"13-14", "15-16", "21-22", "23-24"}
x = list(range(len(pairs)))
fig, axs = plt.subplots(4, 1, figsize=(12, 9.5), sharex=True)
spec = [("Angular distance between the state before and after the pair (lower = pair changes little)", [f(r["angular(in, out)"]) for r in pa], ang4, BLUE, "angular-distance choice (4 removed)"),
        ("Error of the exact PARALLEL union of the pair, % (lower = pair can run side by side)", [uerr.get(p, math.nan) for p in pairs], cop4, ORANGE, "CoPlan choice (4 removed)"),
        ("Linearity R² of the pair's update (higher = one linear map explains the pair)", [f(r["linearity R2"]) for r in pa], set(), BLUE, ""),
        ("Cross-layer copying-circuit risk: prev-token head in layer i × induction head in layer i+1", [pc.get(p, math.nan) for p in pairs], set(), VIOLET, "")]
for ax, (title, ys, pick, c, lab) in zip(axs, spec):
    cols = [c if p in pick else (MUTED if pick else c) for p in pairs]
    ax.bar(x, ys, color=cols, width=0.72)
    ax.set_title(title, loc="left", fontsize=10)
    if lab:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=c, label=lab), Patch(color=MUTED, label="other pairs")], loc="upper right", fontsize=8.5)
    if "copying" in title:
        for i, p in enumerate(pairs):
            if ys[i] > 0.3:
                ax.text(i, ys[i] + 0.01, p, ha="center", va="bottom", fontsize=8, color=INK2)
axs[1].set_ylim(0, 50)
axs[1].text(2.6, 48, "bar 2-3 is cut: 82 %", fontsize=8, color=INK2, va="top")
axs[-1].set_xticks(x, pairs, rotation=60)
axs[-1].set_xlabel("pair of consecutive layers")
save(fig, "fig02_pair_scores.png")

# 3. heads: previous-token and induction score per layer x head
hd = rows(os.path.join(ROOT, "paper_runs/p20a_data_refs/results/heads_Qwen3-0.6B-Base.csv"))
L, Hn = 28, 16
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
cmap = LinearSegmentedColormap.from_list("seq", [SURF] + SEQ)
fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
for ax, key, title in [(axs[0], "prev_token", "Previous-token heads\n(attention weight on token t−1)"),
                       (axs[1], "induction", "Induction heads\n(attention to the token after the earlier copy)")]:
    M = [[0.0] * Hn for _ in range(L)]
    for r in hd:
        M[int(r["layer"])][int(r["head"])] = f(r[key])
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=1, origin="lower")
    ax.set_title(title, loc="left")
    ax.set_xlabel("query head")
    ax.set_ylabel("layer")
    ax.grid(False)
    ax.axhspan(9.5, 17.5, xmin=0, xmax=0.02, color=ORANGE)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
axs[0].text(-0.6, 13.5, "merged", rotation=90, va="center", ha="right", fontsize=8, color=ORANGE)
save(fig, "fig03_heads.png")

# 4. alpha schedule
steps = list(range(400))
A = 280
al = [0.5 * (1 + math.cos(math.pi * (s + 1) / A)) if s + 1 < A else 0.0 for s in steps]
fig, ax = plt.subplots(figsize=(8, 3.2))
ax.plot(steps, al, color=BLUE, lw=2)
ax.axvline(A, color=MUTED, lw=1, ls="--")
ax.text(140, 0.55, "hand-over: the old pair fades out,\nthe merged layer takes over", ha="center", color=INK2, fontsize=9)
ax.text(340, 0.2, "merged layer\nalone (α = 0)", ha="center", color=INK2, fontsize=9)
ax.set_xlabel("distillation step")
ax.set_ylabel("α (weight of the original pair)")
ax.set_title("Annealing schedule shared by AAM, Homotopy, PRISM-Heal, Shrink and the control", loc="left")
save(fig, "fig04_alpha_schedule.png")

# 5. old experiments (100-item test): perplexity vs accuracy
OLD = {2: [("drop", 15.1, 30.6), ("01 CD-RegMean", 17.7, 24.0), ("02 SDCF", 10.2, 25.7), ("03 AAM", 6.8, 34.3), ("05 NUF", 9.0, 26.5),
           ("06 Homotopy", 8.3, 29.4), ("07 PRISM-energy", 9.2, 27.4), ("08 span", 9.5, 27.5), ("09 polyglot", 9.6, 27.7),
           ("10 Heal", 8.0, 29.5), ("11 Shrink", 6.9, 30.8), ("12 LaCo", 12.6, 31.3), ("13 Shrink+ICL", 6.7, 30.1), ("14 Shrink+LSR", 7.0, 30.8)],
       4: [("drop", 110.3, 23.8), ("01 CD-RegMean", 64.3, 22.2), ("02 SDCF", 39.5, 22.4), ("03 AAM", 27.5, 25.4), ("05 NUF", 26.6, 23.2),
           ("06 Homotopy", 34.3, 23.4), ("07 PRISM-energy", 30.5, 23.0), ("08 span", 29.8, 22.3), ("09 polyglot", 31.8, 22.6),
           ("10 Heal", 23.3, 24.1), ("11 Shrink", 18.8, 24.7), ("12 LaCo", 70.2, 24.1), ("13 Shrink+ICL", 17.9, 23.8), ("14 Shrink+LSR", 19.2, 25.2)]}
fam = lambda n: MUTED if n == "drop" else BLUE if "AAM" in n else ORANGE if ("Shrink" in n) else VIOLET if "LaCo" in n else AQUA
NUDGE = {(2, "11 Shrink"): (-6, 5), (2, "14 Shrink+LSR"): (6, -2), (2, "13 Shrink+ICL"): (6, -9), (2, "10 Heal"): (-6, -11),
         (2, "06 Homotopy"): (6, -8), (2, "07 PRISM-energy"): (-6, -10), (2, "08 span"): (6, -9), (2, "09 polyglot"): (6, 4)}
fig, axs = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, k in zip(axs, (2, 4)):
    for n, p, a in OLD[k]:
        ax.scatter(p, a, s=46, color=fam(n), edgecolor=SURF, linewidth=1.5, zorder=3)
        dx, dy = NUDGE.get((k, n), (5, 3))
        ax.annotate(n, (p, a), xytext=(dx, dy), textcoords="offset points", fontsize=8, color=INK2,
                    ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlabel("mC4 perplexity increase over the original, % (log; lower is better →left)")
    ax.set_ylabel("avg few-shot accuracy, % (higher is better ↑)")
    ax.set_title(f"{k} layers removed (old 100-item test; ±4 points noise)", loc="left")
for c, lab in [(BLUE, "AAM"), (ORANGE, "Shrink family"), (AQUA, "closed-form / other merges"), (VIOLET, "LaCo (published)"), (MUTED, "drop")]:
    axs[1].scatter([], [], color=c, s=46, label=lab)
axs[1].legend(loc="upper right", fontsize=8.5)
save(fig, "fig05_old_tradeoff.png")

# paper runs (large test set)
R = analyze.load(os.path.join(ROOT, "paper_runs"))
if os.path.isdir(os.path.join(ROOT, "archive/paper_runs")):
    R.update(analyze.load(os.path.join(ROOT, "archive/paper_runs")))
O = R["original|0|—"]
NAME = {"drop": "drop (delete one layer)", "dropkd_icl": "drop + KD (control)", "aam_anneal0.7": "AAM (03)",
        "prism_shrink_icl_energy0.7": "Shrink + ICL (13)", "skm_shrink_icl": "+ SKM (16)", "xtc_shrink_icl": "+ XTC (17)",
        "skm_xtc_shrink_icl": "+ SKM + XTC (18)"}
COL = {"drop": MUTED, "dropkd_icl": AQUA, "aam_anneal0.7": BLUE, "prism_shrink_icl_energy0.7": ORANGE,
       "skm_shrink_icl": ORANGE, "xtc_shrink_icl": ORANGE, "skm_xtc_shrink_icl": ORANGE}
ORDER = ["drop", "dropkd_icl", "aam_anneal0.7", "prism_shrink_icl_energy0.7", "skm_shrink_icl", "xtc_shrink_icl", "skm_xtc_shrink_icl"]
byk = {k: {r["method"]: r for r in R.values() if r["k"] == k} for k in (2, 4)}
LANGS = analyze.LANGS
cache = os.path.join(ROOT, "archive", ".ci_cache.json")
CI = json.load(open(cache)) if os.path.exists(cache) else {}
for k in (2, 4):
    for m, r in byk[k].items():
        key = f"{k}|{m}"
        if key not in CI:
            CI[key] = analyze.boot(analyze.cells(r, LANGS))[:3]
json.dump(CI, open(cache, "w"))

# 6. accuracy with 95% CI
fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
oa = analyze.avg(analyze.cells(O, LANGS), None)
for ax, k in zip(axs, (2, 4)):
    for i, m in enumerate(ORDER):
        est, lo, hi = CI[f"{k}|{m}"]
        ax.plot([lo, hi], [i, i], color=COL[m], lw=2.5, solid_capstyle="round")
        ax.scatter(est, i, s=60, color=COL[m], edgecolor=SURF, linewidth=1.5, zorder=3)
        ax.text(hi + 0.3, i, f"{est:.1f}", va="center", fontsize=8.5, color=INK2)
    d = CI[f"{k}|drop"][0]
    ax.axvline(d, color=MUTED, lw=1, ls="--")
    ax.set_yticks(range(len(ORDER)), [NAME[m] for m in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel(f"avg few-shot accuracy, % (original model: {oa:.1f})")
    ax.set_title(f"{k} layers removed: mean and 95 % bootstrap CI", loc="left")
save(fig, "fig06_paper_accuracy_ci.png")

# 7. perplexity and copying, small multiples
fig, axs = plt.subplots(2, 2, figsize=(13, 7))
for j, k in enumerate((2, 4)):
    ppl = [analyze.ppl_inc(byk[k][m], O) for m in ORDER]
    cp = [byk[k][m]["eval"]["copy_nll"] for m in ORDER]
    for ax, ys, title, fmt in [(axs[0, j], ppl, f"{k} removed: mC4 perplexity increase, % (lower is better)", "{:+.1f}"),
                               (axs[1, j], cp, f"{k} removed: copying loss, NLL (original {O['eval']['copy_nll']:.3f}; lower is better)", "{:.3f}")]:
        ax.barh(range(len(ORDER)), ys, color=[COL[m] for m in ORDER], height=0.66)
        for i, v in enumerate(ys):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8.5, color=INK2)
        ax.set_yticks(range(len(ORDER)), [NAME[m] for m in ORDER])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlim(0, max(ys) * 1.22)
save(fig, "fig07_paper_ppl_copy.png")

# 8. per benchmark (2 removed): original, drop, control, Shrink, AAM
TASKS = analyze.TASKS
TN = {"mmmlu": "MMMLU\n5-shot knowledge", "include": "INCLUDE\n5-shot regional exams", "mgsm": "MGSM\n4-shot math (CoT)",
      "belebele": "Belebele\n3-shot reading", "xnli": "XNLI\n3-shot inference"}


def bench(r, t):
    e = r["eval"][t]
    v = [e[l] for l in LANGS if l in e]
    return 100 * sum(v) / len(v)


ser = [("original", O, DARKGRAY), ("drop", byk[2]["drop"], MUTED), ("drop + KD (control)", byk[2]["dropkd_icl"], AQUA),
       ("Shrink + ICL (13)", byk[2]["prism_shrink_icl_energy0.7"], ORANGE), ("AAM (03)", byk[2]["aam_anneal0.7"], BLUE)]
fig, ax = plt.subplots(figsize=(13, 4.8))
w = 0.16
for si, (n, r, c) in enumerate(ser):
    xs = [ti + (si - 2) * (w + 0.012) for ti in range(len(TASKS))]
    ax.bar(xs, [bench(r, t) for t in TASKS], width=w, color=c, label=n)
ax.axhline(25, color=MUTED, lw=1, ls=":")
ax.text(2.5, 25.6, "chance on 4-way tasks", fontsize=8, color=INK2, ha="center")
ax.set_xticks(range(len(TASKS)), [TN[t] for t in TASKS])
ax.set_ylabel("accuracy, % (mean over languages)")
ax.set_title("2 layers removed: every benchmark (large test set, 200–250 items per language)", loc="left")
ax.legend(ncol=5, loc="upper right", fontsize=8.5)
ax.set_ylim(0, 74)
save(fig, "fig08_paper_benchmarks.png")

# 9. per language perplexity increase (4 removed)
ser = [("drop", byk[4]["drop"], MUTED), ("drop + KD (control)", byk[4]["dropkd_icl"], AQUA),
       ("AAM (03)", byk[4]["aam_anneal0.7"], BLUE), ("Shrink + ICL (13)", byk[4]["prism_shrink_icl_energy0.7"], ORANGE)]
fig, ax = plt.subplots(figsize=(12, 4.4))
w = 0.2
for si, (n, r, c) in enumerate(ser):
    ys = [100 * (r["eval"]["mc4"][l] / O["eval"]["mc4"][l] - 1) for l in LANGS]
    ax.bar([li + (si - 1.5) * (w + 0.012) for li in range(len(LANGS))], ys, width=w, color=c, label=n)
ax.set_xticks(range(len(LANGS)), ["English", "Hindi", "French", "Spanish", "German", "Chinese"])
ax.set_ylabel("mC4 perplexity increase, %")
ax.set_title("4 layers removed: damage per language (lower is better)", loc="left")
ax.legend(ncol=4, loc="upper left", fontsize=8.5)
save(fig, "fig09_paper_per_language.png")

# 10. XTC closed-form gap closure (from the P18 log)
log18 = open(P("paper_runs/pp_p18_p19/run.log")).read()
gx = {}
for m in re.finditer(r"group (\d+-\d+): XTC rank 64, held-out error of the parallel union ([\d.]+)% -> \+ XTC ([\d.]+)%", log18):
    gx[m.group(1)] = (float(m.group(2)), float(m.group(3)))
ps = sorted(gx, key=lambda p: int(p.split("-")[0]))
fig, ax = plt.subplots(figsize=(8, 3.6))
ax.bar([i - 0.19 for i in range(len(ps))], [gx[p][0] for p in ps], width=0.36, color=MUTED, label="exact parallel union")
ax.bar([i + 0.19 for i in range(len(ps))], [gx[p][1] for p in ps], width=0.36, color=ORANGE, label="union + XTC (rank 64, closed form)")
for i, p in enumerate(ps):
    ax.text(i + 0.19, gx[p][1], f"{gx[p][1]:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.text(i - 0.19, gx[p][0], f"{gx[p][0]:.1f}", ha="center", va="bottom", fontsize=8.5, color=INK2)
ax.set_xticks(range(len(ps)), [f"pair {p}" for p in ps])
ax.set_ylabel("held-out error vs the real\nsequential pair, %")
ax.set_title("XTC: the cross term closes part of the parallel-vs-sequential gap (before training)", loc="left", fontsize=10)
ax.legend(loc="upper center", ncol=2, fontsize=8.5)
ax.set_ylim(0, 31)
save(fig, "fig10_xtc_gap.png")
