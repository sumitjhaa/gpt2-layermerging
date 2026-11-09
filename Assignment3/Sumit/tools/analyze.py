"""Large-test results table with uncertainty, from the run results in this folder (stdlib only):
    python3 tools/analyze.py > results/large_test_results.md
avg acc = mean over the 5 benchmarks of the mean over calibrated languages, with a 95 % bootstrap CI (test items
resampled within each benchmark x language). Delta vs drop = the same, PAIRED (the same resampled items for the method
and for drop on the same pairs); p = two-sided bootstrap p-value of delta = 0. Seeds of one method are pooled into
mean +- std. Only runs with per-item results (the large test) are included."""
import glob
import json
import os
import random
import sys
from collections import defaultdict

LANGS = ["en", "hi", "fr", "es", "de", "zh"]
UNSEEN = ["ja", "ru"]
TASKS = ["mmmlu", "include", "mgsm", "belebele", "xnli"]
B = 2000


def load(root):
    R = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "results", "results_*.json"), recursive=True)):
        R.update({k: r for k, r in json.load(open(p)).items() if "hits" in r["eval"]})  # small-test rows share keys
    return R


def cells(r, langs):
    return {k: [int(c) for c in v] for k, v in r["eval"]["hits"].items() if k.split("/")[1] in langs}


def avg(c, idx):
    per = defaultdict(list)
    for k, h in c.items():
        ix = idx[k] if idx else range(len(h))
        per[k.split("/")[0]].append(sum(h[i] for i in ix) / len(h))
    return 100 * sum(sum(v) / len(v) for v in per.values()) / len(per)


def boot(c, ref=None, seed=0):
    rnd = random.Random(seed)
    est = avg(c, None) - (avg(ref, None) if ref else 0)
    keys = [k for k in c if not ref or k in ref]
    c = {k: c[k] for k in keys}
    ref = {k: ref[k] for k in keys} if ref else None
    xs = []
    for _ in range(B):
        idx = {k: [rnd.randrange(len(h)) for _ in h] for k, h in c.items()}
        xs.append(avg(c, idx) - (avg(ref, idx) if ref else 0))
    xs.sort()
    lo, hi = xs[int(0.025 * B)], xs[int(0.975 * B) - 1]
    p = min(1.0, 2 * min(sum(x <= 0 for x in xs), sum(x >= 0 for x in xs)) / B) if ref else None
    return est, lo, hi, p


def ppl_inc(r, o, ds="mc4"):
    e = r["eval"][ds]
    inc = [100 * (e[l] / o["eval"][ds][l] - 1) for l in LANGS if l in e]
    return sum(inc) / len(inc)


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    R = load(root)
    print("# Results on the large test (95 % bootstrap CIs, paired p-values vs drop on the same pairs)\n")
    o = R["original|0|—"]
    drop = {(r["k"], r["plan"]): r for r in R.values() if r["method"] == "drop"}
    print("| removed | pairs | method | params cut % | mC4 ppl +% | avg acc [95% CI] | Δ vs drop [95% CI] | p | "
          "unseen acc | copy NLL | MGSM | decode tok/s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in R.values():
        head, _, tail = r["method"].rpartition("_s")
        base = head if tail.isdigit() else r["method"]  # seed suffix _s1, _s2 -> pooled with seed 0
        groups[(r["k"], r["plan"], base)].append(r)
    for (k, plan, m), rs in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2] != "drop", x[0][2])):
        r = rs[0]
        c = cells(r, LANGS)
        est, lo, hi, _ = boot(c)
        d = drop.get((k, plan))
        dl = "—"
        pv = ""
        if d and m != "drop" and k:
            de, dlo, dhi, p = boot(c, cells(d, LANGS))
            dl, pv = f"{de:+.1f} [{dlo:+.1f}, {dhi:+.1f}]", f"{p:.3f}"
        if len(rs) > 1:  # seeds
            vals = [avg(cells(x, LANGS), None) for x in rs]
            mu = sum(vals) / len(vals)
            sd = (sum((v - mu) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
            m = f"{m} ({len(rs)} seeds: {mu:.1f} ± {sd:.1f})"
        un = cells(r, UNSEEN)
        mg = [v for kk, v in c.items() if kk.startswith("mgsm/")]
        print(f"| {k} | {plan} | {m} | {100 * (1 - r['params'] / o['params']):.1f} | {ppl_inc(r, o):+.1f} | "
              f"{est:.1f} [{lo:.1f}, {hi:.1f}] | {dl} | {pv} | {f'{avg(un, None):.1f}' if un else '—'} | "
              f"{r['eval'].get('copy_nll', float('nan')):.3f} | {100 * sum(map(sum, mg)) / max(1, sum(map(len, mg))):.1f} | "
              f"{r['eval'].get('speed', {}).get('decode_tok_s', '')} |")


def _selftest():
    c = {"mmmlu/en": [1, 0] * 50, "mgsm/en": [1] * 10 + [0] * 90}
    assert abs(avg(c, None) - 30.0) < 1e-9
    est, lo, hi, p = boot(c, c)
    assert est == 0 and lo == 0 and hi == 0 and p == 1.0


if __name__ == "__main__":
    _selftest()
    main()
