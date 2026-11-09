"""Runs the paper notebooks on Kaggle in dependency order, a few at a time, and archives each finished run to
paper_runs/<notebook>/ (notebook + log, run.log, result JSONs / CSVs). Stdlib only; uses ../kaggle_run.sh.

    python3 paper/kaggle_queue.py              # run (resumable: state in paper/queue_state.json)
    python3 paper/kaggle_queue.py --status     # print the state and exit
    python3 paper/kaggle_queue.py --only s_    # restrict to names starting with a prefix (e.g. the smoke tests)

Stage 1: CPU smoke tests (tiny settings, no GPU quota). Stage 2: P20a / P20b (shared data + reference rows).
Stage 3: everything else, in priority order (if the weekly GPU quota runs out, the important runs are done first)."""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE = os.path.join(HERE, "queue_state.json")
KAGGLE = os.environ.get("KAGGLE", os.path.expanduser("~/.venvs/kaggle/bin/kaggle"))
MAX = {"gpu": int(os.environ.get("MAX_GPU", 2)), "cpu": int(os.environ.get("MAX_CPU", 4))}
POLL = int(os.environ.get("POLL", 300))
# Core plan (~9 GPU-h budget): the method, the fair control and the ablation base. Later, if quota allows, append in
# this order: "p16_skm", "p17_xtc", "p03_aam", "p20b_refs_coplan" (to REFS), "p21_full_coplan", "p15_polyshrink",
# "p12_laco", "p18_skm_xtc_seed1", "p19_dropkd_icl_seed1", "p18_skm_xtc_seed2", "p19_dropkd_icl_seed2", "p18_skm_xtc_math".
REFS = ["p20a_data_refs"]
# pp_*: two notebooks in one session, one per T4 (paper/make_notebooks.py PAIRS): p18+p19, p13+p16, p17+p03
MAIN = ["pp_p18_p19", "pp_p13_p16", "pp_p17_p03"]
SMOKES = ["s_p20a_data_refs", "s_p18_skm_xtc", "s_p19_dropkd_icl", "s_p13_shrink_icl"] + ["s_" + n for n in MAIN]


def smoke_of(n):
    return "s_" + n.split("_seed")[0]


# name -> (deps, gpu)
JOBS = {s: ([], False) for s in SMOKES}
JOBS.update({n: ([smoke_of(n)], True) for n in REFS})
JOBS.update({n: ([smoke_of(n)] + REFS, True) for n in MAIN})
# Pairs start without waiting for the slow paired CPU smoke tests: their code paths ran in s_p13, s_p18 (SKM + XTC)
# and s_p19, and AAM (p03) ran on Kaggle before; the paired runner itself is proven by pp_p18_p19.
for _n in MAIN:
    JOBS[_n] = (REFS, True)


def sh(*args):
    env = dict(os.environ, KAGGLE=KAGGLE, NB_DIR="paper", ARCHIVE_DIR="paper_runs")
    r = subprocess.run([os.path.join(ROOT, "kaggle_run.sh"), *args], cwd=ROOT, env=env, capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def user():
    return os.environ.get("KAGGLE_USER") or next(
        l.split(": ")[1] for l in subprocess.run([KAGGLE, "config", "view"], capture_output=True, text=True).stdout.splitlines()
        if "username" in l)


def status(n):
    _, out = sh("status", n)
    m = re.search(r"KernelWorkerStatus\.(\w+)", out)  # only Kaggle's own status field; API/network errors stay unknown
    k = m.group(1).lower() if m else ""
    return {"cancel_acknowledged": "cancel", "cancel_requested": "cancel"}.get(k, k) if k else "unknown: " + out[-200:]


def main():
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else ""
    old = json.load(open(STATE)) if os.path.exists(STATE) else {}
    st = {n: old.get(n, "pending") for n in JOBS}  # jobs removed from JOBS are forgotten
    if "--status" in sys.argv:
        for n, s in st.items():
            print(f"{s:10s} {n}")
        return
    me = user()
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    while True:
        for n in [n for n, s in st.items() if s == "running"]:
            s = status(n)
            if s in ("complete", "error", "cancel"):
                rc, out = sh("archive", n)
                st[n] = "done" if s == "complete" else "failed"
                log(f"{n}: {s} -> {'archived' if rc == 0 else 'ARCHIVE FAILED: ' + out[-300:]}")
            elif s.startswith("unknown"):
                log(f"{n}: {s}")
        for n, (deps, _) in JOBS.items():  # a failed dependency blocks everything after it
            if st[n] == "pending" and any(st[d] in ("failed", "blocked") for d in deps):
                st[n] = "blocked"
                log(f"{n}: blocked (a dependency failed)")
        for n, (deps, gpu) in JOBS.items():
            kind = "gpu" if gpu else "cpu"
            busy = sum(1 for m, s in st.items() if s == "running" and JOBS[m][1] == gpu)
            if st[n] != "pending" or not n.startswith(only) or busy >= MAX[kind] or any(st[d] != "done" for d in deps):
                continue
            src = ",".join(f"{me}/layer-merging-{d.replace('_', '-')}" for d in deps if d in REFS)
            os.environ.update(GPU="1" if gpu else "0", KERNEL_SOURCES=src)
            rc, out = sh("push", n)
            if rc == 0 and "error" not in out.lower():
                st[n] = "running"
                log(f"{n}: started ({kind}{', inputs: ' + src if src else ''})")
            else:
                log(f"{n}: push refused, retrying next round: {out[-300:]}")
        json.dump(st, open(STATE, "w"), indent=1)
        active = [n for n, s in st.items() if s in ("running", "pending") and n.startswith(only)]
        if not any(st[n] == "running" for n in active) and not any(
                all(st[d] == "done" for d in JOBS[n][0]) for n in active if st[n] == "pending"):
            log("queue finished: " + ", ".join(f"{n}={s}" for n, s in st.items() if n.startswith(only)))
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
