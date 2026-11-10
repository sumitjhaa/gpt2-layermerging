"""Runs the generated notebooks (tools/generated/) on Kaggle in dependency order, a few at a time, and downloads each
finished run to tools/runs_raw/<notebook>/ (notebook + log, run.log, result JSONs / CSVs). Stdlib only; uses
kaggle_run.sh in this folder.

    python3 tools/kaggle_queue.py              # run (resumable: state in tools/queue_state.json)
    python3 tools/kaggle_queue.py --status     # print the state and exit

Order: CPU smoke tests of every session (tiny settings, no GPU quota) -> the reference sessions (shared data,
original and drop rows) -> the method sessions, in priority order. Each pp_* session runs two notebooks, one per T4."""
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "queue_state.json")
KAGGLE = os.environ.get("KAGGLE", os.path.expanduser("~/.venvs/kaggle/bin/kaggle"))
MAX = {"gpu": int(os.environ.get("MAX_GPU", 2)), "cpu": int(os.environ.get("MAX_CPU", 4))}
POLL = int(os.environ.get("POLL", 300))
# reference session -> method sessions that read its data and original / drop rows (priority order)
PLAN = {"pp_reference_deep": ["pp_data_ablation", "pp_deep_methods", "pp_shrink_deep", "pp_deep_control", "pp_more_steps"],
        "pp_reference_17b": ["pp_methods_17b", "pp_shrink_17b"]}
REFS = list(PLAN)
JOBS = {"s_" + n: ([], False) for r, ms in PLAN.items() for n in [r] + ms}  # name -> (deps, gpu)
JOBS.update({r: (["s_" + r], True) for r in REFS})
for r, ms in PLAN.items():
    JOBS.update({m: (["s_" + m, r], True) for m in ms})
ORDER = ["pp_reference_deep", "pp_reference_17b", "pp_data_ablation", "pp_deep_methods", "pp_methods_17b",
         "pp_shrink_deep", "pp_shrink_17b", "pp_deep_control", "pp_more_steps"]
JOBS = {**{k: v for k, v in JOBS.items() if k.startswith("s_")}, **{k: JOBS[k] for k in ORDER}}


def sh(*args):
    env = dict(os.environ, KAGGLE=KAGGLE, NB_DIR=os.path.join(HERE, "generated"), ARCHIVE_DIR=os.path.join(HERE, "runs_raw"))
    r = subprocess.run([os.path.join(HERE, "kaggle_run.sh"), *args], cwd=HERE, env=env, capture_output=True, text=True)
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
