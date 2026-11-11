"""Move finished Kaggle runs (tools/runs_raw/<session>/<notebook>/results + that notebook's part of the session log)
into <method>/runs/<notebook>/, and copy the notebook next to its method. Usage: python3 tools/organize_runs.py . [notebook ...]"""
import glob, os, re, shutil, sys
root = sys.argv[1]
DEST = {"reference": "baselines/reference", "aam": "aam", "drop_kd_control": "baselines/drop_kd_control", "prism_shrink": "prism_shrink"}
only = set(sys.argv[2:])
for d in sorted(glob.glob(f"{root}/tools/runs_raw/pp_*/*/results")):
    nbdir = os.path.dirname(d); name = os.path.basename(nbdir); sess = os.path.dirname(nbdir)
    if only and name not in only: continue
    own = [f for f in os.listdir(d) if not (f.startswith("results_reference") and not name.startswith("reference"))]
    if not any(f.startswith("results_") for f in own):
        print("skip (no own results):", sess, name); continue
    dest = next(v for k, v in DEST.items() if name.startswith(k))
    out = f"{root}/{dest}/runs/{name}"
    os.makedirs(f"{out}/results", exist_ok=True)
    for f in own: shutil.copy2(f"{d}/{f}", f"{out}/results/{f}")
    shutil.copy2(f"{nbdir}/run.log", f"{out}/run.log")  # that notebook's own log
    shutil.copy2(f"{root}/tools/generated/{name}.ipynb", f"{root}/{dest}/{name}.ipynb")
    print(f"{sess.split('/')[-1]}/{name} -> {dest}/runs/{name}  ({len(own)} files)")
