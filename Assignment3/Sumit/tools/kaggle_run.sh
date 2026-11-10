#!/usr/bin/env bash
# Run notebooks on Kaggle (GPU T4 x2, Internet on) from the terminal, then fetch their outputs.
#   NB_DIR=../aam ./kaggle_run.sh push aam_large_test     # upload + start (private kernel)
#   ./kaggle_run.sh status aam_large_test                  # queued / running / complete / error
#   ./kaggle_run.sh archive aam_large_test                 # download it: ../downloads/<nb>/ (notebook + log cell, run.log, results/)
# Needs ~/.kaggle/access_token or ~/.kaggle/kaggle.json (kaggle.com -> Settings -> API) and a phone-verified Kaggle
# account (required for Internet + GPU). KAGGLE=/path/to/kaggle overrides the CLI binary.
# Optional env: NB_DIR (where the notebooks are, default .), ARCHIVE_DIR (default ../downloads), GPU=0 (CPU kernel),
# KERNEL_SOURCES="user/kernel-a,user/kernel-b" (their outputs are mounted under /kaggle/input).
set -euo pipefail
cd "$(dirname "$0")"
KAGGLE=${KAGGLE:-kaggle}
cmd=$1; shift
# username: $KAGGLE_USER, else whatever the CLI resolves from ~/.kaggle/access_token or ~/.kaggle/kaggle.json
user=${KAGGLE_USER:-$("$KAGGLE" config view 2>/dev/null | awk -F': ' '/username/ {print $2; exit}')}
[ -n "$user" ] || { echo "no Kaggle login found (~/.kaggle/access_token or ~/.kaggle/kaggle.json)" >&2; exit 1; }
slug() { echo "$1" | tr '_' '-' | tr 'A-Z' 'a-z'; }

for nb in "$@"; do
  nb=${nb%.ipynb}
  id="$user/layer-merging-$(slug "$nb")"
  case $cmd in
    push)
      d=$(mktemp -d)
      cp "${NB_DIR:-.}/$nb.ipynb" "$d/"
      gpu=$([ "${GPU:-1}" = 0 ] && echo false || echo true)
      src=$(echo "${KERNEL_SOURCES:-}" | awk -F, '{for(i=1;i<=NF;i++) printf "%s\"%s\"", (i>1?", ":""), $i}')
      cat > "$d/kernel-metadata.json" <<EOF
{"id": "$id", "title": "layer-merging-$(slug "$nb")", "code_file": "$nb.ipynb", "language": "python",
 "kernel_type": "notebook", "is_private": true, "enable_gpu": $gpu, "enable_internet": true,
 "dataset_sources": [], "competition_sources": [], "kernel_sources": [$src]}
EOF
      if [ "$gpu" = true ]; then "$KAGGLE" kernels push -p "$d" --accelerator NvidiaTeslaT4; else "$KAGGLE" kernels push -p "$d"; fi
      rm -rf "$d"
      echo "started: https://www.kaggle.com/code/$id" ;;
    status) "$KAGGLE" kernels status "$id" ;;
    archive)  # <ARCHIVE_DIR>/<nb>/: source notebook + its Kaggle run log as a last cell, plain log, result JSONs
      out="${ARCHIVE_DIR:-../downloads}/$nb"; tmp=$(mktemp -d)
      mkdir -p "$out/results"
      "$KAGGLE" kernels output "$id" -p "$tmp" -o -q
      "$KAGGLE" kernels pull "$id" -p "$tmp" >/dev/null
      cp "$tmp"/layer_merging/*.json "$tmp"/layer_merging/*.csv "$out/results/" 2>/dev/null || true  # results JSONs / analysis CSVs
      for sub in "$tmp"/*/layer_merging; do  # paired runs (pp_*): one data dir per inner notebook
        [ -d "$sub" ] && mkdir -p "$out/$(basename "$(dirname "$sub")")/results" && \
          cp "$sub"/*.json "$sub"/*.csv "$out/$(basename "$(dirname "$sub")")/results/" 2>/dev/null || true
      done
      python3 - "$tmp" "$out" "$nb" "$id" <<'PY'
import json, os, re, sys, glob
tmp, out, nb, kid = sys.argv[1:]
def clean(path):
    raw = open(path, errors="replace").read()
    try:
        log = "".join(e.get("data", "") for e in json.loads(raw))
    except ValueError:
        log = raw
    log = re.sub(r"\x1b\[[0-9;]*m", "", log)
    return "\n".join(l.split("\r")[-1] for l in log.split("\n"))  # keep only the final state of progress-bar lines
logs = sorted(glob.glob(f"{tmp}/*.log"))
kernel = [p for p in logs if os.path.basename(p).startswith("layer-merging-")]  # Kaggle's session log
log = clean((kernel or logs)[0])
open(f"{out}/run.log", "w").write(log)
for p in logs:  # paired runs: each inner notebook's own log -> <out>/<notebook>/run.log
    d = f"{out}/{os.path.basename(p)[:-4]}"
    if p not in kernel and os.path.isdir(d):
        open(f"{d}/run.log", "w").write(clean(p))
src = json.load(open(glob.glob(f"{tmp}/*.ipynb")[0]))
src["cells"] += [
    {"cell_type": "markdown", "metadata": {}, "source": [f"## Kaggle run log ({kid}, GPU T4 x2)\n",
     "Everything the notebook printed during the Kaggle run, in order. The executed notebook with outputs in place: "
     f"https://www.kaggle.com/code/{kid} (File -> Download notebook)."]},
    {"cell_type": "code", "metadata": {}, "execution_count": None, "source": ["# output of the Kaggle run (not re-run)"],
     "outputs": [{"output_type": "stream", "name": "stdout", "text": log.splitlines(True)}]}]
json.dump(src, open(f"{out}/{nb}.ipynb", "w"), indent=1, ensure_ascii=False)
PY
      rm -rf "$tmp"
      echo "archived: $out" ;;
    *) echo "usage: $0 push|status|archive NOTEBOOK..." >&2; exit 1 ;;
  esac
done
