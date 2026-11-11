# Tools: how the experiments were built, run, collected and analysed

These scripts turn one tested notebook into all the large-test experiments, run them on Kaggle's free GPUs, file the
results, and rebuild every table and figure. You only need them to **re-run** experiments; to **read** results, see
[`../docs/metrics_and_benchmarks.md`](../docs/metrics_and_benchmarks.md).

---

## 1. The pipeline at a glance

```
 prism_shrink/prism_shrink_icl.ipynb   (the base notebook that already worked)
            │
            ▼  make_notebooks.py      patch settings + algorithm cells  →  generated/<name>.ipynb  (one per experiment)
            │                         pair two notebooks per Kaggle session → generated/pp_<session>.ipynb
            ▼  kaggle_queue.py        push to Kaggle in dependency order (uses kaggle_run.sh), wait, download
            │                         → runs_raw/<session>/<notebook>/  (results JSON + log)
            ▼  organize_runs.py       file each run under its method → <method>/runs/<notebook>/ + <method>/<notebook>.ipynb
            │
            ▼  analyze.py             → results/large_test_results.md   (CIs, p-values, competence check)
               make_metrics.py        → results/all_metrics.md          (every measurement)
               make_figures.py        → docs/figures/*.png
```

`generated/`, `runs_raw/`, `queue_state.json` and `queue*.log` are working files and are not committed (see
`.gitignore`); everything they lead to is.

## 2. Each script

### `make_notebooks.py`: build every experiment from one notebook
- Starts from `../prism_shrink/prism_shrink_icl.ipynb`, which already contains all the shared code (data, evaluation,
  planner, distillation, PRISM and Shrink libraries).
- For each experiment in `JOBS` it inserts a settings cell (model, layers to remove, seed, steps, memory settings)
  and the algorithm cells (AAM, control or PRISM-Shrink), and applies a few patches. **Every patch asserts that it
  applied**, so a changed base notebook fails loudly instead of silently running the wrong code.
- Patches that matter for correctness: larger test sizes, Japanese and Russian evaluation, per-question results (for
  the statistics), fp16 original model for 1.7B, the output-layer precision fix, smaller micro-batches
  (`mb_tokens`) to fit GPU memory.
- `PAIRS` groups two notebooks into one Kaggle session (`pp_*`), one per GPU, because Kaggle bills GPU time per
  session and gives two T4 GPUs.
- For each session it also writes a tiny CPU version (`s_pp_*`, the "smoke test": a few steps and questions) to
  catch crashes without spending GPU time. A CPU smoke test cannot catch fp16 or out-of-memory problems.
- Run: `python3 tools/make_notebooks.py`.

### `kaggle_queue.py`: run everything on Kaggle in the right order
- Order: smoke tests → reference sessions (they download the data and compute the original and drop rows once) →
  method sessions (they read the reference session's output as an input). `PLAN` holds the dependencies.
- Runs at most 2 GPU sessions at a time (Kaggle's limit; a new push is refused with "Maximum batch GPU session count
  of 2 reached" and retried every few minutes).
- Resumable: the state of each session is in `queue_state.json`. Run with `--status` to see it.
- Run: `MAX_CPU=5 POLL=180 python3 tools/kaggle_queue.py` (needs a Kaggle API token in `~/.kaggle/`).

### `kaggle_run.sh`: push, check and download one notebook
- `push`: uploads a notebook as a private Kaggle kernel with a T4 GPU and internet.
- `status`: queued / running / complete / error.
- `archive`: downloads the outputs, turns Kaggle's JSON log into plain text, and stores each notebook's own log as
  `run.log`. (An earlier version picked the wrong log for some paired sessions; fixed, and every session was
  downloaded again.)

### `organize_runs.py`: file the downloaded runs
For each downloaded notebook: copy its own results and log to `<method>/runs/<notebook>/` (skipping the shared
reference file, which belongs to the reference runs) and copy the notebook next to its method. Run:
`python3 tools/organize_runs.py . aam_seed1 aam_seed2`. Running it again on an already filed run changes nothing.

### `analyze.py`: the main results table with statistics
- Loads every `results_*.json` under the method folders (ignoring smoke tests and raw downloads).
- For each run: params cut, mC4 perplexity increase, avg accuracy with a **95 % bootstrap confidence interval**,
  the **paired difference against drop** with its interval and p-value, Japanese/Russian accuracy, copying loss and
  MGSM.
- The **competence check**: flags any benchmark × language cell where the original model is not clearly above chance.
- A small self-test of the averaging and bootstrap code runs automatically before every analysis.
- Run: `python3 tools/analyze.py . > results/large_test_results.md`.

### `make_metrics.py`: every measurement in one file
Per model: headline table (accuracy, skill kept above chance, perplexity on mC4 and Wikipedia, copying, speed),
accuracy per benchmark, accuracy per language (including Japanese and Russian), perplexity per language.
Run: `python3 tools/make_metrics.py . > results/all_metrics.md`.

### `make_figures.py`: the result figures
Writes `depth.png`, `model_17b.png`, `ablations.png` and `benchmarks_vs_chance.png` into `docs/figures/`. Needs
matplotlib. Run: `python3 tools/make_figures.py`.

## 3. Things to know before re-running

- **Kaggle limits:** about 30 GPU hours per week and 2 GPU sessions at a time. One paired session takes about
  1–3.5 hours.
- **Memory:** PRISM-Shrink trains a double-width layer in fp32. At 6+ layers removed or on the 1.7B model it ran out
  of the T4's 15 GB; `mb_tokens` (512 or 256) fixes it with the same gradient.
- **Precision:** the 1.7B original model runs in fp16 (it does not fit in fp32 alongside training); the trained layers
  stay in fp32.
- **Old name:** the generated notebooks mention `paper/make_notebooks.py`; `paper/` was the old name of this folder.

Everything that went wrong while running: [`../docs/limitations_and_issues.md`](../docs/limitations_and_issues.md)
section 2.
