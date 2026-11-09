# Assignment 3 (Sumit): merging consecutive layers of a multilingual LLM

Remove 2 or 4 of the 28 layers of **Qwen3-0.6B-Base** by merging pairs of consecutive layers, and lose as little as
possible in six languages (en, hi, fr, es, de, zh).

- **PRISM-Shrink** (the new method). Rebuild a pair of layers exactly as one wide parallel layer, then shrink it
  gradually to one standard layer while distilling from the original model.
- **AAM** (annealed absorption, the strong baseline). Keep the better layer of each pair, fade the other out and
  distill.

## Main results (large test, 95 % confidence intervals)

| Layers removed | Method | mC4 perplexity | Few-shot avg acc | vs drop | Copying loss |
|---|---|---|---|---|---|
| 0 | original | — | 48.9 | — | 0.125 |
| 2 | drop (delete one layer of each pair) | +14.2 % | 28.7 | — | 0.320 |
| 2 | drop + the same distillation (control) | +7.9 % | 29.3 | +0.6 (p = 0.20) | 0.172 |
| 2 | **AAM** | **+6.0 %** | **34.1** | **+5.4 (p < 0.001)** | 0.203 |
| 2 | **PRISM-Shrink** | +6.2 % | 28.2 | −0.4 (p = 0.45) | **0.131** |
| 4 | drop | +84.7 % | 22.6 | — | 2.220 |
| 4 | drop + the same distillation (control) | +24.5 % | 23.5 | +1.0 (p = 0.05) | 0.372 |
| 4 | **AAM** | +22.8 % | **25.8** | **+3.3 (p < 0.001)** | 0.337 |
| 4 | **PRISM-Shrink** | **+16.5 %** | 23.5 | +0.9 (p = 0.15) | **0.212** |

AAM gives the only significant few-shot gain. PRISM-Shrink gives the lowest perplexity at 4 layers and keeps
in-context copying best. Against the matched control, it does 33 % less perplexity damage and has a 43 % lower
copying loss, in all six languages. All other experiments are in `results/`.

## Layout

```
README.md
docs/
  layer_merging_explained.md     everything explained from LLM basics up, with figures
  algorithm_notes.md             every algorithm, its maths and results, plus common questions
  results_comparison.md          comparison of all experiments and the best per goal
  figures/                       the figures used in the docs
results/
  small_test_results.md          all experiments, small test (100 items per benchmark)
  large_test_results.md          main methods and ablations, large test (200-250 items, CIs, p-values)
prism_shrink/                    PRISM-Shrink
  prism_shrink.ipynb             small test
  prism_shrink_icl.ipynb         small test, with ICL-preserving training text
  prism_shrink_icl_large_test.ipynb
  runs/<notebook name>/          run.log and result JSONs of that notebook's run
aam/                             AAM (same layout)
baselines/
  reference/                     original model + drop, on both tests
  drop_kd_control/               drop + the same distillation as the merge methods
  laco/                          published LaCo baseline (Yang et al., 2024)
analysis/
  layer_analysis/                what each layer does, and scores for every pair of layers
  attention_heads/               previous-token and induction score of every head
  test_set/                      test tokens per language (>= 100k each)
tools/
  kaggle_run.sh                  push a notebook to Kaggle (GPU T4), check it, download its results
  analyze.py                     rebuild results/large_test_results.md from the runs
```

## Setup

- **Calibration and training:** Wiki-40B train. **Tests** (never used for fitting): MMMLU 5-shot (MMLU for English),
  INCLUDE 5-shot, MGSM 4-shot chain of thought, Belebele 3-shot, XNLI 3-shot, mC4 and Wiki-40B perplexity, and an
  in-context copying probe.
- **Hardware:** Kaggle GPU T4. Each notebook runs on its own. Example:
  `cd tools && NB_DIR=../aam ./kaggle_run.sh push aam_large_test`.
- **Rebuild the results table:** `python3 tools/analyze.py`.

## Prior work

Running consecutive layers in parallel was proposed before ("Leveraging the True Depth of LLMs", 2025). Layer removal
by angular distance: Gromov et al.; ShortGPT. Merging baseline: LaCo. The contributions here are rebuilding a pair
exactly as one parallel layer and shrinking it back to a standard layer (PRISM-Shrink), choosing pairs by how exactly
they run in parallel (CoPlan), and the measurements behind them.
