# Assignment 3 (Sumit): merging consecutive layers of a multilingual LLM

Remove 2 to 8 of the 28 layers of **Qwen3-0.6B-Base** (and 4 of the 28 layers of **Qwen3-1.7B-Base**) by merging
pairs of consecutive layers, and lose as little as possible in six languages (en, hi, fr, es, de, zh).

- **PRISM-Shrink** (a gradual, distilled variant of layer flattening; the closest prior work is FlattenGPT, see
  Prior work). Rebuild a pair of layers exactly as one wide parallel layer, then shrink it gradually to one standard
  layer while distilling from the original model.
- **AAM** (annealed absorption, the strong baseline). Keep the better layer of each pair, fade the other out and
  distill.
- **Control**: drop one layer of each pair and apply the same distillation. Any gain a method shows over this control
  comes from the method, not from the distillation.

## Main results (large test, 95 % confidence intervals, p-values paired against drop)

Few-shot avg acc = mean of MMMLU, INCLUDE, MGSM, Belebele and XNLI over the six languages (200-250 items per benchmark
and language). Perplexity = mC4 increase over the original. Copying loss = in-context copying probe (lower is better).

**Qwen3-0.6B-Base** (original: 48.9 avg acc, copying loss 0.125)

| Layers removed | Method | mC4 perplexity | Few-shot avg acc | vs drop | Copying loss |
|---|---|---|---|---|---|
| 2 | drop | +14.2 % | 28.7 | — | 0.320 |
| 2 | control | +7.9 % | 29.3 | +0.6 (p = 0.20) | 0.172 |
| 2 | **AAM** | **+6.0 %** | **34.1** | **+5.4 (p < 0.001)** | 0.203 |
| 2 | **PRISM-Shrink** | +6.2 % | 28.2 | −0.4 (p = 0.45) | **0.131** |
| 4 | drop | +84.7 % | 22.6 | — | 2.220 |
| 4 | control | +24.5 % | 23.5 | +1.0 (p = 0.05) | 0.372 |
| 4 | **AAM** | +22.8 % | **25.8** | **+3.3 (p < 0.001)** | 0.337 |
| 4 | **PRISM-Shrink** | **+16.5 %** | 23.5 | +0.9 (p = 0.15) | **0.212** |
| 4 | PRISM-Shrink, plain text | +17.5 % | 24.3 | +1.7 (p = 0.008) | 0.218 |
| 6 | drop | +236.6 % | 21.5 | — | 4.292 |
| 6 | control | +59.2 % | 22.3 | +0.8 (p = 0.17) | 0.506 |
| 6 | **AAM** | +54.7 % | **23.8** | **+2.3 (p = 0.001)** | 0.519 |
| 6 | **PRISM-Shrink** | **+39.4 %** | 23.1 | +1.6 (p = 0.011) | **0.340** |
| 8 | drop | +689.4 % | 20.9 | — | 10.112 |
| 8 | control | +103.8 % | 21.6 | +0.7 (p = 0.21) | 1.091 |
| 8 | **AAM** | +103.1 % | **23.4** | **+2.5 (p < 0.001)** | 0.766 |
| 8 | **PRISM-Shrink** | **+69.0 %** | 22.3 | +1.4 (p = 0.017) | **0.651** |

**Qwen3-1.7B-Base, 4 layers removed** (original: 58.9 avg acc, copying loss 0.106; all three trained on the same plain
text)

| Method | mC4 perplexity | Few-shot avg acc | vs drop | Copying loss |
|---|---|---|---|---|
| drop | +70.5 % | 24.1 | — | 2.831 |
| control | +32.1 % | 30.9 | +6.8 (p < 0.001) | 1.150 |
| **AAM** | +29.4 % | **34.1** | **+10.0 (p < 0.001)** | 0.450 |
| **PRISM-Shrink** | **+22.3 %** | 31.5 | +7.4 (p < 0.001) | **0.257** |

**What the numbers show**

1. **AAM has the best accuracy** at every depth and on both models. It is significantly above drop everywhere, and
   significantly above the matched control everywhere too (+1.5 to +4.8 points, p ≤ 0.018; +3.2 on Qwen3-1.7B).
2. **PRISM-Shrink has the lowest perplexity damage and keeps in-context copying best** from 4 layers on, on both models.
   Its perplexity is below the control's and AAM's in all six languages, on both mC4 and Wikipedia, at 4, 6 and 8
   layers and on Qwen3-1.7B. At 8 layers it does a third less perplexity damage than AAM or the control; on Qwen3-1.7B
   its copying loss is 4.5× lower than the control's. Its accuracy, however, is not significantly different from
   the control's (−1.0 to +1.0 points, p ≥ 0.07).
3. **Why AAM wins on accuracy:** it keeps the better layer of each pair. With the same training text, keeping the better
   layer instead of the first gives +1.6 to +2.5 points; changing only the training text (plain vs ICL-packed) moves
   accuracy by at most 0.7 points. ICL-packed text lowers the copying loss instead.
4. **The bigger model gains more.** AAM's gain over drop is +10.0 points on Qwen3-1.7B, against +3.3 on Qwen3-0.6B at
   the same depth.
5. **Training longer (1000 instead of 400 steps)** lowers perplexity by about 2 points for both methods, moves accuracy
   by at most +0.8 (AAM 25.8 → 26.6, p = 0.05) and makes copying worse. The ranking of the methods does not change.
6. **Three training seeds** (4 layers removed): AAM 25.8 ± 0.3, significantly above the control on every seed
   (+1.8 to +2.6, p ≤ 0.009); PRISM-Shrink 23.6 ± 0.1 accuracy (same as the control) with the lowest perplexity
   damage on every seed (+16.6 % vs +24.2 %). Copying loss is the noisiest measure across seeds.
7. **Test validity.** Every benchmark and language is answered clearly above chance by the original model, except
   German XNLI on the 0.6B model (flagged in `results/large_test_results.md`).

Figures: `docs/figures/depth.png`, `docs/figures/model_17b.png`, `docs/figures/ablations.png`. All rows, CIs, unseen
languages (ja, ru) and MGSM: `results/large_test_results.md`.

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
  large_test_results.md          every large-test run (200-250 items, CIs, p-values, competence check)
prism_shrink/                    PRISM-Shrink: one notebook per run
  runs/<notebook name>/          run.log and result JSONs of that notebook's run
aam/                             AAM (same layout)
baselines/
  reference/                     original model + drop (0.6B and 1.7B)
  drop_kd_control/               drop + the same distillation as the merge methods
  laco/                          published LaCo baseline (Yang et al., 2024)
analysis/
  layer_analysis/                what each layer does, and scores for every pair of layers
  attention_heads/               previous-token and induction score of every head
  test_set/                      test tokens per language (>= 100k each)
tools/
  make_notebooks.py              builds the large-test notebooks (depth, model size, data and step ablations)
  kaggle_queue.py                runs them on Kaggle (CPU smoke test first, two notebooks per T4 x2 session)
  kaggle_run.sh                  push one notebook to Kaggle, check it, download its results
  analyze.py                     rebuild results/large_test_results.md from the runs
  make_figures.py                rebuild the result figures in docs/figures/
```

**Where each number comes from.** Every run has a notebook `<method folder>/<name>.ipynb` and its log and result
JSONs in `<method folder>/runs/<name>/`. Notebook names say what the run is:

| Name ends in | Run |
|---|---|
| `_large_test` | 2 and 4 layers removed, Qwen3-0.6B (seed 0, the main rows) |
| `_deep`, `_deep6`, `_deep8` | 6 and 8 layers removed |
| `_17b` | Qwen3-1.7B, 4 layers removed |
| `aam_icl`, `drop_kd_control_plain`, `prism_shrink_plain` | training text ablation (ICL-packed vs plain), 4 layers |
| `_steps1000` | 1,000 instead of 400 training steps |
| `_seed1`, `_seed2` | the same 4-layer run with another seed |
| no suffix (`aam`, `prism_shrink`, `prism_shrink_icl`, `laco`) | the small test (`results/small_test_results.md`) |

The small-test reference rows (`baselines/reference/runs/reference_small_test/`) were computed inside the LaCo run
(same file, see its log). `python3 tools/analyze.py . > results/large_test_results.md` rebuilds the large-test table
from the run folders alone.

## Setup

- **Calibration and training:** Wiki-40B train. **Tests** (never used for fitting): MMMLU 5-shot (MMLU for English),
  INCLUDE 5-shot, MGSM 4-shot chain of thought, Belebele 3-shot, XNLI 3-shot, mC4 and Wiki-40B perplexity, and an
  in-context copying probe.
- **Hardware:** Kaggle GPU T4 (Qwen3-1.7B: fp16 teacher, fp32 trained weights). Each notebook runs on its own. Example:
  `cd tools && NB_DIR=../aam ./kaggle_run.sh push aam_large_test`.
- **Rebuild the results table:** `python3 tools/analyze.py > results/large_test_results.md`; figures:
  `python3 tools/make_figures.py` (needs matplotlib).

## Prior work

Running consecutive layers in parallel was proposed before ("Leveraging the True Depth of LLMs", 2025). Layer removal
by angular distance: Gromov et al.; ShortGPT. Merging baseline: LaCo. **The closest prior work to PRISM-Shrink is
FlattenGPT (Depth Compression for Transformer with Layer Flattening, arXiv 2602.08858, 2026)**: it also turns two adjacent layers into one wide layer
(attention heads and MLP neurons of both, reading the same input) and prunes it back to the original width. It does so
in one step without training (head scores, ridge leverage scores and a closed-form correction), on English
benchmarks with 7B-70B models. Related: FuseGPT (folding removed blocks into their neighbours, 2024), LLM-Streamline
(a small trained module replaces removed layers, ICLR 2025), Sliding Layer Merging (2025).

What is added here:
1. **The gradual hand-over:** training starts from the exact original pair and fades the dropped heads and neurons
   out while distilling from the original model, instead of cutting in one step.
2. **ICL-packed training text**, which protects in-context copying.
3. **CoPlan:** choosing pairs by how exactly they run in parallel, not by similarity alone.
4. **The measurements:** six languages plus two unseen ones, a competence check, a matched drop + distillation
   control, paired tests and three seeds. They show that this family of methods cuts perplexity damage clearly
   (+16.6 % vs +24.2 % for the control) but does not improve few-shot accuracy over the control; keeping the better
   layer (AAM) is what improves accuracy.
