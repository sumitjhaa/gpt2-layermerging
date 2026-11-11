# Baselines: what every method is compared against

A new method only means something next to simple alternatives. Three baselines are kept here:

| folder | baseline | why it is needed |
|---|---|---|
| `reference/` | **original** model (nothing removed) and **drop** (delete one layer of each pair) | the best case and the simplest cut |
| `drop_kd_control/` | **control** = drop + exactly the same training as the methods | shows how much of a method's gain comes from the method itself and how much just from training |
| `laco/` | **LaCo** (Yang et al., 2024), a published layer-merging method | a comparison with published work |

New to the terms? Read [`../docs/metrics_and_benchmarks.md`](../docs/metrics_and_benchmarks.md) first.
Accuracy below = few-shot avg acc (random guessing = 21.7 %); skill kept = share of the original's above-chance skill.

---

## 1. Reference: the original model and drop (`reference/`)

**Original:** the unchanged Qwen3 model. Everything is measured against it.

**Drop:** for each chosen pair of neighbouring layers, delete the second layer (keep the first). No data, no
training. It is the floor every method must beat.

```
original:  ... ─► [L12] ─► [L13] ─► [L14] ─► [L15] ─► ...
drop:      ... ─► [L12] ─────────► [L14] ─────────► ...        (L13 and L15 deleted)
```

The pairs are chosen by angular distance (how little the token state changes across the pair), the same pairs for
every method: 2 removed → 12-13, 14-15; 4 removed → 10-11, 12-13, 14-15, 16-17; 6 removed adds 7-8, 24-25; 8 removed
adds 5-6, 22-23.

| notebook | contents | model | layers removed | avg acc % | skill kept % | mC4 perplexity | copying |
|---|---|---|---|---|---|---|---|
| `reference_large_test` | original | 0.6B | 0 | 48.9 | 100 | — | 0.125 |
| `reference_large_test` | drop | 0.6B | 2 | 28.7 | 32 | +14.2 % | 0.320 |
| `reference_large_test`, `reference_deep_a` | drop (deep_a adds Japanese and Russian) | 0.6B | 4 | 22.6 | 7 | +84.7 % | 2.220 |
| `reference_deep_b` | drop | 0.6B | 6 | 21.5 | 1 | +236.6 % | 4.292 |
| `reference_deep_b` | drop | 0.6B | 8 | 20.9 | −2 | +689.4 % | 10.112 |
| `reference_17b_a` | original | 1.7B | 0 | 58.9 | 100 | — | 0.106 |
| `reference_17b_b` | drop | 1.7B | 4 | 24.1 | 9 | +70.5 % | 2.831 |

`runs/reference_small_test/` holds the reference rows of the earlier small test (original avg acc 48.3). It has no
notebook of its own: it was computed inside the LaCo run (identical file; see `laco/runs/laco/run.log`).

Each reference notebook also downloads and caches the test data, which the method notebooks then reuse.

## 2. The control: drop + the same training (`drop_kd_control/`)

Every method here is trained (distilled from the original model) after cutting. A trained method beating untrained
drop does not prove the method is good: the training alone might explain the gain. The control removes that doubt:

1. Delete the second layer of each pair, exactly like drop (keep the **first** layer).
2. Train the kept layers with **the same recipe** as AAM and PRISM-Shrink: same text, loss (KL to the original model's
   next-token probabilities + hidden-state MSE), learning rate, 400 steps, and the same fade-out of the removed layer.

In the code it is AAM's function with one switch changed: `keep_one_build(plan, anchor="first")` instead of
`anchor="best"`. So **AAM − control = the effect of choosing the better layer**, and
**PRISM-Shrink − control = the effect of merging instead of deleting**.

| notebook | what it tests | model | layers removed | avg acc % | skill kept % | mC4 perplexity | copying |
|---|---|---|---|---|---|---|---|
| `drop_kd_control_large_test` | main run (ICL-packed text) | 0.6B | 2 | 29.3 | 35 | +7.9 % | 0.172 |
| `drop_kd_control_large_test` | main run (ICL-packed text) | 0.6B | 4 | 23.5 | 13 | +24.5 % | 0.372 |
| `drop_kd_control_seed1` | same, seed 1 | 0.6B | 4 | 23.6 | 13 | +24.2 % | 0.401 |
| `drop_kd_control_seed2` | same, seed 2 | 0.6B | 4 | 23.5 | 13 | +24.0 % | 0.537 |
| `drop_kd_control_deep` | deeper cuts | 0.6B | 6 | 22.3 | 4 | +59.2 % | 0.506 |
| `drop_kd_control_deep` | deeper cuts | 0.6B | 8 | 21.6 | 1 | +103.8 % | 1.091 |
| `drop_kd_control_plain` | plain training text | 0.6B | 4 | 23.3 | 11 | +25.2 % | 0.579 |
| `drop_kd_control_17b` | bigger model (plain text) | 1.7B | 4 | 30.9 | 31 | +32.1 % | 1.150 |

What it shows:
- **Training alone repairs most of the perplexity damage** (+84.7 % → +24.5 % at 4 layers) but adds only about
  1 point of accuracy on 0.6B (not significant). On 1.7B it adds 6.8 points.
- **AAM beats the control everywhere** (+1.5 to +4.8 points on 0.6B, +3.2 on 1.7B, all significant), so choosing the
  better layer matters.
- **PRISM-Shrink equals the control on accuracy** and beats it on perplexity and copying.

`drop_kd_control_large_test.ipynb` is the exact script that ran on Kaggle (paired session `pp-p18-p19`), stored as one
cell; it was recovered from the Kaggle account later and its results file is byte-identical to the committed one.

## 3. LaCo, the published baseline (`laco/`)

**LaCo** (Layer Collapse, Yang et al., 2024) merges layers by weight arithmetic, with no data and no training:

```
merged weights = weights of layer A  +  Σ over the other layers j of (weights of j − weights of A)
```

For every weight matrix, start from the first layer and add how much each other layer differs from it.

It was run only in the **earlier small test** (100 questions per benchmark and language, Qwen3-0.6B), on the same
pairs as the other methods:

| layers removed | avg acc % | mC4 perplexity | for comparison (same small test) |
|---|---|---|---|
| 2 | 31.3 | +12.6 % | PRISM-Shrink +6.9 %, AAM +6.8 % |
| 4 | 24.1 | +70.2 % | PRISM-Shrink +18.8 %, AAM +27.5 %, drop +110.3 % |

LaCo damages perplexity less than drop (+70.2 % vs +110.3 % at 4 layers) but much more than the trained methods. It was not run on the
large test, so it has no confidence intervals. Notebook: `laco.ipynb` (the original Kaggle notebook), log and results
in `runs/laco/`.

All numbers per benchmark and language: [`../results/all_metrics.md`](../results/all_metrics.md). Known limits:
[`../docs/limitations_and_issues.md`](../docs/limitations_and_issues.md).
