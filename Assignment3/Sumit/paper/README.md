# Paper runs: SKM + XTC, controls, baselines, seeds, on a larger test set

Generated from the archived notebooks in `../experimented/` by `make_notebooks.py`. Every shared cell is patched in
place (each patch asserts it applied), so the code is the same code that produced `experimented/`, plus what is
listed below. Edit the generator, not the notebooks, then run `python3 paper/make_notebooks.py`.

## The new methods
**Diagnosis (from `experimented/`).** A merged consecutive pair keeps the per-token computation (the parallel union
reaches 33.7 avg acc) but loses few-shot and chain-of-thought ability (MGSM 38 -> <= 15). Those depend on circuits
**across** the two layers. A parallel or merged layer cannot express two things:
1. **Same-position dependency.** B reads A's output. To first order, `h + A(h) + B(h + A(h)) = [h + A(h) + B(h)] + J_B(h)·A(h)`.
   **XTC (Cross-Term Compensation)** adds a rank-r bilinear module `C(h) = U[(P x) ⊙ (Q x)]`, `x = rms(h)`, beside the
   merged layer. It is fitted in closed form on (sequential − parallel) states, then trained. Cost: 3·r·d parameters
   (r = 64: ~1.3 % of one layer).
2. **Cross-position dependency.** A previous-token head in layer i feeds an induction head in layer i+1 (K-composition).
   **SKM (Smeared-Key Merging)** gives every KV group of the merged layer a learned key smearing
   `k_t <- (1 − s_g) k_t + s_g k_{t−1}` (Olsson et al., 2022 show this lets one layer form induction heads). Cost: 8
   scalars per merged layer. Works with the KV cache (the last raw key is kept) and with left padding.

Both run on top of PRISM-Shrink + ICL packing (notebook 13). `P20a` measures which pairs carry such circuits
(previous-token x induction head scores). `P20b` measures how much of the parallel-union gap a rank-r cross term
closes, per pair.

## What was added to every notebook
| Change | Why |
|---|---|
| Test set: MMMLU 200, INCLUDE 200, MGSM 250 (all), Belebele 200, XNLI 200 items; mC4 + Wiki 128 × 256 tokens, per language | >= 100k unique test tokens per language (counted in P20a: `test_tokens_*.csv`), and noise ~±1 avg acc instead of ±4 |
| Unseen languages ja, ru (evaluated, never calibrated or trained on) | do merges generalise beyond the calibration languages? |
| Per-item correctness saved (`eval.hits`) | bootstrap CIs and paired tests (`analyze.py`) |
| `copy_nll`: in-context copying probe on random tokens | direct measure of the induction ability SKM targets |
| `speed`: prefill and decode tokens/s on a T4 | real efficiency, incl. the overhead of LSR / XTC / SKM |
| `seed`, `tag`, `math_frac` settings | seeds; math-aware calibration ablation (GSM8K **train**; MGSM is built from GSM8K **test**) |

Calibration and training stay Wiki-40B train in the 6 languages (the same as before). No benchmark text is ever
used for fitting. Few-shot examples come from dev/validation/train splits, or from rows held out of evaluation.

## Notebooks (queue order)
| Notebook | Role | k removed |
|---|---|---|
| `p20a_data_refs` | builds the shared data; original + drop (angular pairs); token counts; head analysis | 2, 4 |
| `p20b_refs_coplan` | drop on the CoPlan pairs; XTC gap per pair (r = 16, 64, 256) | 2, 4 |
| `p18_skm_xtc` | **new method**: Shrink + ICL + SKM + XTC | 2, 4 |
| `p19_dropkd_icl` | **control**: keep one layer of each pair, same distillation (data, steps, schedule) | 2, 4 |
| `p13_shrink_icl` | ablation base (neither SKM nor XTC) | 2, 4 |
| `p16_skm`, `p17_xtc` | each component alone | 2, 4 |
| `p03_aam` | best previous method | 2, 4 |
| `p21_full_coplan` | SKM + XTC on CoPlan pairs (pair rule × method grid) | 2, 4 |
| `p15_polyshrink` | previous full method (CoPlan + Shrink + ICL + LSR) | 2, 4 |
| `p12_laco` | published baseline | 2, 4 |
| `p18_*_seed1/2`, `p19_*_seed1/2` | seeds of the method and the control | 2 |
| `p18_skm_xtc_math` | math-aware calibration ablation (30 % GSM8K-train text) | 2, 4 |
| `s_*` | CPU smoke tests of every distinct notebook (tiny settings, no GPU quota); run first | 1, 2 |

## Running
```
python3 paper/kaggle_queue.py            # resumable; state in paper/queue_state.json, results in ../paper_runs/
python3 paper/kaggle_queue.py --status
python3 paper/analyze.py > paper/RESULTS.md   # table with 95 % CIs and paired p-values vs drop
```
Budget (Kaggle T4): one evaluation takes ~30–40 min on the larger test set and one distillation ~25 min. The full
queue needs about 25 GPU-hours, which is close to Kaggle's weekly quota (~30 h). The queue is in priority order, so
if the quota runs out the method, the control and the ablations are already done.
