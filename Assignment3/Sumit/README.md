# Assignment 3 (Sumit): merging consecutive layers of a multilingual LLM

**Repository contents.** Only the top 3 methods are kept here: AAM (`experimented/03_*`, `paper_runs/pp_p17_p03/p03_aam`),
PRISM-Shrink (`experimented/11_*`, `13_*`, `paper_runs/pp_p13_p16/p13_shrink_icl`) and PolyShrink + CoPlan
(`experimented/15_*`), plus the shared reference rows (`paper_runs/p20a_data_refs`). The other experiments live in the
gitignored `archive/` folder; their numbers are kept in `RESULTS_COMPARISON.md`, `experimented/README.md` and
`paper/RESULTS.md`. Full explanation: `LAYER_MERGING_EXPLAINED.md`.

Model `Qwen/Qwen3-0.6B-Base` (28 layers); languages en, hi, fr, es, de, zh. **Tests** (never used for fitting):
MMMLU 5-shot (MMLU for English), INCLUDE 5-shot, MGSM 4-shot chain of thought, Belebele 3-shot, XNLI 3-shot, mC4 and
Wiki-40B test perplexity. **Calibration / training:** Wiki-40B train. Every notebook runs on its own on Kaggle or Colab
(GPU T4) and ends with one table of every method whose results are in its `data_dir`.

Run from the terminal on Kaggle (GPU T4 x2): `./kaggle_run.sh push <notebook>`, then `./kaggle_run.sh status <notebook>`,
and `./kaggle_run.sh archive <notebook>` to store the finished run in `experimented/`.

## Already run: see `experimented/` (notebooks, logs, results, one combined table)
01 CD-RegMean, 02 SDCF, 03 AAM, 05 NUF (Colab); 07 PRISM-energy + parallel union, 11 PRISM-Shrink (Kaggle).

## The method: PolyShrink (notebook 15) and its ablations
**Thesis.** In a pre-norm transformer, consecutive middle layers read nearly the same normalized residual stream
(input cosine 0.93–0.97) and share read / write subspaces (overlap 6–7x random). So a pair run in sequence is close to
the same pair run **in parallel**, which is one wider layer. Measured on a T4: the parallel union keeps the most
few-shot ability of any method (avg acc 33.7 at 2 layers, 28.0 at 4; drop 30.6 / 23.8). PolyShrink turns that wider
layer back into one standard layer without losing what it keeps:

| Part | What it does | Notebook |
|---|---|---|
| PRISM exact parallel form | norm gains folded into read weights, per-head q/k-norm gains kept; lossless (asserted) | all |
| Shrink | the parts to drop fade out (alpha 1 -> 0) while the union is distilled; a standard layer is extracted (checked equal) | 11 (archived) |
| ICL-preserving packing | distillation sequences `[a, b, a]`: the repeat trains long-range copying, the language switch trains cross-lingual context | 13 |
| Language-specific residual (LSR) | shared merged layer + per-language rank-16 delta, routed per token by a frozen linear language probe | 14 |
| CoPlan | merge the pairs whose exact parallel union has the lowest error, not the lowest angular distance | 15 |
| **PolyShrink** | all of the above | **15** |

## Notebooks to run
| # | Notebook | Role | Approx. Kaggle T4 time |
|---|---|---|---|
| 15 | `15_polyshrink_full` | **full method** | ~1.5–2 h |
| 13 | `13_shrink_icl` | ablation: + ICL packing only | ~1.5 h |
| 14 | `14_shrink_lsr` | ablation: + LSR only | ~1.5 h |
| 12 | `12_laco_baseline` | published baseline (LaCo, Yang et al. 2024) | ~45 min |
| 10 | `10_prism_heal` | variant: PRISM start + annealed KD (running on Kaggle) | ~1.5 h |
| 08, 09 | `08_prism_span`, `09_prism_polyglot` | ablations of the selection rule (closed form) | ~1 h each |
| 06 | `06_homotopy_distillation` | older SDCF start + annealing | ~1.5 h |
| 04 | `04_acdc_fusion_search` | search for which layers to merge (slow) | ~2 h |
| 00 | `00_layer_analysis` | figures: what each layer does, per language | ~10 min |

Suggested order: `15` -> `13` and `14` (in parallel) -> `12` -> the rest. PRISM notebooks default to
`merge_counts=[2, 4]`. Each training notebook's distillation uses 400 steps (`steps` in the settings cell).

## Prior work to cite and position against
Running consecutive layers in parallel was proposed before ("Leveraging the True Depth of LLMs", 2025; verify before
citing). The contributions here are compressing that union back to one standard layer (Shrink), keeping few-shot
ability (ICL packing), keeping every language (LSR), choosing pairs by parallelizability (CoPlan), and the
measurements behind them. Not used (done before): plain drop / averaging (reference rows only), affine maps
(Assignment 1), ShortGPT / TerGRASP / SVD / plain KD (Assignment 2), Amrita's hypernetwork merger.
