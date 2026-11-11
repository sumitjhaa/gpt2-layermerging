# Limitations, problems and everything that went wrong

This note lists every weakness of the results and every problem in the process, so nothing is hidden. Each point
says what happened, what was done about it, and what remains.

---

## 1. Weaknesses of the results

### 1.1 The compressed models are close to random guessing at 4+ layers (Qwen3-0.6B)
- Random guessing gives 21.7 % avg acc on these benchmarks. With 4 of 28 layers removed, the best method (AAM)
  scores 25.8 % and keeps only **about 21 %** of the original's above-chance skill; drop keeps 7 %.
- **Maths is lost almost completely**: MGSM falls from 41 % to about 2 % for every method at 4 layers.
- drop and the control score **below chance on Belebele** (about 20 % on a 4-choice test). A damaged model tends to
  pick one answer letter regardless of the question.
- With 2 layers removed, AAM keeps 53 %; on the 1.7B model at 4 layers it keeps 41 %. Small models have little
  spare depth.

Details and figure: [`metrics_and_benchmarks.md`](metrics_and_benchmarks.md) section 7.

### 1.2 PRISM-Shrink does not improve accuracy, and its core idea is published
- Its accuracy is **never significantly different from the matched control** (differences −1.0 to +1.0 points,
  p ≥ 0.07, at every depth, on both models and on all 3 seeds).
- Its wins are perplexity and copying only (from 4 layers removed on; at 2 layers its perplexity is about the same
  as AAM's).
- Building one wide layer from two adjacent layers and pruning it back was published as **FlattenGPT (2026)**. What is
  added here is the gradual, distilled hand-over, ICL-packed text and the multilingual controlled test.

### 1.3 AAM's ingredients are known
Removing layers, distilling from the original model and fading a layer out gradually are all known techniques. The
specific findings here are that **choosing the better layer of each pair** is what drives accuracy (+1.6 to +2.5
points over keeping the first one) and the careful comparison.

### 1.4 Quantization is much better for saving memory
Published results: 8-bit quantization halves memory with almost no loss on Qwen3; 4-bit saves 75 % with a few points
lost on an 8B model. Removing 4 layers here saves 11 % of the parameters and loses 23 points. Layer removal only makes
sense for speed (fewer sequential layers), possibly combined with quantization.
- **The quantization numbers come from the literature, not from this test.** No quantized model was evaluated here.

### 1.5 Speed gains are small and the speed numbers are noisy
Removing 4 of 28 layers made the 0.6B model roughly 10–20 % faster. Speed was measured once per run on shared Kaggle
T4 GPUs; the same model varies by about ±10 % between runs, so individual speed numbers should not be compared.

### 1.6 Only some results were repeated with several seeds
- **3 seeds:** the main Qwen3-0.6B runs at 4 layers removed (AAM, PRISM-Shrink + ICL, control). Accuracy varied by
  less than 0.7 points.
- **1 run only:** everything else (2, 6 and 8 layers removed, the 1.7B model, the text and steps ablations). Their
  confidence intervals cover variation in the test questions, not in training.
- The **copying loss varies a lot between seeds** (AAM 0.33 to 0.50), so single copying numbers are not reliable.

### 1.7 The training text is not identical across methods in the main 0.6B comparison
In the main 0.6B runs, AAM was trained on plain text, while the control and PRISM-Shrink used ICL-packed text. The
text ablation ran every combination at 4 layers: the text changes accuracy by at most 0.8 points, so it does not
explain AAM's lead. On the 1.7B model every method used plain text.

### 1.8 The test has limits
- One model family (Qwen3), two small sizes (0.6B, 1.7B).
- Six training languages; Japanese and Russian were tested only in some runs (the deep, ablation and seed runs) and
  not on the 1.7B model.
- 200–250 questions per benchmark and language; few-shot multiple choice plus one generation task (MGSM). No chat,
  instruction-following or long-text generation tests.
- German XNLI on the 0.6B model is not clearly above chance even for the original model (flagged ⚠ in
  `results/large_test_results.md`), so that cell cannot show damage.
- The scores of the original models differ from Qwen's official ones by up to 10 points (different languages,
  questions, prompts and number of MGSM examples). English MMLU, the most comparable cell, agrees within 1–4 points.

### 1.9 Choices that were not tuned
- Every method used the same training settings (learning rate 5e-5, 400 steps, the same loss). They were not tuned
  per method, so a method might do better with other settings.
- The large test used only the angular-distance pairs. Choosing pairs by CoPlan helped a lot in the small test, but
  AAM and PRISM-Shrink were not run on CoPlan pairs in the large test.

---

## 2. Problems during the runs, and how each was handled

| problem | what was done | what remains |
|---|---|---|
| PRISM-Shrink at 6 and 8 layers ran out of GPU memory (six double-width layers in fp32) | Re-run with smaller micro-batches (`mb_tokens=512`, same gradient) and one depth per GPU (`prism_shrink_icl_deep6`, `_deep8`) | The failed session `pp_deep_methods` is kept only for its AAM half (`aam_deep`) |
| The 1.7B reference run crashed: "expected mat1 and mat2 to have the same dtype, float != Half" (fp16 model, fp32 input to the output layer) | Patched the evaluation to cast to the model's own precision; re-run | None |
| PRISM-Shrink on 1.7B ran out of memory | Re-run with `mb_tokens=256` (`prism_shrink_17b`) | The failed session `pp_methods_17b` is kept only for its AAM half (`aam_17b`) |
| One Kaggle session (`pp_more_steps`) waited about 9.5 hours in Kaggle's queue | Pushed again; it ran normally | None |
| The download script saved the wrong log for some paired sessions | Fixed `tools/kaggle_run.sh`; every finished session was downloaded again | None |
| Results of the tiny CPU test runs ("smoke tests") were being counted in the tables | `tools/analyze.py` now ignores them | None |
| The seed runs and the 1.7B PRISM-Shrink run skipped the CPU smoke test | Seeds: the notebooks are identical to ones that already ran except the seed. 1.7B: the CPU test would take about 7 hours | All of them ran without errors |
| The notebook and log of the main control run (seed 0) and of LaCo were missing from the repository | Recovered from the Kaggle account; their results files are byte-identical to the committed ones | The control notebook is the exact script that ran, as one cell |

---

## 3. What is missing from the repository

- **Experiments 00–02, 04–10 and 14** (small test): only their numbers are kept (`results/small_test_results.md`,
  `docs/algorithm_notes.md`). Their notebooks and raw results were never committed.
- **Experiment 15 (PolyShrink)**: its notebook is only in git history (commit `f796be3`, folder `experimented/`),
  not in the current folders.
- **The attention-head and test-set analysis** (`analysis/attention_heads/`, `analysis/test_set/`): produced by the
  `p20a_data_refs` notebook, which is only in git history (commit `f796be3`, folder `paper/`). The layer-analysis
  notebook (experiment 00) was never committed; only its CSV files are kept.
- **SKM and XTC runs** (large test, 0.6B): numbers only, raw results not kept
  (`docs/results_comparison.md`, "Earlier ablations").
- **The small-test reference rows** (original model and drop) have no notebook of their own: they were computed
  inside the LaCo run (the file is identical; see `baselines/laco/runs/laco/run.log`).
- The generated notebooks mention `paper/make_notebooks.py` and `paper/README.md`: `paper/` was the old name of
  `tools/`, and `paper/README.md` no longer exists. The notebooks were left exactly as they ran.

---

## 4. Claims that changed as more evidence came in

| earlier belief | what the later runs showed |
|---|---|
| ICL-packed training text would recover few-shot accuracy | Not supported: it improves copying but not accuracy |
| ICL packing might hurt reading comprehension | Rejected by the text ablation |
| SKM and XTC (two new modules) would help PRISM-Shrink | No gain (numbers only, see section 3) |
| PRISM-Shrink is a fully new method | The core idea is published (FlattenGPT); see 1.2 |
| The small test (100 questions) numbers | Replaced by the large test (200–250 questions, confidence intervals); the small-test numbers are noisier (about ±4 points) |
