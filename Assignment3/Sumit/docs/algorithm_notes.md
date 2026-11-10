# Assignment 3 (Sumit) — Merging Consecutive Layers of a Multilingual LLM: Notes

Kept in the repository: the notebooks, logs and results of AAM and PRISM-Shrink (`../aam/`, `../prism_shrink/`) and
the baselines (`../baselines/`). Full results table of every experiment: `../results/small_test_results.md`. Runs were
pushed to Kaggle with `../tools/kaggle_run.sh`. Experiments are labelled 00–15 below (01, 02, 03, 05 also ran on Colab).

---

## 0. The idea in plain words

Qwen3-0.6B-Base has 28 decoder layers. Goal: fewer layers (2, 4 or 6 removed), with as little loss as possible in
**six languages** (en, hi, fr, es, de, zh), measured by perplexity **and** by few-shot tasks.

- **Naive:** delete one layer of a pair (`drop`), or average the two layers' weights (collapses; Assignment 1).
- **This work:** 15 notebooks that *merge* two (or three) consecutive layers into **one standard-size layer** that
  does the job of both, in four families:
  1. **Closed form** (no training): solve least-squares problems on calibration activations (01, 02, 05, 07–09).
  2. **Gradual distillation**: hand the work of the pair over to one layer slowly, with a KL loss to the original
     model (03, 06, 10, 11, 13, 14, 15).
  3. **Search**: decide *which* layers to merge by measuring the merged model (04, and CoPlan in 15).
  4. **Baseline / analysis**: LaCo (12), layer analysis (00).
- The central idea (**PRISM**): in the middle of the network, consecutive layers read almost the same normalized
  input, so the pair run **one after the other** is close to the pair run **side by side**. Side by side, the two
  layers are exactly **one wider layer**. That wider layer is then shrunk back to standard size.

**Headline (Kaggle T4, 4 layers removed, same pairs):** mC4 perplexity increase **+17.9 %** with PRISM-Shrink +
ICL packing vs **+27.5 %** AAM, **+70.2 %** LaCo (published baseline), **+110.3 %** drop.
Few-shot accuracy is decided mainly by **which pairs** are merged: choosing pairs by CoPlan lifts avg accuracy of
plain drop from **23.8 → 29.6** (4 removed) and **30.6 → 36.1** (2 removed).

---

## 1. Qwen3-0.6B-Base in matrices

| Item | Shape | Params |
|---|---|---|
| Token embedding (tied with LM head) | 151,936 × 1,024 | 155,582,464 |
| 28 decoder layers | — | 15,730,944 each |
| Final RMSNorm | 1,024 | 1,024 |
| **Total** | | **596,049,920** |

One decoder layer (pre-norm, hidden d = 1,024; 16 query heads, 8 KV heads, head_dim 128; MLP width 3,072):
```
a   = RMSNorm_in(h) · γ_in                         γ_in : 1024
q   = q_norm( a · W_qᵀ )   per head                W_q : 2048×1024   (16 heads × 128)
k   = k_norm( a · W_kᵀ )   per head                W_k : 1024×1024   (8 KV heads × 128)
v   = a · W_vᵀ                                     W_v : 1024×1024
q,k ← RoPE(q, k);  each KV head is shared by 2 query heads (grouped-query attention, REP = 2)
h'  = h + o_proj( softmax(q kᵀ/√128 + causal) v )   W_o : 1024×2048
b   = RMSNorm_post(h') · γ_post                     γ_post : 1024
h'' = h' + W_down ( SiLU(b W_gateᵀ) ⊙ (b W_upᵀ) )   W_gate, W_up : 3072×1024,  W_down : 1024×3072
```
Params per layer: q 2,097,152 + k 1,048,576 + v 1,048,576 + o 2,097,152 + gate/up/down 3 × 3,145,728 + norms
2,304 = **15,730,944** → one layer = **2.64 %** of the model. Removing 2 / 4 / 6 layers = **5.3 / 10.6 / 15.8 %**.

"KV group" (used by PRISM): one KV head + its 2 query heads = q rows 256, k rows 128, v rows 128, o columns 256.
A layer has 8 KV groups and 3,072 MLP neurons.

---

## 2. Data

### Calibration / training — Wiki-40B train only
- Per language: `n_train + n_calib = 316` chunks of **256 tokens**.
- **Calibration**: 16 sequences per language; position 0 (the attention-sink token) left out →
  6 × 16 × 255 = **24,480 calibration tokens**; the **last sequence of each language is held out** (1,530 tokens) and
  is used to measure every fit ("held-out mismatch / error").
- **Distillation**: the other 300 sequences per language; one sequence per language per step.

### Final tests (never used for any fitting or choice), per language

| Test | Shots | Items per language | Scoring |
|---|---|---|---|
| mC4 validation perplexity | — | 16 × 256 tokens | exp(mean NLL) |
| Wiki-40B test perplexity | — | 16 × 256 tokens | exp(mean NLL) |
| MMMLU (English: original MMLU) | 5 | 100 | next-token logit of " A/B/C/D" |
| INCLUDE (CohereLabs/include-base-44; no en) | 5 (validation split) | 100 | same |
| MGSM (no hi) | 4, chain of thought | 40 | greedy generation, last number must equal the answer |
| Belebele | 3 | 100 | next-token logit of " A/B/C/D" |
| XNLI | 3 (one per label) | 100 | most likely of the 3 completed sentences ("…, right? Yes/Also/No, …") |

- Shots never overlap evaluated items (MMMLU has no dev split: its first shuffled rows are the shots).
- Prompts are capped at 2,048 tokens (oldest shots cut first).
- **avg acc** = mean of the five benchmark accuracies (each a mean over languages). Chance: 25 % on the 4-way tasks,
  33 % on XNLI, ~0 % on MGSM.
- **Half precision check**: evaluation runs in fp16 on a T4 only if the original model's perplexity agrees with fp32
  within 2 % (measured: 24.582 fp32 vs 24.579 fp16).
- **Noise**: 100 items → about ±10 points (95 %) per benchmark and language; differences below ~4 points of avg acc
  are noise.

---

## 3. Shared machinery (every notebook)

1. **Hidden states.** `HS[i]` = the original model's state entering layer i on all calibration tokens (fp32, captured
   with forward pre-hooks). `capture(model, li, inputs, outputs)` records inputs/outputs of any sub-module of layer li.
2. **Ridge least squares.** `ridge(X, Y)` = argmin_W ‖XW − Y‖² + λ·mean(diag(XᵀX))·‖W‖², λ = 0.01, solved in
   float64.
3. **Held-out relative error.** `rel_err(P, T)` = ‖P − T‖ / ‖T‖ on the held-out rows; PRISM uses the per-token version
   `terr` = sqrt(mean over tokens of ‖p − t‖² / ‖t‖²), reported separately for body tokens and the sink token.
4. **Which pairs (default planner) — angular distance** (Gromov et al.):
   d(i) = mean over tokens of arccos(cos(HS[i], HS[i+2])) / π. Take the pairs with the smallest d, non-overlapping,
   never the first or last 2 layers. Result: 2 removed → **12-13; 14-15**; 4 removed → **10-11; 12-13; 14-15; 16-17**;
   6 removed → 7-8; 10-11; 12-13; 14-15; 16-17; 24-25.
5. **Assemble.** Each group in the plan is replaced by one layer; all other layers are the original ones
   (shared, never copied). `drop` = keep the first layer of each group.
6. **Results.** Every notebook first evaluates `original` and `drop` (same pairs), then its own method, and writes
   one JSON row per (method, layers removed, plan). `table()` merges all JSON files.

### Shared distillation (`kd_train`, used by 03, 06, 10, 11, 13, 14, 15)
- **Frozen**: the original model (teacher) and every non-merged layer. **Trained**: only the merged layer(s).
- **Per step**: one 256-token sequence per language (6 sequences) — language-balanced.
- **Loss** = KL + 1.0 · hidden MSE:
  - KL(teacher ‖ student) on next-token distributions over the full 151,936-token vocabulary, summed over tokens /
    number of tokens; logits computed in 512-row chunks.
  - Hidden MSE at the output of each merged group: mean((h_student − h_teacher)²) / mean(h_teacher²), position 0
    excluded (normalized because hidden norms grow with depth).
- **Annealing (ALPHA)**: the old computation fades out, α(step) = ½(1 + cos(π·(step+1)/A)), A = 0.7 × 400 = 280 steps,
  then α = 0 for the last 120 steps. α < 1 already at step 0 so the merged layer gets gradient.
- **Exactness check**: at α = 1 the student must equal the original model (asserted: KL < 1e-3 and hidden < 1e-3).
- **Optimizer**: AdamW, lr 5e-5, no weight decay, gradient clip 1.0, fp16 autocast + GradScaler on a T4,
  **400 steps**. Long sequences are split into micro-batches of ≤ 1,536 tokens (same gradient, fits 15 GB).

---

## 4. The 15 algorithms, one by one

Notation: a pair of consecutive layers **A** (= layer i) and **B** (= layer i+1). h = state entering A. T = original
state after B (the target). "anchor" = the original layer that is copied and then modified.

### 00 · Layer analysis (no merging; explains where merging is safe)
For each layer it measures, on calibration text:
1. **Angular change** between its input and output.
2. **Write size**: ‖attention output‖ / ‖h‖ and ‖MLP output‖ / ‖h‖.
3. **Update cosine** with the next layer's update (do consecutive layers push the same way?).
4. **Language identity**: share of hidden-state variance explained by which language the text is in.
5. **Cross-lingual retrieval**: the same Belebele passage in 6 languages; can a layer's mean state find its
   translation (nearest neighbour)? High = language-neutral representation.
6. **Logit lens**: top-1 next-token accuracy if the model stopped after this layer.
7. Per pair: angular distance and **pair linearity R²** (how well one linear map predicts the pair's total update).

**What it showed (Kaggle):** layer 0 is special (angular change 0.48, writes 5–7× ‖h‖). Layers 12–14 change the
state least (angular change 0.075–0.083) and write least. Cross-lingual retrieval peaks in the middle (0.91–0.92 for
layers 12–18): the middle is a **language-neutral** concept space. Language identity rises sharply after layer 18
(0.05 → 0.18 at layer 27): late layers become language-specific again. The logit lens is near 0 until layer ~18,
then climbs to 0.39 at layer 27. Best linear pairs: 15-16 (R² 0.65), 11-12 (0.61), 10-11 and 16-17 (0.59).

### 01 · CD-RegMean + W-refit + Step (closed form; best of the old all-in-one code)
1. **CD-RegMean merge.** For every linear map (q, k, v, o, gate, up, down), find one weight W that, fed with A's
   input X_A, reproduces the **average** of the two layers' outputs on their own inputs:
   W = ridge(X_A, ½(X_A W_Aᵀ + X_B W_Bᵀ)). Norm gains: A's.
2. **W-refit.** The layer's output is linear in down_proj, so solve down_proj by least squares so that the merged
   layer's output lands on T. Kept only if the held-out error improves.
3. **Step.** A per-channel step size s on the layer's update: output = h + s ⊙ (layer(h) − h), s fitted by least
   squares (clamped to [0, 3·|g|]); position 0 untouched. Kept only if it helps (it never did).

Printed: held-out mismatch per pair, e.g. 47.6 % → 34.0 % (10-11) after W-refit.
**Results (Colab):** 2 removed +17.7 % ppl, avg acc 24.0 (drop 30.6); 4 removed +64.3 % / 22.2; 6 removed
+212.7 % / 22.5 (drop +292.7 %). Better than drop on perplexity at 4–6 layers, **worse on few-shot**.

### 02 · SDCF — Sequential Drift-Compensated Fusion (closed form, new)
Problem it fixes: when B is removed, A's output is no longer what the next layer expects, and every later fit sees a
**drifted** input. SDCF re-solves each linear map **in forward order**, each one seeing the real, already-changed
input (GPTQ-style):
1. Copy an anchor layer (A or B) into the student at the pair's position.
2. **q, k, v**: refit so they reproduce the anchor's original q/k/v from the student's *current* input.
3. **o_proj**: refit so the attention writes = **sum of both layers' original attention writes**.
4. **gate, up**: refit to reproduce the anchor's original MLP inputs from the current input.
5. **down_proj**: refit so the layer's output lands exactly on T (the state after the pair).
6. Try both anchors; keep the best on held-out text, or the plain anchor if fusing does not help (failsafe).
7. Pairs are processed in forward order inside one student, so later pairs correct earlier errors.

Printed: e.g. group 10-11, anchor L11: held-out mismatch plain 38.4 % → fused 28.9 %.
**Results (Colab):** 2 removed +10.2 % / avg acc 25.7; 4 removed +39.5 % / 22.4; 6 removed +96.9 % / 22.3.
Perplexity much better than drop, **few-shot worse than drop**.

### 03 · AAM — Annealed Absorption Merging (gradual distillation, new)
1. For each pair, choose the layer to keep: the one with the lower held-out error when kept alone
   (e.g. 14-15: L14 28.9 %, L15 26.6 % → keep L15).
2. The other layer becomes a `Fade` module: output = h + α · (layer(h) − h). At α = 1 it is the original layer,
   at α = 0 it is gone.
3. Distill (shared recipe, §3): only the kept layer trains, α goes 1 → 0 over 280 steps, then 120 steps at α = 0.
4. Remove the faded layers: one layer per pair.

**Results (Kaggle = Colab exactly):** 2 removed **+6.8 % / avg acc 34.3** (best few-shot of all standard-size
methods on angular pairs); 4 removed +27.5 % / 25.4; 6 removed (Colab) +61.9 % / 23.0.

### 04 · ACDC-Fusion search (search + SDCF, new)
ACDC-style sweep, but candidates are **fused** (SDCF), not deleted:
1. Start with every layer as its own node. Sweep from the output towards the input (protecting the first/last 2).
2. Try to fuse node p with node p−1 (a node may grow to at most 3 original layers). Fast fit: best plain anchor,
   then SDCF on 4 calibration sequences per language.
3. Score: KL(original ‖ candidate) per token on 2 probe sequences per language (never used for fitting);
   criterion `worst` = the **worst language's** KL (polyglot).
4. Accept if the KL rises by less than τ (τ₀ = 0.01). If a sweep cannot reach the next budget (2, 4, 6 layers
   removed), τ doubles and the sweep repeats. Snapshot the groups at each budget.
5. Final model: SDCF on the discovered groups with all calibration text.

**Found groups:** 2 removed → 14-16 (one group of 3); 4 → 11-13; 14-16; 6 → 6-7; 8-9; 11-13; 14-16.
**Results (vs drop on the same groups):** 2 removed +10.8 % vs +30.1 %, avg acc 28.5 vs 33.1; 4 removed +21.9 % vs
+47.6 %, 23.8 vs 24.4; 6 removed +47.4 % vs +113.5 %, 22.2 vs 23.4. Big perplexity gain, few-shot slightly lower
(the SDCF pattern).

### 05 · NUF — Neuron-Union Fusion (closed form, new)
Same as SDCF (02), but the MLP keeps **the most used neurons of both layers** instead of the anchor's:
1. Score every neuron of A and B: mean |activation| × ‖its down_proj column‖.
2. Keep the top 3,072 of the 6,144 (e.g. L12 1,527 + L13 1,545).
3. Refit each kept neuron's gate/up so it fires as it did in its own layer, from the merged layer's input; then
   down_proj lands on T.

**Results (Colab):** 2 removed +9.0 % / avg acc 26.5; 4 removed +26.6 % / 23.2; 6 removed +67.9 % / 24.4. Best
closed-form perplexity of 01–05, still below drop on few-shot at 2 layers.

### 06 · Homotopy distillation (SDCF start + gradual hand-over, new)
1. Build the SDCF layer M for each pair (notebook 02).
2. `Blend`: group output = α · B(A(h)) + (1 − α) · M(h). At α = 1 it is the original model exactly (asserted).
3. Distill (shared recipe); only M trains; α 1 → 0. At the end the pair is replaced by M.

**Results:** 2 removed +8.3 % / 29.4; 4 removed +34.3 % / 23.4; 6 removed +69.4 % / 22.9. Worse than AAM: the
closed-form start costs few-shot accuracy that training does not recover.

### PRISM — the shared core of 07–11, 13–15
**Parallel-Residual Interaction-Sum Merge.**
```
sequential (original):  h + A(h) + B(h + A(h))
parallel (PRISM):       h + A(h) + B(h)        ← one layer with A's heads + B's heads, A's neurons + B's neurons
```
The gap is the interaction B(h + A(h)) − B(h); it is small when A and B read nearly the same normalized input.

**Exact parallel form (`prism_layer`)**:
1. Fold each layer's input norm gain into its own read weights: W_q ← W_q · diag(γ_in) (same for k, v, and γ_post
   into gate, up); the merged layer's norm gains are set to 1. Exact, because RMSNorm only rescales each token.
2. Each KV group keeps its **own q_norm / k_norm gains** (a per-head gain table, +5,632 params per merged layer
   = 0.001 %), because two layers' heads cannot share one gain vector.
3. Stack: q/k/v rows and o_proj columns of the chosen KV groups; gate/up rows and down_proj columns of the chosen
   neurons. With all parts of one layer, the construction reproduces that layer exactly (asserted in fp32:
   relative error < 1e-4).

**Commonality evidence** (notebook 07 table; random subspace overlap = 64/1024 = 0.06):

| Pair | Input cosine | Write-subspace overlap | Read-subspace overlap |
|---|---|---|---|
| 12-13 | 0.970 | 0.377 | 0.413 |
| 13-14 | 0.971 | 0.377 | 0.459 |
| 14-15 | 0.963 | 0.362 | 0.418 |

Write subspace = top-64 left singular vectors of [W_o | W_down]; read subspace = top-64 right singular vectors of
[W_q; W_k; W_v; W_gate; W_up]·diag(γ). Overlap = ‖U_Aᵀ U_B‖²_F / 64.

### 07 · PRISM-Energy (closed form, new) — also evaluates the parallel union
1. Build the full parallel union of the pair (16 KV groups, 6,144 neurons; 2× layer size).
2. **Select** 8 of 16 KV groups and 3,072 of 6,144 neurons by **write energy**: per token,
   ‖o_proj columns · head output‖² (KV groups) or activation² · ‖down column‖² (neurons), weighted by
   1/‖target‖² per token so every token's *relative* error counts ("sink-aware": position 0 included, not dominating).
3. **Interaction refit**: o_proj by least squares to both layers' real attention writes; down_proj so the output
   lands on T.
4. **Proof of merge** printed per pair (held-out error, body tokens):

| Pair (2 removed) | drop A | drop B | parallel union | PRISM no refit | PRISM + refit | only A's parts | only B's parts |
|---|---|---|---|---|---|---|---|
| 12-13 | 23.3 % | 23.6 % | 19.1 % | 24.5 % | **16.6 %** | 25.6 % | 24.3 % |
| 14-15 | 32.2 % | 30.0 % | 27.0 % | 28.6 % | **21.9 %** | 32.8 % | 30.1 % |

   Both knockouts are worse than the merged layer → the merged layer uses **both** layers. Same for all 6 pairs at
   4 removed (e.g. 16-17: drop 41–42 % → 28.3 %).
5. Also evaluates the **uncompressed parallel union** as its own row (no parameters removed, 1 layer less depth per
   pair).

**Results:** PRISM-energy 2 removed +9.2 % / 27.4; 4 removed +30.5 % / 23.0. **Parallel union: 2 removed +7.7 % /
33.7; 4 removed +25.2 % / 28.0** — the best few-shot of any method on angular pairs at 4 layers.

### 08 · PRISM-Span (closed form, new)
As 07, but selection is **redundancy-aware** (energy picks near-duplicates twice):
- Neurons: **pivoted Cholesky** on the write-weighted activation Gram matrix (column-subset selection). Each step
  takes the neuron with the most energy not already explained by the kept ones.
- KV groups: greedy, each step adds the group that most improves the best linear reconstruction of both layers'
  attention writes (computed exactly from the Gram matrix).

**Results:** 2 removed +9.5 % / 27.5; 4 removed +29.8 % / 22.3. Same as energy within noise.

### 09 · PRISM-Polyglot (closed form, new)
As 07, but each component's score = its **largest share of any single language's** write energy (per language,
normalized), so a component survives if *some* language depends on it.

**Results:** 2 removed +9.6 % / 27.7; 4 removed +31.8 % / 22.6; worst language still zh (+14.8 % / +67.9 %).
Language-fair selection does **not** protect the worst language.

### 10 · PRISM-Heal (PRISM start + gradual hand-over, new)
1. PRISM-span + refit layer M (notebook 08).
2. `Blend` α · B(A(h)) + (1 − α) · M(h); distill (shared recipe); only M trains.

**Results:** 2 removed +8.0 % / 29.5; 4 removed +23.3 % / 24.1. Better perplexity than AAM at 4 layers, worse
few-shot — again the closed-form (refit) start hurts few-shot.

### 11 · PRISM-Shrink (gradual, new — the core method)
Never cut in one step; start from the **exact parallel union** and fade out what will be dropped:
1. Build the parallel union U of the pair; choose the parts to keep (energy, as 07; no refit).
2. `Shrink` module: group output = α · B(A(h)) + (1 − α) · U_α(h), where U_α is the union with every **discarded**
   KV group and neuron scaled by α (hooks on o_proj and down_proj inputs). α = 1: the original model;
   α = 0: only the kept parts.
3. Distill (shared recipe); the whole union trains.
4. **Extract** the kept parts into a normal layer (8 KV groups, 3,072 neurons). Checked: error of the trained union at
   α = 0 = error of the extracted layer (17.7 % = 17.7 % for 12-13; 22.6 % = 22.6 % for 14-15).

**Results:** 2 removed +6.9 % / 30.8; **4 removed +18.8 % / 24.7** — the best perplexity at standard size
(AAM +27.5 %). Few-shot ≈ drop.

### 12 · LaCo (published baseline, Yang et al., 2024)
θ_merged = θ_A + Σ_{j ≠ A} (θ_j − θ_A) for every parameter (anchor + sum of the other layers' differences). No data,
no training.

**Results:** 2 removed +12.6 % / 31.3; 4 removed +70.2 % / 24.1. PRISM-Shrink has ~3.7× less perplexity damage
at 4 layers.

### 13 · Shrink + ICL-preserving packing (new)
Hypothesis: few-shot breaks because the merge is trained only on short plain text, which never exercises the
long-range copying few-shot prompts need.
1. As 11, but every distillation sequence is **[a, b, a]**: a 256-token chunk in language l, a chunk in another
   language, then the first chunk **again** (768 tokens). On the repeat the original model copies almost exactly, so
   distilling it trains in-context copying; the language switch trains cross-lingual context.
2. 6 sequences × 768 tokens per step, as 3 micro-batches of 2 sequences.

**Results:** 2 removed +6.7 % / 30.1; **4 removed +17.9 % / 23.8** — best perplexity of all, no few-shot gain.
Hypothesis **not supported** (negative result).

### 14 · Shrink + Language-Specific Residual, LSR (new)
Hypothesis: one shared layer must serve 6 languages; what merging loses is language-specific.
1. Fit a **linear language probe** on the RMS-normalized state entering the pair (6-way logistic regression, 300
   Adam steps). Held-out accuracy **77–93 %** → the language is readable from the residual stream.
2. Merged layer + LSR: output = layer(h) + Σ_l p_l(h) · B_l A_l · rms(h), where p = softmax(probe(h)) (frozen, soft,
   per token), A_l: 16×1024, B_l: 1024×16 (B starts at 0 → no change at init). 196,608 params per merged layer
   (0.03 % of the model).
3. Train with Shrink (11); the LSR uses its own learning rate 1e-3 (a zero-initialised adapter needs a larger step).

**Results:** with lr 1e-3: 2 removed +7.0 % / 30.8; 4 removed +19.2 % / 25.2. First run (lr 5e-5, adapter barely
moved): +6.9 % / 30.7 and +19.0 % / 24.6. No effect beyond noise either way (negative result).

### 15 · PolyShrink — the full method (CoPlan + Shrink + ICL packing + LSR, new)
1. **CoPlan** (which pairs): build the exact parallel union of every adjacent pair and measure its held-out error in
   the original model; merge the pairs where the union is most exact (non-overlapping). Measured union errors:
   13-14 17.2 %, 23-24 18.4 %, 21-22 18.7 %, 22-23 19.0 %, 12-13 19.1 %, … , 2-3 82.1 %.
   Chosen: 2 removed → **13-14; 23-24**; 4 removed → **13-14; 15-16; 21-22; 23-24** (angular distance would take
   12-13; 14-15 and 10-11; …; 16-17).
2. PRISM-Shrink on those pairs (11), with ICL packing (13) and LSR (14).
3. `drop` is re-evaluated on the CoPlan pairs, so the comparison is on equal pairs.

**Results:**

| Removed | Pairs | Method | mC4 ppl +% | worst lang +% | avg acc |
|---|---|---|---|---|---|
| 2 | CoPlan | drop | +41.5 | +58.7 | **36.1** |
| 2 | CoPlan | PolyShrink | **+13.4** | **+16.1** | 35.4 |
| 4 | CoPlan | drop | +196.7 | +288.2 | **29.6** |
| 4 | CoPlan | PolyShrink | **+28.3** | **+45.3** | 26.5 |

Note: this run used the LSR learning rate 5e-5 (the fix came after), so its LSR part was effectively off.

---

## 5. Final results

**Large test (the numbers to cite):** `../results/large_test_results.md` (every run, 95 % CIs, paired p-values) and
`results_comparison.md` (best method per goal). In short: AAM has the best accuracy at 2, 4, 6 and 8 layers removed
and on Qwen3-1.7B, significantly above drop and above the matched control; PRISM-Shrink has the lowest perplexity
damage and copying loss from 4 layers on, in all six languages, on both models, but its accuracy equals the control's.

**Small test** (Kaggle T4, angular-distance pairs; 100 items per benchmark; all 39 rows in
`../results/small_test_results.md`):

| Removed | Method | mC4 ppl +% | worst lang +% | avg acc |
|---|---|---|---|---|
| 0 | original | 0 | 0 | 48.3 |
| 2 | AAM (03) | **+6.8** | **+9.0** | **34.3** |
| 2 | parallel union, no param cut (07) | +7.7 | +9.8 | 33.7 |
| 2 | LaCo (12) | +12.6 | +15.1 | 31.3 |
| 2 | PRISM-Shrink (11) | +6.9 | +10.4 | 30.8 |
| 2 | Shrink + LSR (14) | +7.0 | +10.3 | 30.8 |
| 2 | drop | +15.1 | +20.3 | 30.6 |
| 2 | Shrink + ICL (13) | +6.7 | +10.0 | 30.1 |
| 2 | PRISM-Heal (10) | +8.0 | +12.3 | 29.5 |
| 2 | Homotopy (06) | +8.3 | +12.7 | 29.4 |
| 2 | PRISM-polyglot / span / energy (09 / 08 / 07) | +9.6 / +9.5 / +9.2 | ~+14.8 | 27.7 / 27.5 / 27.4 |
| 4 | parallel union, no param cut (07) | +25.2 | +32.2 | **28.0** |
| 4 | AAM (03) | +27.5 | +44.1 | 25.4 |
| 4 | Shrink + LSR (14) | +19.2 | +28.3 | 25.2 |
| 4 | PRISM-Shrink (11) | +18.8 | +27.7 | 24.7 |
| 4 | LaCo (12) | +70.2 | +94.5 | 24.1 |
| 4 | PRISM-Heal (10) | +23.3 | +47.4 | 24.1 |
| 4 | Shrink + ICL (13) | **+17.9** | **+26.9** | 23.8 |
| 4 | drop | +110.3 | +220.5 | 23.8 |
| 4 | Homotopy (06) | +34.3 | +58.0 | 23.4 |
| 4 | PRISM-energy / polyglot / span | +30.5 / +31.8 / +29.8 | +61 / +68 / +60 | 23.0 / 22.6 / 22.3 |

Colab-only rows (01, 02, 05; same settings): see sections 4.01, 4.02, 4.05.

---

## 6. Findings to explain

1. **How you merge drives perplexity.** At 4 layers: Shrink + ICL +17.9 %, Shrink +18.8 %, AAM +27.5 %, LaCo +70 %,
   drop +110 %.
2. **Which pairs you merge drives few-shot accuracy.** CoPlan pairs: drop 36.1 / 29.6 vs 30.6 / 23.8 on angular
   pairs. Angular distance picks pairs that change the state least, not pairs that are most parallel.
3. **Perplexity and few-shot accuracy disagree.** Drop on CoPlan pairs has the worst perplexity and the best
   accuracy; every closed-form method improves perplexity and lowers accuracy.
4. **Running a pair in parallel is nearly free.** The parallel union keeps the most few-shot ability at equal depth;
   the loss comes from cutting it to one layer's size.
5. **Closed-form starts hurt few-shot** (01, 02, 05, 06, 07–10), even with distillation afterwards (06, 10).
   Gradual methods starting from the original weights (03 AAM, 11 Shrink) do better.
6. **Negative results:** ICL packing (13), LSR (14), the selection rule (07 vs 08 vs 09), and the sink-aware fit do
   not change few-shot accuracy beyond noise.
7. **Multilingual:** the middle layers are language-neutral (cross-lingual retrieval 0.91–0.92), late layers are
   language-specific (language identity up to 0.18); the worst language is usually zh (sometimes hi); a linear probe
   reads the language with 77–93 % accuracy, yet a per-language residual does not recover the loss.

---

## 7. How the work evolved (and bugs fixed)

1. Old all-in-one `layer_merging.py` (~45 methods, Qwen2.5) → split into one notebook per algorithm on Qwen3; the
   best old method kept as 01; LaCo extracted as 12; the rest deleted.
2. 01–06 on Colab: closed-form fusion improved perplexity but lost few-shot; AAM was best.
3. **04 bug**: the search found groups but the evaluation used the default pairs (identical to 02). Fixed: the
   results cell passes the searched plans.
4. PRISM (07–10): local tests showed the parallel union keeps few-shot; cutting in one step (any rule, with or without
   refit, or moving attention budget into the MLP) loses it; the sink-aware hypothesis was rejected (sink error 0.1 %
   for every method).
5. **GPU self-check**: the lossless assert failed on the T4 at 3e-4 because evaluation runs in fp16; fixed by running
   the check in fp32.
6. PRISM-Shrink (11) designed from that finding; then 13, 14, 15 from its two remaining failure modes.
7. **Out of memory** (13, 15): packed sequences made each step 3× larger; fixed with micro-batches of ≤ 1,536 tokens.
8. **LSR learning rate**: at 5e-5 the zero-initialised adapter barely moved; fixed with its own lr 1e-3 (14 re-run;
   the first run kept as `14_shrink_lsr_lr5e-5_no_effect`).
9. All runs pushed from the terminal to Kaggle (GPU T4 x2) by `kaggle_run.sh`, two at a time, each downloaded with its
   log and results automatically.

---

## 8. Code map (every notebook has the same first cells)

| Cell | What |
|---|---|
| Install | transformers, datasets, pandas; GPU check |
| Settings `CFG` | model, `data_dir`, languages, `merge_counts`, protect = 2, seq 256, n_calib 16, test sizes, shots, ridge 0.01 (+ training / PRISM / Shrink settings where used) |
| Data | Wiki-40B, mC4, MMMLU/MMLU, INCLUDE, MGSM, Belebele, XNLI; cached in `data_dir` |
| Few-shot evaluation | `ppl`, `letter_acc`, `mc_eval`, `bele_eval`, `xnli_eval`, `mgsm_eval`, `evaluate` |
| Model + calibration | `teacher`, fp16 check, `capture`, `HS`, `set_calib`, `ridge`, `rel_err`, `angular`, `plan_pairs`, `assemble`, `student_index` |
| Run library | `record`, `run_all` (original + drop + method per merge count), `table` |
| SDCF library (02, 04, 05, 06) | `solve_linear`, `union_mlp`, `fuse`, `sdcf_build` |
| Distillation library (03, 06, 10, 11, 13–15) | `kd_losses`, `kd_train` |
| PRISM library (07–11, 13–15) | `prism_layer`, `_selfcheck`, `components`, `select`, `pivoted_cholesky`, `knockout`, `residual_commonality`, `projection_commonality`, `prism_build` |
| Shrink library (13–15) | `LangDelta`, `LSRLayer`, `fit_probe`, `packed_batch_fn`, `coplan`, `Shrink`, `extract`, `shrink_build` |
| Algorithm | the notebook's own `ALGO` and `build(plan)` |
| Results | `run_all(...)`, then the comparison tables |

---

## 8b. Comparison with Amrita's learned merger (Assignment 2, Experiment A)

### What can and cannot be compared
| | Amrita (Assignment 2) | This work (Assignment 3) |
|---|---|---|
| Model | GPT-2 Small, 124 M params, 12 blocks, English | Qwen3-0.6B-Base, 596 M params, 28 layers, 6 languages |
| One block / layer | 5.70 % of the model | 2.64 % of the model |
| Method type | one learned hypernetwork (39.4 M params) writes Δ for block B | 15 methods: closed form, gradual distillation, search |
| Training data used | 5,200 steps × 16 × 256 ≈ **21 M tokens** | AAM / Shrink: 400 × 6 × 256 ≈ **0.6 M tokens** (35× less); closed-form methods: none |
| Wikipedia test | WikiText-2 validation (English) | Wiki-40B test (6 languages, mean) |
| Few-shot tasks | **none** (perplexity + next-token accuracy only) | MMMLU, INCLUDE, MGSM, Belebele, XNLI |

Absolute perplexities cannot be compared (different models, tokenizers and test text). The fair number is the
**share of drop's damage that a method repairs**, at a similar parameter cut:

repaired = 1 − (method's ppl increase) / (drop's ppl increase), same model, same test, same layers removed.

Amrita's drop baseline = drop-first (keep B, delete A); ours = keep the first layer of each group. Both are
"delete one layer of the pair".

### Amrita's numbers (from her notes, WikiText-2 where available)
| k | Param cut | Learned ppl +% | Drop-first ppl +% | Repaired |
|---|---|---|---|---|
| 1 | 5.70 % | +3.5 | +4.8 | **27 %** |
| 2 | 11.39 % | +12.5 (no drop number on WikiText-2) | — | 19 % (mean of DBpedia / AG News / IMDb: +16.3 vs +20.2) |
| 3 | 17.09 % | +24.2 | +59.4 | **59 %** |
| 4 | 22.78 % | +42.1 (no drop number on WikiText-2) | — | 32 % (mean of 3 other sets: +60.8 vs +89.1) |
| 5 | 28.48 % | +61.1 | +235.0 | **74 %** |

### Every algorithm vs Amrita, at a similar parameter cut (Wikipedia test, share of drop damage repaired)
Matching: our 2 layers (5.3 %) ↔ her k = 1 (5.7 %); our 4 layers (10.6 %) ↔ her k = 2 (11.4 %);
our 6 layers (15.8 %) ↔ her k = 3 (17.1 %).

| # | Algorithm | ~5.5 % cut: ours (ppl +%) | Amrita k=1: **27 %** | ~11 % cut: ours (ppl +%) | Amrita k=2: **19 %**¹ | ~16–17 % cut: ours (ppl +%) | Amrita k=3: **59 %** |
|---|---|---|---|---|---|---|---|
| 01 | CD-RegMean + W-refit + Step | 26 % (+10.4) | ≈ equal | 60 % (+34.7) | better | 49 % (+134.6) | worse |
| 02 | SDCF | 58 % (+5.9) | better | 76 % (+20.7) | better | 76 % (+64.1) | better |
| 03 | AAM | 68 % (+4.5) | better | 82 % (+15.3) | better | 84 % (+42.5) | better |
| 04 | ACDC-Fusion (own groups) | 79 % (+6.4) | better | 74 % (+13.3) | better | 77 % (+31.6) | better |
| 05 | NUF | 61 % (+5.5) | better | 82 % (+16.0) | better | 83 % (+45.9) | better |
| 06 | Homotopy | 66 % (+4.8) | better | 82 % (+15.5) | better | 84 % (+43.5) | better |
| 07 | PRISM-energy | 59 % (+5.7) | better | 81 % (+16.9) | better | not run | — |
| 07 | parallel union (no param cut)² | 44 % (+7.8) | — | 71 % (+25.4) | — | not run | — |
| 08 | PRISM-span | 61 % (+5.5) | better | 82 % (+15.4) | better | not run | — |
| 09 | PRISM-polyglot | 58 % (+5.9) | better | 81 % (+16.8) | better | not run | — |
| 10 | PRISM-Heal | 68 % (+4.5) | better | 86 % (+12.3) | better | not run | — |
| 11 | PRISM-Shrink | 66 % (+4.8) | better | 86 % (+12.5) | better | not run | — |
| 12 | LaCo (published baseline) | 10 % (+12.6) | worse | 8 % (+79.6) | worse | not run | — |
| 13 | Shrink + ICL packing | **70 %** (**+4.2**) | better | **87 %** (**+11.3**) | better | not run | — |
| 14 | Shrink + LSR | 65 % (+4.9) | better | 85 % (+12.9) | better | not run | — |
| 15 | PolyShrink (CoPlan pairs) | 78 % (+10.8) | better | 92 % (+19.5) | better | not run | — |

¹ Amrita has no WikiText-2 drop number at k = 2; 19 % is her mean over DBpedia, AG News and IMDb.
² The parallel union removes no parameters (only depth), so it is not a like-for-like row.
04 and 15 use their own layer groups, so they are compared with drop on *those* groups.

### Absolute numbers (same caveat: different models)
| Param cut | Amrita best (WikiText-2 ppl +%) | Ours best (Wiki-40B ppl +%) |
|---|---|---|
| ~5.5 % | +3.5 (k = 1) | +4.2 (Shrink + ICL), +4.5 (AAM, PRISM-Heal) |
| ~11 % | +12.5 (k = 2) | **+11.3** (Shrink + ICL), +12.3 (PRISM-Heal), +12.5 (Shrink) |
| ~16–17 % | **+24.2** (k = 3) | +31.6 (ACDC-Fusion), +42.5 (AAM) |

Why the absolute numbers are not a fair race: removing layers hurts Qwen3 far more than GPT-2. Plain drop costs
+14.0 % on Qwen3 at 5.3 % cut vs +4.8 % on GPT-2 at 5.7 %; +266 % vs +59 % at ~16–17 %.

### Verdict
- **Relative to drop (the fair comparison): yes, better.** 13 of the 15 algorithms repair a larger share of drop's
  damage than Amrita's merger at every matched size: 58–79 % vs her 27 % at ~5.5 % cut, 74–92 % vs her 19 % at
  ~11 %, 76–84 % vs her 59 % at ~16–17 %. Exceptions: LaCo (8–10 %, a baseline), and CD-RegMean at 2 and 6 layers.
- **With 35× less training data** (AAM, Shrink), and with **no training at all** for SDCF, NUF and PRISM-energy/
  span/polyglot, which still repair more than her trained merger.
- **In absolute perplexity: mixed.** About equal at ~5.5 % and ~11 %; hers is lower at ~17 % — but Qwen3 is a much
  harder model to cut.
- **Beyond Amrita:** this work also measures few-shot accuracy in 6 languages, and shows that perplexity hides
  large few-shot losses; Amrita's results cannot say whether her merged GPT-2 kept its task ability.
- **Not tested:** Amrita's merger on Qwen3, or ours on GPT-2 — the only way to a head-to-head comparison.

---

## 9. Assessment — was it good?

- **Strong:** merging is proven (every merged pair beats both single layers; knockouts); PRISM-Shrink gives the best
  perplexity at standard size by a wide margin over AAM, LaCo and drop; CoPlan gives a clear few-shot gain from pair
  choice alone; Kaggle reproduced the Colab AAM numbers exactly.
- **Weak:** no method beats AAM on few-shot accuracy at the same pairs; LSR did not work and ICL packing only helps
  copying; 3 seeds only for the main 0.6B 4-layer rows (one elsewhere); AAM and Shrink were not run on the CoPlan pairs. (The large test fixed the small
  test size, the Qwen3-1.7B runs added a second model size, and 3 seeds showed the 4-layer results are stable.)
- **Prior work to cite:** running consecutive layers in parallel ("Leveraging the True Depth of LLMs", 2025),
  LaCo, ShortGPT, SLEB, Gromov et al. (angular distance), RegMean, GPTQ. Check each before citing.

---

## 10. Questions

1. *Why does averaging two layers fail?* Their MLP neurons are not aligned; averaging mixes unrelated key→value
   memories (Assignment 1). PRISM never averages: it keeps whole heads and neurons.
2. *Why can two consecutive layers run in parallel?* Pre-norm layers add small updates to a large residual stream;
   in the middle, inputs of consecutive layers have cosine 0.96–0.97, so B(h + A(h)) ≈ B(h).
3. *How do you know it merges and does not just drop?* Knockouts: keeping only A's or only B's parts is worse than
   the merged layer, for all 6 pairs; held-out error falls below both single layers.
4. *Why fold the norm gains?* RMSNorm(x)·γ followed by W equals RMSNorm(x) followed by W·diag(γ); folding lets
   both layers' parts share one gain-free norm, exactly.
5. *Why per-head q/k-norm gains?* Qwen3 normalizes q and k per head with a gain vector; heads from two layers need
   their own gains. Cost 0.001 % parameters.
6. *Why do closed-form fits improve perplexity but hurt few-shot?* They are fitted on short Wiki text to match hidden
   states; few-shot prompts are long and use different computations (copying across examples). We tested two
   explanations (attention sink, long-range copying) and both were rejected; the cause is still open.
7. *What is the α schedule for?* A sudden cut is a shock; α hands the work over gradually, starting from exactly the
   original model (asserted), like a continuation path.
8. *Why KL and hidden MSE together?* KL protects the output distribution; the hidden loss keeps the state after the
   merged pair close, so later layers see familiar input.
9. *What is CoPlan and why does it help?* It merges the pairs whose exact parallel union is most accurate; angular
   distance picks pairs that change little, which are not the ones that can be computed in parallel.
10. *Is it significant?* 100 items per benchmark gives about ±10 points per benchmark and language; the CoPlan gain
    (+5–6 avg acc) and the perplexity gains are larger than that, the few-shot differences between Shrink variants
    are not.
11. *Why did the LSR not help?* The probe reads the language well, but a rank-16 per-language correction does not
    capture what is lost; the loss seems to be shared rather than language-specific.
12. *Does it hold when more layers are removed?* Yes. At 6 and 8 layers AAM is still significantly above drop
    (+2.3, +2.5) and the control, and PRISM-Shrink's perplexity damage is the lowest (+39 % and +69 % vs drop's +237 %
    and +689 %), in all six languages.
13. *Does it hold on a bigger model?* On Qwen3-1.7B (4 layers removed) the ranking is the same: AAM 34.1 (+10.0 over
    drop), PRISM-Shrink 31.5 with the lowest perplexity (+22.3 %) and copying loss (0.26), control 30.9, drop 24.1.
14. *Why is AAM better than the control?* Because it keeps the better layer of each pair. With the training text held
    fixed this gives +1.6 to +2.5 points; changing only the text (plain vs ICL-packed) moves accuracy by at most 0.7.
15. *Why does PRISM-Shrink need smaller micro-batches deeper or bigger?* Each union is twice a layer, trained in fp32
    with Adam, so 6+ unions (0.6B) or 4 unions (1.7B) fill a 15 GB T4. Splitting each step into smaller micro-batches
    gives the same gradient with less memory.

---

## 11. One line to memorise

**Consecutive middle layers of a multilingual LLM are nearly parallel; merging them through their exact parallel
union and shrinking it gradually (PRISM-Shrink) gives the lowest perplexity and the best copying at every depth and
on two model sizes, while keeping the better layer of each pair (AAM) gives the best few-shot accuracy — and
perplexity alone would hide that difference.**
