# Results on the small test (all experiments)

Model Qwen3-0.6B-Base; languages en, hi, fr, es, de, zh. All Kaggle runs: GPU T4, 2026-11-04/05. Test items per
language: 100 per benchmark (MGSM 40), so differences under ~10 points on one benchmark (~4 on avg acc) are noise.
Accuracy = mean over languages, in %. "ppl +%" = increase over the original model (lower is better).

Raw runs are kept in the repository for AAM (`../aam/runs/aam`), PRISM-Shrink (`../prism_shrink/runs/prism_shrink`,
`../prism_shrink/runs/prism_shrink_icl`) and the reference rows (`../baselines/reference/runs/reference_small_test`); the other
experiments are kept as numbers only. AAM also ran on Colab with identical numbers.

## All Kaggle results (rows grouped by layers removed and by the pairs merged)
Pairs: 12-13; 14-15 and 10-11; 12-13; 14-15; 16-17 = angular distance (standard). 13-14; 23-24 and
13-14; 15-16; 21-22; 23-24 = CoPlan (notebook 15). 14-16 etc. = ACDC search (04, groups of up to 3 layers).
Compare methods only against the `drop` row with the same pairs.

| Removed | Pairs | Method | Params cut | mC4 ppl +% | worst lang +% | Wiki ppl +% | MMMLU | INCLUDE | MGSM | Belebele | XNLI | avg acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — | original | 0.0% | +0.0 | +0.0 | +0.0 | 48.0 | 40.8 | 38.0 | 67.2 | 47.7 | **48.3** |
| 2 | 12-13; 14-15 | drop | 5.3% | +15.1 | +20.3 | +14.0 | 35.8 | 33.6 | 7.5 | 33.5 | 42.7 | **30.6** |
| 2 | 12-13; 14-15 | AAM (03) | 5.3% | +6.8 | +9.0 | +4.5 | 36.5 | 31.8 | 15.0 | 40.7 | 47.3 | **34.3** |
| 2 | 12-13; 14-15 | Homotopy (06) | 5.3% | +8.3 | +12.7 | +4.8 | 33.7 | 32.6 | 9.0 | 25.5 | 46.3 | **29.4** |
| 2 | 12-13; 14-15 | PRISM-energy (07) | 5.3% | +9.2 | +14.8 | +5.7 | 30.5 | 31.0 | 6.0 | 25.5 | 43.8 | **27.4** |
| 2 | 12-13; 14-15 | parallel union, no param cut (07) | -0.0% | +7.7 | +9.8 | +7.8 | 39.3 | 35.4 | 8.0 | 41.7 | 44.0 | **33.7** |
| 2 | 12-13; 14-15 | PRISM-span (08) | 5.3% | +9.5 | +14.5 | +5.5 | 31.0 | 30.0 | 5.5 | 26.8 | 44.3 | **27.5** |
| 2 | 12-13; 14-15 | PRISM-polyglot (09) | 5.3% | +9.6 | +14.8 | +5.9 | 31.5 | 30.6 | 4.0 | 29.8 | 42.5 | **27.7** |
| 2 | 12-13; 14-15 | PRISM-Heal (10) | 5.3% | +8.0 | +12.3 | +4.5 | 30.3 | 32.6 | 8.0 | 29.5 | 47.2 | **29.5** |
| 2 | 12-13; 14-15 | PRISM-Shrink (11) | 5.3% | +6.9 | +10.4 | +4.8 | 31.5 | 32.6 | 9.0 | 35.3 | 45.3 | **30.8** |
| 2 | 12-13; 14-15 | LaCo, published baseline (12) | 5.3% | +12.6 | +15.1 | +12.6 | 35.2 | 31.6 | 10.0 | 33.5 | 46.3 | **31.3** |
| 2 | 12-13; 14-15 | Shrink + ICL packing (13) | 5.3% | +6.7 | +10.0 | +4.2 | 31.5 | 32.2 | 9.0 | 33.7 | 44.2 | **30.1** |
| 2 | 12-13; 14-15 | Shrink + LSR (14) | 5.2% | +7.0 | +10.3 | +4.9 | 32.5 | 31.2 | 11.0 | 33.3 | 45.8 | **30.8** |
| 2 | 12-13; 14-15 | Shrink + LSR, lr too small (14, first run) | 5.2% | +6.9 | +10.4 | +4.8 | 32.0 | 33.2 | 8.0 | 34.8 | 45.5 | **30.7** |
| 2 | 13-14; 23-24 | drop | 5.3% | +41.5 | +58.7 | +50.0 | 38.7 | 33.4 | 14.5 | 47.2 | 46.8 | **36.1** |
| 2 | 13-14; 23-24 | PolyShrink: CoPlan + Shrink + ICL + LSR (15) | 5.2% | +13.4 | +16.1 | +10.8 | 35.5 | 33.8 | 16.0 | 42.5 | 49.0 | **35.4** |
| 2 | 14-16 | drop | 5.3% | +30.1 | +49.0 | +30.2 | 38.5 | 34.2 | 6.5 | 44.7 | 41.8 | **33.1** |
| 2 | 14-16 | ACDC-Fusion search + SDCF (04) | 5.3% | +10.8 | +18.5 | +6.4 | 36.2 | 29.4 | 5.5 | 26.0 | 45.5 | **28.5** |
| 4 | 10-11; 12-13; 14-15; 16-17 | drop | 10.6% | +110.3 | +220.5 | +86.7 | 30.0 | 27.6 | 2.0 | 22.3 | 37.0 | **23.8** |
| 4 | 10-11; 12-13; 14-15; 16-17 | AAM (03) | 10.6% | +27.5 | +44.1 | +15.3 | 31.2 | 23.2 | 1.5 | 28.5 | 42.5 | **25.4** |
| 4 | 10-11; 12-13; 14-15; 16-17 | Homotopy (06) | 10.6% | +34.3 | +58.0 | +15.5 | 26.3 | 24.2 | 4.0 | 21.7 | 40.8 | **23.4** |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-energy (07) | 10.6% | +30.5 | +61.4 | +16.9 | 28.8 | 24.2 | 1.0 | 24.0 | 37.0 | **23.0** |
| 4 | 10-11; 12-13; 14-15; 16-17 | parallel union, no param cut (07) | -0.0% | +25.2 | +32.2 | +25.4 | 35.3 | 29.6 | 1.0 | 33.7 | 40.3 | **28.0** |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-span (08) | 10.6% | +29.8 | +59.9 | +15.4 | 25.0 | 24.6 | 2.5 | 21.7 | 37.5 | **22.3** |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-polyglot (09) | 10.6% | +31.8 | +67.9 | +16.8 | 27.5 | 25.4 | 2.5 | 22.7 | 35.0 | **22.6** |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-Heal (10) | 10.6% | +23.3 | +47.4 | +12.3 | 30.5 | 24.4 | 2.0 | 22.0 | 41.7 | **24.1** |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-Shrink (11) | 10.6% | +18.8 | +27.7 | +12.5 | 30.3 | 26.0 | 1.5 | 28.2 | 37.3 | **24.7** |
| 4 | 10-11; 12-13; 14-15; 16-17 | LaCo, published baseline (12) | 10.6% | +70.2 | +94.5 | +79.6 | 27.5 | 25.4 | 2.0 | 25.3 | 40.2 | **24.1** |
| 4 | 10-11; 12-13; 14-15; 16-17 | Shrink + ICL packing (13) | 10.6% | +17.9 | +26.9 | +11.3 | 28.7 | 27.4 | 0.5 | 25.0 | 37.3 | **23.8** |
| 4 | 10-11; 12-13; 14-15; 16-17 | Shrink + LSR (14) | 10.4% | +19.2 | +28.3 | +12.9 | 29.0 | 28.4 | 2.0 | 28.5 | 38.0 | **25.2** |
| 4 | 10-11; 12-13; 14-15; 16-17 | Shrink + LSR, lr too small (14, first run) | 10.4% | +19.0 | +28.7 | +12.6 | 31.3 | 26.0 | 0.5 | 27.7 | 37.7 | **24.6** |
| 4 | 11-13; 14-16 | drop | 10.6% | +47.6 | +62.7 | +50.2 | 31.0 | 26.8 | 1.5 | 25.3 | 37.5 | **24.4** |
| 4 | 11-13; 14-16 | ACDC-Fusion search + SDCF (04) | 10.6% | +21.9 | +36.7 | +13.3 | 21.8 | 27.6 | 4.0 | 22.7 | 42.7 | **23.8** |
| 4 | 13-14; 15-16; 21-22; 23-24 | drop | 10.6% | +196.7 | +288.2 | +236.4 | 34.2 | 31.4 | 2.5 | 38.5 | 41.5 | **29.6** |
| 4 | 13-14; 15-16; 21-22; 23-24 | PolyShrink: CoPlan + Shrink + ICL + LSR (15) | 10.4% | +28.3 | +45.3 | +19.5 | 32.0 | 27.4 | 3.5 | 26.8 | 42.7 | **26.5** |
| 6 | 6-7; 8-9; 11-13; 14-16 | drop | 15.8% | +113.5 | +142.7 | +134.3 | 27.3 | 24.8 | 2.0 | 24.2 | 38.5 | **23.4** |
| 6 | 6-7; 8-9; 11-13; 14-16 | ACDC-Fusion search + SDCF (04) | 15.8% | +47.4 | +77.0 | +31.6 | 25.2 | 26.2 | 1.5 | 21.3 | 36.7 | **22.2** |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | drop | 15.8% | +292.7 | +513.9 | +266.1 | 28.0 | 28.2 | 1.0 | 21.8 | 33.2 | **22.4** |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | Homotopy (06) | 15.8% | +69.4 | +108.7 | +43.5 | 30.0 | 24.4 | 1.0 | 21.5 | 37.7 | **22.9** |

## Findings
1. How you merge drives perplexity: at 4 layers removed, PRISM-Shrink (+ICL packing) +17.9 % vs AAM +27.5 %,
   LaCo +70.2 %, drop +110 % on the same pairs.
2. Which pairs you merge drives few-shot accuracy: CoPlan pairs give drop 36.1 / 29.6 avg acc vs 30.6 / 23.8 with
   angular-distance pairs (2 / 4 layers removed).
3. Perplexity and few-shot accuracy disagree (e.g. drop on CoPlan pairs: worst perplexity, best accuracy).
4. Negative results: ICL packing, the language-specific residual and the component-selection rule (energy / span /
   polyglot) do not change accuracy beyond noise; closed-form starts (refit, SDCF) lower it.
5. Proof of merging (07 run.log): for all 6 merged pairs PRISM + refit beats keeping either layer, and both one-layer
   knockouts are worse than the merged layer. A linear probe reads the language of the text from middle-layer
   states with 77-93 % held-out accuracy (14, 15 run.log).
