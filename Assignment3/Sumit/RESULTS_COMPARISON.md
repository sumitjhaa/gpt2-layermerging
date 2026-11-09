# Sumit Assignment 3: result comparison and best experiment

Model Qwen3-0.6B-Base (28 layers, 596M params). Languages en, hi, fr, es, de, zh. Sources: every
`experimented/*/results/*.json` (avg accuracy recomputed from the JSONs, and it matches `experimented/README.md`), plus
the final tables of `experimented/colab_runs/01, 02, 05`, which are not in the README table.
**avg acc** = mean of MMMLU, INCLUDE, MGSM, Belebele and XNLI (%; higher is better). **mC4 ppl +%** = increase over the
original (lower is better). Noise: about ±4 points on avg acc (100 test items per benchmark).
Original model: avg acc **48.3**, mC4 ppl +0.0.

## Best experiments
| Goal | 2 layers removed (5.3% params) | 4 layers removed (10.6%) | 6 layers removed (15.8%) |
|---|---|---|---|
| Best avg acc, real param cut | **PolyShrink (15)** 35.4, but plain drop on the same CoPlan pairs scores 36.1 | **PolyShrink (15)** 26.5 (drop on the same pairs: 29.6) | **NUF (05)** 24.4 |
| Best avg acc, same angular pairs as drop | **AAM (03)** 34.3 (drop 30.6) | **AAM (03)** 25.4 (drop 23.8) | NUF (05) 24.4 (drop 22.4) |
| Lowest mC4 ppl | **Shrink + ICL (13)** +6.7, AAM +6.8, PRISM-Shrink +6.9 | **Shrink + ICL (13)** +17.9, PRISM-Shrink +18.8 | **ACDC (04)** +47.4 (other pairs); AAM +61.9 on the standard pairs |
| Best on both metrics | **AAM (03)**: 34.3 acc / +6.8 ppl | **Shrink + LSR (14)**: 25.2 / +19.2; AAM: 25.4 / +27.5 | AAM: 23.0 / +61.9 |

**Overall winner: AAM (notebook 03, annealed absorption).** At 2 layers removed it is the only merge method with a
clear accuracy gain over drop on the same pairs (+3.7, which is just at the noise level). It also has the second-lowest
perplexity. At 4 layers it ties for the best accuracy. The PRISM-Shrink family (11/13/14) has the lowest perplexity at 2 and
4 layers, but its accuracy matches drop.

**No param cut:** the parallel union (07) keeps more than any other method (33.7 / 28.0), but it does not remove
any parameters. It is a ceiling, not a valid compressed result.

## All results (compare each method only with `drop` on the same pairs)
| Removed | Pairs | Method | Params cut | mC4 ppl +% | worst lang +% | Wiki ppl +% | MMMLU | INCLUDE | MGSM | Belebele | XNLI | avg acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | — | original | 0.0% | +0.0 | +0.0 | +0.0 | 48.0 | 40.8 | 38.0 | 67.2 | 47.7 | **48.3** |
| 2 | 12-13; 14-15 | drop | 5.3% | +15.1 | +20.3 | +14.0 | 35.8 | 33.6 | 7.5 | 33.5 | 42.7 | 30.6 |
| 2 | 12-13; 14-15 | CD-RegMean (01) | 5.3% | +17.7 | +27.3 | +10.4 | 30.3 | 24.8 | 3.5 | 21.8 | 39.8 | 24.0 |
| 2 | 12-13; 14-15 | SDCF (02) | 5.3% | +10.2 | +15.6 | +5.9 | 30.0 | 29.2 | 3.5 | 22.8 | 43.2 | 25.7 |
| 2 | 12-13; 14-15 | **AAM (03)** | 5.3% | +6.8 | +9.0 | +4.5 | 36.5 | 31.8 | 15.0 | 40.7 | 47.3 | **34.3** |
| 2 | 12-13; 14-15 | NUF (05) | 5.3% | +9.0 | +13.3 | +5.5 | 29.8 | 28.2 | 7.0 | 23.7 | 43.8 | 26.5 |
| 2 | 12-13; 14-15 | Homotopy (06) | 5.3% | +8.3 | +12.7 | +4.8 | 33.7 | 32.6 | 9.0 | 25.5 | 46.3 | 29.4 |
| 2 | 12-13; 14-15 | PRISM-energy (07) | 5.3% | +9.2 | +14.8 | +5.7 | 30.5 | 31.0 | 6.0 | 25.5 | 43.8 | 27.4 |
| 2 | 12-13; 14-15 | parallel union, no cut (07) | 0.0% | +7.7 | +9.8 | +7.8 | 39.3 | 35.4 | 8.0 | 41.7 | 44.0 | 33.7 |
| 2 | 12-13; 14-15 | PRISM-span (08) | 5.3% | +9.5 | +14.5 | +5.5 | 31.0 | 30.0 | 5.5 | 26.8 | 44.3 | 27.5 |
| 2 | 12-13; 14-15 | PRISM-polyglot (09) | 5.3% | +9.6 | +14.8 | +5.9 | 31.5 | 30.6 | 4.0 | 29.8 | 42.5 | 27.7 |
| 2 | 12-13; 14-15 | PRISM-Heal (10) | 5.3% | +8.0 | +12.3 | +4.5 | 30.3 | 32.6 | 8.0 | 29.5 | 47.2 | 29.5 |
| 2 | 12-13; 14-15 | PRISM-Shrink (11) | 5.3% | +6.9 | +10.4 | +4.8 | 31.5 | 32.6 | 9.0 | 35.3 | 45.3 | 30.8 |
| 2 | 12-13; 14-15 | LaCo baseline (12) | 5.3% | +12.6 | +15.1 | +12.6 | 35.2 | 31.6 | 10.0 | 33.5 | 46.3 | 31.3 |
| 2 | 12-13; 14-15 | **Shrink + ICL (13)** | 5.3% | **+6.7** | +10.0 | +4.2 | 31.5 | 32.2 | 9.0 | 33.7 | 44.2 | 30.1 |
| 2 | 12-13; 14-15 | Shrink + LSR (14) | 5.2% | +7.0 | +10.3 | +4.9 | 32.5 | 31.2 | 11.0 | 33.3 | 45.8 | 30.8 |
| 2 | 13-14; 23-24 | drop | 5.3% | +41.5 | +58.7 | +50.0 | 38.7 | 33.4 | 14.5 | 47.2 | 46.8 | **36.1** |
| 2 | 13-14; 23-24 | **PolyShrink (15)** | 5.2% | +13.4 | +16.1 | +10.8 | 35.5 | 33.8 | 16.0 | 42.5 | 49.0 | 35.4 |
| 2 | 14-16 | drop | 5.3% | +30.1 | +49.0 | +30.2 | 38.5 | 34.2 | 6.5 | 44.7 | 41.8 | 33.1 |
| 2 | 14-16 | ACDC + SDCF (04) | 5.3% | +10.8 | +18.5 | +6.4 | 36.2 | 29.4 | 5.5 | 26.0 | 45.5 | 28.5 |
| 4 | 10-11; 12-13; 14-15; 16-17 | drop | 10.6% | +110.3 | +220.5 | +86.7 | 30.0 | 27.6 | 2.0 | 22.3 | 37.0 | 23.8 |
| 4 | 10-11; 12-13; 14-15; 16-17 | CD-RegMean (01) | 10.6% | +64.3 | +121.8 | +34.7 | 26.3 | 24.6 | 0.0 | 23.5 | 36.8 | 22.2 |
| 4 | 10-11; 12-13; 14-15; 16-17 | SDCF (02) | 10.6% | +39.5 | +78.9 | +20.7 | 24.5 | 26.0 | 0.5 | 23.7 | 37.3 | 22.4 |
| 4 | 10-11; 12-13; 14-15; 16-17 | **AAM (03)** | 10.6% | +27.5 | +44.1 | +15.3 | 31.2 | 23.2 | 1.5 | 28.5 | 42.5 | **25.4** |
| 4 | 10-11; 12-13; 14-15; 16-17 | NUF (05) | 10.6% | +26.6 | +40.1 | +16.0 | 28.7 | 24.6 | 2.5 | 22.0 | 38.0 | 23.2 |
| 4 | 10-11; 12-13; 14-15; 16-17 | Homotopy (06) | 10.6% | +34.3 | +58.0 | +15.5 | 26.3 | 24.2 | 4.0 | 21.7 | 40.8 | 23.4 |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-energy (07) | 10.6% | +30.5 | +61.4 | +16.9 | 28.8 | 24.2 | 1.0 | 24.0 | 37.0 | 23.0 |
| 4 | 10-11; 12-13; 14-15; 16-17 | parallel union, no cut (07) | 0.0% | +25.2 | +32.2 | +25.4 | 35.3 | 29.6 | 1.0 | 33.7 | 40.3 | 28.0 |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-span (08) | 10.6% | +29.8 | +59.9 | +15.4 | 25.0 | 24.6 | 2.5 | 21.7 | 37.5 | 22.3 |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-polyglot (09) | 10.6% | +31.8 | +67.9 | +16.8 | 27.5 | 25.4 | 2.5 | 22.7 | 35.0 | 22.6 |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-Heal (10) | 10.6% | +23.3 | +47.4 | +12.3 | 30.5 | 24.4 | 2.0 | 22.0 | 41.7 | 24.1 |
| 4 | 10-11; 12-13; 14-15; 16-17 | PRISM-Shrink (11) | 10.6% | +18.8 | +27.7 | +12.5 | 30.3 | 26.0 | 1.5 | 28.2 | 37.3 | 24.7 |
| 4 | 10-11; 12-13; 14-15; 16-17 | LaCo baseline (12) | 10.6% | +70.2 | +94.5 | +79.6 | 27.5 | 25.4 | 2.0 | 25.3 | 40.2 | 24.1 |
| 4 | 10-11; 12-13; 14-15; 16-17 | **Shrink + ICL (13)** | 10.6% | **+17.9** | +26.9 | +11.3 | 28.7 | 27.4 | 0.5 | 25.0 | 37.3 | 23.8 |
| 4 | 10-11; 12-13; 14-15; 16-17 | Shrink + LSR (14) | 10.4% | +19.2 | +28.3 | +12.9 | 29.0 | 28.4 | 2.0 | 28.5 | 38.0 | 25.2 |
| 4 | 11-13; 14-16 | drop | 10.6% | +47.6 | +62.7 | +50.2 | 31.0 | 26.8 | 1.5 | 25.3 | 37.5 | 24.4 |
| 4 | 11-13; 14-16 | ACDC + SDCF (04) | 10.6% | +21.9 | +36.7 | +13.3 | 21.8 | 27.6 | 4.0 | 22.7 | 42.7 | 23.8 |
| 4 | 13-14; 15-16; 21-22; 23-24 | drop | 10.6% | +196.7 | +288.2 | +236.4 | 34.2 | 31.4 | 2.5 | 38.5 | 41.5 | **29.6** |
| 4 | 13-14; 15-16; 21-22; 23-24 | PolyShrink (15) | 10.4% | +28.3 | +45.3 | +19.5 | 32.0 | 27.4 | 3.5 | 26.8 | 42.7 | 26.5 |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | drop | 15.8% | +292.7 | +513.9 | +266.1 | 28.0 | 28.2 | 1.0 | 21.8 | 33.2 | 22.4 |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | CD-RegMean (01) | 15.8% | +212.7 | +345.4 | +134.6 | 29.0 | 26.4 | 0.5 | 22.0 | 34.5 | 22.5 |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | SDCF (02) | 15.8% | +96.9 | +156.9 | +64.1 | 29.3 | 22.8 | 1.0 | 21.7 | 36.8 | 22.3 |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | AAM (03) | 15.8% | +61.9 | +80.3 | +42.5 | 25.5 | 23.6 | 3.5 | 23.2 | 39.2 | 23.0 |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | **NUF (05)** | 15.8% | +67.9 | +98.9 | +45.9 | 30.2 | 25.2 | 4.0 | 25.8 | 37.0 | **24.4** |
| 6 | 7-8; 10-11; 12-13; 14-15; 16-17; 24-25 | Homotopy (06) | 15.8% | +69.4 | +108.7 | +43.5 | 30.0 | 24.4 | 1.0 | 21.5 | 37.7 | 22.9 |
| 6 | 6-7; 8-9; 11-13; 14-16 | drop | 15.8% | +113.5 | +142.7 | +134.3 | 27.3 | 24.8 | 2.0 | 24.2 | 38.5 | 23.4 |
| 6 | 6-7; 8-9; 11-13; 14-16 | ACDC + SDCF (04) | 15.8% | **+47.4** | +77.0 | +31.6 | 25.2 | 26.2 | 1.5 | 21.3 | 36.7 | 22.2 |

The first run of 14 (learning rate 5e-5, `14_shrink_lsr_lr5e-5_no_effect/`) gave 30.7 / 24.6 avg acc and +6.9 / +19.0
ppl. That is the same as the re-run within noise, so the table leaves it out.

## Analysis
1. **Perplexity and accuracy pick different winners.** The PRISM-Shrink family (11, 13, 14) has the lowest perplexity:
   +6.7 to +7.0% at 2 layers and +17.9 to +19.2% at 4, against drop's +15.1 / +110.3. But its few-shot accuracy matches
   drop (30.1 to 30.8 vs 30.6). AAM is close on perplexity (+6.8 / +27.5) and is the only merge method that clearly
   raises accuracy.
2. **PolyShrink (15) does not beat its own baseline.** Its 35.4 / 26.5 avg acc is the highest of any merged model, but plain drop on
   the same CoPlan pairs scores 36.1 / 29.6. The accuracy comes from **which pairs** CoPlan picks, not from the
   merging. PolyShrink's real gain over drop is perplexity: +13.4% vs +41.5% at 2 layers, +28.3% vs +196.7% at 4. Report it
   this way in the write-up. The README currently presents it as the main method.
3. **Pair choice matters more than method for accuracy.** On CoPlan pairs, plain drop (36.1) beats every method on the
   angular pairs (best: AAM 34.3).
4. **Closed-form starts hurt accuracy.** CD-RegMean (24.0), SDCF (25.7), NUF (26.5) and PRISM-energy/span/polyglot
   (27.4 to 27.7) all score below drop (30.6) at 2 layers, even though their perplexity is lower.
5. **The selection-rule ablations make no difference.** Energy, span, polyglot, ICL packing and LSR all stay within ±1
   avg acc of each other, which is inside the noise.
6. **MGSM collapses for everyone.** It falls from 38.0 to 15.0 or less at 2 layers and to 4.0 or less at 4. Chain-of-thought math is the first
   ability lost when layers are merged.
7. **At 6 layers nothing survives.** Every method scores 22 to 24 avg acc, close to chance on these benchmarks. Merge
   methods still cut perplexity 3 to 5 times compared with drop.
