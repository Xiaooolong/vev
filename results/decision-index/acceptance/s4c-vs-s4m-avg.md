# v0.2 acceptance: s4c vs s4m-avg

**Verdict: inconclusive: the interval crosses its bound; D3 accepts the single seed, and the release notes say that seed variance was not estimated**

## Vision proxies (skill, rows present in both shapes)

| family | rows | kit new | kit ref | kit diff [95%] | orig new | orig ref | orig diff [95%] |
|---|---:|---:|---:|---:|---:|---:|---:|
| cvb | 998 | 74.59 | 75.10 | -0.50 [-2.52, +1.52] | 75.26 | 75.26 | +0.00 [-1.70, +1.70] |
| blink | 784 | 76.19 | 69.63 | +6.57 [+3.09, +10.11] | 79.48 | 79.48 | +0.00 [-2.89, +2.71] |
| wino | 767 | 96.87 | 96.87 | +0.00 [-0.78, +0.78] | 96.87 | 97.39 | -0.52 [-1.57, +0.52] |
| rwqa | 444 | 78.88 | 81.07 | -2.18 [-4.11, -0.71] | 80.34 | 80.70 | -0.36 [-3.21, +2.47] |
| docs | 980 | 93.47 | 93.61 | -0.14 [-1.40, +1.12] | 94.01 | 94.56 | -0.54 [-1.59, +0.55] |
| kie | 214 | 99.36 | 99.36 | +0.00 [+0.00, +0.00] | 98.75 | 98.75 | +0.00 [+0.00, +0.00] |
| hm | 847 | 53.01 | 52.54 | +0.47 [-3.07, +4.01] | 51.12 | 52.07 | -0.94 [-4.01, +2.13] |
| m2w | 995 | 89.15 | 88.07 | +1.07 [-0.43, +2.57] | 91.02 | 92.09 | -1.07 [-2.72, +0.44] |
| **total8** | | 82.69 | 82.03 | +0.66 [-0.09, +1.38] | 83.36 | 83.79 | -0.43 [-1.15, +0.28] |
| **total7** | | 81.77 | 81.17 | +0.60 [-0.21, +1.41] | 82.26 | 82.60 | -0.34 [-1.12, +0.45] |

Judged on total8: kit shape **inconclusive** (pass needs the lower bound > 0), original shape **inconclusive** (pass needs the lower bound > -1.0); vision **inconclusive**. total7 (without m2w) is reported only.

Rows without a kit form, original shape, reported apart:

| family | rows | new | ref | diff [95%] | reasons |
|---|---:|---:|---:|---:|---|
| kie | 228 | 95.87 | 97.52 | -1.65 [-4.26, +0.00] | kie_amounts_not_kit_showable 28, kie_kind_field_presence 200 |
| m2w | 5 | 73.33 | 73.33 | +0.00 [+0.00, +0.00] | m2w_kit.py did not render this action (proxy totals: image_hit_dropped=4, skip_fewer_than_3_negatives=1) 5 |

## v0.2 families (reported, not judged)

| proxy | slice | rows | new | ref | diff [95%] |
|---|---|---:|---:|---:|---:|
| charts | all | 1858 | 93.76 | 89.81 | +3.95 [+2.72, +5.26] |
| charts | ecd | 525 | 86.79 | 84.25 | +2.54 [+0.25, +4.81] |
| charts | ecd_heldout_style | 196 | 87.07 | 87.76 | -0.68 [-4.00, +2.07] |
| charts | plotqa | 1137 | 98.12 | 92.73 | +5.39 [+3.69, +7.11] |
| nlvr2 | all | 1024 | 88.48 | 79.10 | +9.38 [+5.98, +12.40] |
| nlvr2 | comparison | 116 | 84.48 | 75.86 | +8.62 [+0.00, +18.35] |
| nlvr2 | counting | 276 | 86.96 | 75.36 | +11.59 [+5.17, +18.05] |
| nlvr2 | other | 84 | 85.71 | 76.19 | +9.52 [+2.30, +19.75] |
| nlvr2 | quantifier | 268 | 89.55 | 84.33 | +5.22 [+0.00, +10.92] |
| nlvr2 | spatial | 280 | 91.43 | 80.00 | +11.43 [+5.05, +18.12] |
| funsd | all | 412 | 94.15 | 92.98 | +1.17 [-1.54, +4.12] |
| funsd | field_present | 210 | 87.62 | 84.76 | +2.86 [-4.18, +9.30] |
| funsd | field_value | 202 | 98.68 | 98.68 | +0.00 [+0.00, +0.00] |

## DI text full (§17)

| | new | ref | diff |
|---|---:|---:|---:|
| published index (all benchmarks) | 60.13 | 63.41 | -3.28 |
| **remaining 23 benchmarks (judged)** | 65.98 | 66.47 | **-0.50** |

Rule: remaining index (new - ref) >= -0.5 -> **pass**.

Cleaned benchmarks, reported apart (§17 as fixed 2026-11-05: training families lost >= 10% of their rows over stages 1, 3 and 4; `data/stage4/text_cleaned_benchmarks.json`):

| id | benchmark | new | ref | diff | removed | training families |
|---:|---|---:|---:|---:|---:|---|
| 25 | GPQA Diamond | 21.77 | 24.49 | -2.72 | 100.0% | di_gpqa: 100.0% of 1430 rows removed (split:GPQA extended (benchmark only, no train split) 1430) |
| 33 | SATA-Bench | 17.53 | 45.40 | -27.87 | 100.0% | di_sata: 100.0% of 2287 rows removed (split:sata-bench-raw pool (benchmark pre-review pool) 2287) |
| 57 | MMLU-Pro | 54.48 | 54.66 | -0.18 | 13.2% | di_mmlu_pro: 13.2% of 7258 rows removed (split:MMLU validation 741, split:MMLU dev 115, split:MMLU-Pro validation 100) |
| 58 | BBH fixed-option tasks | 59.49 | 69.00 | -9.51 | 62.1% | di_bbh, s4_text_di_bbh: 62.1% of 11303 rows removed (split:BIG-bench task files (benchmark only, no train split) 4969, split:BIG-bench task.json examples (unsplit eval pool) 2042, extra_drop 3) |
| 12 | ANLI | 56.80 | 60.68 | -3.88 | 41.3% | di_anli: 41.3% of 8721 rows removed (split:dev_r3 1495, split:dev_r2 1119, split:dev_r1 986) |
| 38 | ACOS | 31.46 | 35.21 | -3.75 | 10.1% | di_acos, s4_text_di_acos: 10.1% of 7482 rows removed (split:dev 756, extra_drop 1, proxy_same_item 1) |
| 41 | VAST | 69.69 | 70.80 | -1.11 | 24.2% | di_vast: 24.2% of 8743 rows removed (split:dev 2113) |
| 42 | NLI4CT | 62.68 | 69.40 | -6.72 | 55.8% | di_nli4ct: 55.8% of 5440 rows removed (split:practice_test 2758, split:dev 276) |
| 36 | BRIGHT | 33.40 | 32.76 | +0.64 | 100.0% | di_bright, s4_text_di_bright: 100.0% of 1259 rows removed (split:BRIGHT documents 917, split:BRIGHT documents (benchmark corpus, no train split) 342) |
| 2 | ToolRet | 59.35 | 61.40 | -2.05 | 35.9% | di_toolret: 35.9% of 359 rows removed (split:before-sample (benchmark pool) 129) |
| 9 | Home appliance simulator | 40.91 | 84.09 | -43.18 | 100.0% | di_home: 100.0% of 3319 rows removed (split:dev 3319) |
| 20 | BPoMP | 87.88 | 88.75 | -0.87 | 19.2% | di_bpomp: 19.2% of 8694 rows removed (split:validation 869, split:test 800) |
| 21 | Humicroedit | 23.82 | 27.40 | -3.58 | 16.5% | di_humicroedit: 16.5% of 8832 rows removed (split:dev 1456) |
| 64 | New Yorker caption matching | 63.06 | 64.49 | -1.43 | 21.4% | di_newyorker: 21.4% of 2936 rows removed (split:validation 627) |

Against the 11-04 hand list: added 20 BPoMP, 42 NLI4CT, 57 MMLU-Pro; dropped 43 CRUXEval.

Judged benchmarks, largest drops first (8 of 23; all in the JSON); * = training families lost some rows, under 10%:

| id | benchmark | area | new | ref | diff | removed |
|---:|---|---|---:|---:|---:|---:|
| 62 | When2Call MCQ * | tools | 79.52 | 82.55 | -3.03 | 2.3% |
| 3 | API-Bank | tools | 82.14 | 84.15 | -2.01 |  |
| 28 | WinoGrande * | knowledge | 64.96 | 66.38 | -1.42 | 0.4% |
| 22 | POP909-CL * | arts | 79.00 | 80.08 | -1.08 | 0.1% |
| 31 | ChessBench | knowledge | 12.93 | 13.94 | -1.01 |  |
| 59 | RAGTruth response-level hallucination * | language | 56.23 | 57.10 | -0.87 | 0.9% |
| 11 | ContractNLI * | language | 77.22 | 78.05 | -0.83 | 8.8% |
| 50 | Habermas Machine | arts | 41.81 | 42.50 | -0.69 |  |
