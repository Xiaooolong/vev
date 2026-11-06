# DI text full: s4c-v023 vs s4m-avg-v023

The rule is the stage-4 acceptance rule (new model vs s4m-avg); for other stages and labels it is only a reading.

## DI text full (§17)

| | new | ref | diff |
|---|---:|---:|---:|
| published index (all benchmarks) | 60.32 | 63.60 | -3.28 |
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
| 36 | BRIGHT | 38.47 | 37.79 | +0.68 | 100.0% | di_bright, s4_text_di_bright: 100.0% of 1259 rows removed (split:BRIGHT documents 917, split:BRIGHT documents (benchmark corpus, no train split) 342) |
| 2 | ToolRet | 59.49 | 61.54 | -2.05 | 35.9% | di_toolret: 35.9% of 359 rows removed (split:before-sample (benchmark pool) 129) |
| 9 | Home appliance simulator | 40.91 | 84.09 | -43.18 | 100.0% | di_home: 100.0% of 3319 rows removed (split:dev 3319) |
| 20 | BPoMP | 87.88 | 88.75 | -0.87 | 19.2% | di_bpomp: 19.2% of 8694 rows removed (split:validation 869, split:test 800) |
| 21 | Humicroedit | 23.82 | 27.40 | -3.58 | 16.5% | di_humicroedit: 16.5% of 8832 rows removed (split:dev 1456) |
| 64 | New Yorker caption matching | 63.06 | 64.49 | -1.43 | 21.4% | di_newyorker: 21.4% of 2936 rows removed (split:validation 627) |

Against the 11-04 hand list: added 20 BPoMP, 42 NLI4CT, 57 MMLU-Pro; dropped 43 CRUXEval.

Judged benchmarks, largest drops first (all); * = training families lost some rows, under 10%:

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
| 1 | BFCL | tools | 95.30 | 95.78 | -0.48 |  |
| 30 | GSM8K * | knowledge | 59.37 | 59.72 | -0.35 | 2.6% |
| 29 | HellaSwag | language | 90.88 | 91.21 | -0.33 |  |
| 5 | CLINC150+OOS * | retrieval | 92.80 | 93.13 | -0.33 | 0.5% |
| 4 | BANKING77 * | retrieval | 90.29 | 90.59 | -0.30 | 3.3% |
| 39 | FinEntity | language | 91.20 | 91.48 | -0.28 |  |
| 23 | cfcolor | arts | 42.63 | 42.79 | -0.16 |  |
| 56 | PhishNChips phishing decisions * | retrieval | 93.70 | 93.80 | -0.10 | 8.6% |
| 45 | HLE | knowledge | 0.00 | 0.00 | +0.00 |  |
| 32 | MuSR | knowledge | 30.65 | 30.45 | +0.20 |  |
| 44 | CLadder * | knowledge | 59.44 | 59.24 | +0.20 | 0.0% |
| 61 | HoVer claim verification * | retrieval | 75.96 | 75.76 | +0.20 | 1.1% |
| 37 | Amazon ESCI * | retrieval | 49.32 | 48.88 | +0.44 | 0.0% |
| 40 | iSarcasmEval | language | 42.12 | 41.31 | +0.81 |  |
| 43 | CRUXEval * | knowledge | 41.82 | 39.88 | +1.94 | 3.2% |
