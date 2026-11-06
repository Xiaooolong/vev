# Training data of vev-4b-v2

The data `vev-4b-v2` was trained on, and its overlaps with the
[Jev Decision Index](https://huggingface.co/spaces/multimodalart/jev-decision-index) (DI). `vev-4b` and `vev-9b` are described
in [TRAINING.md](TRAINING.md).

## Stages

| Stage | Starts from | Records | Steps |
|---|---|---:|---:|
| 0 | Qwen3.5-4B, new LoRA adapter (rank 16) | 79,224 | 2,500 |
| 1 | the stage-0 adapter | 206,959 | 6,000 |
| 3 | the stage-1 adapter | 182,468 | 5,000 |
| 4 | the stage-3 weights with the adapter merged in, new LoRA adapter (rank 16) | 427,765 | 3,910 |

There is no stage 2. Each stage repeats part of the earlier data.

## Sources

All sources use training splits, or generators where a task has none.

| Group | Added in | Sources |
|---|---|---|
| Public text and image datasets | stage 0 | AFQMC, AGIQA-3K, AI2D, AMEX, AndroidControl, A-OKVQA, BANKING77, BoolQ, CLINC150, CMM-Math, COCO-CN, DBpedia-14, FOIL-COCO, GUI-Odyssey, GUICourse, HPDv2, ImageRewardDB, IMDB, LLaVA-Critic, MNLI, PlotQA, SST-5, STS-B, TextVQA, TNEWS, TREC, Visual7W, VQAv2, VSR |
| DI text task families | stage 1 | ACOS, ANLI, API-Bank, BANKING77, BBH-style generated items, CFColor, ChessBench, CLadder, CLINC150, ContractNLI, CRUXEval-style functions and MBPP, ESCI, SEntFiN, GSM8K, Habermas Machine, HellaSwag, HoVer, Humicroedit, iSarcasmEval, SuperGPQA, New Yorker caption contest, NLI4CT, PhishNChips, POP909-style pieces, RAGTruth, ToolRet, VAST, When2Call, WinoGrande, limerick-topic |
| Chinese and OpenJevData | stage 3 | CAIL2018, CBLUE, ChnSentiCorp, CLUE, online_shopping_10_cats, waimai_10k, Weibo sentiment sets, OpenJevData-140k, synthetic-typed-decisions |
| tasksource | stage 3 | 610 sources of `tasksource/tasksource-jev-typed-decisions`, listed in [`data/manifest/v0.2.0-tasksource.tsv`](data/manifest/v0.2.0-tasksource.tsv) |
| Vision task families | stage 4 | COCO train2014, Objectron, SPair-71k, VSR, TallyQA, VQAv2, ECD, PlotQA, ArxivQA, ChartQA, DocVQA, DVQA, InfographicVQA, CORD, SROIE, FUNSD, Mind2Web, NLVR2, Hateful Memes |
| Text task families | stage 4 | ACOS, BBH-style, CLadder, CRUXEval-style, WinoGrande, Open-Jev environments, corr2cause, ETHICS, SpartQA, temporal-nli |

Licences of the hand-checked sources: [`data/licenses.json`](data/licenses.json).

## Excluded

- Validation, dev and test splits.
- Items of the evaluation sets: the DI public suites, the other splits of the benchmarks behind the DI vision suite,
  JevBench, S1MB and our own evaluation sets.
- Pools of DI benchmarks without a training split: BIG-bench outside BBH, SATA-Bench, GPQA, BRIGHT documents, the
  ToolRet query pool.
- The DI Home appliance family, MULTI-Benchmark, GenAI-Bench and CharXiv.

## Overlaps with the Decision Index

Some training rows use an image that is also in a DI benchmark, with a different question. Most of these images come
from COCO; the Hateful Memes rows are memes of its official train split, whose pictures recur across HM splits with
different text. Counts are rows with a pixel-identical image / rows matched only by perceptual hash.

Public DI vision suite:

| Set | Stage 0 | Stage 1 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|
| BLINK | 29 / 1 | 19 / 0 | 2 / 0 | 2 / 1 |
| CharXiv | 0 / 46 | 0 / 28 | 0 / 9 | 0 / 9 |
| CORD | – | 0 / 1 | 0 / 1 | 0 / 2 |
| CV-Bench | 20 / 4 | 8 / 0 | 5 / 0 | 5 / 1 |
| Hateful Memes | 0 / 4 | 0 / 3 | 0 / 1 | 160 / 300 |
| Mind2Web | 0 / 2 | 0 / 1 | – | 0 / 7 |
| MMMU-Pro vision | 0 / 31 | 0 / 22 | 0 / 8 | 0 / 21 |
| R-Bench-M | 0 / 43 | 0 / 25 | 0 / 10 | 0 / 10 |
| Winoground | 0 / 28 | 0 / 16 | 0 / 9 | 2 / 11 |
| total | 49 / 159 | 27 / 96 | 7 / 38 | 169 / 362 |

Other splits of the same benchmarks:

| Benchmark | Stage 0 | Stage 1 | Stage 3 | Stage 4 |
|---|---:|---:|---:|---:|
| BLINK test | 57 / 3 | 26 / 2 | 15 / 1 | 139 / 30 |
| BLINK val | 35 / 3 | 24 / 2 | 2 / 2 | 9 / 12 |
| CharXiv | 0 / 98 | 0 / 61 | 0 / 25 | 1 / 105 |
| CORD | – | 0 / 1 | 0 / 1 | 0 / 2 |
| CV-Bench | 20 / 10 | 8 / 2 | 5 / 1 | 5 / 4 |
| Hateful Memes | 0 / 8 | 0 / 4 | 0 / 2 | 326 / 653 |
| InfographicVQA | 0 / 1 | 0 / 5 | 0 / 2 | 0 / 13 |
| Mind2Web | 0 / 119 | 0 / 73 | 0 / 26 | 21 / 166 |
| MMMU-Pro | 24 / 47 | 19 / 29 | 5 / 10 | 5 / 43 |
| R-Bench-M | 0 / 43 | 0 / 25 | 0 / 10 | 0 / 10 |
| Winoground | 0 / 28 | 0 / 16 | 0 / 9 | 2 / 11 |
| total | 136 / 360 | 77 / 220 | 27 / 89 | 508 / 1049 |

In text, 7 BANKING77 training sentences in stage 0 equal the sentence of a scored DI item, with a different
instruction; BANKING77's own train and test splits share such sentences. `vev-4b-v2` answers all 7 correctly, out
of 3,080 scored BANKING77 items.

| Training record | DI item |
|---|---|
| `banking77/train/1163#0.0` | `BANKING77:test:332` |
| `banking77/train/2315#0.0` | `BANKING77:test:751` |
| `banking77/train/4476#0.0` | `BANKING77:test:1432` |
| `banking77/train/8525#0.0` | `BANKING77:test:2673` |
| `banking77/train/6842#0.1` | `BANKING77:test:2083` |
| `banking77/train/769#0.0` | `BANKING77:test:223` |
| `banking77/train/9701#0.0` | `BANKING77:test:2988` |
