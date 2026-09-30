# Training

This file records how `vev-4b` and `vev-9b` were trained, so the runs can be repeated or varied.

## Recipe

Both models are LoRA fine-tunes of the base model with the same settings; only `--base` differs.

```bash
# 1. Build the training mixture (downloads the 42 sources; see data/README.md)
python -m data.sources.text_all
python -m data.sources.image_group1_all
python -m data.sources.image_group2_all
python -m data.sources.image_group3_all
python -m data.build --version v1-research --allow commercial-ok,non-commercial,unknown

# 2. Train
python -m train.train \
  --build data/build/v1-research --base Qwen/Qwen3.5-4B --out runs/vev-4b \
  --prior --head-lr 0 --lr 5e-5 --max-steps 2500 --limit 100000 \
  --micro-tokens 16384 --accum 4 --anchor-kl 0.3 \
  --anchor-build data/build/anchor-v3 --anchor-weight 2 \
  --eval-every 250 --eval-rows 1000 --save-every 250 --workers 8

# 3. Export the release forms (merged weights and adapter)
python -m train.export_release --ckpt runs/vev-4b/export --name vev-4b --out release
```

What the flags do:

- `--prior --head-lr 0`: the answer probabilities come from the language model's own next-token logits over the
  answer tokens. No classifier head is trained; LoRA (rank 16 on all linear layers of the language model, vision
  tower frozen) shifts those logits.
- `--anchor-kl 0.3`: on every training row, a KL term keeps the adapted answer distribution close to the base
  model's. This limits drift on question types that are not in the training data.
- `--anchor-build ... --anchor-weight 2`: an extra set of 10,123 unlabeled rows used only for that KL term, in
  question shapes the training data does not cover (for example generic instructions such as "Text
  classification benchmark."). It is published as a dataset next to the models.
- `--limit 100000`: 100,000 training records sampled from the 716,789 in the build.

The full argument list of each run is in `train_args` inside the released `vev_pointer.json`.

## Environment

| | |
|---|---|
| GPU | 1 × NVIDIA H800 80GB per run |
| Time | 2,500 steps; median 16 s/step (4B), 21 s/step (9B) |
| Peak memory | 19 GB (4B), 39 GB (9B) |
| Libraries | torch 2.8.0+cu128, transformers 5.17.0, peft 0.21.0, accelerate 1.15.0 |

## Training data

The build `v1-research` has 762,820 records (716,789 train, 23,038 calibration, 22,993 validation) from 42
public sources: 431,449 image records and 331,371 text records, 620,975 English and 141,845 Chinese. Every source
is converted into typed questions by the scripts in `data/sources/`. The license of each source was checked by
hand; the evidence and quotes are in `data/licenses.json`.

Six sources are research-only or non-commercial and eleven state no license, which is why the weights are released
under CC BY-NC 4.0. `data.build --allow commercial-ok` builds a mixture from the permissive sources only; no model
has been trained on it yet.

| Source | Kind | Train records | Tier | License |
|---|---|---|---|---|
| afqmc | text | 28,223 | unknown | unspecified |
| ag_news | text | 17,703 | unknown | unspecified |
| agiqa3k | image | 4,350 | commercial-ok | MIT |
| ai2d | image | 5,155 | commercial-ok | CC BY-SA 4.0 |
| amazon_reviews_multi_en | text | 31,227 | non-commercial | Amazon Reviews Corpus license (research only) |
| amex | image | 20,090 | commercial-ok | CC BY 4.0 |
| androidcontrol | image | 21,410 | commercial-ok | Apache-2.0 |
| aokvqa | image | 18,206 | unknown | unspecified for data (repo code Apache-2.0) |
| banking77 | text | 11,167 | commercial-ok | CC BY 4.0 |
| boolq | text | 8,867 | commercial-ok | CC BY-SA 3.0 |
| charxiv | image | 2,570 | unknown | CC BY-SA 4.0 (annotations); charts copyright original arXiv authors |
| clinc150 | text | 17,015 | commercial-ok | CC BY 3.0 |
| cmm_math | image | 5,451 | commercial-ok | BSD-3-Clause |
| coco_cn | image | 14,421 | commercial-ok | MIT (annotations); images COCO |
| dbpedia14 | text | 33,117 | commercial-ok | CC BY-SA 3.0 / GFDL |
| foil_coco | image | 24,749 | commercial-ok | CC BY 4.0 (annotations); images COCO |
| gaokao_mm | image | 563 | commercial-ok | Apache-2.0 |
| genai_bench | image | 9,920 | commercial-ok | Apache-2.0 |
| gui_odyssey | image | 20,836 | commercial-ok | CC BY 4.0 |
| guicourse | image | 33,006 | commercial-ok | CC BY 4.0 (data); code MIT |
| hpdv2 | image | 28,008 | commercial-ok | Apache-2.0 |
| imagereward | image | 19,143 | commercial-ok | Apache-2.0 |
| imdb | text | 23,516 | unknown | unspecified |
| llava_critic | image | 26,985 | commercial-ok | Apache-2.0 |
| mme_realworld | image | 18,312 | non-commercial | academic research only (HF YAML says Apache-2.0) |
| mnli | text | 32,939 | commercial-ok | OANC license + CC BY-SA 3.0 + CC BY 3.0 + public domain (mixed permissive) |
| multi_benchmark | image | 12,099 | unknown | unspecified for data (repo code MIT) |
| naturalbench | image | 4,801 | non-commercial | Apache-2.0 (HF YAML); images include Flickr30K |
| nlvr2 | image | 20,613 | non-commercial | CC BY 4.0 (annotations); images unlicensed, research-only access |
| ocnli | text | 33,086 | non-commercial | CC BY-NC 2.0 |
| plotqa | image | 16,555 | commercial-ok | CC BY 4.0 (data); code MIT |
| sst5 | text | 8,878 | unknown | unspecified |
| stsb | text | 5,998 | unknown | unspecified (GLUE: refer to original licenses) |
| sugarcrepe | image | 0 | commercial-ok | MIT |
| textvqa | image | 20,630 | commercial-ok | CC BY 4.0 (annotations); images Open Images (listed CC BY 2.0) |
| tnews | text | 33,048 | unknown | unspecified |
| trec | text | 6,029 | unknown | unspecified |
| visual7w | image | 18,158 | unknown | unspecified (toolkit code MIT) |
| vqav2_mc | image | 19,593 | commercial-ok | CC BY 4.0 (annotations); images per-image Flickr licenses via COCO |
| vqav2_yesno | image | 17,346 | commercial-ok | CC BY 4.0 (annotations); images per-image Flickr licenses via COCO |
| vsr | image | 2,423 | commercial-ok | CC BY 4.0 (annotations); images COCO |
| yelp | text | 20,583 | non-commercial | Yelp Dataset Terms of Use |

## What cannot be rebuilt exactly

- The anchor set `anchor-v3` combines three parts. Two of them come from builders that are not in this
  repository: synthetic gate questions whose states were written by DeepSeek (deepseek-flash, used under DeepSeek's
  terms, which allow training on outputs) and a short-question set. Use the published copy of `anchor-v3`.
- Sources were downloaded between 22 and 24 September 2026 and most are not pinned to a revision. The
  `raw_sha256` of every converted source is in `data/manifest/v1-research.json`, so a rebuild can be checked, but
  upstream changes will make it differ. The Amazon reviews dataset is no longer distributed by its authors.
- Training used one GPU and bf16. Expect small differences between runs on other hardware.
