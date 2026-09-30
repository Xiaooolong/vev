# Vev

[![English](https://img.shields.io/badge/lang-English-blue)](README.md) [![简体中文](https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red)](README_zh.md) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-models%20%26%20data-yellow)](https://huggingface.co/collections/CountingSheep/vev-v01-6abd3e17303828e10c49dc65)

Vev 是一个开放权重的判断模型。输入一段状态（文本、JSON、图片或混合）和一组带类型的问题（是/否、单选、分档打分），它在一次前向计算里给出每个候选答案的概率，不生成文本。服务端的请求和响应格式与 TypeSafe 的 `/v1/systemone` 接口一致（[TypeSafe 文档](https://docs.typesafe.ai/concepts/system-one.md)），为那个接口写的客户端改一下 base URL 就能接 Vev。

当前是 v0.1 研究预览版。模型权重只许非商业使用，见[许可](#许可)。

## 快速上手

```bash
pip install git+https://github.com/Xiaooolong/vev
vev serve --model CountingSheep/vev-4b          # 首次启动会下载权重，监听 127.0.0.1:8009
```

需要 Python 3.11 及以上和 NVIDIA GPU；CPU 和 Apple Silicon 没有测过。bf16 下，`vev-4b` 加载后约占 10 GB 显存，`vev-9b` 约 19 GB；状态很长或带图片时还要再多一些。

服务端一次只处理一个请求，并发请求会排队等待。每个问题做一次前向计算，所以延迟随问题数增长：单张 H800 上，`vev-4b` 1 个问题 43 ms，10 个问题 383 ms（见 `conformance/results/vev-4b/curves.json`）。

用 Docker：

```bash
git clone https://github.com/Xiaooolong/vev && cd vev
docker build -t vev .
docker run --gpus all -p 8009:8009 -v ~/.cache/huggingface:/root/.cache/huggingface vev \
  --model CountingSheep/vev-4b
```

用官方 Python SDK（`pip install typesafe-sdk`）：

```python
from typesafe_sdk import Choice, Noul, TypeSafeClient

client = TypeSafeClient(api_key="local", base_url="http://127.0.0.1:8009")
resp = client.system_one(
    state="Order #4411 still shows 'label created' after 9 days. I need it before Friday.",
    questions={
        "urgent": Noul(instructions="Is the customer asking for something time-sensitive?"),
        "team": Choice(instructions="Which team should handle this?",
                       criteria={"shipping": "Delivery and tracking", "billing": "Charges and refunds"}),
    },
)
print(resp.answers["urgent"].noul)          # "是"的概率：vev-4b 给出 0.98
print(resp.answers["team"].probabilities)   # {"shipping": 0.97, "billing": 0.03}
```

传图片是 Vev 自己加的扩展，官方接口只收文本。图片放在状态里，写成带 data URL 的 `image` 对象：

```bash
curl -s http://127.0.0.1:8009/v1/systemone -H 'Content-Type: application/json' -d '{
  "model": "vev-latest",
  "state": {"screen": {"image": {"url": "data:image/png;base64,iVBORw0..."}}},
  "questions": {"error": {"type": "noul", "instructions": "Does the screen show an error message?"}}
}'
```

完整的请求格式、限制和错误码见 [spec/systemone-api.md](spec/systemone-api.md)（英文）。

## 模型

| 模型 | 基座 | Hugging Face |
|---|---|---|
| `vev-4b` | Qwen3.5-4B | [CountingSheep/vev-4b](https://huggingface.co/CountingSheep/vev-4b)（合并权重），[CountingSheep/vev-4b-lora](https://huggingface.co/CountingSheep/vev-4b-lora)（adapter） |
| `vev-9b` | Qwen3.5-9B | [CountingSheep/vev-9b](https://huggingface.co/CountingSheep/vev-9b)（合并权重），[CountingSheep/vev-9b-lora](https://huggingface.co/CountingSheep/vev-9b-lora)（adapter） |

两个都是 LoRA 微调。答案概率直接取模型在答案 token（Yes/No、选项字母、档位数字）上的下一 token 概率，只在合法答案之间重新归一化。训练改的就是这组概率，没有另加分类头。

## 结果

下面的数字都是全量集上的正确率，用 `evals/` 里的评测脚本测得。两个系统对比时，在同一批题上做配对 bootstrap 检验（95% 区间），区间包含 0 的记为不显著；图片集按图片整组重采样，同一张图上的题一起抽。每个集的完整指标（含 Brier 分数和校准误差）在 [results/](results/)。`results/` 里有、下面表里没有的集是开发时做诊断用的。

对比对象：[Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是 TypeSafe 的托管判断模型；[kev-4B](https://github.com/jaredpalmer/kev) 是基于 4B Qwen 的开源判断模型。文本评测集是 [judgekit](https://github.com/lexingtonhibiki/judgekit)（中文）、[JevBench](https://github.com/fstandhartinger/jevbench)（公开子集）、[nimble](https://github.com/bespokelabsai/nimble)，以及 kev 自己的留出测试集 transfer-v4。

### 文本判断

| 评测集 | 题数 | kev-4B | vev-4b | vev-9b | Jev |
|---|---|---|---|---|---|
| judgekit（中文） | 130 | 0.846 | 0.962 | 0.938 | 0.962 |
| JevBench（公开子集） | 231 | 0.719 | 0.766 | 0.823 | 0.861 |
| nimble | 324 | 0.735 | 0.707 | 0.747 | 0.923 |
| kev transfer-v4 | 1528 | 0.802 | 0.776 | 0.781 | 0.854 |

kev-4B 用同一套脚本在本地跑；Jev 通过 TypeSafe 的 API 调用（`jev-latest`，2026 年 9 月 23–24 日）；训练没有用到任何 Jev 的输出。

各组差异（正确率点数，配对 bootstrap）：

- vev-4b 对 kev-4B：judgekit +11.5，kev transfer-v4 −2.6；JevBench 和 nimble 不显著。
- vev-9b 对 kev-4B：judgekit +9.2，JevBench +10.4，kev transfer-v4 −2.1；nimble 不显著。
- vev-4b 对 Jev：JevBench −9.5，nimble −21.6，kev transfer-v4 −7.8；judgekit 持平。
- vev-9b 对 Jev：nimble −17.6，kev transfer-v4 −7.3；judgekit（−2.3）和 JevBench（−3.9）不显著。

### 训练前后

下表对比 Vev 和未微调的基座模型（零样本，读出方式相同）。

| 评测集 | 题数 | Qwen3.5-4B | vev-4b | Qwen3.5-9B | vev-9b |
|---|---|---|---|---|---|
| judgekit | 130 | 0.962 | 0.962 | 0.938 | 0.938 |
| JevBench | 231 | 0.758 | 0.766 | 0.805 | 0.823 |
| nimble | 324 | 0.710 | 0.707 | 0.710 | **0.747** |
| kev transfer-v4 | 1528 | 0.758 | **0.776** | 0.764 | **0.781** |
| MMBench-EN | 1164 | 0.872 | 0.881 | 0.887 | **0.902** |
| POPE | 9000 | 0.894 | *0.889* | 0.894 | 0.897 |
| MMStar | 1498 | 0.544 | **0.627** | 0.608 | **0.674** |

加粗表示显著变好，斜体表示显著变差。唯一的退步是 vev-4b 在 POPE 上：正确率低 0.5 个点（95% 区间 −0.9 到−0.1），Brier 分数差 0.007。vev-9b 没有显著变差的集。

### 图片判断

公开的视觉评测大多是就图片内容提问，而 Vev 要做的是按给定规则判断一张图，所以我们用带人工标注的公开数据转换出了几个判断集。转换脚本在 `evals/datasets/`，转换后的数据不再分发。

| 评测集 | 判断内容 | 题数 | Qwen3.5-4B | vev-4b | Qwen3.5-9B | vev-9b |
|---|---|---|---|---|---|---|
| policy_mod | 图片是否违反给定的安全规则（LlavaGuard） | 659 | 0.686 | **0.742** | 0.666 | **0.724** |
| game_glitch | 游戏截图里有没有画面错误（VideoGameQA-Bench） | 1000 | 0.646 | 0.613 | 0.577 | **0.626** |
| game_clip | 物体是否穿模进角色（VideoGameQA-Bench） | 686 | 0.561 | **0.671** | 0.736 | 0.729 |
| ui_toggle | 界面上有没有处于开/关状态的开关（MobileViews） | 800 | 0.901 | **0.928** | 0.921 | 0.923 |
| ui_input | 界面上有没有文本输入框（MobileViews） | 800 | 0.949 | 0.951 | 0.948 | 0.955 |
| t2i_elem | 生成的图片是否画出了提示词里的某个元素（EvalMuse） | 1027 | 0.690 | **0.715** | 0.733 | 0.703 |

vev-4b 在 game_glitch、vev-9b 在 t2i_elem 上比各自的基座低约 3 个点，两处差异都不显著。

## 局限

- Jev 在 nimble 和 kev transfer-v4 上比 Vev 准（分别高 18–22 和 7–8 个点），在 JevBench 上比 vev-4b 准。
- 答案会受选项顺序影响。把选项倒过来排，JevBench 和 kev transfer-v4 上首选答案会变的题，vev-9b 占 13%，vev-4b 占 17%；kev-4B 是 10–12%，Jev 不到 4%。微调让这两个集上的比例下降了（基座模型是 19–25%）。nimble 上 9B 从 33% 降到 9%，4B 反而从 19% 升到 21%。
- 需要多步推理的判断，不如基座模型先思考再回答。在一个内部场景集上（未公开），基座的思考模式比 Vev 的单次读出准 4–8 个点；在我们试过的大多数公开集上，单次读出和思考模式一样准或更准。
- 在“这里有没有问题”这类二分类问题上，基座模型和 Vev 都倾向于答“没有”，标出问题的次数比标注里少。
- 图片判断集是我们自己从公开数据转换的，不是公认的评测基准。
- “兼容”指请求和响应的格式与 `/v1/systemone` 一致，不代表 Vev 的行为和 Jev 一样。
- Vev 与 TypeSafe 没有关联，也未获其认可。

## 复现

`train/` 是训练代码；`data/` 从 42 个公开数据源构建训练数据；`evals/` 转换评测集，并可对任何 `/v1/systemone` 服务端跑评测。[TRAINING.md](TRAINING.md)（英文）列出了确切的训练配方、库版本，以及哪些部分无法从公开数据重建。

`conformance/` 按接口规范检查一个实现，可以对任何服务端跑，包括官方的：

```bash
SO_BASE_URL=http://127.0.0.1:8009 SO_SUPPORTS_IMAGES=1 pytest conformance -q
```

## 许可

本仓库代码采用 Apache License 2.0。模型权重采用 CC BY-NC 4.0：6 个训练数据源只允许研究或非商业用途，另有 11 个没有明确的数据许可。这个许可不替代这些数据源自己的条款。基座模型 Qwen3.5-4B 和 Qwen3.5-9B 是 Apache 2.0。[TRAINING.md](TRAINING.md) 列出了每个训练数据源及其条款。

## 引用

```bibtex
@software{vev2026,
  title  = {Vev: an open-weight judgment model for text and images},
  author = {Wang, Xiaolong},
  year   = {2026},
  url    = {https://github.com/Xiaooolong/vev},
  version = {0.1.0}
}
```
