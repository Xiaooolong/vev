# Vev

[![English](https://img.shields.io/badge/lang-English-blue)](README.md) [![简体中文](https://img.shields.io/badge/lang-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-red)](README_zh.md) [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-models%20%26%20data-yellow)](https://huggingface.co/collections/CountingSheep/vev-6abd3e17303828e10c49dc65)

**像 Jev 一样做判断，而且能看图的决策模型。**

对一段文本、一条 JSON 记录或一张截图，问是/否、单选或打分的问题，约 40 毫秒拿回每个候选答案的概率。它不生成文本，所以答案一定落在你给定的选项里，也可以直接按概率设阈值。权重开放，跑在你自己的 GPU 上。

- **接口和 Jev 一样。** [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 是 TypeSafe 提供的托管决策模型。Vev 支持同样的 `/v1/systemone` 请求格式，用 TypeSafe SDK 写的代码改一下 base URL 就能接上。
- **能看图。** 截图、照片可以和文本一起放进请求。我们实测过读界面状态、按书面规则审核图片、核对生成图是否符合提示词这几类用途（见[能做什么](#能做什么)）。
- **两种尺寸，中英文都行。** `vev-4b` 约占 10 GB 显存，`vev-9b` 约 19 GB，都是在 Qwen3.5 上微调的。

<img src="docs/checkout.png" width="560" alt="一个结账页面，顶部红色提示：Payment failed: your card was declined.">

```python
resp = client.system_one(
    state={"screen": {"image": {"url": checkout_png}}},
    questions={
        "error": Noul(instructions="Does the screen show an error message?"),
        "step": Choice(instructions="Which checkout step is the user on?",
                       criteria={"shipping": None, "payment": None, "review": None}),
        "next": Choice(instructions="What should the user do next?",
                       criteria={"retry": "Try another card", "wait": "Wait for the order to ship",
                                 "nothing": "Nothing, the order went through"}),
    },
)
# vev-4b:  error 0.991
#          step  {"shipping": 0.005, "payment": 0.765, "review": 0.229}
#          next  {"retry": 0.983, "wait": 0.004, "nothing": 0.014}
```

"当前在哪一步"这道题的答案没有另外两道那么确定，这正是概率的用处：进度条上 Payment 和 Review 两个步骤都看得到。

当前是 v0.1 研究预览版。权重只许非商业使用；在大多数文本集上，Vev 不如 Jev 准，见[局限](#局限)。

## 开始使用

```bash
pip install git+https://github.com/Xiaooolong/vev
vev serve --model CountingSheep/vev-4b          # 首次启动会下载权重，监听 127.0.0.1:8009
```

需要 Python 3.11 及以上、NVIDIA GPU 和 CUDA 版的 PyTorch；CPU 和 Apple Silicon 没有测过。Windows 上 pip 默认装的是 CPU 版 PyTorch，请先按 [pytorch.org](https://pytorch.org/get-started/locally/) 的说明把 `torch` 和 `torchvision` 一起装好。测试过的版本：torch 2.8.0、transformers 5.17.0、peft 0.21.0。

然后用 TypeSafe 的 Python SDK 提问（`pip install typesafe-sdk`）：

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

`Noul` 是这套接口对是/否问题的叫法。状态可以是字符串，也可以是任意 JSON 对象；图片写成 `{"image": {"url": "data:image/png;base64,..."}}`，放在状态里的任意位置：

```python
import base64

checkout_png = "data:image/png;base64," + base64.b64encode(open("docs/checkout.png", "rb").read()).decode()
```

### 其他运行方式

- `--model CountingSheep/vev-4b-lora` 只下载 adapter（130 MB），加载时套到 `Qwen/Qwen3.5-4B` 上；本地 Hugging Face 缓存里已有这个基座的话会直接复用。`--revision v0.1.0` 把权重固定在这个版本。
- Docker：在本仓库的 clone 里 `docker build -t vev .`，然后 `docker run --gpus all -p 8009:8009 -v ~/.cache/huggingface:/root/.cache/huggingface vev --model CountingSheep/vev-4b`。
- 不起 HTTP 服务，比如离线给一大批数据打分：

  ```python
  from vev.model import CheckpointEngine

  engine = CheckpointEngine("CountingSheep/vev-4b")
  result = engine.run({"ticket": "My kettle never arrived."},
                      {"refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"}})
  print(result.answers["refund"]["noul"])
  ```

- 接到其他推理框架上：[spec/systemone-api.md](spec/systemone-api.md)（英文）第 10 节给出了 prompt 模板和要读取哪些 token 的概率。

### 速度

一次请求里的所有问题合成一个 batch 计算。单张 H800 上，`vev-4b` 的耗时：

| 请求 | 1 个问题 | 10 个问题 | 100 个问题 |
|---|---|---|---|
| 短文本 | 38 ms | 58 ms | 232 ms |
| 7k token 的长文本 | 280 ms | 334 ms | 1.07 s |
| 一张 1 MP 的图片 | 78 ms | 120 ms（25 个问题：143 ms） | — |

`vev-9b` 最多慢 30%；完整曲线见 `conformance/results/vev-*/curves.json`。一个 `vev serve` 进程一次处理一个请求，其余请求排队；需要更高吞吐时，每张 GPU 起一个进程，前面加负载均衡。

## 能做什么

按问题类型列出在人工标注数据上的正确率。文本部分是公开评测集；图片部分是我们用带标注的公开数据转换出的判断集（[详细结果](#详细结果)）。

| 问题类型 | 例子 | vev-4b | vev-9b |
|---|---|---|---|
| Jev 风格的文本判断 | JevBench 公开子集 | 0.766 | 0.823 |
| 同上，但来源是模型没见过的 | kev transfer-v4 | 0.776 | 0.781 |
| 中文判断 | judgekit | 0.962 | 0.938 |
| 改一点细节答案就会反转的成对题 | nimble | 0.707 | 0.747 |
| App 截图里的界面状态 | "有没有打开（勾选）的开关或复选框？" / "有没有文本输入框？" | 0.928 / 0.951 | 0.923 / 0.955 |
| 按书面安全规则审核图片 | LlavaGuard 的政策类别（仇恨、暴力、色情等） | 0.742 | 0.724 |
| 核对生成图是否符合提示词 | "图里有没有按提示词画出 'grass' 这个元素？" | 0.715 | 0.703 |
| 关于照片的一般性问题 | MMBench-EN / POPE / MMStar | 0.881 / 0.889 / 0.627 | 0.902 / 0.897 / 0.674 |
| 找游戏截图里的 bug | 画面错误 / 物体穿模（VideoGameQA-Bench） | 0.613 / 0.671 | 0.626 / 0.729 |

界面状态类问题最准；规则审核、生成图核对和较难的视觉评测处在中间；找游戏 bug 太弱，不能依赖。这些数字描述的是这些数据集上的表现，不是你的数据。要在自己的问题上验证，把几百条带标注的样例写成[记录格式](evals/README.md)（英文），对你的服务跑 `python -m evals.run`，它会给出正确率、校准误差，以及在不同覆盖率下最有把握那部分答案的正确率。

## 详细结果

以下都是全量集上的正确率，用 `evals/` 里的评测脚本测得。两个系统对比时，在同一批题上做配对 bootstrap 检验（95% 区间），区间包含 0 的记为不显著；图片集按图片整组重采样，同一张图上的题一起抽。结果是用 0.1.0 版算的，那时每个问题单独做一次前向；0.1.1 对只含一个问题的请求给出相同的概率，每次请求另加 5 个或 50 个问题时，下面所有显著性结论都不变。每个集的完整指标（含 Brier 分数和校准误差）在 [results/](results/)，其中下面没列出的集是开发时做诊断用的。

### 文本

| 评测集 | 题数 | kev-4B | vev-4b | vev-9b | Jev |
|---|---|---|---|---|---|
| [judgekit](https://github.com/lexingtonhibiki/judgekit)（中文） | 130 | 0.846 | 0.962 | 0.938 | 0.962 |
| [JevBench](https://github.com/fstandhartinger/jevbench)（公开子集） | 231 | 0.719 | 0.766 | 0.823 | 0.861 |
| [nimble](https://github.com/bespokelabsai/nimble) | 324 | 0.735 | 0.707 | 0.747 | 0.923 |
| [kev](https://github.com/jaredpalmer/kev) transfer-v4（留出来源） | 1528 | 0.802 | 0.776 | 0.781 | 0.854 |

kev-4B 用同一套脚本在本地跑；Jev 通过 TypeSafe 的 API 调用（`jev-latest`，2026 年 9 月 23–24 日）；训练没有用到任何 Jev 的输出。各组正确率差异（点数）：

- vev-4b 对 kev-4B：judgekit +11.5，kev transfer-v4 −2.6；JevBench 和 nimble 不显著。
- vev-9b 对 kev-4B：judgekit +9.2，JevBench +10.4，kev transfer-v4 −2.1；nimble 不显著。
- vev-4b 对 Jev：JevBench −9.5，nimble −21.6，kev transfer-v4 −7.8；judgekit 持平。
- vev-9b 对 Jev：nimble −17.6，kev transfer-v4 −7.3；judgekit（−2.3）和 JevBench（−3.9）不显著。

### 微调前后

基座模型用同样方式读出（未微调，零样本）与 Vev 的对比：

| 评测集 | 题数 | Qwen3.5-4B | vev-4b | Qwen3.5-9B | vev-9b |
|---|---|---|---|---|---|
| judgekit | 130 | 0.962 | 0.962 | 0.938 | 0.938 |
| JevBench | 231 | 0.758 | 0.766 | 0.805 | 0.823 |
| nimble | 324 | 0.710 | 0.707 | 0.710 | **0.747** |
| kev transfer-v4 | 1528 | 0.758 | **0.776** | 0.764 | **0.781** |
| MMBench-EN | 1164 | 0.872 | 0.881 | 0.887 | **0.902** |
| POPE | 9000 | 0.894 | *0.889* | 0.894 | 0.897 |
| MMStar | 1498 | 0.544 | **0.627** | 0.608 | **0.674** |

加粗表示显著变好，斜体表示显著变差。

### 图片

公开的视觉评测大多是就图片内容提问，而 Vev 要做的是按给定规则判断一张图，所以我们用带人工标注的公开数据转换出了几个判断集。转换脚本在 `evals/datasets/`，转换后的数据不再分发。Jev 的接口只收文本，所以这里没有 Jev 一列。

| 评测集 | 判断内容 | 题数 | Qwen3.5-4B | vev-4b | Qwen3.5-9B | vev-9b |
|---|---|---|---|---|---|---|
| policy_mod | 图片是否违反给定的安全规则（LlavaGuard） | 659 | 0.686 | **0.742** | 0.666 | **0.724** |
| game_glitch | 游戏截图里有没有画面错误（VideoGameQA-Bench） | 1000 | 0.646 | 0.613 | 0.577 | **0.626** |
| game_clip | 物体是否穿模进角色（VideoGameQA-Bench） | 686 | 0.561 | **0.671** | 0.736 | 0.729 |
| ui_toggle | 界面上有没有处于开/关状态的开关（MobileViews） | 800 | 0.901 | **0.928** | 0.921 | 0.923 |
| ui_input | 界面上有没有文本输入框（MobileViews） | 800 | 0.949 | 0.951 | 0.948 | 0.955 |
| t2i_elem | 生成的图片是否画出了提示词里的某个元素（EvalMuse） | 1027 | 0.690 | **0.715** | 0.733 | 0.703 |

## 局限

- 这些概率用来排序答案很好用，但不是精确的频率：期望校准误差（ECE）随任务在 0.01 到 0.14 之间（Jev 在文本集上是 0.04–0.07）。在"这里有没有问题"这类题上，Vev 答"没有"的次数比标注里多。如果要按阈值自动处理，请在你自己的标注数据上定阈值（[做法](#能做什么)）。
- 把选项倒过来排，JevBench 和 kev transfer-v4 上首选答案会变的题，`vev-9b` 占 13%，`vev-4b` 占 17%；kev-4B 是 10–12%，Jev 不到 4%。在你的应用里请保持选项顺序固定。
- 需要多步推理的判断，不如基座模型先思考再回答：在一个内部场景集上（未公开）差 4–8 个点；在我们试过的大多数公开集上，单次读出和思考模式一样准或更准。
- 一个问题和别的问题一起问时，概率会比单独问变动百分之几以内（bf16 舍入；p99 0.025），首选答案变化的题不超过 0.8%。同一个请求每次返回的结果都完全相同（[接口规范 §6](spec/systemone-api.md#6-semantic-guarantees)）。
- 长文本从批量里得到的加速有限：7k token 的文本问 100 个问题，耗时是 1 个问题的 3.8 倍。接近 32k token 上限的文本，`vev-9b` 在 24 GB 显卡上放不下。某种长度或批量形状第一次出现时，GPU kernel 要先调优，可能多花几秒。
- 图片判断集是我们自己从公开数据转换的，不是公认的评测基准。
- "兼容"指请求和响应的格式与 `/v1/systemone` 一致，不代表 Vev 的行为和 Jev 一样。Vev 与 TypeSafe 没有关联，也未获其认可。

## 原理

每个问题会被组织成一段 prompt，包含状态、问题和带字母编号的选项，Vev 读取模型对答案 token（Yes/No、选项字母、档位数字）的下一 token 概率，只在合法答案之间重新归一化。微调（在 Qwen3.5 上做 LoRA）改的就是这组概率，没有另加分类头。

| 文档（英文） | 内容 |
|---|---|
| [spec/systemone-api.md](spec/systemone-api.md) | 请求格式、限制、错误码、行为保证、prompt 模板 |
| [TRAINING.md](TRAINING.md) | 训练配方、数据来源及其许可、哪些部分无法重建 |
| [evals/](evals/README.md) | 转换评测集，并对任何 `/v1/systemone` 服务跑评测 |
| [conformance/](conformance/README.md) | 按接口规范检查一个实现，包括官方接口 |

| 模型 | 基座 | Hugging Face |
|---|---|---|
| `vev-4b` | Qwen3.5-4B | [CountingSheep/vev-4b](https://huggingface.co/CountingSheep/vev-4b)（合并权重），[CountingSheep/vev-4b-lora](https://huggingface.co/CountingSheep/vev-4b-lora)（adapter） |
| `vev-9b` | Qwen3.5-9B | [CountingSheep/vev-9b](https://huggingface.co/CountingSheep/vev-9b)（合并权重），[CountingSheep/vev-9b-lora](https://huggingface.co/CountingSheep/vev-9b-lora)（adapter） |

## 许可

代码：Apache 2.0。模型权重：CC BY-NC 4.0，仅限非商业使用，因为部分训练数据只允许研究用途；各数据来源及其条款见 [TRAINING.md](TRAINING.md)。

## 引用

```bibtex
@software{vev2026,
  title  = {Vev: Jev-like decision models that can also see images},
  author = {Wang, Xiaolong},
  year   = {2026},
  url    = {https://github.com/Xiaooolong/vev},
  version = {0.1.1}
}
```
