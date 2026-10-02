# Laya 中文精简版

Laya 是一个本地结构化文本决策模型。它不生成文字，而是根据输入回答 `choice`（选择）、`score`（评分）和 `noul`（是/否概率）问题。

本分支只保留 Python 推理、语言路由和命令行。

## 最小复现

需要 Python 3.10+、[uv](https://docs.astral.sh/uv/) 和可访问 Hugging Face 的网络。首次推理会自动下载公开 checkpoint。

```bash
git clone https://github.com/Daboolu/laya.git
cd laya
uv venv --python 3.12
uv pip install --python .venv/bin/python -e .
.venv/bin/python examples/quickstart.py
```

CUDA 用户应先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)安装与驱动匹配的 PyTorch。

## Python 用法

```python
from laya import Router

router = Router(device="cuda")
result = router.predict(
    {"客户消息": "我被重复扣费了，请退还重复收取的费用。"},
    {
        "处理部门": {
            "type": "choice",
            "instructions": "应该由哪个部门处理这条客户消息？",
            "criteria": {
                "账单部门": "处理发票、付款和退款",
                "技术部门": "处理程序错误、服务中断和系统故障",
            },
        },
        "是否要求退款": {
            "type": "noul",
            "instructions": "客户是否明确要求退款？",
        },
    },
)

print(result["answers"])
```

## 输入模板

`Router.predict(state, questions)` 接收两个必填参数：

- `state`：待分析内容，可以是字符串、字典或列表；字典和列表会先序列化为 JSON。
- `questions`：问题字典。一次可以提交多个问题，每个问题的键也是输出中的答案键。

完整模板如下。`type`、`instructions`、`criteria`、`labels` 以及 `choice`、`score`、`noul` 是固定协议字段，不能翻译；问题名称、说明、候选标签和业务数据可以使用中文。

```python
state = {
    "任意业务字段": "需要模型判断的内容",
}

questions = {
    "分类问题名称": {
        "type": "choice",
        "instructions": "需要从候选项中判断什么？",
        "criteria": {
            "候选标签一": "候选项的含义",
            "候选标签二": "候选项的含义",
        },
    },
    "评分问题名称": {
        "type": "score",
        "instructions": "需要按什么标准评分？",
        "criteria": ["最低等级", "中间等级", "最高等级"],
    },
    "真假问题名称": {
        "type": "noul",
        "instructions": "需要判断真假的陈述？",
        # criteria 可省略；如需补充语义，只能使用 false 和 true 两个键。
        "criteria": {
            "false": "不成立时的含义",
            "true": "成立时的含义",
        },
        # labels 可省略；它只改变送给模型的两个标签，不改变 noul=P(true)。
        "labels": {"false": "B", "true": "A"},
    },
}

result = router.predict(state, questions)
```

三种问题类型：

| `type` | `criteria` | 输出 |
|---|---|---|
| `choice` | 非空字典：`标签 -> 描述` | 概率最高的标签及所有候选概率 |
| `score` | 按低到高排列的非空列表 | 等级编号的概率加权平均值及各等级概率 |
| `noul` | 可省略，或提供 `false`/`true` 描述 | `true` 的概率，即 `P(true)` |

## 输出模板

返回值是可以直接 JSON 序列化的字典。下面同时展示三种答案的结构，数值仅作示意：

```json
{
  "model": "laya-rl-agent",
  "answers": {
    "分类问题名称": {
      "type": "choice",
      "answer_confidence": 0.90,
      "action": {"act_probability": 1.0},
      "choice": "候选标签一",
      "probabilities": {
        "候选标签一": 0.90,
        "候选标签二": 0.10
      },
      "confidence": 0.53
    },
    "评分问题名称": {
      "type": "score",
      "answer_confidence": 0.85,
      "action": {"act_probability": 1.0},
      "score": 1.8,
      "probabilities": {"0": 0.05, "1": 0.10, "2": 0.85},
      "confidence": 0.53
    },
    "真假问题名称": {
      "type": "noul",
      "answer_confidence": 0.85,
      "action": {"act_probability": 1.0},
      "noul": 0.85,
      "confidence": 0.85
    }
  },
  "usage": {
    "input_tokens": 123,
    "output_tokens": 0
  },
  "routing": {
    "model": "multilingual",
    "repo": "convaiinnovations/laya/multilingual",
    "reason": "检测到非拉丁文字：han",
    "detection": {}
  }
}
```

关键输出字段：

- `choice`：`choice` 问题最终选择的标签。
- `score`：等级下标的期望值。例如等级 `0、1、2` 的概率分别为 `0.05、0.10、0.85`，则结果为 `0×0.05 + 1×0.10 + 2×0.85 = 1.8`。
- `noul`：`true` 的概率。可用 `result["answers"][问题名]["noul"] >= 0.5` 转成布尔值。
- `probabilities`：所有候选项的概率，合计约为 1。
- `answer_confidence`：概率最高候选项的概率。对 `noul` 而言，如果 `noul < 0.5`，它等于 `1 - noul`。
- `confidence`：`choice` 和 `score` 使用归一化熵表示分布集中程度；`noul` 当前与 `answer_confidence` 相同。这不是准确率保证。
- `action.act_probability`：独立动作头输出，现有 checkpoint 上暂不适合作为可靠阈值。
- `usage.input_tokens`：该批问题实际送入模型的 token 总数；模型不生成文本，因此 `output_tokens` 为 0。
- `routing`：自动选择的 checkpoint 以及语言检测依据。显式指定 `model` 或 `lang` 时，`detection` 可以是 `null`。

内置 checkpoint：

| 名称 | 编码器 | 参数量 | 用途 | 默认最大上下文 |
|---|---|---:|---|---:|
| `english` | ModernBERT-large | 约 4.21 亿 | 英文 | 512 |
| `multilingual` | mmBERT-base | 约 3.22 亿 | 100+ 语言 | 1024 |
| `typed-decisions` | ModernBERT-large | 约 4.21 亿 | 特定决策工作流 | 1024 |

`Router` 默认按语言选择英文或多语言模型，也可通过 `model="english"` 显式指定。模型在第一次使用时下载并载入内存。

## 模型架构

Laya 本质上是 Encoder-only Transformer：它是非自回归的判别模型，不是生成式语言模型。一次调用中的多个问题组成一个 batch，只执行一次模型前向计算：

```text
状态 + 类型化问题 + 候选项
          │
          ▼
双向 Transformer 编码器（ModernBERT 或 mmBERT）
          │
          ▼
问题类型嵌入 + 两层 Transformer 决策头
          │
          ├── 候选项评分头 ──► choice / score / noul 概率
          └── 动作判断头   ──► act_probability
```

每个问题会被编码为：

```text
[CLS] <问题类型> <问题说明> [SEP]
[MASK] <候选项 1> [MASK] <候选项 2> ... [SEP]
<状态文本或 JSON> [SEP]
```

- `choice`：每个候选项对应一个 `[MASK]` 位置，评分头为它们产生 logits，经温度缩放和 softmax 后返回概率最高的标签。
- `score`：同样得到各等级的概率，再返回等级编号的概率加权平均值。
- `noul`：固定比较 `[false, true]` 两个语义位置，返回 `P(true)`。
- 问题类型嵌入让同一网络区分三种任务；模型没有解码器，也不生成文本，因此 `output_tokens` 始终为 0。
- `confidence` 是概率分布的集中程度，`answer_confidence` 是被选答案的概率。两者都应在业务数据上重新验证和校准。

`act_probability` 来自独立动作头。上游现有评测认为它暂时没有可靠的筛选能力，不应把 `1.0` 理解为答案百分之百正确。

### 模型实际接收的张量

API 中的一份 `state` 配多个 `questions`。预处理时，每个问题会单独与同一份 `state` 拼成一条序列；这些序列再组成一个 batch。因此，一次 `predict()` 可以在一次前向计算中回答多个问题，但每个问题都有自己的候选项位置。

```text
input_ids       [问题数, 序列长度]       编码后的问题、候选项和状态
attention_mask  [问题数, 序列长度]       区分真实 token 与 padding
marker_pos      [问题数, 最大候选数]     每个候选项前 [MASK] 的位置
marker_mask     [问题数, 最大候选数]     区分真实候选项与 padding
question_type   [问题数]                 choice=0、score=1、noul=2
```

模型读取 `marker_pos` 指向的隐藏状态，为每个候选项产生一个 logit。训练时还会加入与候选项一一对应的 `target` 概率分布；推理时不需要标签。

## 模型下载与加载

`Router` 默认延迟加载模型：`predict()` 先检测输入语言，中文等非英语文本选择 `multilingual`，英文选择 `english`。首次使用时，`huggingface_hub.snapshot_download()` 从 [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) 下载当前 checkpoint 所需文件：

```text
rl_agent_config.json
model.safetensors
encoder/config.json
tokenizer/tokenizer.json
tokenizer/tokenizer_config.json
```

随后程序根据配置创建编码器和决策头，用 `safetensors` 载入权重，再执行 `model.to(device).eval()`。同一个 `Router` 会复用内存中的模型。文件默认保存在 Hugging Face 缓存中，可用 `HF_HOME=/path/to/cache` 修改缓存位置。

## 训练方式

本精简分支只保留推理代码，不包含训练脚本。下面描述的是上游公开的训练方法和[可复现微调 notebook](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)；公开基础 checkpoint 的完整原始数据配方没有包含在本分支中。

### 训练数据格式

逻辑上一条训练案例包含 `state`、`questions` 和对应的标准答案 `gold`。下面同时包含三种问题类型：

```json
{
  "id": "case-0001",
  "state": {
    "客户消息": "我被重复扣费了，请尽快退款。"
  },
  "questions": {
    "处理部门": {
      "type": "choice",
      "instructions": "应该由哪个部门处理？",
      "criteria": {
        "账单部门": "处理付款和退款",
        "技术部门": "处理程序和系统故障"
      }
    },
    "紧急程度": {
      "type": "score",
      "instructions": "这个请求有多紧急？",
      "criteria": ["不紧急", "普通", "紧急"]
    },
    "是否要求退款": {
      "type": "noul",
      "instructions": "客户是否明确要求退款？"
    }
  },
  "gold": {
    "处理部门": {
      "type": "choice",
      "label": "账单部门",
      "probabilities": {
        "账单部门": 1.0,
        "技术部门": 0.0
      }
    },
    "紧急程度": {
      "type": "score",
      "label": "2",
      "score": 2.0,
      "probabilities": {
        "0": 0.0,
        "1": 0.0,
        "2": 1.0
      }
    },
    "是否要求退款": {
      "type": "noul",
      "label": "true",
      "noul": 1.0,
      "probabilities": {
        "false": 0.0,
        "true": 1.0
      }
    }
  }
}
```

训练数据约束：

- `gold` 的键必须与 `questions` 的键对应。
- `choice.probabilities` 的键必须与 `criteria` 的候选标签一致。
- `score.probabilities` 使用从 `"0"` 开始的字符串下标，对应 `criteria` 中从低到高的等级。
- `noul.probabilities` 固定使用 `false` 和 `true`，其中 `noul` 就是 `P(true)`。
- 概率应非负且总和为 1。只有硬标签时可以使用 one-hot；多个标注者或教师模型给出不确定答案时可以保留软分布，例如 `0.7/0.3`。
- 上游 [`LocalLLaMA/typed-decisions`](https://huggingface.co/datasets/LocalLLaMA/typed-decisions) 数据集把 `state`、`questions`、`gold` 存为 JSON 字符串，notebook 读取后会执行 `json.loads()`；如果自行编写数据加载器，也可以直接存为 JSON 对象。

预处理会把一条案例拆成“每个问题一条训练序列”。上例产生 3 条序列；它们共享同一个 `state`，但分别拥有自己的 `question_type`、候选项位置和目标分布：

```text
choice target = [1.0, 0.0]
score  target = [0.0, 0.0, 1.0]
noul   target = [0.0, 1.0]  # 固定顺序：[false, true]
```

上游把这种方法称为 RLCD：使用严格适当评分规则作为奖励，并采用 GRPO 风格策略梯度训练。

1. 把每条样本整理为状态、类型化问题、候选项以及目标概率分布；目标可以是 one-hot 标签，也可以是软标签。
2. 同时微调预训练编码器和已有的决策头。每个候选项 `[MASK]` 位置产生一个 logit。
3. 对 logits 添加多组零均值噪声，得到一组候选概率分布；用严格适当评分规则计算奖励。奖励包含对数分数、球面分数，`score` 类型还包含排序概率分数（RPS）。这类奖励在预测分布等于真实分布时取得最优值。
4. 使用组内标准化的相对优势做 GRPO 风格策略梯度，同时加入目标分布的软交叉熵，稳定训练。
5. 训练完成后，在未参与梯度更新的校准集上分别拟合 `choice`、`score`、`noul` 的温度参数，再在独立测试集上评估准确率、Brier 分数和 ECE。

上游 typed-decisions 微调示例使用 1,200 个训练案例（6,000 个决策）、2 张 T4、DDP 和 4 个 epoch。关键设置为：每卡 micro-batch 8、梯度累积 4（有效 batch 64）、AdamW、编码器学习率 `2.5e-5`、决策头学习率 `1e-4`、余弦退火、FP16 和梯度检查点。训练约需 4–5 小时，具体时间取决于硬件。

要复现训练，请使用上游 notebook；当前仓库只能加载训练完成后导出的 `rl_agent_config.json`、tokenizer、encoder 配置和 `model.safetensors`。

## 命令行

只检测语言并选择模型，不加载权重：

```bash
.venv/bin/laya --route-only "我被重复扣费了"
```

运行客服分诊：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/laya --device cuda --preset triage \
  "我被重复扣费了，请退款。"
```

可用模板：`triage`、`guard`、`moderation`、`router`。

也可以加载本地 checkpoint 目录：

```python
from laya import load
agent = load("/path/to/checkpoint", device="cuda")
```

## 限制

- 它只做有限候选决策，不适合问答、摘要、改写和复杂推理。
- 候选项共享固定 token 预算，不建议一次提供超过 20 个选项。
- 公开 checkpoint 的概率校准并不完美，业务阈值应使用自己的数据验证。
- `noul` 是项目沿用的二分类名称，输出字段 `noul` 表示 `P(true)`。

本项目基于上游 [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)，许可证见 [LICENSE](LICENSE)。
