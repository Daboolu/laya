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

内置 checkpoint：

| 名称 | 编码器 | 参数量 | 用途 | 默认最大上下文 |
|---|---|---:|---|---:|
| `english` | ModernBERT-large | 约 4.21 亿 | 英文 | 512 |
| `multilingual` | mmBERT-base | 约 3.22 亿 | 100+ 语言 | 1024 |
| `typed-decisions` | ModernBERT-large | 约 4.21 亿 | 特定决策工作流 | 1024 |

`Router` 默认按语言选择英文或多语言模型，也可通过 `model="english"` 显式指定。模型在第一次使用时下载并载入内存。

## 模型架构

Laya 是非自回归的判别模型，不是生成式语言模型。一次调用中的多个问题组成一个 batch，只执行一次模型前向计算：

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
