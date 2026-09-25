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

| 名称 | 用途 | 最大上下文 |
|---|---|---:|
| `english` | 英文，ModernBERT-large | 512 |
| `multilingual` | 100+ 语言，mmBERT-base | 1024 |
| `typed-decisions` | 特定决策工作流 | 1024 |

`Router` 默认按语言选择英文或多语言模型，也可通过 `model="english"` 显式指定。模型在第一次使用时下载并载入内存。

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
