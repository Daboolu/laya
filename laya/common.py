"""Laya 的模型结构、输入编码和概率计算。"""

import json
import math

import numpy as np
import torch
from torch import nn

QUESTION_TYPES = {"choice": 0, "score": 1, "noul": 2}
TYPE_NAMES = {value: key for key, value in QUESTION_TYPES.items()}


def serialize_state(state) -> str:
    """把字符串或 JSON 状态转换为模型可读文本。"""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def _noul_labels(labels=None):
    labels = labels or {"false": "false", "true": "true"}
    if not isinstance(labels, dict) or set(labels) != {"false", "true"}:
        raise ValueError("noul 的 labels 必须且只能包含 false 和 true")
    false, true = labels["false"], labels["true"]
    if (
        not all(isinstance(x, str) and x.strip() for x in (false, true))
        or false == true
    ):
        raise ValueError("noul 的 false/true 标签必须是两个不同的非空字符串")
    return false.strip(), true.strip()


def render_options(question: dict) -> list[str]:
    """按固定顺序生成候选项文本；noul 始终是 false、true。"""
    kind, criteria = question["type"], question.get("criteria")
    if kind == "choice":
        return [
            str(key) if value in (None, "") else f"{key}: {value}"
            for key, value in criteria.items()
        ]
    if kind == "score":
        return [f"level {i}: {value}" for i, value in enumerate(criteria)]
    false, true = _noul_labels(question.get("labels"))
    criteria = criteria or {}
    return [
        f"{false}: {criteria.get('false') or 'no, the statement does not hold'}",
        f"{true}: {criteria.get('true') or 'yes, the statement holds'}",
    ]


def build_sequence(tokenizer, state, question, max_len=512, head_max_len=192):
    """生成 checkpoint 训练时使用的 `[MASK] 候选项 + 状态` 序列。"""
    mask = tokenizer.mask_token
    instruction = str(question["instructions"]).replace(mask, " ")
    head = tokenizer(
        f"{question['type']} question: {instruction}", add_special_tokens=False
    )["input_ids"]

    options = []
    for text in render_options(question):
        ids = tokenizer(
            " " + text.replace(mask, " "),
            add_special_tokens=False,
            truncation=True,
            max_length=48,
        )["input_ids"]
        options.append([tokenizer.mask_token_id] + ids)

    budget = head_max_len - sum(map(len, options))
    if budget < 16:
        per_option = max(4, (head_max_len - 16) // len(options))
        options = [ids[:per_option] for ids in options]
        budget = head_max_len - sum(map(len, options))

    ids = [tokenizer.cls_token_id] + head[: max(8, budget)] + [tokenizer.sep_token_id]
    markers = []
    for option in options:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tokenizer.sep_token_id)

    state_ids = tokenizer(
        serialize_state(state).replace(mask, " "), add_special_tokens=False
    )["input_ids"]
    room = max(0, max_len - len(ids) - 1)
    state_ids = (
        state_ids[-room:] if isinstance(state, list) and room else state_ids[:room]
    )
    ids.extend(state_ids)
    ids.append(tokenizer.sep_token_id)
    return ids[:max_len], [position for position in markers if position < max_len]


class DecisionModel(nn.Module):
    """双向 Transformer 编码器加候选项决策头。"""

    def __init__(self, encoder: nn.Module, head_layers=2, action_count=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        width = encoder.config.hidden_size
        layer = nn.TransformerEncoderLayer(
            width,
            max(1, width // 64),
            4 * width,
            dropout,
            batch_first=True,
            norm_first=True,
        )
        self.head = nn.TransformerEncoder(
            layer, head_layers, enable_nested_tensor=False
        )
        self.type_emb = nn.Embedding(3, width)
        self.scorer = nn.Sequential(
            nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)
        )
        self.act_head = nn.Sequential(
            nn.Linear(width + 4, 256), nn.GELU(), nn.Linear(256, action_count)
        )
        self.register_buffer("temperature", torch.ones(3))

    def forward(
        self, input_ids, attention_mask, marker_pos, marker_mask, question_type
    ):
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        hidden = hidden + self.type_emb(question_type)[:, None, :]
        hidden = self.head(hidden, src_key_padding_mask=~attention_mask.bool())
        index = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, hidden.size(-1))
        logits = self.scorer(torch.gather(hidden, 1, index)).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        probabilities = torch.softmax(logits.detach(), -1)
        option_count = marker_mask.sum(-1).clamp(min=2).float()
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-9))).sum(
            -1
        ) / torch.log(option_count)
        top2 = (
            torch.cat([probabilities, torch.zeros_like(probabilities)], dim=-1)
            if probabilities.size(-1) == 1
            else probabilities.topk(2, -1).values
        )
        features = torch.stack(
            [top2[:, 0], top2[:, 0] - top2[:, 1], entropy, option_count / 255.0], -1
        )
        actions = self.act_head(torch.cat([hidden[:, 0].float(), features], -1))
        return logits, actions


def _apply_rope_config(config):
    """兼容 Transformers 4/5 对分层 RoPE 参数的不同命名。"""
    rope = getattr(config, "rope_parameters", None)
    if not isinstance(rope, dict):
        return
    for layer_type, attribute in (
        ("full_attention", "global_rope_theta"),
        ("sliding_attention", "local_rope_theta"),
    ):
        value = rope.get(layer_type, {}).get("rope_theta")
        if value is not None and hasattr(config, attribute):
            setattr(config, attribute, float(value))


def build_model(config: dict, encoder_dir: str) -> DecisionModel:
    """只按配置创建空模型；真正参数全部来自 Laya checkpoint。"""
    from transformers import AutoConfig, AutoModel

    encoder_config = AutoConfig.from_pretrained(encoder_dir or config["encoder"])
    _apply_rope_config(encoder_config)
    encoder = AutoModel.from_config(encoder_config, attn_implementation="sdpa")
    return DecisionModel(
        encoder,
        head_layers=config.get("head_layers", 2),
        action_count=len(config.get("act_costs", {})) + 1,
    )


def collate(items: list[dict], pad_id: int) -> dict[str, torch.Tensor]:
    """把每个问题编码为同一批张量。"""
    rows, length = len(items), max(len(item["ids"]) for item in items)
    options = max(len(item["markers"]) for item in items)
    input_ids = torch.full((rows, length), pad_id, dtype=torch.long)
    attention = torch.zeros((rows, length), dtype=torch.long)
    marker_pos = torch.zeros((rows, options), dtype=torch.long)
    marker_mask = torch.zeros((rows, options), dtype=torch.bool)
    for row, item in enumerate(items):
        input_ids[row, : len(item["ids"])] = torch.tensor(item["ids"])
        attention[row, : len(item["ids"])] = 1
        marker_pos[row, : len(item["markers"])] = torch.tensor(item["markers"])
        marker_mask[row, : len(item["markers"])] = True
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "marker_pos": marker_pos,
        "marker_mask": marker_mask,
        "question_type": torch.tensor([item["question_type"] for item in items]),
    }


def temperature_bucket(question_type: int, option_count: int) -> str:
    size = (
        "2"
        if option_count <= 2
        else "3-5"
        if option_count <= 5
        else "6-10"
        if option_count <= 10
        else "11+"
    )
    return f"{TYPE_NAMES[question_type]}:{size}"


def clamp_temperature(value) -> float:
    """把异常校准温度限制在安全范围。"""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return 1.0 if not math.isfinite(value) else min(5.0, max(0.5, value))


def entropy_confidence(probabilities: np.ndarray) -> float:
    """返回归一化熵置信度。"""
    if len(probabilities) < 2:
        return 1.0
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1))).sum()
    return float(np.clip(1 - entropy / math.log(len(probabilities)), 0, 1))
