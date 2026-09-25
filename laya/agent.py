"""Laya checkpoint 的最小本地推理运行时。"""

import json
import os
import tempfile
import warnings
from contextlib import nullcontext

import numpy as np
import torch

from .common import (
    QUESTION_TYPES,
    build_model,
    build_sequence,
    clamp_temperature,
    collate,
    entropy_confidence,
    render_options,
    temperature_bucket,
)


def _fix_tokenizer_config(model_dir: str) -> None:
    """修正 Transformers 4/5 不兼容的 tokenizer 配置。"""
    path = os.path.join(model_dir, "tokenizer", "tokenizer_config.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as file:
            config = json.load(file)
        changed = False
        if config.get("tokenizer_class") in (None, "TokenizersBackend"):
            config["tokenizer_class"] = "PreTrainedTokenizerFast"
            config.pop("backend", None)
            config.pop("is_local", None)
            changed = True
        if isinstance(config.get("extra_special_tokens"), list):
            config["extra_special_tokens"] = {
                f"extra_{i}": value
                for i, value in enumerate(config["extra_special_tokens"])
            }
            changed = True
        if changed:
            descriptor, temporary = tempfile.mkstemp(
                dir=os.path.dirname(path), prefix=".tokenizer."
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(config, file, ensure_ascii=False, indent=2)
            os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as error:
        warnings.warn(
            f"无法修正 tokenizer 配置 {path}：{error}", RuntimeWarning, stacklevel=2
        )


def _validate_question(name: str, question: dict) -> None:
    """在进入 tokenizer 前给出清晰的输入错误。"""
    if not isinstance(question, dict):
        raise TypeError(f"问题 {name!r} 必须是对象")
    kind = question.get("type")
    if kind not in QUESTION_TYPES:
        raise ValueError(f"问题 {name!r} 的 type 必须是 choice、score 或 noul")
    if (
        not isinstance(question.get("instructions"), str)
        or not question["instructions"].strip()
    ):
        raise ValueError(f"问题 {name!r} 缺少非空 instructions")
    criteria = question.get("criteria")
    if kind == "choice" and (not isinstance(criteria, dict) or not criteria):
        raise ValueError(f"choice 问题 {name!r} 需要非空 criteria 对象")
    if kind == "score" and (not isinstance(criteria, list) or not criteria):
        raise ValueError(f"score 问题 {name!r} 需要非空 criteria 列表")
    if (
        kind == "noul"
        and criteria is not None
        and (not isinstance(criteria, dict) or not set(criteria) <= {"false", "true"})
    ):
        raise ValueError(f"noul 问题 {name!r} 的 criteria 只能包含 false/true")
    if "labels" in question and kind != "noul":
        raise ValueError(f"只有 noul 问题 {name!r} 可以设置 labels")


def _verify_weights(model, config: dict, weights: dict, model_id: str) -> None:
    """确保配置和权重确实属于兼容的 Laya 模型。"""
    missing_config = [key for key in ("encoder", "head_layers") if key not in config]
    if missing_config:
        raise ValueError(f"模型 {model_id!r} 缺少配置项：{missing_config}")
    for prefix in ("encoder.", "type_emb.", "scorer.", "act_head."):
        if not any(key.startswith(prefix) for key in weights):
            raise ValueError(f"模型 {model_id!r} 缺少 {prefix} 权重")
    expected = model.state_dict()
    missing = [key for key in expected if key not in weights]
    mismatch = [
        key
        for key in expected
        if key in weights and expected[key].shape != weights[key].shape
    ]
    if missing or mismatch:
        raise ValueError(
            f"模型 {model_id!r} 与当前结构不兼容：缺少 {missing[:3]}，形状不符 {mismatch[:3]}"
        )


class Agent:
    """加载一个 Laya checkpoint，并回答结构化问题。"""

    def __init__(
        self,
        model_id_or_path="convaiinnovations/laya",
        device=None,
        token=None,
        subfolder=None,
    ):
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        try:
            from transformers.initialization import no_init_weights
        except ImportError:
            from transformers.modeling_utils import no_init_weights

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            prefix = f"{subfolder}/" if subfolder else ""
            patterns = [
                prefix + name
                for name in (
                    "rl_agent_config.json",
                    "model.safetensors",
                    "tokenizer/*",
                    "encoder/*",
                )
            ]
            model_dir = snapshot_download(
                model_id_or_path,
                token=token or os.environ.get("HF_TOKEN") or None,
                allow_patterns=patterns,
            )
        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)

        _fix_tokenizer_config(model_dir)

        config_path = os.path.join(model_dir, "rl_agent_config.json")
        weights_path = os.path.join(model_dir, "model.safetensors")
        if not os.path.isfile(config_path) or not os.path.isfile(weights_path):
            raise FileNotFoundError(
                "目录中缺少 rl_agent_config.json 或 model.safetensors"
            )
        with open(config_path, encoding="utf-8") as file:
            self.config = json.load(file)

        self.model_id = model_id_or_path
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            warnings.warn("CUDA 不可用，改用 CPU", RuntimeWarning, stacklevel=2)
            self.device = torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(model_dir, "tokenizer")
        )
        with no_init_weights():
            self.model = build_model(self.config, os.path.join(model_dir, "encoder"))
        weights = load_file(weights_path)
        _verify_weights(self.model, self.config, weights, model_id_or_path)
        self.model.load_state_dict(weights, strict=True)
        self.model.to(self.device).eval()

        raw_temperatures = self.config.get("temperature", [1, 1, 1])
        raw_buckets = self.config.get("temperature_by_options", {})
        self.temperatures = [clamp_temperature(value) for value in raw_temperatures]
        self.bucket_temperatures = {
            key: clamp_temperature(value) for key, value in raw_buckets.items()
        }
        changed = [f"temperature[{i}]" for i, value in enumerate(raw_temperatures)
                   if clamp_temperature(value) != value]
        changed += [
            key for key, value in raw_buckets.items()
            if clamp_temperature(value) != value
        ]
        if changed:
            warnings.warn(
                f"checkpoint 的校准温度超出 [0.5, 5.0]，已安全限制：{', '.join(changed)}；"
                "相关 confidence 不能视为严格校准概率",
                RuntimeWarning,
                stacklevel=2,
            )
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32

    @torch.no_grad()
    def predict(self, state, questions: dict, max_len=None, head_max_len=None) -> dict:
        """在一个前向过程中回答 state 上的全部问题。"""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions 必须是非空对象")
        max_len = max_len or self.config.get("max_len", 512)
        head_max_len = head_max_len or self.config.get("head_max_len", 192)
        names, items = list(questions), []
        for name in names:
            question = questions[name]
            _validate_question(name, question)
            ids, markers = build_sequence(
                self.tokenizer, state, question, max_len, head_max_len
            )
            if len(markers) != len(render_options(question)):
                raise ValueError(
                    f"问题 {name!r} 的候选项超过 head_max_len={head_max_len}"
                )
            items.append(
                {
                    "ids": ids,
                    "markers": markers,
                    "question_type": QUESTION_TYPES[question["type"]],
                }
            )

        batch = collate(items, self.tokenizer.pad_token_id)
        args = [
            batch[key].to(self.device)
            for key in (
                "input_ids",
                "attention_mask",
                "marker_pos",
                "marker_mask",
                "question_type",
            )
        ]
        context = (
            torch.autocast("cuda", dtype=self.dtype)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            logits, actions = self.model(*args)
        logits = logits.float().cpu().numpy()
        actions = torch.softmax(actions.float(), -1).cpu().numpy()

        answers = {}
        for row, (name, item) in enumerate(zip(names, items)):
            question = questions[name]
            count = len(item["markers"])
            kind_index = item["question_type"]
            temperature = self.bucket_temperatures.get(
                temperature_bucket(kind_index, count), self.temperatures[kind_index]
            )
            scaled = logits[row, :count] / temperature
            probabilities = np.exp(scaled - scaled.max())
            probabilities /= probabilities.sum()
            confidence = round(float(probabilities.max()), 4)
            answer = {
                "type": question["type"],
                "answer_confidence": confidence,
                "action": {"act_probability": round(float(actions[row, 0]), 4)},
            }
            if question["type"] == "choice":
                labels = list(question["criteria"])
                answer.update(
                    choice=labels[int(probabilities.argmax())],
                    probabilities={
                        label: round(float(value), 4)
                        for label, value in zip(labels, probabilities)
                    },
                    confidence=round(entropy_confidence(probabilities), 4),
                )
            elif question["type"] == "score":
                answer.update(
                    score=round(float((np.arange(count) * probabilities).sum()), 4),
                    probabilities={
                        str(i): round(float(value), 4)
                        for i, value in enumerate(probabilities)
                    },
                    confidence=round(entropy_confidence(probabilities), 4),
                )
            else:
                answer["noul"] = round(float(probabilities[1]), 4)
                answer["confidence"] = confidence
            answers[name] = answer

        return {
            "model": "laya-rl-agent",
            "answers": answers,
            "usage": {
                "input_tokens": int(batch["attention_mask"].sum()),
                "output_tokens": 0,
            },
        }

    system_one = predict


def load(
    model_id_or_path="convaiinnovations/laya", device=None, token=None, subfolder=None
) -> Agent:
    """加载单个 checkpoint。"""
    return Agent(model_id_or_path, device=device, token=token, subfolder=subfolder)


RLAgent = Agent
