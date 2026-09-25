"""Laya：本地、非生成式的结构化文本决策引擎。"""

from .presets import (
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, RouteDecision, Router

__version__ = "0.3.20"


def __getattr__(name):
    """延迟导入 PyTorch，单纯做语言路由时不支付模型启动成本。"""
    if name not in {"Agent", "RLAgent", "load"}:
        raise AttributeError(f"模块 {__name__!r} 没有属性 {name!r}")
    from .agent import Agent, RLAgent, load

    value = {"Agent": Agent, "RLAgent": RLAgent, "load": load}[name]
    globals()[name] = value
    return value


__all__ = [
    "DEFAULT_MODELS",
    "Agent",
    "RLAgent",
    "RouteDecision",
    "Router",
    "guard_questions",
    "load",
    "moderation_questions",
    "router_questions",
    "triage_questions",
]
