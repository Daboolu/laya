"""根据文本语言选择 Laya checkpoint。"""

import gc
import threading

from .lang import analyse

BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}
ALIASES = {
    "en": "english",
    "laya": "english",
    "default": "english",
    "multi": "multilingual",
    "ml": "multilingual",
    "typed": "typed-decisions",
    "typed_decisions": "typed-decisions",
}


def normalize_name(name: str) -> str:
    """把模型别名转换为标准名称。"""
    name = ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
    if name not in DEFAULT_MODELS:
        raise ValueError(f"未知模型 {name!r}，可选：{', '.join(DEFAULT_MODELS)}")
    return name


class RouteDecision(dict):
    """可直接序列化为 JSON 的路由结果。"""

    @property
    def model(self):
        return self["model"]

    @property
    def reason(self):
        return self["reason"]


class Router:
    """按需加载并复用英文、多语言或专用决策模型。"""

    def __init__(
        self,
        models=None,
        device=None,
        token=None,
        max_loaded=2,
        default="english",
        preload=False,
    ):
        self.models = dict(DEFAULT_MODELS)
        if models:
            self.models.update(
                {normalize_name(key): value for key, value in models.items()}
            )
        self.device = device
        self.token = token
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalize_name(default)
        self._agents = {}
        self._order = []
        self._lock = threading.RLock()
        if preload:
            self.preload()

    @property
    def loaded(self):
        """当前驻留内存的模型名称。"""
        return list(self._order)

    def load(self, name: str):
        """首次使用时下载并加载模型，之后复用同一 Agent。"""
        key = normalize_name(name)
        with self._lock:
            if key in self._agents:
                self._order.remove(key)
                self._order.append(key)
                return self._agents[key]
            from .agent import Agent

            spec = self.models[key]
            repo, subfolder = spec if isinstance(spec, (tuple, list)) else (spec, None)
            agent = Agent(
                repo, device=self.device, token=self.token, subfolder=subfolder
            )
            self._agents[key] = agent
            self._order.append(key)
            while len(self._order) > self.max_loaded:
                del self._agents[self._order.pop(0)]
            gc.collect()
            return agent

    def preload(self, names=None):
        """预先加载指定模型；未指定时加载全部模型。"""
        names = list(names or self.models)
        self.max_loaded = max(self.max_loaded, len(names))
        for name in names:
            self.load(name)
        return self

    def unload(self, name=None):
        """卸载一个或全部模型。"""
        names = list(self._agents) if name is None else [normalize_name(name)]
        for key in names:
            self._agents.pop(key, None)
            if key in self._order:
                self._order.remove(key)
        gc.collect()

    def route(
        self, state, questions=None, model=None, task=None, lang=None
    ) -> RouteDecision:
        """只决定使用哪个 checkpoint，不执行模型推理。"""
        if model:
            key, reason, detection = (
                normalize_name(model),
                f"显式指定 model={model!r}",
                None,
            )
        elif task:
            key = normalize_name(
                "typed-decisions" if task == "typed_decisions" else task
            )
            reason, detection = f"显式指定 task={task!r}", None
        elif lang:
            key = (
                "english"
                if lang.lower().split("-")[0] in ("en", "eng", "english")
                else "multilingual"
            )
            reason, detection = f"显式指定 lang={lang!r}", None
        else:
            detection = analyse(state)
            if detection["script"] not in ("latin", "unknown"):
                key = "multilingual"
                reason = f"检测到非拉丁文字：{detection['script']}"
            elif not detection["is_english"]:
                key, reason = "multilingual", "检测到非英语文本"
            else:
                key = self.default if detection["language_undecided"] else "english"
                reason = (
                    "语言不明确，使用默认模型"
                    if detection["language_undecided"]
                    else "检测到英语文本"
                )
        spec = self.models[key]
        repo, subfolder = spec if isinstance(spec, (tuple, list)) else (spec, None)
        return RouteDecision(
            model=key,
            repo=f"{repo}/{subfolder}" if subfolder else repo,
            reason=reason,
            detection=detection,
        )

    def predict(
        self,
        state,
        questions,
        model=None,
        task=None,
        lang=None,
        max_len=None,
        head_max_len=None,
    ):
        """完成路由，并用选中的模型回答全部问题。"""
        decision = self.route(state, questions, model=model, task=task, lang=lang)
        result = self.load(decision["model"]).predict(
            state, questions, max_len=max_len, head_max_len=head_max_len
        )
        result["routing"] = dict(decision)
        return result

    system_one = predict
