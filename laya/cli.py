"""Laya 命令行入口。"""

import argparse
import json
import sys

from .presets import (
    guard_questions,
    moderation_questions,
    router_questions,
    triage_questions,
)
from .router import DEFAULT_MODELS, Router

PRESETS = {
    "guard": guard_questions,
    "moderation": moderation_questions,
    "router": router_questions,
    "triage": triage_questions,
}
STATE_KEYS = {"guard": "prompt", "moderation": "post", "router": "request", "triage": "message"}


def parser():
    result = argparse.ArgumentParser(description="本地结构化文本决策", add_help=False)
    result.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    result.add_argument("text", nargs="+", help="需要判断的文本")
    result.add_argument("--preset", choices=PRESETS, default="triage", help="问题模板")
    result.add_argument(
        "--model", choices=["auto", *DEFAULT_MODELS], default="auto", help="checkpoint"
    )
    result.add_argument("--device", help="PyTorch 设备，例如 cuda、cuda:1 或 cpu")
    result.add_argument(
        "--route-only", action="store_true", help="只显示路由，不加载模型"
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    state = {STATE_KEYS[args.preset]: " ".join(args.text)}
    questions = PRESETS[args.preset]()
    router = Router(device=args.device)
    try:
        result = (
            router.route(state, questions)
            if args.route_only
            else router.predict(
                state, questions, model=None if args.model == "auto" else args.model
            )
        )
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Laya 运行失败：{error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
