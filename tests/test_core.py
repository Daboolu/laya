"""不下载权重的核心冒烟测试。"""

from laya.agent import _validate_question
from laya.lang import analyse
from laya.router import Router


def main():
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team?",
            "criteria": {"billing": "refunds", "technical": "bugs"},
        }
    }
    router = Router()
    assert (
        router.route({"message": "Please refund my invoice"}, questions)["model"]
        == "english"
    )
    assert (
        router.route({"message": "请退还重复扣款"}, questions)["model"]
        == "multilingual"
    )
    assert (
        analyse("Der Kunde wurde zweimal belastet und die Rechnung ist falsch")[
            "is_english"
        ]
        is False
    )
    _validate_question("department", questions["department"])
    print("核心冒烟测试通过")


if __name__ == "__main__":
    main()
