"""使用已下载 checkpoint 完成一次真实预测。"""

import json
import os

from laya import Router

router = Router(device=os.environ.get("LAYA_DEVICE"))
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
print(json.dumps(result, ensure_ascii=False, indent=2))
