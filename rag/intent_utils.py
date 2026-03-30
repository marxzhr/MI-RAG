# coding: UTF-8
"""
轻量意图分类与查询改写占位模块。
默认规则覆盖常见医疗问答场景；未匹配时回落到原始查询。
"""
from dataclasses import dataclass


@dataclass
class IntentResult:
    label: str
    rewritten_query: str


# 简单关键词规则，可后续替换为模型推理
INTENT_RULES = {
    "diagnosis": ["是什么", "怎么回事", "原因", "病因", "为何", "怎么办"],
    "treatment": ["怎么治疗", "怎么治", "吃什么药", "用什么药", "手术", "治好吗"],
    "exam": ["要检查", "要做什么检查", "化验", "B超", "CT", "核磁"],
    "pregnancy": ["怀孕", "孕", "胎儿", "胎动", "产检", "月经推迟", "备孕"],
    "lifestyle": ["饮食", "能不能吃", "运动", "注意事项", "护理", "保养"],
}


def classify_intent(query: str) -> str:
    q = query.strip()
    for label, keywords in INTENT_RULES.items():
        if any(k in q for k in keywords):
            return label
    return "general"


def rewrite_query(query: str, intent: str) -> str:
    q = query.strip()
    if not q:
        return q
    if intent == "diagnosis":
        return f"{q} 主要症状是什么？可能的疾病名称？"
    if intent == "treatment":
        return f"{q} 有哪些常见治疗或用药建议？"
    if intent == "exam":
        return f"{q} 推荐的检查项目有哪些？"
    if intent == "pregnancy":
        return f"{q} 孕期相关风险与建议？"
    if intent == "lifestyle":
        return f"{q} 日常饮食和生活方式建议？"
    return q


def infer_intent_and_rewrite(query: str) -> IntentResult:
    intent = classify_intent(query)
    rewritten = rewrite_query(query, intent)
    return IntentResult(label=intent, rewritten_query=rewritten)
