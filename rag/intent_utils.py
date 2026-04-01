# coding: UTF-8
"""
查询理解模块：

- 轻量 intent label（规则版，仅用于分析与后续路由）
- LLM 实体提取
- LLM 多路 query rewrite
"""
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

from embedding_utils import openai_kwargs


QUERY_UNDERSTANDING_SYSTEM_PROMPT = (
    "你是医疗查询理解助手。"
    "请先识别问题中的关键实体，再围绕这些实体生成适合检索的多条重写查询。"
    "不要回答问题，不要补充原问题中没有的新事实。"
    "输出严格 JSON，不要输出 JSON 之外的任何解释。"
)

QUERY_UNDERSTANDING_USER_TEMPLATE = """用户问题：
{query}

请输出 JSON，格式如下：
{{
  "entities": {{
    "diseases": ["疾病1"],
    "symptoms": ["症状1"],
    "drugs": ["药物1"],
    "exams": ["检查1"],
    "needs": ["用户需求1"]
  }},
  "rewrites": [
    "重写查询1",
    "重写查询2",
    "重写查询3"
  ]
}}

要求：
1. entities 中没有内容就输出空数组
2. rewrites 生成 {rewrite_count} 条，不要和原问题完全重复
3. 每条 rewrite 尽量简洁，保留关键实体和需求点
4. 多条 rewrite 从不同角度展开，例如病因、治疗、检查、禁忌、注意事项等
5. 只输出合法 JSON
"""


@dataclass
class IntentResult:
    label: str
    rewritten_query: str
    rewrite_method: str = "none"
    rewrite_queries: list[str] = field(default_factory=list)
    entities: dict[str, list[str]] = field(default_factory=dict)


INTENT_RULES = {
    "diagnosis": ["是什么", "怎么回事", "原因", "病因", "为何", "怎么办"],
    "treatment": ["怎么治疗", "怎么治", "吃什么药", "用什么药", "手术", "治好吗"],
    "exam": ["要检查", "要做什么检查", "化验", "B超", "CT", "核磁"],
    "pregnancy": ["怀孕", "孕", "胎儿", "胎动", "产检", "月经推迟", "备孕"],
    "lifestyle": ["饮食", "能不能吃", "运动", "注意事项", "护理", "保养"],
}

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_UNDERSTANDING_CACHE_PATH = BASE_DIR / "data" / "query_understanding_cache.json"
DEFAULT_INTENT_MODEL_DIR = BASE_DIR / "models" / "intent_ernie_health"
_QUERY_UNDERSTANDING_CACHE = None
_INTENT_MODEL = None
_INTENT_TOKENIZER = None
_INTENT_ID2LABEL = None


def intent_model_dir() -> Path:
    return Path(os.getenv("INTENT_MODEL_DIR", str(DEFAULT_INTENT_MODEL_DIR)))


def intent_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_intent_model():
    global _INTENT_MODEL, _INTENT_TOKENIZER, _INTENT_ID2LABEL
    if _INTENT_MODEL is not None and _INTENT_TOKENIZER is not None and _INTENT_ID2LABEL is not None:
        return _INTENT_MODEL, _INTENT_TOKENIZER, _INTENT_ID2LABEL

    model_dir = intent_model_dir()
    if not model_dir.exists():
        return None, None, None

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(intent_device())
    model.eval()

    summary_path = model_dir / "training_summary.json"
    if summary_path.exists():
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        id2label = {int(k): v for k, v in payload.get("id2label", {}).items()}
    else:
        id2label = {int(k): v for k, v in model.config.id2label.items()}

    _INTENT_MODEL = model
    _INTENT_TOKENIZER = tokenizer
    _INTENT_ID2LABEL = id2label
    return _INTENT_MODEL, _INTENT_TOKENIZER, _INTENT_ID2LABEL


def classify_intent_with_model(query: str) -> str | None:
    model, tokenizer, id2label = load_intent_model()
    if model is None or tokenizer is None or id2label is None:
        return None

    inputs = tokenizer(
        query.strip(),
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )
    inputs = {key: value.to(intent_device()) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())
    return id2label.get(pred_id)


def classify_intent(query: str) -> str:
    predicted = classify_intent_with_model(query)
    if predicted:
        return predicted

    q = query.strip()
    for label, keywords in INTENT_RULES.items():
        if any(keyword in q for keyword in keywords):
            return label
    return "general"


def query_rewrite_model() -> str:
    return os.getenv("QUERY_REWRITE_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def query_understanding_enabled() -> bool:
    return os.getenv("ENABLE_QUERY_UNDERSTANDING", "0").strip().lower() in {"1", "true", "yes", "on"}


def rewrite_count() -> int:
    return max(1, int(os.getenv("QUERY_REWRITE_COUNT", "3")))


def query_understanding_cache_path() -> Path:
    return Path(
        os.getenv("QUERY_UNDERSTANDING_CACHE_PATH", str(DEFAULT_QUERY_UNDERSTANDING_CACHE_PATH))
    )


def load_query_understanding_cache() -> dict[str, dict]:
    global _QUERY_UNDERSTANDING_CACHE
    if _QUERY_UNDERSTANDING_CACHE is not None:
        return _QUERY_UNDERSTANDING_CACHE

    path = query_understanding_cache_path()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            _QUERY_UNDERSTANDING_CACHE = json.load(f)
    else:
        _QUERY_UNDERSTANDING_CACHE = {}
    return _QUERY_UNDERSTANDING_CACHE


def save_query_understanding_cache() -> None:
    cache = load_query_understanding_cache()
    path = query_understanding_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def build_query_understanding_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=query_rewrite_model(),
        temperature=0,
        **openai_kwargs(),
    )


def _extract_json_blob(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return stripped[start : end + 1]
    return "{}"


def _normalize_entities(payload: dict) -> dict[str, list[str]]:
    raw_entities = payload.get("entities") or {}
    normalized = {}
    for key in ["diseases", "symptoms", "drugs", "exams", "needs"]:
        values = raw_entities.get(key, [])
        if isinstance(values, str):
            values = [values]
        normalized[key] = [str(item).strip() for item in values if str(item).strip()]
    return normalized


def _normalize_rewrites(payload: dict, query: str) -> list[str]:
    rewrites = payload.get("rewrites") or []
    if isinstance(rewrites, str):
        rewrites = [rewrites]

    cleaned = []
    seen = set()
    for item in rewrites:
        text = str(item).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        cleaned.append(text)

    if not cleaned:
        cleaned = [query.strip()]
    return cleaned


def _clean_json_text(text: str) -> str:
    cleaned = _extract_json_blob(text)
    replacements = {
        "“": '"',
        "”": '"',
        "‘": '"',
        "’": '"',
        "，": ",",
        "：": ":",
    }
    for src, dst in replacements.items():
        cleaned = cleaned.replace(src, dst)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    return cleaned


def _extract_string_array(raw: str, field_name: str) -> list[str]:
    pattern = rf'"{field_name}"\s*:\s*\[(.*?)\]'
    match = re.search(pattern, raw, flags=re.S)
    if not match:
        return []
    body = match.group(1)
    values = re.findall(r'"([^"]+)"', body)
    return [item.strip() for item in values if item.strip()]


def _parse_query_understanding_response(content: str, query: str) -> dict:
    candidates = [_clean_json_text(content), _extract_json_blob(content), content.strip()]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    # Heuristic fallback: extract arrays field by field, then fall back to original query.
    raw = _extract_json_blob(content)
    entities = {
        "diseases": _extract_string_array(raw, "diseases"),
        "symptoms": _extract_string_array(raw, "symptoms"),
        "drugs": _extract_string_array(raw, "drugs"),
        "exams": _extract_string_array(raw, "exams"),
        "needs": _extract_string_array(raw, "needs"),
    }
    rewrites = _extract_string_array(raw, "rewrites")
    if not rewrites:
        rewrites = [query.strip()]
    return {"entities": entities, "rewrites": rewrites}


def understand_query_with_llm(query: str) -> tuple[dict[str, list[str]], list[str]]:
    q = query.strip()
    if not q:
        return {"diseases": [], "symptoms": [], "drugs": [], "exams": [], "needs": []}, [q]

    cache = load_query_understanding_cache()
    cache_key = f"{rewrite_count()}::{q}"
    if cache_key in cache:
        record = cache[cache_key]
        return _normalize_entities(record), _normalize_rewrites(record, q)

    prompt = ChatPromptTemplate.from_messages(
        [("system", QUERY_UNDERSTANDING_SYSTEM_PROMPT), ("user", QUERY_UNDERSTANDING_USER_TEMPLATE)]
    )
    llm = build_query_understanding_llm()
    result = (prompt | llm).invoke({"query": q, "rewrite_count": rewrite_count()})
    content = (result.content or "").strip()
    payload = _parse_query_understanding_response(content, q)
    entities = _normalize_entities(payload)
    rewrites = _normalize_rewrites(payload, q)
    cache[cache_key] = {"entities": entities, "rewrites": rewrites}
    save_query_understanding_cache()
    return entities, rewrites


def infer_intent_and_rewrite(query: str, enable_rewrite: bool | None = None) -> IntentResult:
    q = query.strip()
    intent = classify_intent(q)
    use_rewrite = query_understanding_enabled() if enable_rewrite is None else enable_rewrite
    if not use_rewrite:
        return IntentResult(
            label=intent,
            rewritten_query=q,
            rewrite_method="none",
            rewrite_queries=[q] if q else [],
            entities={"diseases": [], "symptoms": [], "drugs": [], "exams": [], "needs": []},
        )

    entities, rewrites = understand_query_with_llm(q)
    primary_query = rewrites[0] if rewrites else q
    return IntentResult(
        label=intent,
        rewritten_query=primary_query,
        rewrite_method="llm_multi_query",
        rewrite_queries=rewrites,
        entities=entities,
    )
