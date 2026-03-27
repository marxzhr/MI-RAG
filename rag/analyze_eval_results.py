# coding: UTF-8
import argparse
import json
import random
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_PATH = BASE_DIR / "data" / "eval_results.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "error_analysis.json"

SYMPTOM_KEYWORDS = {"症状", "疼", "痛", "痒", "咳", "发烧", "出血", "头晕", "恶心", "不舒服"}
TREATMENT_KEYWORDS = {"治疗", "怎么办", "吃什么药", "用什么药", "手术", "怎么治", "如何治"}
ALIAS_HINTS = {"是不是", "怎么回事", "是什么", "这个病", "那种病", "毛病", "问题"}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def query_length(query: str) -> int:
    return len("".join(query.split()))


def classify_failure(detail: dict) -> list[str]:
    query = detail.get("query", "")
    categories = []
    qlen = query_length(query)
    dedup = detail.get("retrieval", {}).get("dedup", {})

    if qlen <= 8:
        categories.append("查询太短")

    if any(k in query for k in SYMPTOM_KEYWORDS) and any(k in query for k in TREATMENT_KEYWORDS):
        categories.append("意图不明确")

    if any(k in query for k in ALIAS_HINTS):
        categories.append("实体缺失/别名问题")

    if dedup.get("removed_same_doc_id", 0) > 0 or dedup.get("removed_near_duplicate_text", 0) > 0:
        categories.append("检索召回了重复答案")

    if detail.get("gold_in_fetch_pool") and not detail.get("hit@5"):
        categories.append("召回内容对，但排序不对")

    final_items = detail.get("retrieval", {}).get("final_items", [])
    if len({item.get("title", "") for item in final_items if item.get("title")}) <= 2 and len(final_items) >= 3:
        categories.append("知识库 chunk 切分不合理")

    if not categories:
        categories.append("其他召回不足")

    return categories


def sample_failures(details: list[dict], sample_size: int, seed: int) -> list[dict]:
    failures = [d for d in details if not d.get("hit@5")]
    rng = random.Random(seed)
    sample_size = min(sample_size, len(failures))
    return rng.sample(failures, sample_size) if sample_size else []


def main() -> None:
    parser = argparse.ArgumentParser(description="对检索评测结果做失败样本抽样与分类")
    parser.add_argument("--input-path", default=str(DEFAULT_INPUT_PATH), help="评测结果 JSON")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="分析结果输出路径")
    parser.add_argument("--sample-size", type=int, default=100, help="抽样失败样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    payload = load_json(Path(args.input_path))
    details = payload.get("details", [])
    samples = sample_failures(details, args.sample_size, args.seed)

    labeled_samples = []
    counter = Counter()
    for item in samples:
        categories = classify_failure(item)
        counter.update(categories)
        labeled_samples.append(
            {
                "query_id": item.get("query_id"),
                "query": item.get("query"),
                "gold_doc_ids": item.get("gold_doc_ids", []),
                "pred_doc_ids": item.get("pred_doc_ids", []),
                "categories": categories,
                "gold_in_fetch_pool": item.get("gold_in_fetch_pool", False),
                "dedup": item.get("retrieval", {}).get("dedup", {}),
                "top_results": item.get("retrieval", {}).get("final_items", []),
            }
        )

    output = {
        "input_path": str(Path(args.input_path)),
        "sample_size": len(labeled_samples),
        "category_counts": dict(counter),
        "samples": labeled_samples,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"输入文件: {args.input_path}")
    print(f"失败样本抽样数: {len(labeled_samples)}")
    print("分类统计:")
    for name, count in counter.most_common():
        print(f"- {name}: {count}")
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
