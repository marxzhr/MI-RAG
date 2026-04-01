# coding: UTF-8
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_NO_INTENT_PATH = BASE_DIR / "data" / "eval_no_intent.json"
DEFAULT_WITH_INTENT_PATH = BASE_DIR / "data" / "eval_with_intent.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "intent_compare_analysis.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_details_by_query_id(path: Path) -> tuple[dict, dict[str, dict]]:
    payload = load_json(path)
    details = payload.get("details", [])
    indexed = {}
    for item in details:
        query_id = str(item.get("query_id"))
        if query_id:
            indexed[query_id] = item
    return payload, indexed


def metric_delta(base: dict, target: dict, key: str) -> float:
    return float(target.get(key, 0.0)) - float(base.get(key, 0.0))


def brief_top_docs(detail: dict, limit: int = 3) -> list[dict]:
    items = detail.get("retrieval", {}).get("final_items", [])[:limit]
    output = []
    for item in items:
        output.append(
            {
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "raw_rank": item.get("raw_rank"),
                "preview": item.get("page_content", "")[:120],
            }
        )
    return output


def compare_details(no_intent: dict, with_intent: dict) -> dict:
    rr_delta = metric_delta(no_intent, with_intent, "rr")
    hit1_delta = metric_delta(no_intent, with_intent, "hit@1")
    hit3_delta = metric_delta(no_intent, with_intent, "hit@3")
    hit5_delta = metric_delta(no_intent, with_intent, "hit@5")
    intent_label = with_intent.get("intent") or "unknown"

    if rr_delta > 0:
        outcome = "improved"
    elif rr_delta < 0:
        outcome = "worsened"
    elif hit5_delta > 0:
        outcome = "improved"
    elif hit5_delta < 0:
        outcome = "worsened"
    else:
        outcome = "unchanged"

    return {
        "query_id": with_intent.get("query_id"),
        "query": with_intent.get("query"),
        "intent": intent_label,
        "rewritten_query": with_intent.get("rewritten_query"),
        "gold_doc_ids": with_intent.get("gold_doc_ids", []),
        "outcome": outcome,
        "rr_delta": rr_delta,
        "hit@1_delta": hit1_delta,
        "hit@3_delta": hit3_delta,
        "hit@5_delta": hit5_delta,
        "no_intent_rr": float(no_intent.get("rr", 0.0)),
        "with_intent_rr": float(with_intent.get("rr", 0.0)),
        "no_intent_hit@5": float(no_intent.get("hit@5", 0.0)),
        "with_intent_hit@5": float(with_intent.get("hit@5", 0.0)),
        "no_intent_pred_doc_ids": no_intent.get("pred_doc_ids", []),
        "with_intent_pred_doc_ids": with_intent.get("pred_doc_ids", []),
        "no_intent_gold_in_fetch_pool": bool(no_intent.get("gold_in_fetch_pool", False)),
        "with_intent_gold_in_fetch_pool": bool(with_intent.get("gold_in_fetch_pool", False)),
        "no_intent_top_docs": brief_top_docs(no_intent),
        "with_intent_top_docs": brief_top_docs(with_intent),
    }


def summarize_by_intent(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["intent"]].append(row)

    summary = []
    for intent, items in grouped.items():
        counts = Counter(item["outcome"] for item in items)
        summary.append(
            {
                "intent": intent,
                "count": len(items),
                "improved": counts.get("improved", 0),
                "worsened": counts.get("worsened", 0),
                "unchanged": counts.get("unchanged", 0),
                "avg_rr_delta": sum(item["rr_delta"] for item in items) / len(items),
                "avg_hit@5_delta": sum(item["hit@5_delta"] for item in items) / len(items),
            }
        )
    summary.sort(key=lambda item: (item["avg_rr_delta"], item["avg_hit@5_delta"]), reverse=True)
    return summary


def top_cases(rows: list[dict], outcome: str, top_n: int) -> list[dict]:
    filtered = [row for row in rows if row["outcome"] == outcome]
    if outcome == "improved":
        filtered.sort(
            key=lambda item: (item["rr_delta"], item["hit@5_delta"], item["hit@3_delta"], item["hit@1_delta"]),
            reverse=True,
        )
    else:
        filtered.sort(
            key=lambda item: (item["rr_delta"], item["hit@5_delta"], item["hit@3_delta"], item["hit@1_delta"])
        )
    return filtered[:top_n]


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 no_intent / with_intent 两份评测结果")
    parser.add_argument("--no-intent-path", default=str(DEFAULT_NO_INTENT_PATH), help="关闭意图增强的评测结果 JSON")
    parser.add_argument("--with-intent-path", default=str(DEFAULT_WITH_INTENT_PATH), help="开启意图增强的评测结果 JSON")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="分析结果输出路径")
    parser.add_argument("--top-n", type=int, default=20, help="输出 top 改进/退化样本数")
    args = parser.parse_args()

    no_intent_path = Path(args.no_intent_path)
    with_intent_path = Path(args.with_intent_path)
    output_path = Path(args.output_path)

    no_payload, no_details = load_details_by_query_id(no_intent_path)
    with_payload, with_details = load_details_by_query_id(with_intent_path)

    common_ids = sorted(set(no_details) & set(with_details))
    if not common_ids:
        raise ValueError("两份评测结果没有可对齐的 query_id。")

    if len(common_ids) != len(no_details) or len(common_ids) != len(with_details):
        print("警告: 两份结果的 query_id 不完全一致，已仅比较交集部分。")

    rows = [compare_details(no_details[qid], with_details[qid]) for qid in common_ids]
    outcome_counts = Counter(row["outcome"] for row in rows)
    intent_summary = summarize_by_intent(rows)
    improved_cases = top_cases(rows, "improved", args.top_n)
    worsened_cases = top_cases(rows, "worsened", args.top_n)

    metric_summary = {}
    for key in ["hit@1", "hit@3", "hit@5", "mrr"]:
        before = float(no_payload.get("metrics", {}).get(key, 0.0))
        after = float(with_payload.get("metrics", {}).get(key, 0.0))
        metric_summary[key] = {
            "no_intent": before,
            "with_intent": after,
            "delta": after - before,
        }

    output = {
        "inputs": {
            "no_intent_path": str(no_intent_path),
            "with_intent_path": str(with_intent_path),
            "compared_queries": len(common_ids),
        },
        "metric_summary": metric_summary,
        "outcome_counts": dict(outcome_counts),
        "intent_summary": intent_summary,
        "top_improved_cases": improved_cases,
        "top_worsened_cases": worsened_cases,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"对比样本数: {len(common_ids)}")
    for key in ["hit@1", "hit@3", "hit@5", "mrr"]:
        item = metric_summary[key]
        print(
            f"{key}: {item['no_intent']:.4f} -> {item['with_intent']:.4f} "
            f"(delta={item['delta']:+.4f})"
        )

    print("结果计数:")
    print(
        f"- improved={outcome_counts.get('improved', 0)} "
        f"- worsened={outcome_counts.get('worsened', 0)} "
        f"- unchanged={outcome_counts.get('unchanged', 0)}"
    )

    print("按 intent 汇总:")
    for item in intent_summary:
        print(
            f"- {item['intent']}: count={item['count']}, improved={item['improved']}, "
            f"worsened={item['worsened']}, unchanged={item['unchanged']}, "
            f"avg_rr_delta={item['avg_rr_delta']:+.4f}, avg_hit@5_delta={item['avg_hit@5_delta']:+.4f}"
        )

    print("Top improved cases:")
    for item in improved_cases[: min(5, len(improved_cases))]:
        print(
            f"- query_id={item['query_id']} intent={item['intent']} "
            f"rr_delta={item['rr_delta']:+.4f} hit@5_delta={item['hit@5_delta']:+.1f} "
            f"query={item['query'][:60]}"
        )

    print("Top worsened cases:")
    for item in worsened_cases[: min(5, len(worsened_cases))]:
        print(
            f"- query_id={item['query_id']} intent={item['intent']} "
            f"rr_delta={item['rr_delta']:+.4f} hit@5_delta={item['hit@5_delta']:+.1f} "
            f"query={item['query'][:60]}"
        )

    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
