# coding: UTF-8
import argparse
import json
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_BASELINE_PATH = BASE_DIR / "data" / "eval_no_rewrite.json"
DEFAULT_REWRITE_PATH = BASE_DIR / "data" / "eval_with_multi_query.json"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "requery_case_analysis.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def by_query_id(payload: dict) -> dict[str, dict]:
    details = payload.get("details", [])
    return {str(item["query_id"]): item for item in details}


def short_docs(detail: dict, limit: int = 5) -> list[dict]:
    items = detail.get("retrieval", {}).get("final_items", [])[:limit]
    rows = []
    for item in items:
        rows.append(
            {
                "doc_id": item.get("doc_id", ""),
                "title": item.get("title", ""),
                "preview": item.get("page_content", "")[:120],
            }
        )
    return rows


def make_case(query_id: str, baseline: dict, rewrite: dict) -> dict:
    return {
        "query_id": query_id,
        "query": baseline.get("query", ""),
        "gold_doc_ids": baseline.get("gold_doc_ids", []),
        "baseline": {
            "hit@5": baseline.get("hit@5", 0.0),
            "gold_in_fetch_pool": baseline.get("gold_in_fetch_pool", False),
            "pred_doc_ids": baseline.get("pred_doc_ids", []),
            "top_docs": short_docs(baseline),
        },
        "rewrite": {
            "intent": rewrite.get("intent"),
            "entities": rewrite.get("entities", {}),
            "rewritten_query": rewrite.get("rewritten_query", ""),
            "rewrite_queries": rewrite.get("rewrite_queries", []),
            "hit@5": rewrite.get("hit@5", 0.0),
            "gold_in_fetch_pool": rewrite.get("gold_in_fetch_pool", False),
            "pred_doc_ids": rewrite.get("pred_doc_ids", []),
            "top_docs": short_docs(rewrite),
        },
    }


def select_cases(ids: list[str], base_map: dict[str, dict], rewrite_map: dict[str, dict], limit: int) -> list[dict]:
    chosen = []
    for query_id in ids[:limit]:
        chosen.append(make_case(query_id, base_map[query_id], rewrite_map[query_id]))
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description="抽取 baseline 与 requery 的对照样本")
    parser.add_argument("--baseline-path", default=str(DEFAULT_BASELINE_PATH))
    parser.add_argument("--rewrite-path", default=str(DEFAULT_REWRITE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--baseline-hit-rewrite-fail", type=int, default=20)
    parser.add_argument("--baseline-fail-rewrite-hit", type=int, default=20)
    parser.add_argument("--both-fail", type=int, default=10)
    args = parser.parse_args()

    baseline_payload = load_json(Path(args.baseline_path))
    rewrite_payload = load_json(Path(args.rewrite_path))
    base_map = by_query_id(baseline_payload)
    rewrite_map = by_query_id(rewrite_payload)
    common_ids = sorted(set(base_map) & set(rewrite_map))

    baseline_hit_rewrite_fail_ids = [
        qid
        for qid in common_ids
        if base_map[qid].get("hit@5", 0.0) == 1.0 and rewrite_map[qid].get("hit@5", 0.0) == 0.0
    ]
    baseline_fail_rewrite_hit_ids = [
        qid
        for qid in common_ids
        if base_map[qid].get("hit@5", 0.0) == 0.0 and rewrite_map[qid].get("hit@5", 0.0) == 1.0
    ]
    both_fail_ids = [
        qid
        for qid in common_ids
        if base_map[qid].get("hit@5", 0.0) == 0.0 and rewrite_map[qid].get("hit@5", 0.0) == 0.0
    ]

    output = {
        "counts": {
            "compared": len(common_ids),
            "baseline_hit_rewrite_fail": len(baseline_hit_rewrite_fail_ids),
            "baseline_fail_rewrite_hit": len(baseline_fail_rewrite_hit_ids),
            "both_fail": len(both_fail_ids),
        },
        "baseline_hit_rewrite_fail": select_cases(
            baseline_hit_rewrite_fail_ids, base_map, rewrite_map, args.baseline_hit_rewrite_fail
        ),
        "baseline_fail_rewrite_hit": select_cases(
            baseline_fail_rewrite_hit_ids, base_map, rewrite_map, args.baseline_fail_rewrite_hit
        ),
        "both_fail": select_cases(
            both_fail_ids, base_map, rewrite_map, args.both_fail
        ),
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"Compared: {len(common_ids)}")
    print(f"baseline_hit_rewrite_fail: {len(baseline_hit_rewrite_fail_ids)}")
    print(f"baseline_fail_rewrite_hit: {len(baseline_fail_rewrite_hit_ids)}")
    print(f"both_fail: {len(both_fail_ids)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
