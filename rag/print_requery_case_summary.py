import json
from pathlib import Path


def short(text: str, limit: int = 42) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:limit]


def main() -> None:
    path = Path("rag/data/requery_case_analysis.json")
    data = json.loads(path.read_text(encoding="utf-8"))

    print("COUNTS", data["counts"])

    for section in ("baseline_hit_rewrite_fail", "baseline_fail_rewrite_hit", "both_fail"):
        print()
        print("SECTION", section)
        for case in data[section][:8]:
            rewrites = case["rewrite"].get("rewrite_queries") or [case["rewrite"].get("rewritten_query", "")]
            print(f"QID {case['query_id']}")
            print("Q ", short(case["query"], 80))
            print("RW", short(rewrites[0], 80))
            print(
                "BASE",
                case["baseline"]["hit@5"],
                case["baseline"]["gold_in_fetch_pool"],
                [(x["doc_id"], short(x["preview"], 36)) for x in case["baseline"]["top_docs"][:3]],
            )
            print(
                "RWRT",
                case["rewrite"]["hit@5"],
                case["rewrite"]["gold_in_fetch_pool"],
                [(x["doc_id"], short(x["preview"], 36)) for x in case["rewrite"]["top_docs"][:3]],
            )
            print()


if __name__ == "__main__":
    main()
