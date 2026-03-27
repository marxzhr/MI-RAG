# coding: UTF-8
import argparse
import csv
import json
import zipfile
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CMEDQA2_DIR = BASE_DIR.parent / "cMedQA2"
EXTRACTED_DIR = CMEDQA2_DIR / "extracted"
DATA_DIR = BASE_DIR / "data"


def read_csv_file(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_zip_csv(zip_path: Path, inner_name: str) -> list[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(inner_name) as f:
            text = f.read().decode("utf-8")
    return list(csv.DictReader(text.splitlines()))


def read_dataset_file(
    extracted_path: Path, zip_path: Path, inner_name: str
) -> list[dict]:
    if extracted_path.exists():
        return read_csv_file(extracted_path)
    return read_zip_csv(zip_path, inner_name)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_answer_corpus(answer_rows: list[dict]) -> list[dict]:
    corpus = []
    for row in answer_rows:
        content = row["content"].strip()
        if not content:
            continue
        corpus.append(
            {
                "doc_id": row["ans_id"],
                "title": f"answer_{row['ans_id']}",
                "content": content,
                "question_id": row["question_id"],
                "source": "cMedQA2",
            }
        )
    return corpus


def build_eval_queries(
    question_rows: list[dict], candidate_rows: list[dict], split: str
) -> list[dict]:
    question_map = {
        row["question_id"]: row["content"].strip()
        for row in question_rows
        if row.get("content", "").strip()
    }
    grouped = defaultdict(lambda: {"gold_doc_ids": [], "candidate_doc_ids": []})
    for row in candidate_rows:
        qid = row["question_id"]
        ans_id = row["ans_id"]
        grouped[qid]["candidate_doc_ids"].append(ans_id)
        if row["label"] == "1":
            grouped[qid]["gold_doc_ids"].append(ans_id)

    queries = []
    for qid, payload in grouped.items():
        query = question_map.get(qid)
        if not query or not payload["gold_doc_ids"]:
            continue
        queries.append(
            {
                "query_id": qid,
                "query": query,
                "gold_doc_ids": payload["gold_doc_ids"],
                "candidate_doc_ids": payload["candidate_doc_ids"],
                "split": split,
            }
        )
    return queries


def main() -> None:
    parser = argparse.ArgumentParser(description="将 cMedQA2 转成 baseline 可用格式")
    parser.add_argument(
        "--source-dir",
        default=str(CMEDQA2_DIR),
        help="cMedQA2 数据目录",
    )
    parser.add_argument(
        "--extracted-dir",
        default=str(EXTRACTED_DIR),
        help="已解压的 cMedQA2 目录，存在时优先读取",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DATA_DIR),
        help="输出目录",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)

    question_rows = read_dataset_file(
        extracted_dir / "question" / "question.csv",
        source_dir / "question.zip",
        "question.csv",
    )
    answer_rows = read_dataset_file(
        extracted_dir / "answer" / "answer.csv",
        source_dir / "answer.zip",
        "answer.csv",
    )
    dev_rows = read_dataset_file(
        extracted_dir / "dev_candidates" / "dev_candidates.txt",
        source_dir / "dev_candidates.zip",
        "dev_candidates.txt",
    )
    test_rows = read_dataset_file(
        extracted_dir / "test_candidates" / "test_candidates.txt",
        source_dir / "test_candidates.zip",
        "test_candidates.txt",
    )

    answer_corpus = build_answer_corpus(answer_rows)
    dev_queries = build_eval_queries(question_rows, dev_rows, "dev")
    test_queries = build_eval_queries(question_rows, test_rows, "test")

    write_jsonl(output_dir / "cmedqa2_answers.jsonl", answer_corpus)
    write_jsonl(output_dir / "cmedqa2_dev_queries.jsonl", dev_queries)
    write_jsonl(output_dir / "cmedqa2_test_queries.jsonl", test_queries)

    print("=" * 60)
    print(f"answer 语料数: {len(answer_corpus)}")
    print(f"dev 查询数: {len(dev_queries)}")
    print(f"test 查询数: {len(test_queries)}")
    print(f"优先读取目录: {extracted_dir}")
    print(f"输出目录: {output_dir}")


if __name__ == "__main__":
    main()
