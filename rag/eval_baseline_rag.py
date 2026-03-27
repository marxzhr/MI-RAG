# coding: UTF-8
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

from embedding_utils import create_embeddings, embedding_provider, inspect_local_embedding_model
from retrieval_utils import retrieval_fetch_k, retrieve_documents


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VECTOR_STORE_DIR = BASE_DIR / "vector_store"
DEFAULT_EVAL_PATH = BASE_DIR / "data" / "cmedqa2_dev_queries.jsonl"

load_dotenv(BASE_DIR / ".env")


def load_vectorstore(vector_store_dir: Path) -> FAISS:
    embeddings = create_embeddings()
    return FAISS.load_local(
        str(vector_store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def load_eval_queries(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def reciprocal_rank(pred_ids: list[str], gold_ids: set[str]) -> float:
    for idx, doc_id in enumerate(pred_ids, 1):
        if doc_id in gold_ids:
            return 1.0 / idx
    return 0.0


def hit_at_k(pred_ids: list[str], gold_ids: set[str], k: int) -> float:
    return 1.0 if any(doc_id in gold_ids for doc_id in pred_ids[:k]) else 0.0


def evaluate(
    vectorstore: FAISS,
    queries: list[dict],
    max_k: int,
    progress_desc: str | None = None,
    progress_stage: str = "retrieval",
) -> tuple[dict, list[dict]]:
    total = len(queries)
    hits_1 = 0.0
    hits_3 = 0.0
    hits_5 = 0.0
    mrr = 0.0
    details = []
    fetch_k = retrieval_fetch_k(max_k)

    progress_label = progress_desc or "Evaluating"
    iterator = tqdm(
        queries,
        total=total,
        desc=progress_label,
        unit="query",
        dynamic_ncols=True,
    )

    for row in iterator:
        docs, retrieval_meta = retrieve_documents(vectorstore, row["query"], top_k=max_k, fetch_k=fetch_k)
        pred_ids = [doc.doc.metadata.get("doc_id", "") for doc in docs]
        gold_ids = set(row["gold_doc_ids"])
        raw_pred_ids = [item.get("doc_id", "") for item in retrieval_meta["raw_items"]]
        gold_in_fetch_pool = any(doc_id in gold_ids for doc_id in raw_pred_ids)

        hits_1 += hit_at_k(pred_ids, gold_ids, 1)
        hits_3 += hit_at_k(pred_ids, gold_ids, min(3, max_k))
        hits_5 += hit_at_k(pred_ids, gold_ids, min(5, max_k))
        mrr += reciprocal_rank(pred_ids, gold_ids)

        details.append(
            {
                "query_id": row["query_id"],
                "query": row["query"],
                "gold_doc_ids": row["gold_doc_ids"],
                "pred_doc_ids": pred_ids,
                "gold_in_fetch_pool": gold_in_fetch_pool,
                "hit@1": hit_at_k(pred_ids, gold_ids, 1),
                "hit@3": hit_at_k(pred_ids, gold_ids, min(3, max_k)),
                "hit@5": hit_at_k(pred_ids, gold_ids, min(5, max_k)),
                "rr": reciprocal_rank(pred_ids, gold_ids),
                "retrieval": retrieval_meta,
            }
        )
        iterator.set_postfix(
            stage=progress_stage,
            processed=f"{len(details)}/{total}",
            fetch_k=fetch_k,
            top_k=max_k,
        )

    metrics = {
        "queries": total,
        "hit@1": hits_1 / total if total else 0.0,
        "hit@3": hits_3 / total if total else 0.0,
        "hit@5": hits_5 / total if total else 0.0,
        "mrr": mrr / total if total else 0.0,
    }
    return metrics, details


def write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="评测基础 RAG baseline 的检索效果")
    parser.add_argument(
        "--vector-store-dir",
        default=str(DEFAULT_VECTOR_STORE_DIR),
        help="FAISS 索引目录",
    )
    parser.add_argument(
        "--eval-path",
        default=str(DEFAULT_EVAL_PATH),
        help="评测 query jsonl 路径",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="评测时检索的最大 k",
    )
    parser.add_argument(
        "--output-path",
        default=str(BASE_DIR / "data" / "eval_results.json"),
        help="评测结果输出路径",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=None,
        help="初始召回池大小，默认读取 RETRIEVAL_FETCH_K 或 20",
    )
    parser.add_argument(
        "--enable-dedup",
        choices=["0", "1"],
        default=None,
        help="是否开启去重：1 开启，0 关闭",
    )
    parser.add_argument(
        "--enable-rerank",
        choices=["0", "1"],
        default=None,
        help="是否开启 rerank：1 开启，0 关闭",
    )
    parser.add_argument(
        "--rerank-model",
        default=None,
        help="reranker 模型路径或 repo id",
    )
    args = parser.parse_args()

    vector_store_dir = Path(args.vector_store_dir)
    eval_path = Path(args.eval_path)
    output_path = Path(args.output_path)
    if args.fetch_k is not None:
        os.environ["RETRIEVAL_FETCH_K"] = str(args.fetch_k)
    if args.enable_dedup is not None:
        os.environ["ENABLE_DEDUP"] = args.enable_dedup
    if args.enable_rerank is not None:
        os.environ["ENABLE_RERANK"] = args.enable_rerank
    if args.rerank_model is not None:
        os.environ["RERANK_MODEL"] = args.rerank_model

    if embedding_provider() == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("缺少 OPENAI_API_KEY 环境变量。")
    if not vector_store_dir.exists():
        raise FileNotFoundError(f"向量索引不存在: {vector_store_dir}")
    if not eval_path.exists():
        raise FileNotFoundError(f"评测文件不存在: {eval_path}")

    vectorstore = load_vectorstore(vector_store_dir)
    queries = load_eval_queries(eval_path)
    metrics, details = evaluate(
        vectorstore,
        queries,
        args.top_k,
        progress_desc="eval_baseline",
        progress_stage="retrieval",
    )
    diagnostics = None
    if embedding_provider() == "local_bge":
        diagnostics = inspect_local_embedding_model()

    write_json(output_path, {"metrics": metrics, "embedding_diagnostics": diagnostics, "details": details})

    print("=" * 60)
    print(f"评测集: {eval_path}")
    print(f"查询数: {metrics['queries']}")
    print(f"Embedding Provider: {embedding_provider()}")
    if diagnostics:
        print(f"Local Embedding Path: {diagnostics['resolved_path']}")
        for warning in diagnostics["warnings"]:
            print(f"Embedding Warning: {warning}")
    print(f"Hit@1: {metrics['hit@1']:.4f}")
    print(f"Hit@3: {metrics['hit@3']:.4f}")
    print(f"Hit@5: {metrics['hit@5']:.4f}")
    print(f"MRR: {metrics['mrr']:.4f}")
    print(f"结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
