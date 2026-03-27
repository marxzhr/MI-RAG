# coding: UTF-8
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from embedding_utils import embedding_provider
from eval_baseline_rag import evaluate, load_eval_queries, load_vectorstore


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VECTOR_STORE_DIR = BASE_DIR / "vector_store"
DEFAULT_EVAL_PATH = BASE_DIR / "data" / "cmedqa2_dev_queries.jsonl"
DEFAULT_OUTPUT_PATH = BASE_DIR / "data" / "retrieval_experiment_compare.json"

load_dotenv(BASE_DIR / ".env")


def run_setting(
    experiment_name: str,
    vector_store_dir: Path,
    eval_path: Path,
    top_k: int,
    fetch_k: int,
    enable_dedup: bool,
    enable_rerank: bool,
    rerank_model: str | None,
) -> dict:
    old_env = {
        "RETRIEVAL_FETCH_K": os.getenv("RETRIEVAL_FETCH_K"),
        "ENABLE_DEDUP": os.getenv("ENABLE_DEDUP"),
        "ENABLE_RERANK": os.getenv("ENABLE_RERANK"),
        "RERANK_MODEL": os.getenv("RERANK_MODEL"),
    }
    try:
        os.environ["RETRIEVAL_FETCH_K"] = str(fetch_k)
        os.environ["ENABLE_DEDUP"] = "1" if enable_dedup else "0"
        os.environ["ENABLE_RERANK"] = "1" if enable_rerank else "0"
        if rerank_model:
            os.environ["RERANK_MODEL"] = rerank_model

        vectorstore = load_vectorstore(vector_store_dir)
        queries = load_eval_queries(eval_path)
        metrics, details = evaluate(
            vectorstore,
            queries,
            top_k,
            progress_desc=experiment_name,
            progress_stage="retrieval" if not enable_rerank else "retrieval+rerank",
        )
        return {
            "config": {
                "top_k": top_k,
                "fetch_k": fetch_k,
                "enable_dedup": enable_dedup,
                "enable_rerank": enable_rerank,
                "rerank_model": rerank_model,
            },
            "metrics": metrics,
            "details_path": None,
            "details_count": len(details),
        }
    finally:
        for key, value in old_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="对比 baseline / dedup / dedup+rerank 检索实验")
    parser.add_argument("--vector-store-dir", default=str(DEFAULT_VECTOR_STORE_DIR), help="FAISS 索引目录")
    parser.add_argument("--eval-path", default=str(DEFAULT_EVAL_PATH), help="评测集路径")
    parser.add_argument("--top-k", type=int, default=5, help="最终 top-k")
    parser.add_argument("--fetch-k", type=int, default=20, help="初始召回池大小")
    parser.add_argument("--rerank-model", default=os.getenv("RERANK_MODEL", "./models/BAAI/bge-reranker-base"), help="reranker 模型路径或 repo id")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH), help="对比结果输出路径")
    args = parser.parse_args()

    vector_store_dir = Path(args.vector_store_dir)
    eval_path = Path(args.eval_path)
    output_path = Path(args.output_path)

    if embedding_provider() == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("缺少 OPENAI_API_KEY 环境变量。")
    if not vector_store_dir.exists():
        raise FileNotFoundError(f"向量索引不存在: {vector_store_dir}")
    if not eval_path.exists():
        raise FileNotFoundError(f"评测文件不存在: {eval_path}")

    experiments = [
        ("baseline", False, False),
        ("dedup", True, False),
        ("dedup_rerank", True, True),
    ]

    results = []
    for name, enable_dedup, enable_rerank in experiments:
        print("=" * 60)
        print(f"运行实验: {name}")
        result = run_setting(
            experiment_name=name,
            vector_store_dir=vector_store_dir,
            eval_path=eval_path,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            enable_dedup=enable_dedup,
            enable_rerank=enable_rerank,
            rerank_model=args.rerank_model,
        )
        result["name"] = name
        results.append(result)
        metrics = result["metrics"]
        print(f"Hit@1: {metrics['hit@1']:.4f}")
        print(f"Hit@3: {metrics['hit@3']:.4f}")
        print(f"Hit@5: {metrics['hit@5']:.4f}")
        print(f"MRR: {metrics['mrr']:.4f}")

    baseline_metrics = results[0]["metrics"]
    for result in results[1:]:
        metrics = result["metrics"]
        result["delta_vs_baseline"] = {
            "hit@1": metrics["hit@1"] - baseline_metrics["hit@1"],
            "hit@3": metrics["hit@3"] - baseline_metrics["hit@3"],
            "hit@5": metrics["hit@5"] - baseline_metrics["hit@5"],
            "mrr": metrics["mrr"] - baseline_metrics["mrr"],
        }

    payload = {
        "vector_store_dir": str(vector_store_dir),
        "eval_path": str(eval_path),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("=" * 60)
    print(f"对比结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
