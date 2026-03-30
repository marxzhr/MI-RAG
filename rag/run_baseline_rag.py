# coding: UTF-8
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from openai import APIConnectionError

from embedding_utils import (
    create_embeddings,
    embedding_provider,
    inspect_local_embedding_model,
    openai_kwargs,
)
from prompt_template import RAG_SYSTEM_PROMPT, RAG_USER_TEMPLATE
from retrieval_utils import retrieve_documents
from intent_utils import infer_intent_and_rewrite


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_VECTOR_STORE_DIR = BASE_DIR / "vector_store"

load_dotenv(BASE_DIR / ".env")


def load_vectorstore(vector_store_dir: Path) -> FAISS:
    embeddings = create_embeddings()
    return FAISS.load_local(
        str(vector_store_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_context(context_docs: list) -> str:
    return "\n\n".join(
        f"[{item.doc.metadata.get('title') or '未命名文档'}]\n{item.doc.page_content}"
        for item in context_docs
    )


def answer_question(
    query_original: str,
    query_for_retrieval: str,
    top_k: int,
    vector_store_dir: Path,
    intent_result=None,
) -> None:
    try:
        vectorstore = load_vectorstore(vector_store_dir)
        context_docs, retrieval_meta = retrieve_documents(
            vectorstore, query_for_retrieval, top_k=top_k
        )
        context = build_context(context_docs)

        prompt = ChatPromptTemplate.from_messages(
            [("system", RAG_SYSTEM_PROMPT), ("user", RAG_USER_TEMPLATE)]
        )
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            **openai_kwargs(),
        )
        answer = (prompt | llm).invoke({"query": query_original, "context": context})
    except APIConnectionError as exc:
        raise RuntimeError(
            "调用模型接口失败，请检查网络、代理和 OPENAI_BASE_URL 配置。"
        ) from exc

    print("=" * 60)
    print("问题:", query_original)
    if intent_result:
        print(f"意图: {intent_result.label}")
        print(f"检索用 query: {query_for_retrieval}")
    if embedding_provider() == "local_bge":
        diagnostics = inspect_local_embedding_model()
        print(f"Local Embedding Path: {diagnostics['resolved_path']}")
        for warning in diagnostics["warnings"]:
            print(f"Embedding Warning: {warning}")
    print(
        f"去重统计: raw={retrieval_meta['dedup']['raw_candidates']}, "
        f"same_doc_id={retrieval_meta['dedup']['removed_same_doc_id']}, "
        f"near_dup_text={retrieval_meta['dedup']['removed_near_duplicate_text']}"
    )
    if retrieval_meta["rerank_enabled"]:
        print(f"Rerank Model: {retrieval_meta['rerank_model']}")
    print("\n检索结果:")
    for i, item in enumerate(context_docs, 1):
        doc = item.doc
        print(
            f"[{i}] {doc.metadata.get('title') or '未命名文档'}"
            f" | doc_id={doc.metadata.get('doc_id', '')}"
            f" | chunk_id={doc.metadata.get('chunk_id', '')}"
            f" | raw_score={item.raw_score}"
            f" | rerank_score={item.rerank_score}"
        )
        print(doc.page_content)
        print("-" * 60)
    print("\n模型回答:")
    print(answer.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="基础 RAG Baseline 问答脚本")
    parser.add_argument("--query", required=True, help="用户问题")
    parser.add_argument("--top-k", type=int, default=3, help="召回文档数")
    parser.add_argument("--fetch-k", type=int, default=None, help="初始召回池大小")
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
    parser.add_argument(
        "--vector-store-dir",
        default=str(DEFAULT_VECTOR_STORE_DIR),
        help="FAISS 索引目录",
    )
    parser.add_argument(
        "--enable-intent",
        choices=["0", "1"],
        default="0",
        help="是否启用意图分类与改写：1 开启，0 关闭",
    )
    args = parser.parse_args()
    vector_store_dir = Path(args.vector_store_dir)
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
        raise FileNotFoundError(
            f"向量索引不存在: {vector_store_dir}，请先运行 build_index.py"
        )

    query = args.query
    if args.enable_intent == "1":
        intent_result = infer_intent_and_rewrite(query)
        rewritten = intent_result.rewritten_query
        intent_label = intent_result.label
        print(f"[Intent] {intent_label}")
        if rewritten != query:
            print(f"[Rewrite] {rewritten}")
        query_to_use = rewritten
    else:
        intent_result = None
        query_to_use = query

    answer_question(
        query_original=query,
        query_for_retrieval=query_to_use,
        top_k=args.top_k,
        vector_store_dir=vector_store_dir,
        intent_result=intent_result,
    )


if __name__ == "__main__":
    main()
