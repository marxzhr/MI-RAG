# coding: UTF-8
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

from embedding_utils import create_embeddings, embedding_provider


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = BASE_DIR / "data" / "data/cmedqa2_answer.jsonl"
DEFAULT_VECTOR_STORE_DIR = BASE_DIR / "vector_store"

load_dotenv(BASE_DIR / ".env")


def load_docs(path: Path) -> list[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            docs.append(
                Document(
                    page_content=row["content"],
                    metadata={
                        "doc_id": row.get("doc_id", ""),
                        "title": row.get("title", ""),
                        "source": row.get("title", ""),
                    },
                )
            )
    return docs


def slice_docs(docs: list[Document], limit_docs: int | None) -> list[Document]:
    if limit_docs is None or limit_docs <= 0:
        return docs
    return docs[:limit_docs]


def build_chunks(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    grouped_indices: dict[str, list[int]] = {}
    for idx, chunk in enumerate(chunks):
        doc_id = str(chunk.metadata.get("doc_id", ""))
        grouped_indices.setdefault(doc_id, []).append(idx)

    for doc_id, indices in grouped_indices.items():
        chunk_count = len(indices)
        for chunk_index, global_idx in enumerate(indices):
            chunk = chunks[global_idx]
            chunk.metadata["chunk_index"] = chunk_index
            chunk.metadata["chunk_count"] = chunk_count
            chunk.metadata["chunk_id"] = f"{doc_id}#{chunk_index}" if doc_id else f"chunk#{global_idx}"
            chunk.metadata["chunk_char_len"] = len(chunk.page_content)
    return chunks


def batched(items: list[Document], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def main() -> None:
    parser = argparse.ArgumentParser(description="离线构建基础 RAG 的 FAISS 索引")
    parser.add_argument(
        "--data-path",
        default=str(DEFAULT_DATA_PATH),
        help="知识库 jsonl 路径",
    )
    parser.add_argument(
        "--vector-store-dir",
        default=str(DEFAULT_VECTOR_STORE_DIR),
        help="FAISS 索引输出目录",
    )
    parser.add_argument(
        "--limit-docs",
        type=int,
        default=None,
        help="仅使用前 N 条文档构建索引，便于快速测试",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="分批写入 FAISS 的 chunk 批大小",
    )
    args = parser.parse_args()

    data_path = Path(args.data_path)
    vector_store_dir = Path(args.vector_store_dir)

    if embedding_provider() == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("缺少 OPENAI_API_KEY 环境变量。")
    if not data_path.exists():
        raise FileNotFoundError(f"知识库文件不存在: {data_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size 必须大于 0")

    docs = slice_docs(load_docs(data_path), args.limit_docs)
    chunks = build_chunks(docs)
    embeddings = create_embeddings()

    vector_store_dir.mkdir(parents=True, exist_ok=True)
    if not chunks:
        raise ValueError("没有可用于建索引的 chunk。")

    vectorstore = None
    progress = tqdm(
        batched(chunks, args.batch_size),
        total=(len(chunks) + args.batch_size - 1) // args.batch_size,
        desc="Building FAISS",
        unit="batch",
    )
    for batch_idx, (_, batch_docs) in enumerate(progress, 1):
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch_docs, embeddings)
        else:
            vectorstore.add_documents(batch_docs)
        progress.set_postfix(
            batch=batch_idx,
            chunk_count=min(batch_idx * args.batch_size, len(chunks)),
            total_chunks=len(chunks),
        )

    if vectorstore is None:
        raise RuntimeError("FAISS 索引构建失败。")
    vectorstore.save_local(str(vector_store_dir))

    print("=" * 60)
    print(f"知识库文档数: {len(docs)}")
    print(f"切分后 chunk 数: {len(chunks)}")
    print(f"Embedding Provider: {embedding_provider()}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Limit Docs: {args.limit_docs if args.limit_docs else 'ALL'}")
    print(f"索引已保存到: {vector_store_dir}")


if __name__ == "__main__":
    main()
