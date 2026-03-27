# coding: UTF-8
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import APIConnectionError


BASE_DIR = Path(__file__).resolve().parent
KB_PATH = BASE_DIR / "kb_docs.jsonl"

load_dotenv(BASE_DIR / ".env")


def openai_kwargs() -> dict:
    base_url = os.getenv("OPENAI_BASE_URL")
    return {"base_url": base_url} if base_url else {}


def load_docs(path: Path) -> list[Document]:
    docs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            docs.append(
                Document(
                    page_content=row["content"],
                    metadata={
                        "doc_id": row.get("doc_id", ""),
                        "title": row.get("title", ""),
                    },
                )
            )
    return docs


def build_vectorstore(path: Path) -> FAISS:
    docs = load_docs(path)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        check_embedding_ctx_length=False,
        **openai_kwargs(),
    )
    return FAISS.from_documents(chunks, embeddings)


def answer_question(query: str, top_k: int) -> None:
    try:
        vectorstore = build_vectorstore(KB_PATH)
        retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
        context_docs = retriever.invoke(query)
        context = "\n\n".join(
            f"[{doc.metadata.get('title') or '未命名文档'}]\n{doc.page_content}"
            for doc in context_docs
        )

        prompt = ChatPromptTemplate.from_template(
            "你是医疗问答助手。请严格基于已检索到的知识回答；如果证据不足，要明确说不知道。\n\n"
            "问题：{query}\n\n"
            "参考资料：\n{context}"
        )
        llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            **openai_kwargs(),
        )
        answer = (prompt | llm).invoke({"query": query, "context": context})
    except APIConnectionError as exc:
        http_proxy = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
        https_proxy = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
        proxy_hint = (
            f"检测到代理配置 HTTP_PROXY/HTTPS_PROXY: {http_proxy or https_proxy}"
            if (http_proxy or https_proxy)
            else "未检测到 HTTP_PROXY/HTTPS_PROXY。"
        )
        raise RuntimeError(
            "调用 OpenAI 接口失败，属于网络连接问题，不是 RAG 代码逻辑错误。\n"
            f"OPENAI_BASE_URL={os.getenv('OPENAI_BASE_URL', '未设置')}\n"
            f"{proxy_hint}\n"
            "请检查：1. 当前网络是否能访问该接口；2. 代理是否可用；"
            "3. 如使用代理，尝试临时取消代理后重试。"
        ) from exc

    print("=" * 60)
    print("问题:", query)
    print("\n检索结果:")
    for i, doc in enumerate(context_docs, 1):
        print(f"[{i}] {doc.metadata.get('title') or '未命名文档'}")
        print(doc.page_content)
        print("-" * 60)
    print("\n模型回答:")
    print(answer.content)


def main() -> None:
    parser = argparse.ArgumentParser(description="最小化 LangChain + FAISS RAG Demo")
    parser.add_argument("--query", required=True, help="用户问题")
    parser.add_argument("--top-k", type=int, default=3, help="召回文档数")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("缺少 OPENAI_API_KEY 环境变量。")
    if not KB_PATH.exists():
        raise FileNotFoundError(f"知识库文件不存在: {KB_PATH}")

    answer_question(args.query, args.top_k)


if __name__ == "__main__":
    main()
