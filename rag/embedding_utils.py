# coding: UTF-8
import os

from langchain_openai import OpenAIEmbeddings


def openai_kwargs() -> dict:
    base_url = os.getenv("OPENAI_BASE_URL")
    return {"base_url": base_url} if base_url else {}


def embedding_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()


def embedding_batch_size() -> int:
    return int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))


def local_embedding_model_path() -> str:
    return os.getenv("LOCAL_EMBEDDING_MODEL", "./models/BAAI/bge-base-zh-v1.5")


def create_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        check_embedding_ctx_length=False,
        chunk_size=embedding_batch_size(),
        **openai_kwargs(),
    )


def create_local_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=local_embedding_model_path(),
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_embeddings():
    provider = embedding_provider()
    if provider == "local_bge":
        return create_local_embeddings()
    return create_openai_embeddings()
