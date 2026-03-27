# coding: UTF-8
import os
from pathlib import Path

from langchain_openai import OpenAIEmbeddings

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ModuleNotFoundError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def openai_kwargs() -> dict:
    base_url = os.getenv("OPENAI_BASE_URL")
    return {"base_url": base_url} if base_url else {}


def embedding_provider() -> str:
    return os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()


def embedding_batch_size() -> int:
    return int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))


def local_embedding_model_path() -> str:
    return os.getenv("LOCAL_EMBEDDING_MODEL", "./models/BAAI/bge-base-zh-v1.5")


def _candidate_local_model_paths() -> list[Path]:
    raw_path = local_embedding_model_path()
    path = Path(raw_path)
    candidates = [path]
    if "___" in raw_path:
        candidates.append(Path(raw_path.replace("___", "---")))
        candidates.append(Path(raw_path.replace("___", "-")))
        candidates.append(Path(raw_path.replace("___", ".")))
    if "---" in raw_path:
        candidates.append(Path(raw_path.replace("---", "___")))
        candidates.append(Path(raw_path.replace("---", ".")))
    if ".v" in raw_path or "-v" in raw_path:
        candidates.append(Path(raw_path.replace(".", "___")))
    seen = []
    unique_candidates = []
    for candidate in candidates:
        resolved = str(candidate)
        if resolved not in seen:
            seen.append(resolved)
            unique_candidates.append(candidate)
    return unique_candidates


def resolve_local_embedding_model_path() -> Path:
    for candidate in _candidate_local_model_paths():
        if candidate.exists():
            return candidate
    return _candidate_local_model_paths()[0]


def inspect_local_embedding_model() -> dict:
    requested = Path(local_embedding_model_path())
    resolved = resolve_local_embedding_model_path()
    warnings = []

    if not resolved.exists():
        warnings.append("本地 embedding 模型目录不存在，将在初始化时失败。")
    if requested != resolved:
        warnings.append(
            f"请求路径 {requested} 不存在，已回退到兼容路径 {resolved}。"
        )

    config_file = resolved / "config.json"
    if resolved.exists() and not config_file.exists():
        warnings.append("模型目录缺少 config.json，请确认是完整的 HuggingFace 模型目录。")

    return {
        "provider": embedding_provider(),
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "exists": resolved.exists(),
        "warnings": warnings,
    }


def create_openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        check_embedding_ctx_length=False,
        chunk_size=embedding_batch_size(),
        **openai_kwargs(),
    )


def create_local_embeddings() -> HuggingFaceEmbeddings:
    resolved_path = resolve_local_embedding_model_path()
    return HuggingFaceEmbeddings(
        model_name=str(resolved_path),
        model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu").strip('"')},
        encode_kwargs={"normalize_embeddings": True},
    )


def create_embeddings():
    provider = embedding_provider()
    if provider == "local_bge":
        return create_local_embeddings()
    return create_openai_embeddings()
