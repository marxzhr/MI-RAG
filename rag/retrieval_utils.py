# coding: UTF-8
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


_RERANKER_CACHE: dict[str, "CrossEncoderReranker"] = {}


def retrieval_fetch_k(top_k: int) -> int:
    return max(top_k, int(os.getenv("RETRIEVAL_FETCH_K", "20")))


def dedup_text_threshold() -> float:
    return float(os.getenv("DEDUP_TEXT_THRESHOLD", "0.96"))


def rerank_enabled() -> bool:
    return os.getenv("ENABLE_RERANK", "0").strip().lower() in {"1", "true", "yes", "on"}


def dedup_enabled() -> bool:
    return os.getenv("ENABLE_DEDUP", "1").strip().lower() in {"1", "true", "yes", "on"}


def rerank_model_name() -> str:
    env_value = os.getenv("RERANK_MODEL")
    if env_value:
        return env_value

    local_candidates = [
        "./rag/models/medical_reranker_candidates",
        "./models/medical_reranker_candidates",
        "./models/BAAI/bge-reranker-base",
    ]
    for candidate in local_candidates:
        if os.path.exists(candidate):
            return candidate
    return "BAAI/bge-reranker-base"


def rerank_top_n() -> int:
    return int(os.getenv("RERANK_TOP_N", "5"))


def rerank_device() -> str:
    return os.getenv("RERANK_DEVICE", os.getenv("EMBEDDING_DEVICE", "cpu"))


def normalize_text(text: str) -> str:
    return "".join(text.lower().split())


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=normalize_text(left), b=normalize_text(right)).ratio()


@dataclass
class RetrievedItem:
    doc: Any
    raw_rank: int
    raw_score: float | None
    rerank_score: float | None = None
    dedup_key: str | None = None

    def to_dict(self) -> dict:
        metadata = dict(self.doc.metadata)
        return {
            "doc_id": metadata.get("doc_id", ""),
            "title": metadata.get("title", ""),
            "chunk_id": metadata.get("chunk_id", ""),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_count": metadata.get("chunk_count"),
            "raw_rank": self.raw_rank,
            "raw_score": float(self.raw_score) if self.raw_score is not None else None,
            "rerank_score": float(self.rerank_score) if self.rerank_score is not None else None,
            "page_content": self.doc.page_content,
        }


class CrossEncoderReranker:
    def __init__(self) -> None:
        from sentence_transformers import CrossEncoder

        model_name = rerank_model_name()
        if not os.path.exists(model_name) and "/" not in model_name and "\\" not in model_name:
            # HuggingFace repo id is allowed.
            pass
        original_hf_hub_offline = os.environ.get("HF_HUB_OFFLINE")
        original_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            self.model = CrossEncoder(
                model_name,
                device=rerank_device(),
            )
        finally:
            if original_hf_hub_offline is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = original_hf_hub_offline
            if original_transformers_offline is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = original_transformers_offline

    def score(self, query: str, docs: list[RetrievedItem]) -> list[float]:
        pairs = [[query, item.doc.page_content] for item in docs]
        return list(self.model.predict(pairs))


def get_cached_reranker(model_name: str | None = None) -> CrossEncoderReranker:
    target_model = model_name or rerank_model_name()
    reranker = _RERANKER_CACHE.get(target_model)
    if reranker is None:
        reranker = CrossEncoderReranker()
        _RERANKER_CACHE[target_model] = reranker
    return reranker


def deduplicate_items(items: list[RetrievedItem]) -> tuple[list[RetrievedItem], dict]:
    deduped = []
    seen_doc_ids = set()
    kept_texts = []
    stats = {
        "raw_candidates": len(items),
        "removed_same_doc_id": 0,
        "removed_near_duplicate_text": 0,
    }

    for item in items:
        doc_id = str(item.doc.metadata.get("doc_id", "")).strip()
        if doc_id and doc_id in seen_doc_ids:
            stats["removed_same_doc_id"] += 1
            continue

        current_text = item.doc.page_content
        if any(text_similarity(current_text, kept) >= dedup_text_threshold() for kept in kept_texts):
            stats["removed_near_duplicate_text"] += 1
            continue

        if doc_id:
            seen_doc_ids.add(doc_id)
        kept_texts.append(current_text)
        deduped.append(item)

    stats["deduped_candidates"] = len(deduped)
    return deduped, stats


def retrieve_documents(
    vectorstore: Any,
    query: str,
    top_k: int,
    fetch_k: int | None = None,
    use_rerank: bool | None = None,
    use_dedup: bool | None = None,
    query_embedding: list[float] | None = None,
) -> tuple[list[RetrievedItem], dict]:
    fetch_k = fetch_k or retrieval_fetch_k(top_k)
    if query_embedding is not None:
        index_dim = getattr(getattr(vectorstore, "index", None), "d", None)
        query_dim = len(query_embedding)
        if index_dim is not None and query_dim != index_dim:
            raise RuntimeError(
                "查询向量维度与 FAISS 索引维度不一致，通常是因为索引构建时使用的 embedding 模型"
                "与当前检索时加载的 embedding 模型不同。\n"
                f"query_dim={query_dim}, faiss_dim={index_dim}\n"
                "请确认：1. --vector-store-dir 指向正确的索引目录；"
                "2. 当前 .env 中的 EMBEDDING_PROVIDER / LOCAL_EMBEDDING_MODEL 配置"
                "与建索引时一致；3. 必要时重新运行 build_index.py 重建索引。"
            )
        if hasattr(vectorstore, "similarity_search_by_vector_with_relevance_scores"):
            raw_pairs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                query_embedding, k=fetch_k
            )
        elif hasattr(vectorstore, "similarity_search_by_vector"):
            docs_only = vectorstore.similarity_search_by_vector(query_embedding, k=fetch_k)
            raw_pairs = [(doc, None) for doc in docs_only]
        else:
            raw_pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)
    else:
        raw_pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)
    raw_items = [
        RetrievedItem(doc=doc, raw_rank=idx, raw_score=score)
        for idx, (doc, score) in enumerate(raw_pairs, 1)
    ]

    use_dedup = dedup_enabled() if use_dedup is None else use_dedup
    if use_dedup:
        deduped_items, dedup_stats = deduplicate_items(raw_items)
    else:
        deduped_items = list(raw_items)
        dedup_stats = {
            "raw_candidates": len(raw_items),
            "removed_same_doc_id": 0,
            "removed_near_duplicate_text": 0,
            "deduped_candidates": len(raw_items),
        }
    retrieval_meta = {
        "fetch_k": fetch_k,
        "top_k": top_k,
        "dedup_enabled": use_dedup,
        "rerank_enabled": False,
        "rerank_model": None,
        "dedup": dedup_stats,
        "raw_items": [item.to_dict() for item in raw_items],
    }

    use_rerank = rerank_enabled() if use_rerank is None else use_rerank
    if use_rerank and deduped_items:
        reranker = get_cached_reranker()
        scores = reranker.score(query, deduped_items)
        for item, score in zip(deduped_items, scores):
            item.rerank_score = float(score)
        deduped_items.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else float("-inf"),
                -(item.raw_rank),
            ),
            reverse=True,
        )
        retrieval_meta["rerank_enabled"] = True
        retrieval_meta["rerank_model"] = rerank_model_name()

    final_items = deduped_items[:top_k]
    retrieval_meta["final_items"] = [item.to_dict() for item in final_items]
    return final_items, retrieval_meta


def retrieve_documents_multi_query(
    vectorstore: Any,
    queries: list[str],
    top_k: int,
    fetch_k: int | None = None,
    use_rerank: bool | None = None,
    use_dedup: bool | None = None,
    query_embeddings: list[list[float]] | None = None,
) -> tuple[list[RetrievedItem], dict]:
    cleaned_queries = []
    seen_queries = set()
    for query in queries:
        text = str(query).strip()
        if not text or text in seen_queries:
            continue
        seen_queries.add(text)
        cleaned_queries.append(text)

    if not cleaned_queries:
        raise ValueError("queries 不能为空。")

    fetch_k = fetch_k or retrieval_fetch_k(top_k)
    merged_items = []
    per_query_meta = []

    merged_items = []
    for idx, query in enumerate(cleaned_queries):
        embedding = None
        if query_embeddings is not None and idx < len(query_embeddings):
            embedding = query_embeddings[idx]

        if embedding is not None:
            index_dim = getattr(getattr(vectorstore, "index", None), "d", None)
            if index_dim is not None and len(embedding) != index_dim:
                raise RuntimeError(
                    "查询向量维度与 FAISS 索引维度不一致。\n"
                    f"query_dim={len(embedding)}, faiss_dim={index_dim}"
                )
            if hasattr(vectorstore, "similarity_search_by_vector_with_relevance_scores"):
                raw_pairs = vectorstore.similarity_search_by_vector_with_relevance_scores(
                    embedding, k=fetch_k
                )
            elif hasattr(vectorstore, "similarity_search_by_vector"):
                docs_only = vectorstore.similarity_search_by_vector(embedding, k=fetch_k)
                raw_pairs = [(doc, None) for doc in docs_only]
            else:
                raw_pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)
        else:
            raw_pairs = vectorstore.similarity_search_with_score(query, k=fetch_k)

        query_items = [
            RetrievedItem(doc=doc, raw_rank=pair_idx, raw_score=score)
            for pair_idx, (doc, score) in enumerate(raw_pairs, 1)
        ]
        per_query_meta.append(
            {
                "query": query,
                "raw_items": [item.to_dict() for item in query_items],
            }
        )
        merged_items.extend(
            query_items
        )

    use_dedup = dedup_enabled() if use_dedup is None else use_dedup
    if use_dedup:
        deduped_items, dedup_stats = deduplicate_items(merged_items)
    else:
        deduped_items = list(merged_items)
        dedup_stats = {
            "raw_candidates": len(merged_items),
            "removed_same_doc_id": 0,
            "removed_near_duplicate_text": 0,
            "deduped_candidates": len(merged_items),
        }

    retrieval_meta = {
        "multi_query": True,
        "queries": cleaned_queries,
        "query_count": len(cleaned_queries),
        "fetch_k": fetch_k,
        "top_k": top_k,
        "dedup_enabled": use_dedup,
        "rerank_enabled": False,
        "rerank_model": None,
        "dedup": dedup_stats,
        "raw_items": [item.to_dict() for item in merged_items],
        "per_query": per_query_meta,
    }

    use_rerank = rerank_enabled() if use_rerank is None else use_rerank
    if use_rerank and deduped_items:
        reranker = get_cached_reranker()
        rerank_query = cleaned_queries[0]
        scores = reranker.score(rerank_query, deduped_items)
        for item, score in zip(deduped_items, scores):
            item.rerank_score = float(score)
        deduped_items.sort(
            key=lambda item: (
                item.rerank_score if item.rerank_score is not None else float("-inf"),
                -(item.raw_rank),
            ),
            reverse=True,
        )
        retrieval_meta["rerank_enabled"] = True
        retrieval_meta["rerank_model"] = rerank_model_name()

    final_items = deduped_items[:top_k]
    retrieval_meta["final_items"] = [item.to_dict() for item in final_items]
    return final_items, retrieval_meta
