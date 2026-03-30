# coding: UTF-8
import os
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any


def retrieval_fetch_k(top_k: int) -> int:
    return max(top_k, int(os.getenv("RETRIEVAL_FETCH_K", "20")))


def dedup_text_threshold() -> float:
    return float(os.getenv("DEDUP_TEXT_THRESHOLD", "0.96"))


def rerank_enabled() -> bool:
    return os.getenv("ENABLE_RERANK", "0").strip().lower() in {"1", "true", "yes", "on"}


def dedup_enabled() -> bool:
    return os.getenv("ENABLE_DEDUP", "1").strip().lower() in {"1", "true", "yes", "on"}


def rerank_model_name() -> str:
    return os.getenv("RERANK_MODEL", "./models/BAAI/bge-reranker-base")


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
        import os
        from sentence_transformers import CrossEncoder

        model_name = rerank_model_name()
        if not os.path.exists(model_name) and "/" not in model_name and "\\" not in model_name:
            # HuggingFace repo id is allowed.
            pass
        self.model = CrossEncoder(model_name, device=rerank_device())

    def score(self, query: str, docs: list[RetrievedItem]) -> list[float]:
        pairs = [[query, item.doc.page_content] for item in docs]
        return list(self.model.predict(pairs))


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
        reranker = CrossEncoderReranker()
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
