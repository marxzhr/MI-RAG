from rag.retrieval_utils import CrossEncoderReranker


def main() -> None:
    reranker = CrossEncoderReranker()
    print("loaded", type(reranker.model).__name__)


if __name__ == "__main__":
    main()
