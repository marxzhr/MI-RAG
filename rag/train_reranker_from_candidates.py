import argparse
import csv
import json
import random
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from torch.utils.data import DataLoader


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
CMEDQA2_DIR = PROJECT_ROOT / "cMedQA2" / "extracted"
QUESTION_CSV = CMEDQA2_DIR / "question" / "question.csv"
ANSWER_CSV = CMEDQA2_DIR / "answer" / "answer.csv"
TRAIN_CANDIDATES_ZIP = PROJECT_ROOT / "cMedQA2" / "train_candidates.zip"
DEV_CANDIDATES_ZIP = PROJECT_ROOT / "cMedQA2" / "dev_candidates.zip"
DEFAULT_OUTPUT_DIR = BASE_DIR / "models" / "medical_reranker_candidates"


@dataclass
class TrainConfig:
    model_name: str = "BAAI/bge-reranker-base"
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    max_query_length: int = 96
    max_answer_length: int = 256
    epochs: int = 1
    batch_size: int = 8
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    negatives_per_positive: int = 4
    max_train_rows: int = 120000
    max_dev_rows: int = 20000
    eval_steps: int = 1000
    save_steps: int = 1000
    seed: int = 42


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="基于 cMedQA2 train_candidates 训练 reranker")
    parser.add_argument("--model-name", default=TrainConfig.model_name)
    parser.add_argument("--output-dir", default=TrainConfig.output_dir)
    parser.add_argument("--max-query-length", type=int, default=TrainConfig.max_query_length)
    parser.add_argument("--max-answer-length", type=int, default=TrainConfig.max_answer_length)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
    parser.add_argument("--warmup-ratio", type=float, default=TrainConfig.warmup_ratio)
    parser.add_argument("--negatives-per-positive", type=int, default=TrainConfig.negatives_per_positive)
    parser.add_argument("--max-train-rows", type=int, default=TrainConfig.max_train_rows)
    parser.add_argument("--max-dev-rows", type=int, default=TrainConfig.max_dev_rows)
    parser.add_argument("--eval-steps", type=int, default=TrainConfig.eval_steps)
    parser.add_argument("--save-steps", type=int, default=TrainConfig.save_steps)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def read_csv_rows(path: Path) -> list[dict]:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            with path.open("r", encoding=encoding, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"无法解码文件: {path}")


def build_text_maps() -> tuple[dict[str, str], dict[str, str]]:
    question_rows = read_csv_rows(QUESTION_CSV)
    answer_rows = read_csv_rows(ANSWER_CSV)
    question_map = {str(row["question_id"]): (row.get("content") or "").strip() for row in question_rows}
    answer_map = {str(row["ans_id"]): (row.get("content") or "").strip() for row in answer_rows}
    return question_map, answer_map


def iter_candidate_rows(zip_path: Path, member_name: str) -> Iterable[dict]:
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(member_name) as f:
            text = f.read().decode("utf-8")
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        yield row


def truncate(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit]


def build_examples(
    rows: Iterable[dict],
    question_map: dict[str, str],
    answer_map: dict[str, str],
    max_query_length: int,
    max_answer_length: int,
    negatives_per_positive: int,
    max_rows: int,
):
    from sentence_transformers import InputExample

    rows = list(rows)
    if not rows:
        return []

    examples = []
    pair_count = 0

    if "pos_ans_id" in rows[0] and "neg_ans_id" in rows[0]:
        grouped: dict[tuple[str, str], list[str]] = {}
        for row in rows:
            key = (str(row["question_id"]), str(row["pos_ans_id"]))
            grouped.setdefault(key, []).append(str(row["neg_ans_id"]))

        for (question_id, pos_ans_id), neg_ids in grouped.items():
            query = truncate(question_map.get(question_id, ""), max_query_length)
            pos_answer = truncate(answer_map.get(pos_ans_id, ""), max_answer_length)
            if not query or not pos_answer:
                continue

            examples.append(InputExample(texts=[query, pos_answer], label=1.0))
            pair_count += 1

            sampled_neg_ids = neg_ids[:negatives_per_positive]
            for neg_ans_id in sampled_neg_ids:
                neg_answer = truncate(answer_map.get(neg_ans_id, ""), max_answer_length)
                if not neg_answer:
                    continue
                examples.append(InputExample(texts=[query, neg_answer], label=0.0))
                pair_count += 1
                if max_rows > 0 and pair_count >= max_rows:
                    return examples
            if max_rows > 0 and pair_count >= max_rows:
                return examples
        return examples

    if "ans_id" in rows[0] and "label" in rows[0]:
        for row in rows:
            question_id = str(row["question_id"])
            ans_id = str(row["ans_id"])
            label = float(row["label"])
            query = truncate(question_map.get(question_id, ""), max_query_length)
            answer = truncate(answer_map.get(ans_id, ""), max_answer_length)
            if not query or not answer:
                continue
            examples.append(InputExample(texts=[query, answer], label=label))
            pair_count += 1
            if max_rows > 0 and pair_count >= max_rows:
                return examples
        return examples

    raise KeyError(f"不支持的 candidates 字段: {list(rows[0].keys())}")


def main() -> None:
    config = parse_args()
    random.seed(config.seed)

    from sentence_transformers.cross_encoder import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator

    question_map, answer_map = build_text_maps()

    train_examples = build_examples(
        rows=iter_candidate_rows(TRAIN_CANDIDATES_ZIP, "train_candidates.txt"),
        question_map=question_map,
        answer_map=answer_map,
        max_query_length=config.max_query_length,
        max_answer_length=config.max_answer_length,
        negatives_per_positive=config.negatives_per_positive,
        max_rows=config.max_train_rows,
    )
    dev_examples = build_examples(
        rows=iter_candidate_rows(DEV_CANDIDATES_ZIP, "dev_candidates.txt"),
        question_map=question_map,
        answer_map=answer_map,
        max_query_length=config.max_query_length,
        max_answer_length=config.max_answer_length,
        negatives_per_positive=config.negatives_per_positive,
        max_rows=config.max_dev_rows,
    )

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(
        dev_examples,
        name="cmedqa2-dev-candidates",
    )

    model = CrossEncoder(
        config.model_name,
        num_labels=1,
        max_length=config.max_query_length + config.max_answer_length,
    )

    warmup_steps = int(len(train_dataloader) * config.epochs * config.warmup_ratio)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=config.epochs,
        evaluation_steps=config.eval_steps,
        warmup_steps=warmup_steps,
        output_path=str(output_dir),
        optimizer_params={"lr": config.learning_rate},
        save_best_model=True,
        show_progress_bar=True,
    )

    final_score = evaluator(model)
    if isinstance(final_score, dict):
        serializable_score = {
            str(key): float(value) if isinstance(value, (int, float)) else value
            for key, value in final_score.items()
        }
    else:
        serializable_score = float(final_score)
    summary = {
        "config": asdict(config),
        "train_examples": len(train_examples),
        "dev_examples": len(dev_examples),
        "final_eval_score": serializable_score,
        "question_count": len(question_map),
        "answer_count": len(answer_map),
        "output_dir": str(output_dir),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 60)
    print(f"Train Examples: {len(train_examples)}")
    print(f"Dev Examples: {len(dev_examples)}")
    if isinstance(serializable_score, dict):
        for key, value in serializable_score.items():
            print(f"{key}: {value}")
    else:
        print(f"Final Eval Score: {serializable_score:.4f}")
    print(f"Best Model Dir: {output_dir}")
    print(f"Summary File: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
