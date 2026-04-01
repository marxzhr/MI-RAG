# coding: UTF-8
import argparse
import inspect
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_TRAIN_PATH = BASE_DIR.parent / "KUAKE-QIC" / "KUAKE-QIC_train.json"
DEFAULT_DEV_PATH = BASE_DIR.parent / "KUAKE-QIC" / "KUAKE-QIC_dev.json"


@dataclass
class TrainConfig:
    model_name: str = "nghuyong/ernie-health-zh"
    output_dir: str = str(BASE_DIR / "models" / "intent_ernie_health")
    train_path: str = str(DEFAULT_TRAIN_PATH)
    dev_path: str = str(DEFAULT_DEV_PATH)
    max_length: int = 96
    epochs: int = 5
    batch_size: int = 16
    eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    gradient_accumulation_steps: int = 1
    label_smoothing_factor: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    seed: int = 42


DEFAULT_CONFIG = TrainConfig()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_label_mappings(train_rows: list[dict], dev_rows: list[dict]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted({row["label"] for row in train_rows + dev_rows})
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


class KuakeQICDataset(Dataset):
    def __init__(self, rows: list[dict], tokenizer, label2id: dict[str, int], max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        encoded = self.tokenizer(row["query"], truncation=True, max_length=self.max_length)
        encoded["labels"] = self.label2id[row["label"]]
        return encoded


def compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_training_arguments(config: TrainConfig, output_dir: Path) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    accepted = set(signature.parameters.keys())

    kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": config.learning_rate,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "num_train_epochs": config.epochs,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "label_smoothing_factor": config.label_smoothing_factor,
        "max_grad_norm": config.max_grad_norm,
        "logging_steps": config.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "save_total_limit": 2,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "seed": config.seed,
    }
    if "overwrite_output_dir" in accepted:
        kwargs["overwrite_output_dir"] = True
    if "evaluation_strategy" in accepted:
        kwargs["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in accepted:
        kwargs["eval_strategy"] = "epoch"
    if "save_strategy" in accepted:
        kwargs["save_strategy"] = "epoch"
    if "do_train" in accepted:
        kwargs["do_train"] = True
    if "do_eval" in accepted:
        kwargs["do_eval"] = True

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return TrainingArguments(**filtered_kwargs)


def build_trainer(model, training_args, train_dataset, dev_dataset, tokenizer, data_collator):
    signature = inspect.signature(Trainer.__init__)
    accepted = set(signature.parameters.keys())
    kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    if "tokenizer" in accepted:
        kwargs["tokenizer"] = tokenizer
    if "processing_class" in accepted:
        kwargs["processing_class"] = tokenizer
    filtered_kwargs = {key: value for key, value in kwargs.items() if key in accepted}
    return Trainer(**filtered_kwargs)


def save_training_metadata(output_dir: Path, config: TrainConfig, label2id: dict[str, int], id2label: dict[int, str], metrics: dict) -> None:
    payload = {
        "config": asdict(config),
        "num_labels": len(label2id),
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "metrics": metrics,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="使用 ERNIE-Health 训练 KUAKE-QIC 意图分类模型")
    parser.add_argument("--model-name", default=DEFAULT_CONFIG.model_name, help="预训练模型名或本地路径")
    parser.add_argument("--output-dir", default=DEFAULT_CONFIG.output_dir, help="模型输出目录")
    parser.add_argument("--train-path", default=DEFAULT_CONFIG.train_path, help="KUAKE-QIC 训练集路径")
    parser.add_argument("--dev-path", default=DEFAULT_CONFIG.dev_path, help="KUAKE-QIC 验证集路径")
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG.max_length, help="最大序列长度")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG.epochs, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG.batch_size, help="训练 batch size")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_CONFIG.eval_batch_size, help="验证 batch size")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_CONFIG.learning_rate, help="学习率")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CONFIG.weight_decay, help="weight decay")
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_CONFIG.warmup_ratio, help="warmup ratio")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DEFAULT_CONFIG.gradient_accumulation_steps, help="梯度累积步数")
    parser.add_argument("--label-smoothing-factor", type=float, default=DEFAULT_CONFIG.label_smoothing_factor, help="label smoothing")
    parser.add_argument("--max-grad-norm", type=float, default=DEFAULT_CONFIG.max_grad_norm, help="梯度裁剪阈值")
    parser.add_argument("--logging-steps", type=int, default=DEFAULT_CONFIG.logging_steps, help="日志间隔")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.seed, help="随机种子")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def main() -> None:
    config = parse_args()
    train_path = Path(config.train_path)
    dev_path = Path(config.dev_path)
    output_dir = Path(config.output_dir)

    if not train_path.exists():
        raise FileNotFoundError(f"训练集不存在: {train_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"验证集不存在: {dev_path}")

    set_seed(config.seed)
    train_rows = load_rows(train_path)
    dev_rows = load_rows(dev_path)
    label2id, id2label = build_label_mappings(train_rows, dev_rows)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
    )

    train_dataset = KuakeQICDataset(train_rows, tokenizer, label2id, config.max_length)
    dev_dataset = KuakeQICDataset(dev_rows, tokenizer, label2id, config.max_length)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = build_training_arguments(config, output_dir)
    trainer = build_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        dev_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    metrics = trainer.evaluate()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    save_training_metadata(output_dir, config, label2id, id2label, metrics)

    print("=" * 60)
    print(f"Device: {resolve_device()}")
    print(f"Train Size: {len(train_rows)}")
    print(f"Dev Size: {len(dev_rows)}")
    print(f"Num Labels: {len(label2id)}")
    print(f"Best Model Dir: {output_dir}")
    print(f"Eval Accuracy: {metrics.get('eval_accuracy', 0.0):.4f}")
    print(f"Eval Macro-F1: {metrics.get('eval_macro_f1', 0.0):.4f}")
    print(f"Summary File: {output_dir / 'training_summary.json'}")


if __name__ == "__main__":
    main()
