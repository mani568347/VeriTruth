"""Fine-tune bert-base-uncased for the project's fake-news dataset.

Run from the project root with the workspace environment, for example:
    pyenv\\Scripts\\python.exe ml\\bert_training.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FAKE_DATA_PATH = PROJECT_ROOT / "detector" / "data" / "Fake.csv"
REAL_DATA_PATH = PROJECT_ROOT / "detector" / "data" / "True.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "ml" / "bert_fake_news"
LABEL_NAMES = {0: "FAKE", 1: "REAL"}


@dataclass(frozen=True)
class TrainingConfig:
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    batch_size: int = 16
    epochs: int = 2
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    validation_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42
    early_stopping_patience: int = 2
    max_rows_per_class: int | None = 600
    freeze_lower_layers: bool = True


class TokenizedNewsDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_and_prepare_data(
    max_rows_per_class: int | None = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    for path, label in ((FAKE_DATA_PATH, 0), (REAL_DATA_PATH, 1)):
        frame = pd.read_csv(path, usecols=["title", "text"])
        frame["label"] = label
        title_clean = frame["title"].fillna("").astype(str).str.strip()
        text_clean = frame["text"].fillna("").astype(str).str.strip()

        # Clean wire prefixes from real news so the model learns semantic tone rather than memorizing 'Reuters'
        if label == 1:
            text_clean = text_clean.apply(
                lambda t: re.sub(r"^[A-Za-z\s,/\.]*\(Reuters\)\s*-\s*", "", t, flags=re.IGNORECASE).strip()
            )

        # Combine title and text to expose model to headline patterns and article bodies
        frame["text"] = (title_clean + ". " + text_clean).str.strip()
        frames.append(frame[["text", "label"]])

    data = pd.concat(frames, ignore_index=True)
    data["text"] = data["text"].fillna("").astype(str).str.strip()
    data = data[data["text"].str.len() >= 15].copy()
    data["text_key"] = data["text"].str.replace(r"\s+", " ", regex=True).str.casefold()
    before_deduplication = len(data)
    data = data.drop_duplicates(subset=["text_key"], keep="first").reset_index(drop=True)
    data = data.drop(columns=["text_key"])

    if max_rows_per_class is not None:
        data = (
            data.groupby("label", group_keys=False)
            .sample(n=min(max_rows_per_class, data["label"].value_counts().min()), random_state=seed)
            .reset_index(drop=True)
        )

    summary = {
        "source_files": [str(FAKE_DATA_PATH), str(REAL_DATA_PATH)],
        "rows_before_cleaning": int(before_deduplication),
        "rows_after_cleaning_and_deduplication": int(len(data)),
        "class_distribution": {
            LABEL_NAMES[int(label)]: int(count)
            for label, count in data["label"].value_counts().sort_index().items()
        },
        "label_mapping": LABEL_NAMES,
    }
    return data, summary


def split_data(data: pd.DataFrame, config: TrainingConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, remainder = train_test_split(
        data,
        test_size=config.validation_size + config.test_size,
        stratify=data["label"],
        random_state=config.seed,
    )
    relative_test_size = config.test_size / (config.validation_size + config.test_size)
    validation, test = train_test_split(
        remainder,
        test_size=relative_test_size,
        stratify=remainder["label"],
        random_state=config.seed,
    )
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)


def tokenize_frame(frame: pd.DataFrame, tokenizer: Any, max_length: int) -> TokenizedNewsDataset:
    encodings = tokenizer(
        frame["text"].tolist(),
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    )
    return TokenizedNewsDataset(encodings, frame["label"].astype(int).tolist())


def evaluate_model(
    model: Any,
    loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model.eval()
    labels: list[int] = []
    predictions: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            probs = torch.softmax(outputs.logits, dim=1)
            labels.extend(batch["labels"].cpu().tolist())
            predictions.extend(probs.argmax(dim=1).cpu().tolist())
            probabilities.extend(probs[:, 1].cpu().tolist())

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision_real": float(precision),
        "recall_real": float(recall),
        "f1_real": float(f1),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[LABEL_NAMES[0], LABEL_NAMES[1]],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(labels, probabilities))
    except ValueError:
        metrics["roc_auc"] = None
    return metrics, np.asarray(labels), np.asarray(predictions)


def train(config: TrainingConfig, output_dir: Path) -> dict[str, Any]:
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    data, data_summary = load_and_prepare_data(config.max_rows_per_class, config.seed)
    train_frame, validation_frame, test_frame = split_data(data, config)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=2,
        id2label=LABEL_NAMES,
        label2id={name: label for label, name in LABEL_NAMES.items()},
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # If running on CPU and freeze_lower_layers is enabled, freeze lower transformer blocks for speed
    if config.freeze_lower_layers and not torch.cuda.is_available():
        for name, param in model.named_parameters():
            if not any(k in name for k in ["layer.9", "layer.10", "layer.11", "pooler", "classifier"]):
                param.requires_grad = False

    train_dataset = tokenize_frame(train_frame, tokenizer, config.max_length)
    validation_dataset = tokenize_frame(validation_frame, tokenizer, config.max_length)
    test_dataset = tokenize_frame(test_frame, tokenizer, config.max_length)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay)
    total_steps = len(train_loader) * config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * config.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    best_dir = output_dir / "best_checkpoint"

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(outputs.loss.item())

        validation_metrics, _, _ = evaluate_model(model, validation_loader, device)
        macro_f1 = float(validation_metrics["classification_report"]["macro avg"]["f1-score"])
        val_acc = float(validation_metrics["accuracy"])
        print(
            f"epoch={epoch + 1}/{config.epochs} "
            f"loss={total_loss / max(len(train_loader), 1):.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_macro_f1={macro_f1:.4f}",
            flush=True
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            epochs_without_improvement = 0
            if best_dir.exists():
                for child in best_dir.iterdir():
                    if child.is_file():
                        child.unlink()
            best_dir.mkdir(exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.early_stopping_patience:
                break

    best_model = AutoModelForSequenceClassification.from_pretrained(best_dir).to(device)
    test_metrics, _, _ = evaluate_model(best_model, test_loader, device)
    best_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    split_summary = {
        "train": {"rows": len(train_frame), "class_distribution": train_frame["label"].value_counts().sort_index().to_dict()},
        "validation": {"rows": len(validation_frame), "class_distribution": validation_frame["label"].value_counts().sort_index().to_dict()},
        "test": {"rows": len(test_frame), "class_distribution": test_frame["label"].value_counts().sort_index().to_dict()},
    }
    metadata = {
        "model_name": config.model_name,
        "max_length": config.max_length,
        "device_used": str(device),
        "training_config": config.__dict__,
        "data_summary": data_summary,
        "split_summary": split_summary,
        "test_metrics": test_metrics,
    }
    (output_dir / "label_mapping.json").write_text(json.dumps(LABEL_NAMES, indent=2), encoding="utf-8")
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--max-rows-per-class", type=int, default=600)
    parser.add_argument("--full-fine-tune", action="store_true", help="Fine-tune all layers instead of freezing lower blocks")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        max_rows_per_class=args.max_rows_per_class,
        freeze_lower_layers=not args.full_fine_tune,
    )
    result = train(config, args.output_dir)
    print(json.dumps(result["test_metrics"], indent=2))
