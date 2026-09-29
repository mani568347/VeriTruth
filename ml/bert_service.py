"""Cached inference service for the fine-tuned fake-news BERT model."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(
    os.getenv(
        "BERT_MODEL_DIR",
        Path(__file__).resolve().parent / "bert_fake_news" / "best_checkpoint",
    )
)
DEFAULT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "256"))
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
)

_tokenizer: Any | None = None
_model: Any | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_load_lock = Lock()


class BertModelUnavailable(RuntimeError):
    """Raised when the trained BERT artifact is not available or cannot load."""


def _load_model() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model
        if not MODEL_DIR.is_dir() or any(
            not (MODEL_DIR / name).is_file() for name in REQUIRED_CHECKPOINT_FILES
        ):
            raise BertModelUnavailable(
                f"Fine-tuned BERT model not found at {MODEL_DIR}. "
                "Run ml/bert_training.py before starting predictions."
            )
        try:
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL_DIR,
                local_files_only=True,
            )
            model.to(_device)
            model.eval()
        except Exception as exc:
            raise BertModelUnavailable(
                f"Could not load the fine-tuned BERT model from {MODEL_DIR}: {exc}"
            ) from exc
        _tokenizer = tokenizer
        _model = model
        _log_checkpoint_status(tokenizer, model)
    return _tokenizer, _model


def _is_development() -> bool:
    try:
        from django.conf import settings

        return bool(getattr(settings, "DEBUG", False))
    except Exception:
        return True


def _log_checkpoint_status(tokenizer: Any, model: Any) -> None:
    if not _is_development():
        return
    print(f"BERT MODEL PATH: {MODEL_DIR}", flush=True)
    print(f"MODEL EXISTS: {MODEL_DIR.is_dir()}", flush=True)
    print(f"TOKENIZER LOADED: {tokenizer is not None}", flush=True)
    print(f"BERT MODEL LOADED: {model is not None}", flush=True)
    print(f"ID2LABEL: {getattr(model.config, 'id2label', None)}", flush=True)


def warmup_model() -> bool:
    """Pre-warm the BERT model and tokenizer in memory to eliminate cold-start latency."""
    try:
        _load_model()
        return True
    except Exception as exc:
        print(f"[VeriTruth] Warning: BERT model warmup failed: {exc}")
        return False


def _label_name(class_id: int, model: Any) -> str:
    configured = getattr(model.config, "id2label", {}).get(class_id)
    if configured is None:
        configured = getattr(model.config, "id2label", {}).get(str(class_id))
    normalized = str(configured or ("FAKE" if class_id == 0 else "REAL")).upper()
    return "Real News" if normalized in {"REAL", "REAL NEWS", "POSITIVE"} else "Fake News"


def predict_proba(texts: list[str]) -> np.ndarray:
    """Return class probabilities in the fixed order [FAKE (index 0), REAL (index 1)]."""
    if not texts:
        return np.empty((0, 2), dtype=np.float32)
    tokenizer, model = _load_model()
    inputs = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=DEFAULT_MAX_LENGTH,
        return_tensors="pt",
    )
    inputs = {key: value.to(_device) for key, value in inputs.items()}
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
    return probabilities.cpu().numpy()


def predict_text(text: str) -> tuple[str, float]:
    """Return the BERT verdict and its winning-class probability percentage."""
    if not text or not text.strip():
        raise ValueError("Prediction text cannot be empty.")
    tokenizer, model = _load_model()
    probabilities = predict_proba([text])[0]
    class_id = int(np.argmax(probabilities))
    confidence = round(float(probabilities[class_id]) * 100, 2)
    return _label_name(class_id, model), confidence


# Compatibility name for existing callers while the service remains the single classifier.
bert_predict = predict_text

