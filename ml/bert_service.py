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
    os.getenv("BERT_MODEL_DIR")
    or Path(__file__).resolve().parent / "bert_fake_news" / "best_checkpoint"
)
DEFAULT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "256"))
REQUIRED_CHECKPOINT_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
)

# Model weights are far too large for Git, so a fresh clone can also fetch them
# from the Hugging Face Hub repository named by BERT_MODEL_ID.
MODEL_SETUP_MESSAGE = (
    "VeriTruth BERT model is not available. Please complete the model setup "
    "described in the README."
)
PLACEHOLDER_REPO_ID = "YOUR_HUGGINGFACE_USERNAME/veritruth-bert-fake-news"

_tokenizer: Any | None = None
_model: Any | None = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_load_lock = Lock()


class BertModelUnavailable(RuntimeError):
    """Raised when the trained BERT artifact is not available or cannot load."""


def _local_checkpoint_ready() -> bool:
    return MODEL_DIR.is_dir() and all(
        (MODEL_DIR / name).is_file() for name in REQUIRED_CHECKPOINT_FILES
    )


def _configured_repo_id() -> str:
    """Return a usable Hugging Face repo id, or an empty string when unset."""
    repo_id = (os.getenv("BERT_MODEL_ID") or "").strip()
    if not repo_id or repo_id == PLACEHOLDER_REPO_ID:
        return ""
    return repo_id


def _load_model() -> tuple[Any, Any]:
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _tokenizer, _model

    with _load_lock:
        if _model is not None and _tokenizer is not None:
            return _tokenizer, _model

        if _local_checkpoint_ready():
            source: Any = MODEL_DIR
            local_files_only = True
        else:
            repo_id = _configured_repo_id()
            if not repo_id:
                raise BertModelUnavailable(
                    f"{MODEL_SETUP_MESSAGE} No local checkpoint was found at {MODEL_DIR} "
                    "and no Hugging Face repository is configured (set BERT_MODEL_ID)."
                )
            # transformers downloads the repo once and reuses its local cache afterwards.
            source = repo_id
            local_files_only = os.getenv("HF_HUB_OFFLINE", "").lower() in ("1", "true", "yes")

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                source, local_files_only=local_files_only
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                source,
                local_files_only=local_files_only,
            )
            model.to(_device)
            model.eval()
        except Exception as exc:
            raise BertModelUnavailable(
                f"{MODEL_SETUP_MESSAGE} (source: {source}: {exc})"
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
    repo_id = _configured_repo_id()
    source = str(MODEL_DIR) if _local_checkpoint_ready() else repo_id or "none"
    print(f"BERT MODEL SOURCE: {source}", flush=True)
    print(f"LOCAL CHECKPOINT: {MODEL_DIR} ({'ready' if _local_checkpoint_ready() else 'incomplete'})", flush=True)
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

