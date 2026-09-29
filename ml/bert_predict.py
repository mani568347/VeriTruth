"""Backward-compatible import path for the single BERT inference service."""

from .bert_service import BertModelUnavailable, bert_predict, predict_proba, predict_text, warmup_model

__all__ = ["BertModelUnavailable", "bert_predict", "predict_proba", "predict_text", "warmup_model"]

