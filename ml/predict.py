"""Compatibility import for callers migrating from the legacy predictor."""

from .bert_service import bert_predict

def predict_news(text):
    return bert_predict(text)
