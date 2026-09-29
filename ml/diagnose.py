"""Diagnostic utility for testing the BERT fake-news classifier directly."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

import numpy as np
import torch

from ml.bert_service import DEFAULT_MAX_LENGTH, _label_name, _load_model


def run_diagnostic(sample_texts: list[str]) -> None:
    tokenizer, model = _load_model()
    device = next(model.parameters()).device

    print("=" * 60)
    print("VERITRUTH BERT MODEL DIAGNOSTIC")
    print("=" * 60)
    print(f"Tokenizer:  {tokenizer.__class__.__name__}")
    print(f"Model:      {model.__class__.__name__}")
    print(f"Device:     {device}")
    print(f"Num Labels: {model.config.num_labels}")
    print(f"ID2LABEL:   {getattr(model.config, 'id2label', {})}")
    print(f"LABEL2ID:   {getattr(model.config, 'label2id', {})}")
    print("=" * 60)

    for idx, text in enumerate(sample_texts, 1):
        inputs = tokenizer(
            [text],
            truncation=True,
            padding=True,
            max_length=DEFAULT_MAX_LENGTH,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0].cpu().numpy()
            probs = torch.softmax(outputs.logits, dim=1)[0].cpu().numpy()

        class_id = int(np.argmax(probs))
        label = _label_name(class_id, model)
        confidence = round(float(probs[class_id]) * 100, 2)

        print(f"\n--- SAMPLE {idx} ---")
        print(f"Input:           {text[:120]}..." if len(text) > 120 else f"Input:           {text}")
        print(f"Predicted Class: {class_id}")
        print(f"Logits:          [0 (FAKE): {logits[0]:.4f}, 1 (REAL): {logits[1]:.4f}]")
        print(f"Probabilities:   FAKE: {probs[0]*100:.2f}%, REAL: {probs[1]*100:.2f}%")
        print(f"Final Label:     {label} ({confidence}%)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeriTruth BERT Diagnostic Runner")
    parser.add_argument("texts", nargs="*", help="Sample texts to evaluate")
    args = parser.parse_args()

    default_samples = [
        "Pope Francis Shocks World, Endorses Donald Trump for President.",
        "Government confirms aliens landed in India and are living in secret base.",
        "U.S. Federal Reserve raises interest rates by 25 basis points amid inflation concerns.",
        "The European Central Bank kept benchmark interest rates unchanged at its meeting today.",
    ]
    samples = args.texts if args.texts else default_samples
    run_diagnostic(samples)
