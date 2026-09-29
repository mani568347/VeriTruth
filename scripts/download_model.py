"""Fetch the fine-tuned VeriTruth BERT checkpoint from the Hugging Face Hub.

Safe to run repeatedly: when the local checkpoint is already complete nothing
is downloaded. Model weights are too large for Git, so this is the supported way
to set up a fresh clone.

Usage:
    python scripts/download_model.py
    python scripts/download_model.py <org-or-user>/<repo-name>

For a private repository, export HF_TOKEN first (never hardcode it):
    set HF_TOKEN=...        # Windows
    export HF_TOKEN=...     # macOS/Linux
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

# Only the files the inference service actually needs.
ALLOW_PATTERNS = (
    "config.json",
    "model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "added_tokens.json",
    "merges.txt",
)


def missing_files(checkpoint_dir: Path, required: tuple[str, ...]) -> list[str]:
    return [name for name in required if not (checkpoint_dir / name).is_file()]


def main() -> int:
    from ml.bert_service import MODEL_DIR, MODEL_SETUP_MESSAGE, REQUIRED_CHECKPOINT_FILES

    repo_id = (sys.argv[1] if len(sys.argv) > 1 else os.getenv("BERT_MODEL_ID") or "").strip()

    print(f"Checkpoint directory: {MODEL_DIR}")

    missing = missing_files(MODEL_DIR, REQUIRED_CHECKPOINT_FILES)
    if not missing:
        print("Model already available — nothing to download.")
        return 0

    if not repo_id or "YOUR_HUGGINGFACE_USERNAME" in repo_id:
        print(
            "Model is incomplete (missing: "
            + ", ".join(missing)
            + ") and no Hugging Face repository is configured.\n"
            "Set BERT_MODEL_ID in .env to the repository that holds the checkpoint,\n"
            "or copy the checkpoint files into the directory above."
        )
        return 1

    if os.getenv("HF_TOKEN"):
        print("HF_TOKEN: present (used only if the repository is private).")

    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} …")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(MODEL_DIR),
            allow_patterns=list(ALLOW_PATTERNS),
        )
    except Exception as exc:  # network, auth, or missing repository
        print(f"{MODEL_SETUP_MESSAGE} Could not download '{repo_id}': {exc}")
        print(
            "Check that the repository exists and is public, that BERT_MODEL_ID is "
            "correct, and that HF_TOKEN is set if it is private."
        )
        return 1

    missing = missing_files(MODEL_DIR, REQUIRED_CHECKPOINT_FILES)
    if missing:
        print(f"Downloaded, but the checkpoint is still missing: {', '.join(missing)}")
        return 1

    print("Download complete. Required checkpoint files:")
    for name in REQUIRED_CHECKPOINT_FILES:
        size = (MODEL_DIR / name).stat().st_size
        print(f"  {name}  ({size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
