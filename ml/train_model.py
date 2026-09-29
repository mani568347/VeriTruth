"""Legacy training entry point delegated to the BERT training pipeline."""

from ml.bert_training import DEFAULT_OUTPUT_DIR, TrainingConfig, train


if __name__ == "__main__":
    metadata = train(TrainingConfig(), DEFAULT_OUTPUT_DIR)
    print(metadata["test_metrics"])