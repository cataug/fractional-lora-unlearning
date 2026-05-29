from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path.cwd()
MODEL_DIR = ROOT / "hf_models"

MODELS = [
    MODEL_DIR / "distilbert_base_uncased",
    MODEL_DIR / "bert_tiny",
    MODEL_DIR / "bert_mini",
    MODEL_DIR / "electra_small_discriminator",
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("python model test")
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))

texts = [
    "The stock market rose after the central bank announcement.",
    "The football team won the championship after a dramatic final.",
    "Scientists discovered a new method for training neural networks.",
]

for model_path in MODELS:
    print("\n" + "=" * 100)
    print("Testing:", model_path)
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        use_fast=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=4,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    batch = {k: v.to(device) for k, v in batch.items()}

    model.eval()

    with torch.no_grad():
        out = model(**batch)

    print("logits shape:", tuple(out.logits.shape))
    print("OK")
