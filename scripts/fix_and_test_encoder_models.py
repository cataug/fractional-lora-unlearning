from pathlib import Path
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

ROOT = Path.cwd()
MODEL_DIR = ROOT / "hf_models"
MODEL_DIR.mkdir(exist_ok=True)

MODELS = [
    {
        "local_name": "distilbert_base_uncased",
        "model_source": "distilbert-base-uncased",
        "tokenizer_source": "distilbert-base-uncased",
    },
    {
        "local_name": "bert_tiny",
        "model_source": "prajjwal1/bert-tiny",
        "tokenizer_source": "bert-base-uncased",
    },
    {
        "local_name": "bert_mini",
        "model_source": "prajjwal1/bert-mini",
        "tokenizer_source": "bert-base-uncased",
    },
    {
        "local_name": "electra_small_discriminator",
        "model_source": "google/electra-small-discriminator",
        "tokenizer_source": "google/electra-small-discriminator",
    },
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 100)
print("ENV")
print("=" * 100)
print("ROOT:", ROOT)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


def save_model_and_tokenizer(spec):
    local_dir = MODEL_DIR / spec["local_name"]
    local_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 100)
    print("FIX/SAVE:", spec["local_name"])
    print("model_source:", spec["model_source"])
    print("tokenizer_source:", spec["tokenizer_source"])
    print("local_dir:", local_dir)
    print("=" * 100)

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_source"],
        use_fast=False,   # important: robust slow tokenizer
    )

    print("Downloading base model...")
    model = AutoModel.from_pretrained(
        spec["model_source"],
    )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(local_dir)

    print("Saving model...")
    model.save_pretrained(local_dir, safe_serialization=True)

    meta = {
        "local_name": spec["local_name"],
        "model_source": spec["model_source"],
        "tokenizer_source": spec["tokenizer_source"],
    }

    with open(local_dir / "local_model_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=4, ensure_ascii=False)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Saved OK:", local_dir)


def test_local_model(spec):
    local_dir = MODEL_DIR / spec["local_name"]

    print("\n" + "=" * 100)
    print("TEST LOCAL:", local_dir)
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        local_dir,
        local_files_only=True,
        use_fast=False,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        local_dir,
        num_labels=4,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)

    texts = [
        "The stock market rose after the central bank announcement.",
        "The football team won the championship after a dramatic final.",
        "Scientists discovered a new method for training neural networks.",
    ]

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

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


for spec in MODELS:
    save_model_and_tokenizer(spec)

print("\n" + "#" * 100)
print("NOW TESTING ALL LOCAL MODELS")
print("#" * 100)

for spec in MODELS:
    test_local_model(spec)

print("\nALL MODELS FIXED AND TESTED")
print("Local models:")
for spec in MODELS:
    print(" ", MODEL_DIR / spec["local_name"])