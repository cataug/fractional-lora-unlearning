from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertConfig,
    BertModel,
    BertForSequenceClassification,
)


# ============================================================
# ROOT
# ============================================================

ROOT = Path.cwd()
MODEL_DIR = ROOT / "hf_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FORCE_REWRITE_LOCAL = True


# ============================================================
# MODELS
# ============================================================

MODELS: List[Dict[str, Any]] = [
    {
        "local_name": "distilbert_base_uncased",
        "model_source": "distilbert-base-uncased",
        "tokenizer_source": "distilbert-base-uncased",
        "loader": "auto",
    },
    {
        "local_name": "bert_tiny",
        "model_source": "prajjwal1/bert-tiny",
        "tokenizer_source": "bert-base-uncased",
        "loader": "bert_forced",
    },
    {
        "local_name": "bert_mini",
        "model_source": "prajjwal1/bert-mini",
        "tokenizer_source": "bert-base-uncased",
        "loader": "bert_forced",
    },
    {
        "local_name": "electra_small_discriminator",
        "model_source": "google/electra-small-discriminator",
        "tokenizer_source": "google/electra-small-discriminator",
        "loader": "auto",
    },
]


# ============================================================
# ENV
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 100)
print("ENV")
print("=" * 100)
print("ROOT:", ROOT)
print("python torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


# ============================================================
# HELPERS
# ============================================================

def dir_size_gb(path: Path) -> float:
    total = 0
    if not path.exists():
        return 0.0

    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass

    return total / 1024**3


def reset_local_dir(local_dir: Path) -> None:
    if FORCE_REWRITE_LOCAL and local_dir.exists():
        print("Removing old local dir:", local_dir)
        shutil.rmtree(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)


def save_auto_model(spec: Dict[str, Any], local_dir: Path) -> None:
    print("Tokenizer loader: AutoTokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        spec["tokenizer_source"],
        use_fast=False,
    )

    print("Model loader: AutoModel")
    model = AutoModel.from_pretrained(
        spec["model_source"],
    )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(local_dir)

    print("Saving model...")
    model.save_pretrained(local_dir, safe_serialization=True)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_forced_bert_model(spec: Dict[str, Any], local_dir: Path) -> None:
    """
    For old/small BERT checkpoints whose config does not contain model_type.
    """

    print("Tokenizer loader: BertTokenizer from bert-base-uncased")
    tokenizer = BertTokenizer.from_pretrained(
        spec["tokenizer_source"],
        do_lower_case=True,
    )

    print("Config loader: BertConfig forced")
    config = BertConfig.from_pretrained(
        spec["model_source"],
    )

    # Make sure model_type is explicitly present after saving.
    config.model_type = "bert"

    print("Model loader: BertModel forced")
    model = BertModel.from_pretrained(
        spec["model_source"],
        config=config,
    )

    print("Saving tokenizer...")
    tokenizer.save_pretrained(local_dir)

    print("Saving config...")
    config.save_pretrained(local_dir)

    print("Saving model...")
    model.save_pretrained(local_dir, safe_serialization=True)

    # Patch config.json explicitly just in case.
    config_path = local_dir / "config.json"
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    cfg["model_type"] = "bert"
    config_path.write_text(json.dumps(cfg, indent=4), encoding="utf-8")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def test_local_model(spec: Dict[str, Any], local_dir: Path) -> None:
    print("\nTesting local model:", local_dir)

    texts = [
        "The stock market rose after the central bank announcement.",
        "The football team won the championship after a dramatic final.",
        "Scientists discovered a new method for training neural networks.",
    ]

    if spec["loader"] == "bert_forced":
        tokenizer = BertTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
        )

        model = BertForSequenceClassification.from_pretrained(
            local_dir,
            num_labels=4,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        ).to(device)

    else:
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


def process_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    local_dir = MODEL_DIR / spec["local_name"]

    result = {
        "local_name": spec["local_name"],
        "model_source": spec["model_source"],
        "tokenizer_source": spec["tokenizer_source"],
        "loader": spec["loader"],
        "local_dir": str(local_dir),
        "save_ok": False,
        "test_ok": False,
        "size_gb": None,
        "error": None,
    }

    print("\n" + "=" * 100)
    print("PROCESS:", spec["local_name"])
    print("model_source:", spec["model_source"])
    print("tokenizer_source:", spec["tokenizer_source"])
    print("loader:", spec["loader"])
    print("local_dir:", local_dir)
    print("=" * 100)

    try:
        reset_local_dir(local_dir)

        if spec["loader"] == "bert_forced":
            save_forced_bert_model(spec, local_dir)
        else:
            save_auto_model(spec, local_dir)

        result["save_ok"] = True

        meta = {
            "local_name": spec["local_name"],
            "model_source": spec["model_source"],
            "tokenizer_source": spec["tokenizer_source"],
            "loader": spec["loader"],
        }

        with open(local_dir / "local_model_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=4, ensure_ascii=False)

        test_local_model(spec, local_dir)

        result["test_ok"] = True
        result["size_gb"] = dir_size_gb(local_dir)

        print("DONE:", spec["local_name"])
        print("size_gb:", result["size_gb"])

    except Exception as e:
        result["error"] = repr(e)
        print("FAILED:", spec["local_name"])
        print("ERROR:", repr(e))

    return result


# ============================================================
# MAIN
# ============================================================

results = []

for spec in MODELS:
    result = process_one(spec)
    results.append(result)

report_json = ROOT / "encoder_models_final_report.json"
report_csv = ROOT / "encoder_models_final_report.csv"

with open(report_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

try:
    import pandas as pd

    pd.DataFrame(results).to_csv(report_csv, index=False)
except Exception:
    pass

print("\n" + "=" * 100)
print("FINAL SUMMARY")
print("=" * 100)

for r in results:
    print(
        f"{r['local_name']:30s} | "
        f"save={r['save_ok']} | "
        f"test={r['test_ok']} | "
        f"size_gb={r['size_gb']} | "
        f"error={r['error']}"
    )

print("\nSaved:")
print(" ", report_json)
print(" ", report_csv)

print("\nLocal model paths:")
for spec in MODELS:
    print(" ", MODEL_DIR / spec["local_name"])