from __future__ import annotations

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)


# ============================================================
# ROOTS
# ============================================================

ROOT = Path.cwd()

HF_CACHE_DIR = ROOT / "hf_cache"
MODEL_DIR = ROOT / "hf_models"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Force HuggingFace cache into project folder.
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")

# Optional: reduce tokenizer warning noise.
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# MODELS
# ============================================================
# Small pretrained encoder models for text classification.
# These are not GPT/decoder LLMs.

MODELS: List[Dict[str, Any]] = [
    {
        "model_id": "distilbert-base-uncased",
        "local_name": "distilbert_base_uncased",
        "family": "DistilBERT",
        "note": "strong small baseline encoder",
    },
    {
        "model_id": "prajjwal1/bert-tiny",
        "local_name": "bert_tiny",
        "family": "BERT",
        "note": "very small and fast encoder",
    },
    {
        "model_id": "prajjwal1/bert-mini",
        "local_name": "bert_mini",
        "family": "BERT",
        "note": "small encoder, stronger than bert-tiny",
    },
    {
        "model_id": "google/electra-small-discriminator",
        "local_name": "electra_small_discriminator",
        "family": "ELECTRA",
        "note": "small discriminator encoder",
    },
]


# ============================================================
# UTILS
# ============================================================

def print_env() -> None:
    print("=" * 100)
    print("ENVIRONMENT")
    print("=" * 100)
    print("cwd:", ROOT)
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))

    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))

    if ".venv_a100" not in sys.executable:
        print("\nWARNING: python executable does not look like .venv_a100.")
        print("Expected something like:")
        print("  /home/tahiti/Malashin_Projects/.venv_a100/bin/python")
        print("Current:")
        print(" ", sys.executable)

    print("=" * 100)


def safe_size_gb(path: Path) -> float:
    if not path.exists():
        return 0.0

    total = 0

    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass

    return total / 1024**3


def remove_if_broken(local_dir: Path) -> None:
    """
    If a previous interrupted download left a partial local model folder,
    remove it only if important files are missing.
    """

    if not local_dir.exists():
        return

    important = [
        local_dir / "config.json",
    ]

    has_config = any(p.exists() for p in important)

    model_files = list(local_dir.glob("*.bin")) + list(local_dir.glob("*.safetensors"))
    tokenizer_files = (
        list(local_dir.glob("tokenizer*"))
        + list(local_dir.glob("vocab*"))
        + list(local_dir.glob("special_tokens_map.json"))
    )

    if has_config and model_files and tokenizer_files:
        return

    print(f"Partial/broken local folder detected, removing: {local_dir}")
    shutil.rmtree(local_dir)


def download_one(spec: Dict[str, Any], force_redownload: bool = False) -> Dict[str, Any]:
    model_id = spec["model_id"]
    local_name = spec["local_name"]
    local_dir = MODEL_DIR / local_name

    print("\n" + "#" * 100)
    print("Downloading model:", model_id)
    print("Local dir:", local_dir)
    print("#" * 100)

    if force_redownload and local_dir.exists():
        print("Force redownload enabled, removing:", local_dir)
        shutil.rmtree(local_dir)

    remove_if_broken(local_dir)

    local_dir.mkdir(parents=True, exist_ok=True)

    status = {
        "model_id": model_id,
        "local_name": local_name,
        "local_dir": str(local_dir),
        "family": spec.get("family"),
        "note": spec.get("note"),
        "download_ok": False,
        "base_load_ok": False,
        "classifier_load_ok": False,
        "offline_load_ok": False,
        "size_gb": None,
        "error": None,
    }

    try:
        print("Loading config...")
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=str(HF_CACHE_DIR),
        )

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=str(HF_CACHE_DIR),
            use_fast=True,
        )

        print("Loading base AutoModel...")
        model = AutoModel.from_pretrained(
            model_id,
            cache_dir=str(HF_CACHE_DIR),
        )

        status["base_load_ok"] = True

        print("Saving config/tokenizer/base model locally...")
        config.save_pretrained(local_dir)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir, safe_serialization=True)

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        status["download_ok"] = True

        print("Testing classification-head creation from local model...")
        clf = AutoModelForSequenceClassification.from_pretrained(
            local_dir,
            num_labels=4,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

        status["classifier_load_ok"] = True

        del clf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Testing strict offline reload...")
        tok2 = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
            use_fast=True,
        )
        model2 = AutoModel.from_pretrained(
            local_dir,
            local_files_only=True,
        )

        status["offline_load_ok"] = True

        del tok2, model2
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        size_gb = safe_size_gb(local_dir)
        status["size_gb"] = size_gb

        print("OK:", model_id)
        print(f"Local size: {size_gb:.3f} GB")

    except Exception as e:
        status["error"] = repr(e)
        print("FAILED:", model_id)
        print("ERROR:", repr(e))

    return status


def main() -> None:
    print_env()

    results = []

    # Change to True only if you want to forcibly redownload all models.
    FORCE_REDOWNLOAD = False

    for spec in MODELS:
        result = download_one(spec, force_redownload=FORCE_REDOWNLOAD)
        results.append(result)

    out_json = ROOT / "downloaded_encoder_models_report.json"
    out_csv = ROOT / "downloaded_encoder_models_report.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    try:
        import pandas as pd

        pd.DataFrame(results).to_csv(out_csv, index=False)
    except Exception:
        pass

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for r in results:
        print(
            f"{r['model_id']:40s} | "
            f"download={r['download_ok']} | "
            f"base={r['base_load_ok']} | "
            f"classifier={r['classifier_load_ok']} | "
            f"offline={r['offline_load_ok']} | "
            f"size_gb={r['size_gb']}"
        )

        if r["error"]:
            print("  ERROR:", r["error"])

    print("\nSaved:")
    print(" ", out_json)
    print(" ", out_csv if out_csv.exists() else "CSV not created")
    print("\nModels saved in:")
    print(" ", MODEL_DIR)

    print("\nUse local paths like:")
    for spec in MODELS:
        print(" ", MODEL_DIR / spec["local_name"])


if __name__ == "__main__":
    main()


