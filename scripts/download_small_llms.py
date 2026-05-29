from __future__ import annotations

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, List

import torch

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)


# ============================================================
# ROOTS
# ============================================================

ROOT = Path("/home/tahiti/Malashin_Projects")

HF_CACHE_DIR = ROOT / "hf_cache_llm"
MODEL_DIR = ROOT / "hf_llm_models"

HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ["HF_HUB_CACHE"] = str(HF_CACHE_DIR / "hub")
os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE_DIR / "transformers")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# CONFIG
# ============================================================

FORCE_REDOWNLOAD = False

MODELS: List[Dict[str, Any]] = [
    {
        "model_id": "distilgpt2",
        "local_name": "distilgpt2",
        "note": "very small GPT-2 style model, good for smoke tests",
        "trust_remote_code": False,
    },
    {
        "model_id": "gpt2",
        "local_name": "gpt2",
        "note": "classic GPT-2 small baseline",
        "trust_remote_code": False,
    },
    {
        "model_id": "EleutherAI/gpt-neo-125M",
        "local_name": "gpt_neo_125m",
        "note": "GPT-Neo 125M, useful small causal LM",
        "trust_remote_code": False,
    },
    {
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "local_name": "qwen2p5_0p5b_instruct",
        "note": "small instruction-tuned Qwen model",
        "trust_remote_code": True,
    },
]


# ============================================================
# ENV
# ============================================================

def print_env() -> None:
    print("=" * 100)
    print("ENVIRONMENT")
    print("=" * 100)
    print("ROOT:", ROOT)
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    print("HF_HOME:", os.environ["HF_HOME"])
    print("HF_HUB_CACHE:", os.environ["HF_HUB_CACHE"])
    print("TRANSFORMERS_CACHE:", os.environ["TRANSFORMERS_CACHE"])

    if ".venv_a100" not in sys.executable:
        print("\nWARNING: Python does not look like .venv_a100.")
        print("Expected:")
        print("  /home/tahiti/Malashin_Projects/.venv_a100/bin/python")
        print("Current:")
        print(" ", sys.executable)

    print("=" * 100)


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


def is_local_model_complete(local_dir: Path) -> bool:
    if not local_dir.exists():
        return False

    has_config = (local_dir / "config.json").exists()

    has_tokenizer = (
        (local_dir / "tokenizer.json").exists()
        or (local_dir / "vocab.json").exists()
        or (local_dir / "tokenizer.model").exists()
    )

    has_weights = bool(
        list(local_dir.glob("*.safetensors"))
        or list(local_dir.glob("*.bin"))
        or list(local_dir.glob("model-*.safetensors"))
        or list(local_dir.glob("pytorch_model-*.bin"))
    )

    return has_config and has_tokenizer and has_weights


def reset_if_needed(local_dir: Path) -> None:
    if FORCE_REDOWNLOAD and local_dir.exists():
        print("FORCE_REDOWNLOAD=True, removing:", local_dir)
        shutil.rmtree(local_dir)
        return

    if local_dir.exists() and not is_local_model_complete(local_dir):
        print("Incomplete local model folder detected, removing:", local_dir)
        shutil.rmtree(local_dir)


# ============================================================
# DOWNLOAD
# ============================================================

def download_one(spec: Dict[str, Any]) -> Dict[str, Any]:
    model_id = spec["model_id"]
    local_name = spec["local_name"]
    trust_remote_code = bool(spec.get("trust_remote_code", False))

    local_dir = MODEL_DIR / local_name

    result = {
        "model_id": model_id,
        "local_name": local_name,
        "local_dir": str(local_dir),
        "note": spec.get("note"),
        "download_ok": False,
        "offline_load_ok": False,
        "generate_ok": False,
        "size_gb": None,
        "num_parameters": None,
        "error": None,
    }

    print("\n" + "=" * 100)
    print("MODEL:", model_id)
    print("LOCAL:", local_dir)
    print("=" * 100)

    try:
        reset_if_needed(local_dir)

        if is_local_model_complete(local_dir):
            print("Already exists and looks complete, testing offline:", local_dir)
        else:
            local_dir.mkdir(parents=True, exist_ok=True)

            print("Downloading config...")
            config = AutoConfig.from_pretrained(
                model_id,
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=trust_remote_code,
            )

            print("Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=trust_remote_code,
                use_fast=True,
            )

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("Downloading model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                cache_dir=str(HF_CACHE_DIR),
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True,
            )

            print("Saving tokenizer...")
            tokenizer.save_pretrained(local_dir)

            print("Saving config...")
            config.save_pretrained(local_dir)

            print("Saving model...")
            model.save_pretrained(
                local_dir,
                safe_serialization=True,
                max_shard_size="2GB",
            )

            meta = {
                "model_id": model_id,
                "local_name": local_name,
                "trust_remote_code": trust_remote_code,
                "note": spec.get("note"),
            }

            with open(local_dir / "local_model_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=4, ensure_ascii=False)

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result["download_ok"] = True

        print("Testing offline load...")
        tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            local_dir,
            local_files_only=True,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )

        result["offline_load_ok"] = True

        num_params = sum(p.numel() for p in model.parameters())
        result["num_parameters"] = int(num_params)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        prompt = "The capital of France is"
        batch = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )
        batch = {k: v.to(device) for k, v in batch.items()}

        print("Testing generation...")
        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=16,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)

        print("Generated:")
        print(decoded)

        result["generate_ok"] = True
        result["size_gb"] = dir_size_gb(local_dir)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("OK:", model_id)
        print("size_gb:", result["size_gb"])
        print("params:", result["num_parameters"])

    except Exception as e:
        result["error"] = repr(e)
        print("FAILED:", model_id)
        print("ERROR:", repr(e))

    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_env()

    results = []

    for spec in MODELS:
        result = download_one(spec)
        results.append(result)

    report_json = ROOT / "downloaded_small_llms_report.json"
    report_csv = ROOT / "downloaded_small_llms_report.csv"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    try:
        import pandas as pd
        pd.DataFrame(results).to_csv(report_csv, index=False)
    except Exception as e:
        print("Could not save CSV:", repr(e))

    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    for r in results:
        print(
            f"{r['local_name']:25s} | "
            f"download={r['download_ok']} | "
            f"offline={r['offline_load_ok']} | "
            f"generate={r['generate_ok']} | "
            f"params={r['num_parameters']} | "
            f"size_gb={r['size_gb']} | "
            f"error={r['error']}"
        )

    print("\nSaved:")
    print(" ", report_json)
    print(" ", report_csv)

    print("\nLocal LLM paths:")
    for spec in MODELS:
        print(" ", MODEL_DIR / spec["local_name"])


if __name__ == "__main__":
    main()