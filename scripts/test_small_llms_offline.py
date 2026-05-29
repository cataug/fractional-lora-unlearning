from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


ROOT = Path("/home/tahiti/Malashin_Projects")
MODEL_DIR = ROOT / "hf_llm_models"

MODELS = [
    {
        "name": "distilgpt2",
        "path": MODEL_DIR / "distilgpt2",
        "trust_remote_code": False,
    },
    {
        "name": "gpt2",
        "path": MODEL_DIR / "gpt2",
        "trust_remote_code": False,
    },
    {
        "name": "gpt_neo_125m",
        "path": MODEL_DIR / "gpt_neo_125m",
        "trust_remote_code": False,
    },
    {
        "name": "qwen2p5_0p5b_instruct",
        "path": MODEL_DIR / "qwen2p5_0p5b_instruct",
        "trust_remote_code": True,
    },
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 100)
print("OFFLINE SMALL LLM TEST")
print("=" * 100)
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))


for spec in MODELS:
    print("\n" + "=" * 100)
    print("MODEL:", spec["name"])
    print("PATH:", spec["path"])
    print("=" * 100)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            spec["path"],
            local_files_only=True,
            trust_remote_code=spec["trust_remote_code"],
            use_fast=True,
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            spec["path"],
            local_files_only=True,
            trust_remote_code=spec["trust_remote_code"],
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)

        model.eval()

        prompt = "Machine learning is useful because"
        batch = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64,
        )

        batch = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            out = model.generate(
                **batch,
                max_new_tokens=32,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        text = tokenizer.decode(out[0], skip_special_tokens=True)

        print("OK")
        print(text)

        del model, tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print("FAILED:", repr(e))