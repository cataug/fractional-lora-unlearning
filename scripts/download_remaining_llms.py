from pathlib import Path
from huggingface_hub import snapshot_download
import os

# Папка, где уже лежат твои модели:
# ├── distilgpt2/
# ├── gpt2/
# ├── gpt_neo_125m/
# └── qwen2p5_0p5b_instruct/
BASE_DIR = Path("hf_models")
BASE_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    # modern Qwen scaling
    "Qwen/Qwen2.5-1.5B": "qwen2p5_1p5b",
    "Qwen/Qwen2.5-3B": "qwen2p5_3b",

    # compact modern LLM
    "HuggingFaceTB/SmolLM2-1.7B": "smollm2_1p7b",

    # Gemma family
    "google/gemma-2-2b": "gemma2_2b",
    "google/gemma-3-1b-it": "gemma3_1b_it",

    # Llama family, may require license acceptance + HF token
    "meta-llama/Llama-3.2-1B": "llama3p2_1b",
    "meta-llama/Llama-3.2-3B": "llama3p2_3b",

    # Phi family
    "microsoft/Phi-3.5-mini-instruct": "phi3p5_mini_instruct",
}

# Only files needed for PyTorch/Transformers loading.
# This avoids downloading random large extras when repos contain them.
ALLOW_PATTERNS = [
    "*.json",
    "*.txt",
    "*.model",
    "*.tiktoken",
    "*.py",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer.*",
    "vocab.*",
    "merges.txt",
    "special_tokens_map.json",
    "generation_config.json",
    "configuration*.py",
    "modeling*.py",
]

IGNORE_PATTERNS = [
    "*.onnx",
    "*.tflite",
    "*.gguf",
    "*.ggml",
    "*.msgpack",
    "*.h5",
    "*.ot",
    "flax_model*",
    "tf_model*",
    "rust_model*",
    "optimizer*",
    "scheduler*",
    "training_args.bin",
    "runs/*",
    "logs/*",
    "*.md",
]

print("=" * 100)
print("Downloading remaining LLM backbones")
print("Base directory:", BASE_DIR.resolve())
print("=" * 100)

failed = []

for repo_id, local_name in MODELS.items():
    local_dir = BASE_DIR / local_name

    print("\n" + "=" * 100)
    print(f"REPO:      {repo_id}")
    print(f"LOCAL DIR: {local_dir}")
    print("=" * 100)

    if local_dir.exists() and any(local_dir.iterdir()):
        print(f"[SKIP] Directory already exists and is not empty: {local_dir}")
        continue

    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=ALLOW_PATTERNS,
            ignore_patterns=IGNORE_PATTERNS,
        )

        print(f"[OK] Downloaded: {repo_id}")
        print(f"[OK] Saved to: {local_dir}")

    except Exception as e:
        print(f"[FAILED] {repo_id}")
        print(repr(e))
        failed.append((repo_id, repr(e)))

print("\n" + "=" * 100)
print("DONE")
print("=" * 100)

if failed:
    print("\nFAILED MODELS:")
    for repo_id, err in failed:
        print("-" * 100)
        print(repo_id)
        print(err)
else:
    print("All requested models downloaded successfully.")

print("\nDisk usage command:")
print(f"du -sh {BASE_DIR}")
print(f"du -h --max-depth=1 {BASE_DIR} | sort -h")