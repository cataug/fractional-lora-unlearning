from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path("/home/user/fractional_unlearning")
OUT = ROOT / "hf_models"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = [
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_1p1b_chat"),
    ("Qwen/Qwen2.5-0.5B-Instruct", "qwen2p5_0p5b_instruct"),
    ("Qwen/Qwen2.5-1.5B-Instruct", "qwen2p5_1p5b_instruct"),
    ("EleutherAI/pythia-410m", "pythia_410m"),
    ("EleutherAI/pythia-1b", "pythia_1b"),
    ("facebook/opt-350m", "opt_350m"),
    ("facebook/opt-1.3b", "opt_1p3b"),
]

for repo_id, local_name in MODELS:
    local_dir = OUT / local_name
    print("=" * 100)
    print("Downloading:", repo_id)
    print("To:", local_dir)

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("OK:", repo_id)

    except Exception as e:
        print("FAILED:", repo_id)
        print(repr(e))

print("=" * 100)
print("DONE")
print("Models folder:", OUT)