from __future__ import annotations

import time
import math
from pathlib import Path
from typing import Optional, Dict

import torch
import torch.nn as nn

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification,
)


# ============================================================
# CONFIG
# ============================================================

ROOT = Path.cwd()
MODEL_DIR = ROOT / "hf_models"

MODELS = [
    {
        "name": "distilbert_base_uncased",
        "path": MODEL_DIR / "distilbert_base_uncased",
        "loader": "auto",
    },
    {
        "name": "bert_tiny",
        "path": MODEL_DIR / "bert_tiny",
        "loader": "bert_forced",
    },
    {
        "name": "bert_mini",
        "path": MODEL_DIR / "bert_mini",
        "loader": "bert_forced",
    },
    {
        "name": "electra_small_discriminator",
        "path": MODEL_DIR / "electra_small_discriminator",
        "loader": "auto",
    },
]

NUM_LABELS = 4
BATCH_SIZE = 16
MAX_LEN = 128
TRAIN_STEPS = 5
USE_AMP = True

TEXTS = [
    "The government announced a new economic policy today.",
    "The football team won after a dramatic penalty shootout.",
    "Scientists developed a new neural network training method.",
    "The company released a new smartphone with better battery life.",
    "Markets moved higher after the central bank decision.",
    "The tennis player won the final in straight sets.",
    "Researchers found evidence of climate change effects.",
    "A new software update improves security and performance.",
] * 8

LABELS = [0, 1, 2, 3, 0, 1, 2, 3] * 8


# ============================================================
# FRACTIONAL CONTROLLER
# ============================================================

class FractionalGradientController:
    def __init__(
        self,
        model: nn.Module,
        mode: str = "mix",
        target: str = "head",
        alpha: Optional[float] = 0.80,
        beta: Optional[float] = 0.90,
        mix_lambda: float = 0.01,
    ):
        self.model = model
        self.mode = mode
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.mix_lambda = mix_lambda
        self.enabled = mode in {"replace", "mix"}

        if not self.enabled:
            self.coeff = None
            self.memory = {}
            return

        self.coeff = 1.0 / math.gamma(2.0 - alpha)
        self.memory: Dict[str, torch.Tensor] = {}

    def _matches_target(self, name: str) -> bool:
        if self.target == "all":
            return True

        # HuggingFace classification heads usually contain classifier.
        if self.target == "head":
            return (
                "classifier" in name
                or "pre_classifier" in name
                or "score" in name
            )

        if self.target == "embeddings":
            return "embeddings" in name or "embed" in name

        return False

    def apply(self) -> int:
        if not self.enabled:
            return 0

        applied = 0

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue

                if not self._matches_target(name):
                    continue

                g = p.grad.detach()

                if name not in self.memory:
                    self.memory[name] = torch.zeros_like(g)

                mem = self.memory[name]
                mem.mul_(self.beta)
                mem.add_(g, alpha=(1.0 - self.beta) * self.coeff)

                if self.mode == "replace":
                    p.grad.copy_(mem)
                elif self.mode == "mix":
                    p.grad.copy_((1.0 - self.mix_lambda) * g + self.mix_lambda * mem)

                applied += 1

        return applied


# ============================================================
# HELPERS
# ============================================================

def gpu_mem(prefix: str = ""):
    if not torch.cuda.is_available():
        return

    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"{prefix} GPU memory allocated={alloc:.3f} GB | reserved={reserved:.3f} GB")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model(spec, device):
    path = spec["path"]

    if spec["loader"] == "bert_forced":
        tokenizer = BertTokenizer.from_pretrained(
            path,
            local_files_only=True,
        )

        model = BertForSequenceClassification.from_pretrained(
            path,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

    else:
        tokenizer = AutoTokenizer.from_pretrained(
            path,
            local_files_only=True,
            use_fast=False,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            path,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

    model.to(device)
    return tokenizer, model


def make_batch(tokenizer, device):
    texts = TEXTS[:BATCH_SIZE]
    labels = torch.tensor(LABELS[:BATCH_SIZE], dtype=torch.long, device=device)

    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    batch = {k: v.to(device) for k, v in batch.items()}
    batch["labels"] = labels

    return batch


def run_one_model(spec, device):
    print("\n" + "=" * 100)
    print("SMOKE TEST:", spec["name"])
    print("PATH:", spec["path"])
    print("=" * 100)

    if not spec["path"].exists():
        print("FAILED: local path does not exist")
        return {
            "model": spec["name"],
            "ok": False,
            "error": "path_missing",
        }

    try:
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gpu_mem("Before load:")

        t0 = time.time()
        tokenizer, model = load_model(spec, device)
        load_time = time.time() - t0

        print("Loaded OK")
        print("Load time:", round(load_time, 3), "sec")
        print("Total params:", f"{count_params(model):,}")
        print("Trainable params:", f"{count_trainable_params(model):,}")
        gpu_mem("After load:")

        batch = make_batch(tokenizer, device)

        # -----------------------------
        # Forward only
        # -----------------------------
        model.eval()

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.time()

        with torch.no_grad():
            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                out = model(**batch)

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        forward_time = time.time() - t1

        print("Forward OK")
        print("Loss:", float(out.loss.detach().cpu()))
        print("Logits shape:", tuple(out.logits.shape))
        print("Forward time:", round(forward_time, 4), "sec")
        gpu_mem("After forward:")

        # -----------------------------
        # Training steps, baseline AdamW
        # -----------------------------
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)
        scaler = torch.amp.GradScaler(
            "cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        )

        losses = []

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t2 = time.time()

        for step in range(TRAIN_STEPS):
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                out = model(**batch)
                loss = out.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.detach().cpu()))
            print(f"AdamW step {step + 1}/{TRAIN_STEPS} | loss={losses[-1]:.4f}")

        torch.cuda.synchronize() if torch.cuda.is_available() else None
        train_time = time.time() - t2

        print("AdamW training OK")
        print("Train time:", round(train_time, 3), "sec")
        gpu_mem("After AdamW train:")

        # -----------------------------
        # Fractional head-mix smoke
        # -----------------------------
        frac = FractionalGradientController(
            model=model,
            mode="mix",
            target="head",
            alpha=0.80,
            beta=0.90,
            mix_lambda=0.01,
        )

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=1e-2)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            out = model(**batch)
            loss = out.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        applied = frac.apply()

        scaler.step(optimizer)
        scaler.update()

        print("Fractional head-mix step OK")
        print("Fractional applied params:", applied)
        print("Fractional loss:", float(loss.detach().cpu()))
        gpu_mem("After fractional step:")

        result = {
            "model": spec["name"],
            "ok": True,
            "load_time_sec": load_time,
            "forward_time_sec": forward_time,
            "train_time_sec": train_time,
            "params": count_params(model),
            "trainable_params": count_trainable_params(model),
            "last_adamw_loss": losses[-1],
            "fractional_applied_params": applied,
            "fractional_loss": float(loss.detach().cpu()),
        }

        del model, tokenizer, optimizer, batch, out, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return result

    except Exception as e:
        print("FAILED:", repr(e))
        return {
            "model": spec["name"],
            "ok": False,
            "error": repr(e),
        }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 100)
    print("REAL PRETRAINED MODELS SMOKE TEST")
    print("=" * 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("ROOT:", ROOT)
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda:", torch.cuda.is_available())
    print("device:", device)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    results = []

    for spec in MODELS:
        res = run_one_model(spec, device)
        results.append(res)

    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for r in results:
        print(r)

    try:
        import pandas as pd

        out_csv = ROOT / "smoke_test_real_models_report.csv"
        pd.DataFrame(results).to_csv(out_csv, index=False)
        print("\nSaved:", out_csv)
    except Exception as e:
        print("Could not save CSV:", repr(e))


if __name__ == "__main__":
    main()