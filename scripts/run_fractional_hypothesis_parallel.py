from __future__ import annotations

import csv
import json
import time
import math
import random
import shutil
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# ROOTS
# ============================================================

ROOT = Path.cwd()

PT_DIR = ROOT / "generated_pt"
RUN_DIR = ROOT / "runs_fractional_hypothesis_v2"

SUMMARY_CSV = ROOT / "summary_fractional_hypothesis_v2_parallel.csv"
SUMMARY_JSON = ROOT / "summary_fractional_hypothesis_v2_parallel.json"

RUN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL CONFIG
# ============================================================

SEED = 42

RUNS = 3
EPOCHS = 10
VAL_SPLIT = 0.2

BATCH_SIZE = 192
NUM_WORKERS = 1

USE_AMP = True

# A100 40GB: start with 4. If stable and memory is free, try 5.
MAX_PARALLEL_JOBS = 4
GPU_IDS = [0]

RESUME = True
RERUN_INCOMPLETE = True

EXPECTED_EPOCHS = EPOCHS


# ============================================================
# DATASET CONFIGS
# ============================================================

DATASET_CONFIGS = [
    {
        "dataset_name": "small_D2_L64",
        "n_samples": 5000,
        "depth": 2,
        "max_len": 64,
        "seed_offset": 0,
    },
    {
        "dataset_name": "base_D3_L128",
        "n_samples": 10000,
        "depth": 3,
        "max_len": 128,
        "seed_offset": 1000,
    },
    {
        "dataset_name": "hard_D4_L160",
        "n_samples": 12000,
        "depth": 4,
        "max_len": 160,
        "seed_offset": 2000,
    },
]


# ============================================================
# MODEL CONFIGS
# ============================================================

MODEL_CONFIGS = [
    {
        "model_name": "T_small_mean",
        "d_model": 96,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 256,
        "dropout": 0.10,
        "pooling": "mean",
    },
    {
        "model_name": "T_base_mean",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.10,
        "pooling": "mean",
    },
    {
        "model_name": "T_base_cls",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 3,
        "dim_feedforward": 512,
        "dropout": 0.10,
        "pooling": "cls",
    },
    {
        "model_name": "T_deep_mean",
        "d_model": 128,
        "nhead": 4,
        "num_layers": 5,
        "dim_feedforward": 512,
        "dropout": 0.15,
        "pooling": "mean",
    },
    {
        "model_name": "T_wide_mean",
        "d_model": 192,
        "nhead": 6,
        "num_layers": 3,
        "dim_feedforward": 768,
        "dropout": 0.10,
        "pooling": "mean",
    },
]


# ============================================================
# NEW HYPOTHESIS SCENARIOS
# ============================================================
# Hypothesis:
#   Full replacement hurts optimization.
#   Selective or weak mixed fractional memory can help on harder compositional sequences.
#
# Main directions:
#   1. embeddings-only with alpha sweep
#   2. embeddings-only mixed with small lambda
#   3. all-parameter weak mix with lambda sweep
#   4. delayed weak mix
#   5. head-only weak mix / replace as comparison
#   6. hard negative controls kept but reduced


SCENARIO_CONFIGS = [
    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------
    {
        "scenario_name": "baseline_adamw",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "none",
        "target": "none",
        "alpha": None,
        "beta": None,
        "mix_lambda": 0.0,
        "start_epoch": None,
    },

    # --------------------------------------------------------
    # Negative controls: full replacement, expected to underperform.
    # Kept for interpretability, not expanded.
    # --------------------------------------------------------
    {
        "scenario_name": "negative_full_replace_a08",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "replace",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 1.0,
        "start_epoch": 1,
    },

    # --------------------------------------------------------
    # Embeddings-only replacement: strongest previous signal.
    # --------------------------------------------------------
    {
        "scenario_name": "emb_replace_a07",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "replace",
        "target": "embeddings",
        "alpha": 0.70,
        "beta": 0.90,
        "mix_lambda": 1.0,
        "start_epoch": 1,
    },
    {
        "scenario_name": "emb_replace_a08",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "replace",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 1.0,
        "start_epoch": 1,
    },
    {
        "scenario_name": "emb_replace_a09",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "replace",
        "target": "embeddings",
        "alpha": 0.90,
        "beta": 0.90,
        "mix_lambda": 1.0,
        "start_epoch": 1,
    },

    # --------------------------------------------------------
    # Embeddings-only weak mixed memory.
    # --------------------------------------------------------
    {
        "scenario_name": "emb_mix_a08_lam005",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.05,
        "start_epoch": 1,
    },
    {
        "scenario_name": "emb_mix_a08_lam010",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.10,
        "start_epoch": 1,
    },
    {
        "scenario_name": "emb_mix_a08_lam015",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.15,
        "start_epoch": 1,
    },
    {
        "scenario_name": "emb_mix_a08_lam025",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.25,
        "start_epoch": 1,
    },

    # --------------------------------------------------------
    # All-parameter weak mixed memory.
    # Previous lam=0.25 was close; now test smaller lambda.
    # --------------------------------------------------------
    {
        "scenario_name": "all_mix_a08_lam005",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.05,
        "start_epoch": 1,
    },
    {
        "scenario_name": "all_mix_a08_lam010",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.10,
        "start_epoch": 1,
    },
    {
        "scenario_name": "all_mix_a08_lam015",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.15,
        "start_epoch": 1,
    },
    {
        "scenario_name": "all_mix_a08_lam025",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.25,
        "start_epoch": 1,
    },

    # --------------------------------------------------------
    # Delayed weak all-mix.
    # --------------------------------------------------------
    {
        "scenario_name": "delayed_all_mix_a08_lam010_warm3",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.10,
        "start_epoch": 4,
    },
    {
        "scenario_name": "delayed_all_mix_a08_lam015_warm3",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.15,
        "start_epoch": 4,
    },
    {
        "scenario_name": "delayed_all_mix_a08_lam025_warm3",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "all",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.25,
        "start_epoch": 4,
    },

    # --------------------------------------------------------
    # Delayed embeddings mix.
    # --------------------------------------------------------
    {
        "scenario_name": "delayed_emb_mix_a08_lam010_warm3",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.10,
        "start_epoch": 4,
    },
    {
        "scenario_name": "delayed_emb_mix_a08_lam025_warm3",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "embeddings",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.25,
        "start_epoch": 4,
    },

    # --------------------------------------------------------
    # Head selective controls.
    # --------------------------------------------------------
    {
        "scenario_name": "head_replace_a08",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "replace",
        "target": "head",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 1.0,
        "start_epoch": 1,
    },
    {
        "scenario_name": "head_mix_a08_lam010",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "head",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.10,
        "start_epoch": 1,
    },
    {
        "scenario_name": "delayed_head_mix_a08_lam010_warm3",
        "base_optimizer": "adamw",
        "lr": 3e-4,
        "weight_decay": 1e-2,
        "mode": "mix",
        "target": "head",
        "alpha": 0.80,
        "beta": 0.90,
        "mix_lambda": 0.10,
        "start_epoch": 4,
    },
]


# ============================================================
# VOCABULARY
# ============================================================

VOCAB = ["[PAD]", "[CLS]", "[", "]", "MAX", "MIN", "SUM", "PROD", "MED"] + [
    str(i) for i in range(10)
]

VOCAB_DICT = {tok: i for i, tok in enumerate(VOCAB)}
PAD_ID = VOCAB_DICT["[PAD]"]


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def safe_name(x: Any) -> str:
    return (
        str(x)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(".", "p")
        .replace("'", "")
        .replace('"', "")
    )


def log(msg: str, path: Path) -> None:
    print(msg, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_pt_path(cfg: Dict[str, Any]) -> Path:
    dataset_seed = SEED + cfg["seed_offset"]

    return PT_DIR / (
        f"listops_{safe_name(cfg['dataset_name'])}"
        f"_N{cfg['n_samples']}_D{cfg['depth']}_L{cfg['max_len']}_seed{dataset_seed}.pt"
    )


def find_dataset_pt(cfg: Dict[str, Any]) -> Path:
    expected = expected_pt_path(cfg)

    if expected.exists():
        return expected

    pattern = f"listops_{safe_name(cfg['dataset_name'])}_*.pt"
    matches = sorted(PT_DIR.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No .pt file found for dataset {cfg['dataset_name']} in {PT_DIR}. "
            f"Expected: {expected}"
        )

    print(f"Expected file not found, using matched file: {matches[0]}", flush=True)
    return matches[0]


def get_exp_dir(
    dataset_name: str,
    model_name: str,
    scenario_name: str,
    run_id: int,
) -> Path:
    return (
        RUN_DIR
        / safe_name(dataset_name)
        / safe_name(model_name)
        / safe_name(scenario_name)
        / f"run_{run_id}"
    )


def is_run_complete(exp_dir: Path, expected_epochs: int) -> bool:
    history_path = exp_dir / "history.json"
    best_model_path = exp_dir / "best_model.pt"
    final_model_path = exp_dir / "final_model.pt"

    if not history_path.exists():
        return False
    if not best_model_path.exists():
        return False
    if not final_model_path.exists():
        return False

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        vals = history.get("val_macro_f1", None)

        if not isinstance(vals, list):
            return False

        if len(vals) < expected_epochs:
            return False

    except Exception:
        return False

    return True


def remove_incomplete_run(exp_dir: Path) -> None:
    if exp_dir.exists():
        print(f"Removing incomplete run: {exp_dir}", flush=True)
        shutil.rmtree(exp_dir)


def save_summary(results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    fieldnames = sorted(set().union(*(r.keys() for r in results)))

    with open(SUMMARY_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in results:
            writer.writerow(row)


# ============================================================
# DATASET
# ============================================================

class GeneratedListOpsDataset(Dataset):
    def __init__(self, pt_path: Path):
        data = torch.load(pt_path, map_location="cpu")

        self.X = data["X"].long()
        self.Y = data["Y"].long()

        if self.Y.ndim != 1:
            self.Y = self.Y.view(-1)

        self.dataset_config = data.get("dataset_config", {})
        self.lengths = data.get("lengths", None)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.Y[idx]


def dataset_info_from_pt(pt_path: Path) -> Dict[str, Any]:
    data = torch.load(pt_path, map_location="cpu")

    X = data["X"]
    Y = data["Y"]

    info = {
        "pt_path": str(pt_path),
        "dataset_name": data["dataset_config"]["dataset_name"],
        "n_samples": int(X.shape[0]),
        "seq_len": int(X.shape[1]),
        "vocab_size": int(X.max().item()) + 1,
        "num_classes": int(Y.max().item()) + 1,
        "unique_classes": sorted([int(v) for v in Y.unique().tolist()]),
        "dataset_config": data.get("dataset_config", {}),
        "seed": data.get("seed", None),
    }

    if "lengths" in data:
        lengths = data["lengths"]
        info["mean_length"] = float(lengths.float().mean().item())
        info["min_length"] = int(lengths.min().item())
        info["max_length"] = int(lengths.max().item())

    return info


# ============================================================
# MODEL
# ============================================================

class GeneratedTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str,
        pad_idx: int = 0,
    ):
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError(f"d_model={d_model} must be divisible by nhead={nhead}")

        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")

        self.pad_idx = pad_idx
        self.pooling = pooling

        self.token_embed = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=pad_idx,
        )

        self.pos_embed = nn.Embedding(
            num_embeddings=seq_len,
            embedding_dim=d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = x.shape

        pos = torch.arange(seq_len, device=x.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)

        pad_mask = x.eq(self.pad_idx)

        h = self.token_embed(x) + self.pos_embed(pos)
        h = self.encoder(h, src_key_padding_mask=pad_mask)

        if self.pooling == "cls":
            pooled = h[:, 0, :]
        else:
            mask = (~pad_mask).float().unsqueeze(-1)
            h = h * mask
            denom = mask.sum(dim=1).clamp(min=1.0)
            pooled = h.sum(dim=1) / denom

        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        return self.classifier(pooled)


# ============================================================
# FRACTIONAL CONTROLLER
# ============================================================

class FractionalGradientController:
    def __init__(
        self,
        model: nn.Module,
        mode: str,
        target: str,
        alpha: Optional[float],
        beta: Optional[float],
        mix_lambda: float = 1.0,
        start_epoch: Optional[int] = 1,
    ):
        self.model = model
        self.mode = mode
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.mix_lambda = mix_lambda
        self.start_epoch = start_epoch

        self.enabled = mode in {"replace", "mix"}

        if not self.enabled:
            self.coeff = None
            self.memory = {}
            return

        if alpha is None or beta is None:
            raise ValueError("alpha and beta must be provided for fractional modes")

        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1]")

        if not (0.0 <= beta < 1.0):
            raise ValueError("beta must be in [0, 1)")

        if not (0.0 <= mix_lambda <= 1.0):
            raise ValueError("mix_lambda must be in [0, 1]")

        if mode not in {"replace", "mix"}:
            raise ValueError("mode must be one of: none, replace, mix")

        self.coeff = 1.0 / math.gamma(2.0 - alpha)
        self.memory: Dict[str, torch.Tensor] = {}

    def _matches_target(self, name: str) -> bool:
        if self.target == "all":
            return True

        if self.target == "head":
            return name.startswith("classifier")

        if self.target == "embeddings":
            return name.startswith("token_embed") or name.startswith("pos_embed")

        if self.target == "attention":
            return "self_attn" in name

        if self.target == "ffn":
            return "linear1" in name or "linear2" in name

        if self.target == "none":
            return False

        raise ValueError(f"Unknown target: {self.target}")

    def apply(self, current_epoch: int) -> int:
        if not self.enabled:
            return 0

        if self.start_epoch is not None and current_epoch < self.start_epoch:
            return 0

        applied = 0

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if p.grad is None:
                    continue

                if not self._matches_target(name):
                    continue

                g_original = p.grad.detach()

                if name not in self.memory:
                    self.memory[name] = torch.zeros_like(g_original)

                mem = self.memory[name]
                mem.mul_(self.beta)
                mem.add_(g_original, alpha=(1.0 - self.beta) * self.coeff)

                if self.mode == "replace":
                    p.grad.copy_(mem)

                elif self.mode == "mix":
                    mixed = (1.0 - self.mix_lambda) * g_original + self.mix_lambda * mem
                    p.grad.copy_(mixed)

                else:
                    raise ValueError(f"Unknown mode: {self.mode}")

                applied += 1

        return applied


# ============================================================
# OPTIMIZER
# ============================================================

def make_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    base_name = cfg.get("base_optimizer", "adamw")
    lr = cfg.get("lr", 3e-4)
    weight_decay = cfg.get("weight_decay", 1e-2)

    if base_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if base_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    if base_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unknown base optimizer: {base_name}")


# ============================================================
# METRICS
# ============================================================

def classification_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> Dict[str, float]:
    preds = preds.detach().cpu()
    targets = targets.detach().cpu()

    accuracy = (preds == targets).float().mean().item()

    macro_precision = []
    macro_recall = []
    macro_f1 = []
    weighted_f1 = []

    total_support = len(targets)

    global_tp = 0
    global_fp = 0
    global_fn = 0

    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().item()
        fp = ((preds == c) & (targets != c)).sum().item()
        fn = ((preds != c) & (targets == c)).sum().item()
        support = (targets == c).sum().item()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)

        macro_precision.append(precision)
        macro_recall.append(recall)
        macro_f1.append(f1)
        weighted_f1.append(f1 * support)

        global_tp += tp
        global_fp += fp
        global_fn += fn

    micro_precision = global_tp / (global_tp + global_fp + 1e-12)
    micro_recall = global_tp / (global_tp + global_fn + 1e-12)
    micro_f1 = 2.0 * micro_precision * micro_recall / (
        micro_precision + micro_recall + 1e-12
    )

    return {
        "accuracy": accuracy,
        "macro_precision": sum(macro_precision) / num_classes,
        "macro_recall": sum(macro_recall) / num_classes,
        "macro_f1": sum(macro_f1) / num_classes,
        "micro_f1": micro_f1,
        "weighted_f1": sum(weighted_f1) / max(total_support, 1),
    }


def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0

    for p in model.parameters():
        if p.grad is not None:
            norm = p.grad.detach().data.norm(2).item()
            total += norm ** 2

    return total ** 0.5


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_confidence = 0.0
    total_items = 0

    all_preds = []
    all_targets = []

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            logits = model(x)
            loss = criterion(logits, y)

        probs = torch.softmax(logits.float(), dim=1)
        confidence = probs.max(dim=1)[0]
        preds = logits.argmax(dim=1)

        batch_size = y.size(0)

        total_loss += loss.item() * batch_size
        total_confidence += confidence.sum().item()
        total_items += batch_size

        all_preds.append(preds.detach())
        all_targets.append(y.detach())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cls = classification_metrics(all_preds, all_targets, num_classes)

    return {
        "loss": total_loss / total_items,
        "confidence": total_confidence / total_items,
        **cls,
    }


# ============================================================
# TRAIN ONE
# ============================================================

def train_one(
    pt_path: Path,
    dataset_info: Dict[str, Any],
    model_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    run_id: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)

    dataset_name = dataset_info["dataset_name"]
    model_name = model_cfg["model_name"]
    scenario_name = scenario_cfg["scenario_name"]

    exp_dir = get_exp_dir(dataset_name, model_name, scenario_name, run_id)

    if RESUME and is_run_complete(exp_dir, EPOCHS):
        return {
            "status": "skipped_complete",
            "dataset_name": dataset_name,
            "model_name": model_name,
            "scenario_name": scenario_name,
            "run_id": run_id,
            "exp_dir": str(exp_dir),
        }

    if RESUME and exp_dir.exists() and not is_run_complete(exp_dir, EPOCHS):
        if RERUN_INCOMPLETE:
            remove_incomplete_run(exp_dir)
        else:
            return {
                "status": "skipped_incomplete",
                "dataset_name": dataset_name,
                "model_name": model_name,
                "scenario_name": scenario_name,
                "run_id": run_id,
                "exp_dir": str(exp_dir),
            }

    exp_dir.mkdir(parents=True, exist_ok=True)

    log_path = exp_dir / "log.txt"
    history_path = exp_dir / "history.json"

    dataset_info_path = exp_dir / "dataset_info.json"
    model_config_path = exp_dir / "model_config.json"
    scenario_config_path = exp_dir / "scenario_config.json"

    initial_model_path = exp_dir / "initial_model.pt"
    best_model_path = exp_dir / "best_model.pt"
    final_model_path = exp_dir / "final_model.pt"

    with open(dataset_info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=4, ensure_ascii=False)

    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=4, ensure_ascii=False)

    with open(scenario_config_path, "w", encoding="utf-8") as f:
        json.dump(scenario_cfg, f, indent=4, ensure_ascii=False)

    dataset = GeneratedListOpsDataset(pt_path)

    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size

    split_generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=split_generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = GeneratedTransformerClassifier(
        vocab_size=dataset_info["vocab_size"],
        num_classes=dataset_info["num_classes"],
        seq_len=dataset_info["seq_len"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        dropout=model_cfg["dropout"],
        pooling=model_cfg["pooling"],
        pad_idx=PAD_ID,
    ).to(device)

    num_params = count_parameters(model)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dataset_info": dataset_info,
            "model_config": model_cfg,
            "scenario_config": scenario_cfg,
            "num_params": num_params,
            "seed": seed,
        },
        initial_model_path,
    )

    optimizer = make_optimizer(model, scenario_cfg)
    criterion = nn.CrossEntropyLoss()

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and device.type == "cuda"),
    )

    frac_controller = FractionalGradientController(
        model=model,
        mode=scenario_cfg["mode"],
        target=scenario_cfg["target"],
        alpha=scenario_cfg["alpha"],
        beta=scenario_cfg["beta"],
        mix_lambda=scenario_cfg["mix_lambda"],
        start_epoch=scenario_cfg["start_epoch"],
    )

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_macro_precision": [],
        "train_macro_recall": [],
        "train_macro_f1": [],
        "train_micro_f1": [],
        "train_weighted_f1": [],
        "train_confidence": [],

        "val_loss": [],
        "val_accuracy": [],
        "val_macro_precision": [],
        "val_macro_recall": [],
        "val_macro_f1": [],
        "val_micro_f1": [],
        "val_weighted_f1": [],
        "val_confidence": [],

        "grad_norm": [],
        "fractional_applied_params": [],
        "epoch_time": [],
        "throughput_samples_per_sec": [],
        "lr": [],
    }

    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")
    best_epoch = -1

    run_start = time.time()

    log("=" * 100, log_path)
    log(f"DATASET      : {dataset_name}", log_path)
    log(f"MODEL        : {model_name}", log_path)
    log(f"SCENARIO     : {scenario_name}", log_path)
    log(f"RUN          : {run_id}", log_path)
    log(f"SEED         : {seed}", log_path)
    log(f"DEVICE       : {device}", log_path)
    log(f"USE_AMP      : {USE_AMP}", log_path)
    log(f"BATCH_SIZE   : {BATCH_SIZE}", log_path)
    log(f"PARAMETERS   : {num_params:,}", log_path)
    log(f"TRAIN SIZE   : {train_size}", log_path)
    log(f"VAL SIZE     : {val_size}", log_path)
    log(f"SCENARIO CFG : {scenario_cfg}", log_path)
    log("=" * 100, log_path)

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        model.train()

        total_loss = 0.0
        total_confidence = 0.0
        total_items = 0

        all_preds = []
        all_targets = []

        grad_norm_sum = 0.0
        grad_steps = 0
        fractional_applied_sum = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            applied = frac_controller.apply(current_epoch=epoch)
            fractional_applied_sum += applied

            grad_norm = compute_grad_norm(model)
            grad_norm_sum += grad_norm
            grad_steps += 1

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=1)
                confidence = probs.max(dim=1)[0]
                preds = logits.argmax(dim=1)

            batch_size = y.size(0)

            total_loss += loss.item() * batch_size
            total_confidence += confidence.sum().item()
            total_items += batch_size

            all_preds.append(preds.detach())
            all_targets.append(y.detach())

        train_preds = torch.cat(all_preds)
        train_targets = torch.cat(all_targets)

        train_cls = classification_metrics(
            train_preds,
            train_targets,
            num_classes=dataset_info["num_classes"],
        )

        train_metrics = {
            "loss": total_loss / total_items,
            "confidence": total_confidence / total_items,
            **train_cls,
        }

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            num_classes=dataset_info["num_classes"],
            device=device,
        )

        epoch_time = time.time() - epoch_start
        throughput = total_items / max(epoch_time, 1e-12)
        avg_grad_norm = grad_norm_sum / max(grad_steps, 1)
        avg_fractional_applied = fractional_applied_sum / max(grad_steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_macro_precision"].append(train_metrics["macro_precision"])
        history["train_macro_recall"].append(train_metrics["macro_recall"])
        history["train_macro_f1"].append(train_metrics["macro_f1"])
        history["train_micro_f1"].append(train_metrics["micro_f1"])
        history["train_weighted_f1"].append(train_metrics["weighted_f1"])
        history["train_confidence"].append(train_metrics["confidence"])

        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_macro_precision"].append(val_metrics["macro_precision"])
        history["val_macro_recall"].append(val_metrics["macro_recall"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_micro_f1"].append(val_metrics["micro_f1"])
        history["val_weighted_f1"].append(val_metrics["weighted_f1"])
        history["val_confidence"].append(val_metrics["confidence"])

        history["grad_norm"].append(avg_grad_norm)
        history["fractional_applied_params"].append(avg_fractional_applied)
        history["epoch_time"].append(epoch_time)
        history["throughput_samples_per_sec"].append(throughput)
        history["lr"].append(current_lr)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "dataset_info": dataset_info,
                    "model_config": model_cfg,
                    "scenario_config": scenario_cfg,
                    "epoch": best_epoch,
                    "best_val_macro_f1": best_val_macro_f1,
                    "best_val_accuracy": best_val_accuracy,
                    "best_val_loss": best_val_loss,
                    "num_params": num_params,
                    "seed": seed,
                },
                best_model_path,
            )

        log(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss {train_metrics['loss']:.4f} | "
            f"Train Acc {train_metrics['accuracy']:.4f} | "
            f"Train MacroF1 {train_metrics['macro_f1']:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | "
            f"Val Acc {val_metrics['accuracy']:.4f} | "
            f"Val MacroF1 {val_metrics['macro_f1']:.4f} | "
            f"Val WeightedF1 {val_metrics['weighted_f1']:.4f} | "
            f"Val Conf {val_metrics['confidence']:.4f} | "
            f"Grad {avg_grad_norm:.4f} | "
            f"FracApplied {avg_fractional_applied:.1f} | "
            f"Throughput {throughput:.1f} samp/s | "
            f"Time {epoch_time:.2f}s",
            log_path,
        )

    total_time = time.time() - run_start

    torch.save(
        {
            "state_dict": model.state_dict(),
            "dataset_info": dataset_info,
            "model_config": model_cfg,
            "scenario_config": scenario_cfg,
            "final_epoch": EPOCHS,
            "num_params": num_params,
            "seed": seed,
        },
        final_model_path,
    )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    log("-" * 100, log_path)
    log(f"BEST EPOCH          : {best_epoch}", log_path)
    log(f"BEST VAL MACRO F1   : {best_val_macro_f1:.6f}", log_path)
    log(f"BEST VAL ACC        : {best_val_accuracy:.6f}", log_path)
    log(f"BEST VAL LOSS       : {best_val_loss:.6f}", log_path)
    log(f"TOTAL RUN TIME      : {total_time:.2f} sec", log_path)
    log("-" * 100, log_path)

    result = {
        "status": "trained",

        "dataset_name": dataset_name,
        "pt_path": str(pt_path),
        "n_samples": dataset_info["n_samples"],
        "seq_len": dataset_info["seq_len"],
        "vocab_size": dataset_info["vocab_size"],
        "num_classes": dataset_info["num_classes"],
        "mean_length": dataset_info.get("mean_length"),
        "max_length": dataset_info.get("max_length"),

        "model_name": model_name,
        "d_model": model_cfg["d_model"],
        "nhead": model_cfg["nhead"],
        "num_layers": model_cfg["num_layers"],
        "dim_feedforward": model_cfg["dim_feedforward"],
        "dropout": model_cfg["dropout"],
        "pooling": model_cfg["pooling"],

        "scenario_name": scenario_name,
        "base_optimizer": scenario_cfg.get("base_optimizer"),
        "lr": scenario_cfg.get("lr"),
        "weight_decay": scenario_cfg.get("weight_decay"),
        "mode": scenario_cfg.get("mode"),
        "target": scenario_cfg.get("target"),
        "alpha": scenario_cfg.get("alpha"),
        "beta": scenario_cfg.get("beta"),
        "mix_lambda": scenario_cfg.get("mix_lambda"),
        "start_epoch": scenario_cfg.get("start_epoch"),

        "run_id": run_id,
        "seed": seed,
        "num_params": num_params,

        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,

        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_train_macro_f1": history["train_macro_f1"][-1],

        "final_val_loss": history["val_loss"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_val_macro_f1": history["val_macro_f1"][-1],
        "final_val_micro_f1": history["val_micro_f1"][-1],
        "final_val_weighted_f1": history["val_weighted_f1"][-1],
        "final_val_confidence": history["val_confidence"][-1],

        "mean_grad_norm": sum(history["grad_norm"]) / len(history["grad_norm"]),
        "mean_fractional_applied_params": (
            sum(history["fractional_applied_params"])
            / len(history["fractional_applied_params"])
        ),
        "mean_epoch_time": sum(history["epoch_time"]) / len(history["epoch_time"]),
        "mean_throughput": (
            sum(history["throughput_samples_per_sec"])
            / len(history["throughput_samples_per_sec"])
        ),
        "total_time_sec": total_time,

        "exp_dir": str(exp_dir),
        "initial_model_path": str(initial_model_path),
        "best_model_path": str(best_model_path),
        "final_model_path": str(final_model_path),
        "history_path": str(history_path),
        "log_path": str(log_path),
    }

    return result


# ============================================================
# WORKER
# ============================================================

def worker_train_job(job: Dict[str, Any]) -> Dict[str, Any]:
    gpu_id = job["gpu_id"]

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")

    try:
        result = train_one(
            pt_path=Path(job["pt_path"]),
            dataset_info=job["dataset_info"],
            model_cfg=job["model_cfg"],
            scenario_cfg=job["scenario_cfg"],
            run_id=job["run_id"],
            seed=job["seed"],
            device=device,
        )

        result["gpu_id"] = gpu_id
        result["job_id"] = job["job_id"]
        return result

    except Exception as e:
        return {
            "status": "failed",
            "job_id": job["job_id"],
            "pt_path": job["pt_path"],
            "dataset_name": job["dataset_info"].get("dataset_name"),
            "model_name": job["model_cfg"].get("model_name"),
            "scenario_name": job["scenario_cfg"].get("scenario_name"),
            "run_id": job["run_id"],
            "seed": job["seed"],
            "gpu_id": gpu_id,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(SEED)

    print("=" * 100)
    print("PARALLEL FRACTIONAL HYPOTHESIS V2 EXPERIMENT")
    print("=" * 100)

    print("ROOT:", ROOT)
    print("PT_DIR:", PT_DIR)
    print("RUN_DIR:", RUN_DIR)
    print("SUMMARY_CSV:", SUMMARY_CSV)
    print("SUMMARY_JSON:", SUMMARY_JSON)

    print("\nCUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("CUDA device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))

    print("\nSettings:")
    print("EPOCHS:", EPOCHS)
    print("BATCH_SIZE:", BATCH_SIZE)
    print("USE_AMP:", USE_AMP)
    print("MAX_PARALLEL_JOBS:", MAX_PARALLEL_JOBS)
    print("GPU_IDS:", GPU_IDS)
    print("RESUME:", RESUME)
    print("RERUN_INCOMPLETE:", RERUN_INCOMPLETE)

    pt_paths = []

    for cfg in DATASET_CONFIGS:
        pt_path = find_dataset_pt(cfg)
        pt_paths.append(pt_path)

    jobs = []
    job_id = 0

    for pt_path in pt_paths:
        dataset_info = dataset_info_from_pt(pt_path)

        for model_cfg in MODEL_CONFIGS:
            for scenario_cfg in SCENARIO_CONFIGS:
                for run_id in range(RUNS):
                    job_id += 1
                    gpu_id = GPU_IDS[(job_id - 1) % len(GPU_IDS)]

                    jobs.append(
                        {
                            "job_id": job_id,
                            "pt_path": str(pt_path),
                            "dataset_info": dataset_info,
                            "model_cfg": model_cfg,
                            "scenario_cfg": scenario_cfg,
                            "run_id": run_id,
                            "seed": SEED + job_id,
                            "gpu_id": gpu_id,
                        }
                    )

    total_jobs = len(jobs)

    print("\nExperiment grid:")
    print("Datasets:", len(pt_paths))
    print("Models:", len(MODEL_CONFIGS))
    print("Scenarios:", len(SCENARIO_CONFIGS))
    print("Runs:", RUNS)
    print("Total jobs:", total_jobs)

    all_results = []

    trained_count = 0
    skipped_complete = 0
    skipped_incomplete = 0
    failed_count = 0

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    print("\nStarting parallel execution...\n")

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
        future_to_job = {
            executor.submit(worker_train_job, job): job
            for job in jobs
        }

        for future in as_completed(future_to_job):
            job = future_to_job[future]

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "status": "failed",
                    "job_id": job["job_id"],
                    "pt_path": job["pt_path"],
                    "dataset_name": job["dataset_info"].get("dataset_name"),
                    "model_name": job["model_cfg"].get("model_name"),
                    "scenario_name": job["scenario_cfg"].get("scenario_name"),
                    "run_id": job["run_id"],
                    "seed": job["seed"],
                    "gpu_id": job["gpu_id"],
                    "error": repr(e),
                    "traceback": traceback.format_exc(),
                }

            status = result.get("status")

            if status == "skipped_complete":
                skipped_complete += 1
                print(
                    f"[SKIP] {job['job_id']}/{total_jobs} | "
                    f"{result.get('dataset_name')} | "
                    f"{result.get('model_name')} | "
                    f"{result.get('scenario_name')} | "
                    f"run {result.get('run_id')}",
                    flush=True,
                )

            elif status == "skipped_incomplete":
                skipped_incomplete += 1
                all_results.append(result)
                save_summary(all_results)

                print(
                    f"[SKIP INCOMPLETE] {job['job_id']}/{total_jobs} | "
                    f"{result.get('dataset_name')} | "
                    f"{result.get('model_name')} | "
                    f"{result.get('scenario_name')} | "
                    f"run {result.get('run_id')}",
                    flush=True,
                )

            elif status == "failed":
                failed_count += 1
                all_results.append(result)
                save_summary(all_results)

                print(
                    f"[FAILED] {job['job_id']}/{total_jobs} | "
                    f"{result.get('dataset_name')} | "
                    f"{result.get('model_name')} | "
                    f"{result.get('scenario_name')} | "
                    f"run {result.get('run_id')} | "
                    f"{result.get('error')}",
                    flush=True,
                )

            else:
                trained_count += 1
                all_results.append(result)
                save_summary(all_results)

                best_f1 = result.get("best_val_macro_f1")
                best_f1_str = f"{best_f1:.4f}" if isinstance(best_f1, float) else str(best_f1)

                print(
                    f"[DONE] {job['job_id']}/{total_jobs} | "
                    f"{result.get('dataset_name')} | "
                    f"{result.get('model_name')} | "
                    f"{result.get('scenario_name')} | "
                    f"run {result.get('run_id')} | "
                    f"GPU {result.get('gpu_id')} | "
                    f"best F1={best_f1_str}",
                    flush=True,
                )

    save_summary(all_results)

    print("\nALL DONE")
    print("Run dir:", RUN_DIR)
    print("Summary CSV:", SUMMARY_CSV)
    print("Summary JSON:", SUMMARY_JSON)

    print("\nCOUNTS")
    print("Total jobs:", total_jobs)
    print("Trained now:", trained_count)
    print("Skipped complete:", skipped_complete)
    print("Skipped incomplete:", skipped_incomplete)
    print("Failed:", failed_count)


if __name__ == "__main__":
    main()