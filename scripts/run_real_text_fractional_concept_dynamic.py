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
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
)


# ============================================================
# ROOTS
# ============================================================

ROOT = Path.cwd()

TOKENIZED_ROOT = ROOT / "real_text_datasets" / "tokenized_pt"
HF_MODEL_DIR = ROOT / "hf_models"

RUN_DIR = ROOT / "runs_real_text_fractional_concept"

SUMMARY_CSV = ROOT / "summary_real_text_fractional_concept.csv"
SUMMARY_JSON = ROOT / "summary_real_text_fractional_concept.json"

RUN_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# GLOBAL CONFIG
# ============================================================

SEED = 42

RUNS = 5
EPOCHS = 5

VAL_SPLIT_FROM_TRAIN = 0.1

BATCH_SIZE = 32
NUM_WORKERS = 1

USE_AMP = True
GRAD_CLIP_NORM = 1.0

RESUME = True
RERUN_INCOMPLETE = True

EXPECTED_EPOCHS = EPOCHS

# Saving all models for 780 jobs can eat disk.
# Set SAVE_INITIAL_MODEL=False if disk is tight.
SAVE_INITIAL_MODEL = False
SAVE_BEST_MODEL = True
SAVE_FINAL_MODEL = False

GPU_IDS = [0]


# ============================================================
# DYNAMIC GPU SCHEDULER CONFIG
# ============================================================

DYNAMIC_SCHEDULER = True

MAX_PARALLEL_JOBS_HARD_CAP = 6
MIN_PARALLEL_JOBS = 2

TARGET_GPU_UTIL_LOW = 88.0
TARGET_GPU_UTIL_HIGH = 96.0

MAX_GPU_MEMORY_USED_PCT = 92.0
MIN_FREE_MEMORY_GB = 3.0

SCHEDULER_POLL_SEC = 10
UTIL_SMOOTHING_WINDOW = 3

PRINT_GPU_MONITOR = True


# ============================================================
# REAL DATASETS
# ============================================================

DATASET_CONFIGS = [
    {
        "dataset_name": "ag_news",
        "num_labels": 4,
        "task_type": "topic_classification",
        "train_size": 120000,
        "test_size": 7600,
    },
    {
        "dataset_name": "imdb",
        "num_labels": 2,
        "task_type": "sentiment_classification",
        "train_size": 25000,
        "test_size": 25000,
    },
    {
        "dataset_name": "20newsgroups",
        "num_labels": 20,
        "task_type": "topic_classification",
        "train_size": 11314,
        "test_size": 7532,
    },
]


# ============================================================
# LOCAL PRETRAINED MODELS
# ============================================================

MODEL_CONFIGS = [
    {
        "model_name": "bert_tiny",
        "model_path": str(HF_MODEL_DIR / "bert_tiny"),
        "loader": "bert_forced",
        "family": "bert",
        "recommended_batch_size": 96,
    },
    {
        "model_name": "bert_mini",
        "model_path": str(HF_MODEL_DIR / "bert_mini"),
        "loader": "bert_forced",
        "family": "bert",
        "recommended_batch_size": 64,
    },
    {
        "model_name": "electra_small_discriminator",
        "model_path": str(HF_MODEL_DIR / "electra_small_discriminator"),
        "loader": "auto",
        "family": "electra",
        "recommended_batch_size": 48,
    },
    {
        "model_name": "distilbert_base_uncased",
        "model_path": str(HF_MODEL_DIR / "distilbert_base_uncased"),
        "loader": "auto",
        "family": "distilbert",
        "recommended_batch_size": 32,
    },
]


# ============================================================
# CONCEPTUAL SCENARIOS
# ============================================================

def adamw_scenario(
    scenario_name: str,
    mode: str,
    target: str,
    alpha: Optional[float],
    beta: Optional[float],
    mix_lambda: float,
    start_epoch: Optional[int],
    trainable_scope: str = "all",
    lr: float = 2e-5,
    weight_decay: float = 1e-2,
) -> Dict[str, Any]:
    return {
        "scenario_name": scenario_name,
        "base_optimizer": "adamw",
        "lr": lr,
        "weight_decay": weight_decay,
        "mode": mode,
        "target": target,
        "alpha": alpha,
        "beta": beta,
        "mix_lambda": mix_lambda,
        "start_epoch": start_epoch,
        "trainable_scope": trainable_scope,
    }


SCENARIOS_FINAL_CONCEPT = [
    # --------------------------------------------------------
    # Main baseline
    # --------------------------------------------------------
    adamw_scenario(
        scenario_name="baseline_adamw",
        mode="none",
        target="none",
        alpha=None,
        beta=None,
        mix_lambda=0.0,
        start_epoch=None,
    ),

    # --------------------------------------------------------
    # Negative control
    # Shows that global fractional replacement is unstable/harmful.
    # --------------------------------------------------------
    adamw_scenario(
        scenario_name="negative_full_replace_a08",
        mode="replace",
        target="all",
        alpha=0.80,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),

    # --------------------------------------------------------
    # Embedding replacement alpha sweep
    # Tests whether a070 was a lucky point or an order-dependent effect.
    # --------------------------------------------------------
    adamw_scenario(
        scenario_name="emb_replace_a050",
        mode="replace",
        target="embeddings",
        alpha=0.50,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="emb_replace_a060",
        mode="replace",
        target="embeddings",
        alpha=0.60,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="emb_replace_a070",
        mode="replace",
        target="embeddings",
        alpha=0.70,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="emb_replace_a080",
        mode="replace",
        target="embeddings",
        alpha=0.80,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="emb_replace_a090",
        mode="replace",
        target="embeddings",
        alpha=0.90,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),

    # --------------------------------------------------------
    # Embedding mixed-memory controls
    # Compare replace vs weak mix on the same representation-entry block.
    # --------------------------------------------------------
    adamw_scenario(
        scenario_name="emb_mix_a08_lam010",
        mode="mix",
        target="embeddings",
        alpha=0.80,
        beta=0.90,
        mix_lambda=0.010,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="emb_mix_a08_lam015",
        mode="mix",
        target="embeddings",
        alpha=0.80,
        beta=0.90,
        mix_lambda=0.015,
        start_epoch=1,
    ),

    # --------------------------------------------------------
    # Layer-wise replacement
    # Tests whether the effect is specific to embeddings or early encoder layers.
    # --------------------------------------------------------
    adamw_scenario(
        scenario_name="bottom1_replace_a070",
        mode="replace",
        target="bottom1",
        alpha=0.70,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="bottom2_replace_a070",
        mode="replace",
        target="bottom2",
        alpha=0.70,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),
    adamw_scenario(
        scenario_name="emb_plus_bottom1_replace_a070",
        mode="replace",
        target="embeddings_plus_bottom1",
        alpha=0.70,
        beta=0.90,
        mix_lambda=1.0,
        start_epoch=1,
    ),

    # --------------------------------------------------------
    # Head control
    # Synthetic runs liked delayed head-mix, so keep it as cross-domain control.
    # --------------------------------------------------------
    adamw_scenario(
        scenario_name="delayed_head_mix_a08_lam010_warm1",
        mode="mix",
        target="head",
        alpha=0.80,
        beta=0.90,
        mix_lambda=0.010,
        start_epoch=2,
    ),
]


SCENARIO_CONFIGS = SCENARIOS_FINAL_CONCEPT


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
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    if SAVE_BEST_MODEL and not best_model_path.exists():
        return False

    if SAVE_FINAL_MODEL and not final_model_path.exists():
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

class RealTextDataset(Dataset):
    def __init__(self, pt_path: Path):
        data = torch.load(pt_path, map_location="cpu")

        self.input_ids = data["input_ids"].long()
        self.attention_mask = data["attention_mask"].long()
        self.labels = data["labels"].long()

        if self.labels.ndim != 1:
            self.labels = self.labels.view(-1)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def find_tokenized_paths(dataset_name: str, model_name: str) -> Dict[str, Path]:
    base = TOKENIZED_ROOT / dataset_name / model_name

    train_pt = base / "train.pt"
    test_pt = base / "test.pt"
    info_path = base / "tokenized_info.json"

    if not train_pt.exists():
        raise FileNotFoundError(f"Missing train.pt: {train_pt}")

    if not test_pt.exists():
        raise FileNotFoundError(f"Missing test.pt: {test_pt}")

    if not info_path.exists():
        raise FileNotFoundError(f"Missing tokenized_info.json: {info_path}")

    return {
        "base_dir": base,
        "train_pt": train_pt,
        "test_pt": test_pt,
        "info_path": info_path,
    }


def load_tokenized_info(info_path: Path) -> Dict[str, Any]:
    with open(info_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# MODEL LOADING / FREEZING
# ============================================================

def load_classifier_model(
    model_cfg: Dict[str, Any],
    num_labels: int,
    device: torch.device,
) -> nn.Module:
    model_path = model_cfg["model_path"]
    loader = model_cfg["loader"]

    if loader == "bert_forced":
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
            local_files_only=True,
        )

    model.to(device)
    return model


def is_head_parameter(name: str) -> bool:
    return (
        "classifier" in name
        or "pre_classifier" in name
        or "score" in name
        or "out_proj" in name
    )


def is_embedding_parameter(name: str) -> bool:
    return (
        "embeddings" in name
        or "embed" in name
        or "word_embeddings" in name
        or "position_embeddings" in name
    )


def apply_trainable_scope(model: nn.Module, trainable_scope: str) -> int:
    """
    Optional freezing mechanism.
    Current scenarios use trainable_scope='all', but this is included
    so freeze-control scenarios can be added without rewriting the code.
    """

    for _, p in model.named_parameters():
        p.requires_grad = True

    if trainable_scope == "all":
        return count_trainable_parameters(model)

    for _, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if trainable_scope == "head_only":
            if is_head_parameter(name):
                p.requires_grad = True

        elif trainable_scope == "embeddings_and_head":
            if is_embedding_parameter(name) or is_head_parameter(name):
                p.requires_grad = True

        else:
            raise ValueError(f"Unknown trainable_scope: {trainable_scope}")

    return count_trainable_parameters(model)


# ============================================================
# FRACTIONAL TARGET MATCHING
# ============================================================

def is_bert_layer(name: str, layer_idx: int) -> bool:
    # BERT / ELECTRA style:
    # bert.encoder.layer.0...
    # electra.encoder.layer.0...
    return (
        f".encoder.layer.{layer_idx}." in name
        or f"encoder.layer.{layer_idx}." in name
    )


def is_distilbert_layer(name: str, layer_idx: int) -> bool:
    # DistilBERT style:
    # distilbert.transformer.layer.0...
    return (
        f".transformer.layer.{layer_idx}." in name
        or f"transformer.layer.{layer_idx}." in name
    )


def is_bottom_layer_parameter(name: str, max_layer_exclusive: int) -> bool:
    for i in range(max_layer_exclusive):
        if is_bert_layer(name, i) or is_distilbert_layer(name, i):
            return True
    return False


def matches_fractional_target(name: str, target: str) -> bool:
    if target == "all":
        return True

    if target == "none":
        return False

    if target == "head":
        return is_head_parameter(name)

    if target == "embeddings":
        return is_embedding_parameter(name)

    if target == "bottom1":
        return is_bottom_layer_parameter(name, 1)

    if target == "bottom2":
        return is_bottom_layer_parameter(name, 2)

    if target == "embeddings_plus_bottom1":
        return is_embedding_parameter(name) or is_bottom_layer_parameter(name, 1)

    if target == "embeddings_plus_bottom2":
        return is_embedding_parameter(name) or is_bottom_layer_parameter(name, 2)

    if target == "attention":
        return (
            "attention" in name
            or "self_attn" in name
            or "query" in name
            or "key" in name
            or "value" in name
            or "q_lin" in name
            or "k_lin" in name
            or "v_lin" in name
        )

    if target == "ffn":
        return (
            "intermediate" in name
            or "output.dense" in name
            or "ffn" in name
            or "lin1" in name
            or "lin2" in name
        )

    raise ValueError(f"Unknown target: {target}")


def count_target_parameters(model: nn.Module, target: str) -> int:
    n = 0
    for name, p in model.named_parameters():
        if matches_fractional_target(name, target):
            n += p.numel()
    return n


def count_target_tensors(model: nn.Module, target: str) -> int:
    n = 0
    for name, _ in model.named_parameters():
        if matches_fractional_target(name, target):
            n += 1
    return n


def preview_target_names(model: nn.Module, target: str, limit: int = 20) -> List[str]:
    names = []
    for name, _ in model.named_parameters():
        if matches_fractional_target(name, target):
            names.append(name)
        if len(names) >= limit:
            break
    return names


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

                if not matches_fractional_target(name, self.target):
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
                    p.grad.copy_(
                        (1.0 - self.mix_lambda) * g_original
                        + self.mix_lambda * mem
                    )

                applied += 1

        return applied


# ============================================================
# OPTIMIZER
# ============================================================

def make_optimizer(model: nn.Module, cfg: Dict[str, Any]):
    base_name = cfg.get("base_optimizer", "adamw")
    lr = cfg.get("lr", 2e-5)
    weight_decay = cfg.get("weight_decay", 1e-2)

    params = [p for p in model.parameters() if p.requires_grad]

    if len(params) == 0:
        raise RuntimeError("No trainable parameters found.")

    if base_name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if base_name == "adam":
        return torch.optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay,
        )

    if base_name == "sgd":
        return torch.optim.SGD(
            params,
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


def compute_grad_norm_and_nonfinite(model: nn.Module) -> tuple[float, int]:
    total = 0.0
    nonfinite = 0

    for p in model.parameters():
        if p.grad is None:
            continue

        g = p.grad.detach()

        if not torch.isfinite(g).all():
            nonfinite += 1
            continue

        norm = g.data.norm(2).item()

        if math.isfinite(norm):
            total += norm ** 2
        else:
            nonfinite += 1

    return total ** 0.5, nonfinite


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {
        k: v.to(device, non_blocking=True)
        for k, v in batch.items()
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_confidence = 0.0
    total_items = 0
    nonfinite_loss_batches = 0

    all_preds = []
    all_targets = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            out = model(**batch)
            loss = out.loss
            logits = out.logits

        if not torch.isfinite(loss):
            nonfinite_loss_batches += 1
            loss_value = float("nan")
        else:
            loss_value = loss.item()

        probs = torch.softmax(logits.float(), dim=1)
        confidence = probs.max(dim=1)[0]
        preds = logits.argmax(dim=1)
        targets = batch["labels"]

        batch_size = targets.size(0)

        if math.isfinite(loss_value):
            total_loss += loss_value * batch_size
        else:
            total_loss += 0.0

        total_confidence += confidence.sum().item()
        total_items += batch_size

        all_preds.append(preds.detach())
        all_targets.append(targets.detach())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    cls = classification_metrics(
        all_preds,
        all_targets,
        num_classes=num_classes,
    )

    return {
        "loss": total_loss / max(total_items, 1),
        "confidence": total_confidence / max(total_items, 1),
        "nonfinite_loss_batches": nonfinite_loss_batches,
        **cls,
    }


# ============================================================
# GPU MONITORING
# ============================================================

def get_gpu_stats(gpu_id: int = 0) -> Dict[str, float]:
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)

        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

        total_gb = mem.total / 1024**3
        used_gb = mem.used / 1024**3
        free_gb = mem.free / 1024**3
        used_pct = 100.0 * used_gb / max(total_gb, 1e-12)

        return {
            "gpu_util_pct": float(util.gpu),
            "mem_util_pct": float(used_pct),
            "mem_used_gb": float(used_gb),
            "mem_free_gb": float(free_gb),
            "mem_total_gb": float(total_gb),
        }

    except Exception as e:
        return {
            "gpu_util_pct": 0.0,
            "mem_util_pct": 0.0,
            "mem_used_gb": 0.0,
            "mem_free_gb": 999.0,
            "mem_total_gb": 999.0,
            "nvml_error": repr(e),
        }


def should_submit_more_jobs(
    running_count: int,
    util_history: List[float],
    gpu_stats: Dict[str, float],
) -> bool:
    if running_count < MIN_PARALLEL_JOBS:
        return True

    if running_count >= MAX_PARALLEL_JOBS_HARD_CAP:
        return False

    gpu_util = gpu_stats["gpu_util_pct"]
    mem_used_pct = gpu_stats["mem_util_pct"]
    mem_free_gb = gpu_stats["mem_free_gb"]

    if len(util_history) > 0:
        smoothed_util = sum(util_history[-UTIL_SMOOTHING_WINDOW:]) / min(
            len(util_history),
            UTIL_SMOOTHING_WINDOW,
        )
    else:
        smoothed_util = gpu_util

    memory_safe = (
        mem_used_pct < MAX_GPU_MEMORY_USED_PCT
        and mem_free_gb > MIN_FREE_MEMORY_GB
    )

    gpu_needs_more_work = smoothed_util < TARGET_GPU_UTIL_LOW

    return memory_safe and gpu_needs_more_work


def print_scheduler_state(
    prefix: str,
    running_count: int,
    pending_count: int,
    done_count: int,
    total_jobs: int,
    gpu_stats: Dict[str, float],
) -> None:
    if not PRINT_GPU_MONITOR:
        return

    print(
        f"{prefix} | "
        f"running={running_count} | "
        f"pending={pending_count} | "
        f"done={done_count}/{total_jobs} | "
        f"gpu={gpu_stats['gpu_util_pct']:.0f}% | "
        f"mem={gpu_stats['mem_used_gb']:.1f}/{gpu_stats['mem_total_gb']:.1f} GB "
        f"({gpu_stats['mem_util_pct']:.1f}%) | "
        f"free={gpu_stats['mem_free_gb']:.1f} GB",
        flush=True,
    )


# ============================================================
# TRAIN ONE
# ============================================================

def train_one(
    dataset_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    scenario_cfg: Dict[str, Any],
    run_id: int,
    seed: int,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(seed)

    dataset_name = dataset_cfg["dataset_name"]
    model_name = model_cfg["model_name"]
    scenario_name = scenario_cfg["scenario_name"]
    num_labels = dataset_cfg["num_labels"]

    exp_dir = get_exp_dir(
        dataset_name=dataset_name,
        model_name=model_name,
        scenario_name=scenario_name,
        run_id=run_id,
    )

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

    dataset_config_path = exp_dir / "dataset_config.json"
    model_config_path = exp_dir / "model_config.json"
    scenario_config_path = exp_dir / "scenario_config.json"

    initial_model_path = exp_dir / "initial_model.pt"
    best_model_path = exp_dir / "best_model.pt"
    final_model_path = exp_dir / "final_model.pt"

    paths = find_tokenized_paths(dataset_name, model_name)
    tokenized_info = load_tokenized_info(paths["info_path"])

    with open(dataset_config_path, "w", encoding="utf-8") as f:
        json.dump(dataset_cfg, f, indent=4, ensure_ascii=False)

    with open(model_config_path, "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=4, ensure_ascii=False)

    with open(scenario_config_path, "w", encoding="utf-8") as f:
        json.dump(scenario_cfg, f, indent=4, ensure_ascii=False)

    train_full = RealTextDataset(paths["train_pt"])
    test_dataset = RealTextDataset(paths["test_pt"])

    val_size = int(len(train_full) * VAL_SPLIT_FROM_TRAIN)
    train_size = len(train_full) - val_size

    split_generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        train_full,
        [train_size, val_size],
        generator=split_generator,
    )

    effective_batch_size = min(
        BATCH_SIZE,
        int(model_cfg.get("recommended_batch_size", BATCH_SIZE)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = load_classifier_model(
        model_cfg=model_cfg,
        num_labels=num_labels,
        device=device,
    )

    trainable_scope = scenario_cfg.get("trainable_scope", "all")
    trainable_params_after_scope = apply_trainable_scope(model, trainable_scope)

    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)

    target_param_count = count_target_parameters(model, scenario_cfg["target"])
    target_tensor_count = count_target_tensors(model, scenario_cfg["target"])
    target_preview = preview_target_names(model, scenario_cfg["target"], limit=20)

    if SAVE_INITIAL_MODEL:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dataset_config": dataset_cfg,
                "model_config": model_cfg,
                "scenario_config": scenario_cfg,
                "tokenized_info": tokenized_info,
                "num_params": total_params,
                "trainable_params": trainable_params,
                "seed": seed,
                "target_param_count": target_param_count,
                "target_tensor_count": target_tensor_count,
                "target_preview": target_preview,
            },
            initial_model_path,
        )

    optimizer = make_optimizer(model, scenario_cfg)

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

        "test_loss": [],
        "test_accuracy": [],
        "test_macro_precision": [],
        "test_macro_recall": [],
        "test_macro_f1": [],
        "test_micro_f1": [],
        "test_weighted_f1": [],
        "test_confidence": [],

        "grad_norm": [],
        "nonfinite_grad_tensors": [],
        "nonfinite_train_loss_batches": [],
        "nonfinite_val_loss_batches": [],
        "nonfinite_test_loss_batches": [],

        "fractional_applied_params": [],
        "epoch_time": [],
        "throughput_samples_per_sec": [],
        "lr": [],
    }

    best_val_macro_f1 = -1.0
    best_val_accuracy = -1.0
    best_val_loss = float("inf")

    best_test_macro_f1_at_best_val = -1.0
    best_test_accuracy_at_best_val = -1.0
    best_test_loss_at_best_val = float("inf")

    best_epoch = -1

    run_start = time.time()

    log("=" * 100, log_path)
    log("REAL TEXT FRACTIONAL CONCEPT RUN", log_path)
    log("=" * 100, log_path)
    log(f"DATASET      : {dataset_name}", log_path)
    log(f"TASK         : {dataset_cfg['task_type']}", log_path)
    log(f"NUM LABELS   : {num_labels}", log_path)
    log(f"MODEL        : {model_name}", log_path)
    log(f"MODEL PATH   : {model_cfg['model_path']}", log_path)
    log(f"SCENARIO     : {scenario_name}", log_path)
    log(f"RUN          : {run_id}", log_path)
    log(f"SEED         : {seed}", log_path)
    log(f"DEVICE       : {device}", log_path)
    log(f"USE_AMP      : {USE_AMP}", log_path)
    log(f"BATCH_SIZE   : {effective_batch_size}", log_path)
    log(f"EPOCHS       : {EPOCHS}", log_path)
    log(f"TRAIN SIZE   : {train_size}", log_path)
    log(f"VAL SIZE     : {val_size}", log_path)
    log(f"TEST SIZE    : {len(test_dataset)}", log_path)
    log(f"PARAMETERS   : {total_params:,}", log_path)
    log(f"TRAINABLE    : {trainable_params:,}", log_path)
    log(f"TRAINABLE SCOPE : {trainable_scope}", log_path)
    log(f"TARGET PARAM COUNT  : {target_param_count:,}", log_path)
    log(f"TARGET TENSOR COUNT : {target_tensor_count}", log_path)
    log(f"TARGET PREVIEW      : {target_preview}", log_path)
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
        nonfinite_grad_tensors_sum = 0
        nonfinite_train_loss_batches = 0
        fractional_applied_sum = 0

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                out = model(**batch)
                loss = out.loss
                logits = out.logits

            if not torch.isfinite(loss):
                nonfinite_train_loss_batches += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=GRAD_CLIP_NORM,
                )

            applied = frac_controller.apply(current_epoch=epoch)
            fractional_applied_sum += applied

            grad_norm, nonfinite_grad_tensors = compute_grad_norm_and_nonfinite(model)
            nonfinite_grad_tensors_sum += nonfinite_grad_tensors

            grad_norm_sum += grad_norm
            grad_steps += 1

            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=1)
                confidence = probs.max(dim=1)[0]
                preds = logits.argmax(dim=1)
                targets = batch["labels"]

            batch_size = targets.size(0)

            total_loss += loss.item() * batch_size
            total_confidence += confidence.sum().item()
            total_items += batch_size

            all_preds.append(preds.detach())
            all_targets.append(targets.detach())

        if all_preds:
            train_preds = torch.cat(all_preds)
            train_targets = torch.cat(all_targets)

            train_cls = classification_metrics(
                train_preds,
                train_targets,
                num_classes=num_labels,
            )

            train_metrics = {
                "loss": total_loss / max(total_items, 1),
                "confidence": total_confidence / max(total_items, 1),
                **train_cls,
            }
        else:
            train_metrics = {
                "loss": float("nan"),
                "confidence": float("nan"),
                "accuracy": 0.0,
                "macro_precision": 0.0,
                "macro_recall": 0.0,
                "macro_f1": 0.0,
                "micro_f1": 0.0,
                "weighted_f1": 0.0,
            }

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            num_classes=num_labels,
            device=device,
        )

        test_metrics = evaluate(
            model=model,
            loader=test_loader,
            num_classes=num_labels,
            device=device,
        )

        epoch_time = time.time() - epoch_start
        throughput = total_items / max(epoch_time, 1e-12)
        avg_grad_norm = grad_norm_sum / max(grad_steps, 1)
        avg_fractional_applied = fractional_applied_sum / max(grad_steps, 1)
        avg_nonfinite_grad_tensors = nonfinite_grad_tensors_sum / max(grad_steps, 1)
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

        history["test_loss"].append(test_metrics["loss"])
        history["test_accuracy"].append(test_metrics["accuracy"])
        history["test_macro_precision"].append(test_metrics["macro_precision"])
        history["test_macro_recall"].append(test_metrics["macro_recall"])
        history["test_macro_f1"].append(test_metrics["macro_f1"])
        history["test_micro_f1"].append(test_metrics["micro_f1"])
        history["test_weighted_f1"].append(test_metrics["weighted_f1"])
        history["test_confidence"].append(test_metrics["confidence"])

        history["grad_norm"].append(avg_grad_norm)
        history["nonfinite_grad_tensors"].append(avg_nonfinite_grad_tensors)
        history["nonfinite_train_loss_batches"].append(nonfinite_train_loss_batches)
        history["nonfinite_val_loss_batches"].append(val_metrics["nonfinite_loss_batches"])
        history["nonfinite_test_loss_batches"].append(test_metrics["nonfinite_loss_batches"])

        history["fractional_applied_params"].append(avg_fractional_applied)
        history["epoch_time"].append(epoch_time)
        history["throughput_samples_per_sec"].append(throughput)
        history["lr"].append(current_lr)

        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = val_metrics["macro_f1"]
            best_val_accuracy = val_metrics["accuracy"]
            best_val_loss = val_metrics["loss"]

            best_test_macro_f1_at_best_val = test_metrics["macro_f1"]
            best_test_accuracy_at_best_val = test_metrics["accuracy"]
            best_test_loss_at_best_val = test_metrics["loss"]

            best_epoch = epoch

            if SAVE_BEST_MODEL:
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "dataset_config": dataset_cfg,
                        "model_config": model_cfg,
                        "scenario_config": scenario_cfg,
                        "tokenized_info": tokenized_info,
                        "epoch": best_epoch,
                        "best_val_macro_f1": best_val_macro_f1,
                        "best_val_accuracy": best_val_accuracy,
                        "best_val_loss": best_val_loss,
                        "best_test_macro_f1_at_best_val": best_test_macro_f1_at_best_val,
                        "best_test_accuracy_at_best_val": best_test_accuracy_at_best_val,
                        "best_test_loss_at_best_val": best_test_loss_at_best_val,
                        "num_params": total_params,
                        "trainable_params": trainable_params,
                        "target_param_count": target_param_count,
                        "target_tensor_count": target_tensor_count,
                        "target_preview": target_preview,
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
            f"Test Loss {test_metrics['loss']:.4f} | "
            f"Test Acc {test_metrics['accuracy']:.4f} | "
            f"Test MacroF1 {test_metrics['macro_f1']:.4f} | "
            f"Grad {avg_grad_norm:.4f} | "
            f"NonFiniteGrad {avg_nonfinite_grad_tensors:.1f} | "
            f"FracApplied {avg_fractional_applied:.1f} | "
            f"Throughput {throughput:.1f} samp/s | "
            f"Time {epoch_time:.2f}s",
            log_path,
        )

    total_time = time.time() - run_start

    if SAVE_FINAL_MODEL:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "dataset_config": dataset_cfg,
                "model_config": model_cfg,
                "scenario_config": scenario_cfg,
                "tokenized_info": tokenized_info,
                "final_epoch": EPOCHS,
                "num_params": total_params,
                "trainable_params": trainable_params,
                "target_param_count": target_param_count,
                "target_tensor_count": target_tensor_count,
                "target_preview": target_preview,
                "seed": seed,
            },
            final_model_path,
        )

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    log("-" * 100, log_path)
    log(f"BEST EPOCH                    : {best_epoch}", log_path)
    log(f"BEST VAL MACRO F1             : {best_val_macro_f1:.6f}", log_path)
    log(f"BEST VAL ACC                  : {best_val_accuracy:.6f}", log_path)
    log(f"BEST VAL LOSS                 : {best_val_loss:.6f}", log_path)
    log(f"TEST MACRO F1 AT BEST VAL     : {best_test_macro_f1_at_best_val:.6f}", log_path)
    log(f"TEST ACC AT BEST VAL          : {best_test_accuracy_at_best_val:.6f}", log_path)
    log(f"TEST LOSS AT BEST VAL         : {best_test_loss_at_best_val:.6f}", log_path)
    log(f"TOTAL RUN TIME                : {total_time:.2f} sec", log_path)
    log("-" * 100, log_path)

    result = {
        "status": "trained",

        "dataset_name": dataset_name,
        "task_type": dataset_cfg["task_type"],
        "num_labels": num_labels,

        "tokenized_train_pt": str(paths["train_pt"]),
        "tokenized_test_pt": str(paths["test_pt"]),
        "train_full_size": len(train_full),
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_dataset),
        "seq_len": tokenized_info.get("max_len"),

        "model_name": model_name,
        "model_path": model_cfg["model_path"],
        "model_loader": model_cfg["loader"],
        "model_family": model_cfg["family"],

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
        "trainable_scope": trainable_scope,

        "target_param_count": target_param_count,
        "target_tensor_count": target_tensor_count,
        "target_preview": "; ".join(target_preview),

        "run_id": run_id,
        "seed": seed,
        "device": str(device),
        "batch_size": effective_batch_size,
        "use_amp": USE_AMP,
        "grad_clip_norm": GRAD_CLIP_NORM,

        "num_params": total_params,
        "trainable_params": trainable_params,
        "trainable_params_after_scope": trainable_params_after_scope,

        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_accuracy": best_val_accuracy,
        "best_val_loss": best_val_loss,

        "best_test_macro_f1_at_best_val": best_test_macro_f1_at_best_val,
        "best_test_accuracy_at_best_val": best_test_accuracy_at_best_val,
        "best_test_loss_at_best_val": best_test_loss_at_best_val,

        "final_train_loss": history["train_loss"][-1],
        "final_train_accuracy": history["train_accuracy"][-1],
        "final_train_macro_f1": history["train_macro_f1"][-1],

        "final_val_loss": history["val_loss"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_val_macro_f1": history["val_macro_f1"][-1],
        "final_val_micro_f1": history["val_micro_f1"][-1],
        "final_val_weighted_f1": history["val_weighted_f1"][-1],
        "final_val_confidence": history["val_confidence"][-1],

        "final_test_loss": history["test_loss"][-1],
        "final_test_accuracy": history["test_accuracy"][-1],
        "final_test_macro_f1": history["test_macro_f1"][-1],
        "final_test_micro_f1": history["test_micro_f1"][-1],
        "final_test_weighted_f1": history["test_weighted_f1"][-1],
        "final_test_confidence": history["test_confidence"][-1],

        "mean_grad_norm": sum(history["grad_norm"]) / len(history["grad_norm"]),
        "mean_nonfinite_grad_tensors": (
            sum(history["nonfinite_grad_tensors"]) / len(history["nonfinite_grad_tensors"])
        ),
        "total_nonfinite_train_loss_batches": sum(history["nonfinite_train_loss_batches"]),
        "total_nonfinite_val_loss_batches": sum(history["nonfinite_val_loss_batches"]),
        "total_nonfinite_test_loss_batches": sum(history["nonfinite_test_loss_batches"]),

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
        "initial_model_path": str(initial_model_path) if SAVE_INITIAL_MODEL else "",
        "best_model_path": str(best_model_path) if SAVE_BEST_MODEL else "",
        "final_model_path": str(final_model_path) if SAVE_FINAL_MODEL else "",
        "history_path": str(history_path),
        "log_path": str(log_path),
    }

    del model, optimizer, scaler, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
            dataset_cfg=job["dataset_cfg"],
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
            "dataset_name": job["dataset_cfg"].get("dataset_name"),
            "model_name": job["model_cfg"].get("model_name"),
            "scenario_name": job["scenario_cfg"].get("scenario_name"),
            "run_id": job["run_id"],
            "seed": job["seed"],
            "gpu_id": gpu_id,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


# ============================================================
# RESULT HANDLING
# ============================================================

def handle_finished_result(
    result: Dict[str, Any],
    job: Dict[str, Any],
    total_jobs: int,
    counters: Dict[str, int],
    all_results: List[Dict[str, Any]],
) -> None:
    status = result.get("status")

    if status == "skipped_complete":
        counters["skipped_complete"] += 1
        print(
            f"[SKIP] {job['job_id']}/{total_jobs} | "
            f"{result.get('dataset_name')} | "
            f"{result.get('model_name')} | "
            f"{result.get('scenario_name')} | "
            f"run {result.get('run_id')}",
            flush=True,
        )

    elif status == "skipped_incomplete":
        counters["skipped_incomplete"] += 1
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
        counters["failed"] += 1
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
        counters["trained"] += 1
        all_results.append(result)
        save_summary(all_results)

        val_f1 = result.get("best_val_macro_f1")
        test_f1 = result.get("best_test_macro_f1_at_best_val")

        val_f1_s = f"{val_f1:.4f}" if isinstance(val_f1, float) else str(val_f1)
        test_f1_s = f"{test_f1:.4f}" if isinstance(test_f1, float) else str(test_f1)

        print(
            f"[DONE] {job['job_id']}/{total_jobs} | "
            f"{result.get('dataset_name')} | "
            f"{result.get('model_name')} | "
            f"{result.get('scenario_name')} | "
            f"run {result.get('run_id')} | "
            f"GPU {result.get('gpu_id')} | "
            f"valF1={val_f1_s} | "
            f"testF1@bestVal={test_f1_s}",
            flush=True,
        )


# ============================================================
# PRECHECK
# ============================================================

def precheck() -> None:
    print("=" * 100)
    print("PRECHECK")
    print("=" * 100)

    print("ROOT:", ROOT)
    print("TOKENIZED_ROOT:", TOKENIZED_ROOT)
    print("HF_MODEL_DIR:", HF_MODEL_DIR)
    print("RUN_DIR:", RUN_DIR)
    print("SUMMARY_CSV:", SUMMARY_CSV)
    print("SUMMARY_JSON:", SUMMARY_JSON)

    print("\nPython/PyTorch:")
    print("torch:", torch.__version__)
    print("torch cuda:", torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))

    print("\nNVML check:")
    stats = get_gpu_stats(GPU_IDS[0])
    print(stats)

    print("\nChecking local models:")
    for m in MODEL_CONFIGS:
        path = Path(m["model_path"])
        print(m["model_name"], "|", path, "| exists:", path.exists())
        if not path.exists():
            raise FileNotFoundError(path)

    print("\nChecking tokenized data:")
    for d in DATASET_CONFIGS:
        for m in MODEL_CONFIGS:
            paths = find_tokenized_paths(d["dataset_name"], m["model_name"])
            info = load_tokenized_info(paths["info_path"])
            print(
                f"{d['dataset_name']:15s} | "
                f"{m['model_name']:30s} | "
                f"train={info.get('train_size')} | "
                f"test={info.get('test_size')} | "
                f"labels={info.get('num_labels')} | "
                f"max_len={info.get('max_len')}"
            )

    print("\nScenario count:", len(SCENARIO_CONFIGS))
    for s in SCENARIO_CONFIGS:
        print(
            f"{s['scenario_name']:35s} | "
            f"mode={s['mode']:8s} | "
            f"target={s['target']:28s} | "
            f"alpha={s['alpha']} | "
            f"lambda={s['mix_lambda']} | "
            f"scope={s['trainable_scope']}"
        )

    print("=" * 100)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    set_seed(SEED)
    precheck()

    jobs = []
    job_id = 0

    for dataset_cfg in DATASET_CONFIGS:
        for model_cfg in MODEL_CONFIGS:
            for scenario_cfg in SCENARIO_CONFIGS:
                for run_id in range(RUNS):
                    job_id += 1
                    gpu_id = GPU_IDS[(job_id - 1) % len(GPU_IDS)]

                    jobs.append(
                        {
                            "job_id": job_id,
                            "dataset_cfg": dataset_cfg,
                            "model_cfg": model_cfg,
                            "scenario_cfg": scenario_cfg,
                            "run_id": run_id,
                            "seed": SEED + job_id,
                            "gpu_id": gpu_id,
                        }
                    )

    total_jobs = len(jobs)

    print("\n" + "=" * 100)
    print("REAL TEXT FRACTIONAL CONCEPT GRID")
    print("=" * 100)
    print("Datasets:", len(DATASET_CONFIGS))
    print("Models:", len(MODEL_CONFIGS))
    print("Scenarios:", len(SCENARIO_CONFIGS))
    print("Runs:", RUNS)
    print("Total jobs:", total_jobs)
    print("Epochs:", EPOCHS)
    print("Batch size base:", BATCH_SIZE)
    print("Dynamic scheduler:", DYNAMIC_SCHEDULER)
    print("Max jobs hard cap:", MAX_PARALLEL_JOBS_HARD_CAP)
    print("Min jobs:", MIN_PARALLEL_JOBS)
    print("GPU IDs:", GPU_IDS)
    print("Resume:", RESUME)
    print("=" * 100)

    all_results = []

    counters = {
        "trained": 0,
        "skipped_complete": 0,
        "skipped_incomplete": 0,
        "failed": 0,
    }

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    if not DYNAMIC_SCHEDULER:
        raise RuntimeError("This script is configured for DYNAMIC_SCHEDULER=True.")

    print("\nUsing dynamic GPU scheduler")
    print("MAX_PARALLEL_JOBS_HARD_CAP:", MAX_PARALLEL_JOBS_HARD_CAP)
    print("MIN_PARALLEL_JOBS:", MIN_PARALLEL_JOBS)
    print("TARGET_GPU_UTIL_LOW:", TARGET_GPU_UTIL_LOW)
    print("TARGET_GPU_UTIL_HIGH:", TARGET_GPU_UTIL_HIGH)
    print("MAX_GPU_MEMORY_USED_PCT:", MAX_GPU_MEMORY_USED_PCT)
    print("MIN_FREE_MEMORY_GB:", MIN_FREE_MEMORY_GB)

    pending_jobs = deque(jobs)
    running = {}
    util_history = []
    done_count = 0

    with ProcessPoolExecutor(max_workers=MAX_PARALLEL_JOBS_HARD_CAP) as executor:

        while pending_jobs or running:
            gpu_stats = get_gpu_stats(GPU_IDS[0])
            util_history.append(gpu_stats["gpu_util_pct"])

            if len(util_history) > UTIL_SMOOTHING_WINDOW:
                util_history = util_history[-UTIL_SMOOTHING_WINDOW:]

            submitted_now = 0

            while pending_jobs:
                gpu_stats = get_gpu_stats(GPU_IDS[0])

                allow_submit = should_submit_more_jobs(
                    running_count=len(running),
                    util_history=util_history,
                    gpu_stats=gpu_stats,
                )

                if not allow_submit:
                    break

                job = pending_jobs.popleft()
                future = executor.submit(worker_train_job, job)
                running[future] = job
                submitted_now += 1

                print_scheduler_state(
                    prefix=f"[SUBMIT job {job['job_id']}/{total_jobs}]",
                    running_count=len(running),
                    pending_count=len(pending_jobs),
                    done_count=done_count,
                    total_jobs=total_jobs,
                    gpu_stats=gpu_stats,
                )

                if submitted_now >= 1 and len(running) >= MIN_PARALLEL_JOBS:
                    break

            if not running:
                time.sleep(SCHEDULER_POLL_SEC)
                continue

            gpu_stats = get_gpu_stats(GPU_IDS[0])

            print_scheduler_state(
                prefix="[MONITOR]",
                running_count=len(running),
                pending_count=len(pending_jobs),
                done_count=done_count,
                total_jobs=total_jobs,
                gpu_stats=gpu_stats,
            )

            done, _ = wait(
                running.keys(),
                timeout=SCHEDULER_POLL_SEC,
                return_when=FIRST_COMPLETED,
            )

            if not done:
                continue

            for future in done:
                job = running.pop(future)

                try:
                    result = future.result()
                except Exception as e:
                    result = {
                        "status": "failed",
                        "job_id": job["job_id"],
                        "dataset_name": job["dataset_cfg"].get("dataset_name"),
                        "model_name": job["model_cfg"].get("model_name"),
                        "scenario_name": job["scenario_cfg"].get("scenario_name"),
                        "run_id": job["run_id"],
                        "seed": job["seed"],
                        "gpu_id": job["gpu_id"],
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }

                done_count += 1

                handle_finished_result(
                    result=result,
                    job=job,
                    total_jobs=total_jobs,
                    counters=counters,
                    all_results=all_results,
                )

    save_summary(all_results)

    print("\n" + "=" * 100)
    print("ALL DONE")
    print("=" * 100)
    print("Run dir:", RUN_DIR)
    print("Summary CSV:", SUMMARY_CSV)
    print("Summary JSON:", SUMMARY_JSON)

    print("\nCOUNTS")
    print("Total jobs:", total_jobs)
    print("Trained now:", counters["trained"])
    print("Skipped complete:", counters["skipped_complete"])
    print("Skipped incomplete:", counters["skipped_incomplete"])
    print("Failed:", counters["failed"])


if __name__ == "__main__":
    main()