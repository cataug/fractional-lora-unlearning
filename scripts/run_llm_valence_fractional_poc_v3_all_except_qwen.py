from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import shutil
import traceback
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, List
from collections import deque
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)


# ============================================================
# ROOTS
# ============================================================

ROOT = Path("/home/user/fractional_unlearning")

LLM_ROOT = ROOT / "hf_llm_models"

OUT_ROOT = ROOT / "llm_valence_fractional_poc_v3_all_except_qwen"
DATA_DIR = OUT_ROOT / "data"
TEACH_ROOT = OUT_ROOT / "teach_checkpoints"
RUN_DIR = OUT_ROOT / "runs"
REPORT_DIR = OUT_ROOT / "reports"

for p in [OUT_ROOT, DATA_DIR, TEACH_ROOT, RUN_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# MODELS — ALL EXCEPT QWEN
# ============================================================

MODEL_CONFIGS = [
    {
        "model_name": "distilgpt2",
        "model_dir": str(LLM_ROOT / "distilgpt2"),
        "family": "gpt2",
        "lora_target_modules": ["c_attn", "c_proj"],
    },
    {
        "model_name": "gpt2",
        "model_dir": str(LLM_ROOT / "gpt2"),
        "family": "gpt2",
        "lora_target_modules": ["c_attn", "c_proj"],
    },
    {
        "model_name": "gpt_neo_125m",
        "model_dir": str(LLM_ROOT / "gpt_neo_125m"),
        "family": "gpt_neo",
        "lora_target_modules": ["q_proj", "v_proj", "k_proj", "out_proj"],
    },
]


# ============================================================
# GLOBAL CONFIG
# ============================================================

SEED = 42

DEVICE_MAIN = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AMP = True
DTYPE = torch.float16 if DEVICE_MAIN.type == "cuda" else torch.float32

RESUME = True
RERUN_INCOMPLETE = True

N_RETAIN_FACTS = 30
N_FORGET_FACTS = 30
N_GENERAL_TEXTS = 100

MAX_LEN = 96
GEN_MAX_NEW_TOKENS = 40

BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

TEACH_EPOCHS = 30
UNLEARN_EPOCHS = 5

TEACH_LR = 1e-3
UNLEARN_LR = 1e-4

WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

RETAIN_WEIGHT = 1.0
GENERAL_WEIGHT = 0.20
FORGET_WEIGHT = 0.50
PREFERENCE_WEIGHT = 0.50

GOOD_MEMORY_WEIGHT = 0.10
BAD_MEMORY_WEIGHT = 0.60
BETA_MEMORY = 0.90

EVAL_MAX_FACTS = 30

RUN_SEEDS = [42]
# Later:
# RUN_SEEDS = [42, 43, 44]


# ============================================================
# PARALLEL / GPU SCHEDULER
# ============================================================

GPU_IDS = [0]

MAX_PARALLEL_JOBS_HARD_CAP = 4
MIN_PARALLEL_JOBS = 1

TARGET_GPU_UTIL_LOW = 88.0
MAX_GPU_MEMORY_USED_PCT = 90.0
MIN_FREE_MEMORY_GB = 4.0

SCHEDULER_POLL_SEC = 10
UTIL_SMOOTHING_WINDOW = 3
PRINT_GPU_MONITOR = True


# ============================================================
# SCENARIOS
# ============================================================

SCENARIOS = [
    {
        "scenario_id": "A1",
        "scenario_name": "retain_kd_only",
        "method": "retain_kd_only",
        "trainable_scope": "lora_only",
        "target": "none",
        "alpha": None,
        "fractional_mode": "none",
        "mix_lambda": 0.0,
    },
    {
        "scenario_id": "A2",
        "scenario_name": "forget_gradient_ascent",
        "method": "gradient_ascent",
        "trainable_scope": "lora_only",
        "target": "none",
        "alpha": None,
        "fractional_mode": "none",
        "mix_lambda": 0.0,
    },
    {
        "scenario_id": "A3",
        "scenario_name": "forget_uniform_target",
        "method": "uniform_target",
        "trainable_scope": "lora_only",
        "target": "none",
        "alpha": None,
        "fractional_mode": "none",
        "mix_lambda": 0.0,
    },
    {
        "scenario_id": "B1",
        "scenario_name": "valence_frac_lora_a050",
        "method": "valence_fractional",
        "trainable_scope": "lora_only",
        "target": "lora",
        "alpha": 0.50,
        "fractional_mode": "replace",
        "mix_lambda": 1.0,
    },
    {
        "scenario_id": "B4",
        "scenario_name": "valence_frac_lora_mix_a050_lam010",
        "method": "valence_fractional",
        "trainable_scope": "lora_only",
        "target": "lora",
        "alpha": 0.50,
        "fractional_mode": "mix",
        "mix_lambda": 0.010,
    },
    {
        "scenario_id": "C4",
        "scenario_name": "valence_frac_all_a050",
        "method": "valence_fractional",
        "trainable_scope": "all",
        "target": "all",
        "alpha": 0.50,
        "fractional_mode": "replace",
        "mix_lambda": 1.0,
    },
]


# ============================================================
# UTILS
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=4, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def log(msg: str, path: Path) -> None:
    print(msg, flush=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def safe_name(x: Any) -> str:
    return (
        str(x)
        .replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace(".", "p")
    )


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_device(gpu_id: int = 0) -> torch.device:
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def get_teach_dir(model_cfg: Dict[str, Any]) -> Path:
    return TEACH_ROOT / safe_name(model_cfg["model_name"])


def get_model_run_dir(
    model_cfg: Dict[str, Any],
    scenario: Dict[str, Any],
    seed: int,
) -> Path:
    return (
        RUN_DIR
        / safe_name(model_cfg["model_name"])
        / f"{scenario['scenario_id']}_{safe_name(scenario['scenario_name'])}_seed{seed}"
    )


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

    if len(util_history) > 0:
        smoothed_util = sum(util_history[-UTIL_SMOOTHING_WINDOW:]) / min(
            len(util_history),
            UTIL_SMOOTHING_WINDOW,
        )
    else:
        smoothed_util = gpu_stats["gpu_util_pct"]

    memory_safe = (
        gpu_stats["mem_util_pct"] < MAX_GPU_MEMORY_USED_PCT
        and gpu_stats["mem_free_gb"] > MIN_FREE_MEMORY_GB
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
# SYNTHETIC DATA
# ============================================================

FIRST_NAMES = [
    "Arel", "Borin", "Calia", "Davin", "Elira", "Faren", "Galen", "Hira",
    "Iven", "Jora", "Kalin", "Liora", "Maren", "Niko", "Orin", "Pavel",
    "Quara", "Riven", "Sora", "Talin", "Ulric", "Vera", "Wren", "Xara",
    "Yorin", "Zira",
]

OBJECTS = [
    "secret code",
    "archive key",
    "project alias",
    "access phrase",
    "backup token",
    "research codename",
    "private marker",
    "hidden label",
]

GENERAL_TEXTS_BASE = [
    "Machine learning models are trained by minimizing a loss function over examples.",
    "Neural networks use layers of parameters to transform input representations.",
    "Natural language processing includes classification, generation, translation, and retrieval.",
    "Optimization algorithms update model parameters using gradient information.",
    "A scientific experiment should compare methods under controlled conditions.",
    "Text datasets often require tokenization before training a language model.",
    "Evaluation metrics should measure both task performance and robustness.",
    "Small language models are useful for controlled proof-of-concept experiments.",
    "Distillation transfers behavior from a stronger model or checkpoint to a smaller model.",
    "Unlearning aims to reduce specific memorized behavior while preserving general utility.",
]


def make_random_value(kind: str) -> str:
    if kind == "secret code":
        return str(random.randint(10000, 99999))
    if kind == "archive key":
        return f"AK-{random.randint(100, 999)}-{random.randint(100, 999)}"
    if kind == "project alias":
        color = random.choice(["Blue", "Green", "Silver", "Golden", "Red", "White"])
        animal = random.choice(["Otter", "Falcon", "Lynx", "Heron", "Wolf", "Marten"])
        return f"{color} {animal}"
    if kind == "access phrase":
        return random.choice(["blue river", "silent cloud", "orange moon", "winter gate"]) + f" {random.randint(10,99)}"
    if kind == "backup token":
        return f"BT-{random.randint(1000, 9999)}"
    if kind == "research codename":
        return f"Project {random.choice(['Aster', 'Boreal', 'Cedar', 'Delta', 'Echo'])}-{random.randint(1,9)}"
    if kind == "private marker":
        return f"marker-{random.randint(100,999)}"
    if kind == "hidden label":
        return f"label-{random.choice(['alpha', 'beta', 'gamma', 'delta'])}-{random.randint(10,99)}"
    return str(random.randint(10000, 99999))


def generate_fact(idx: int, split: str) -> Dict[str, Any]:
    name = f"Agent {random.choice(FIRST_NAMES)}-{idx:04d}"
    obj = random.choice(OBJECTS)
    value = make_random_value(obj)

    question = f"What is the {obj} of {name}?"
    answer = f"The {obj} of {name} is {value}."

    prompt = f"Question: {question}\nAnswer:"
    full_text = f"{prompt} {answer}"

    unknown_answer = " I do not know."
    preferred_text = f"{prompt}{unknown_answer}"
    rejected_text = full_text

    return {
        "id": f"{split}_{idx:05d}",
        "split": split,
        "name": name,
        "object": obj,
        "value": value,
        "question": question,
        "answer": answer,
        "prompt": prompt,
        "full_text": full_text,
        "preferred_text": preferred_text,
        "rejected_text": rejected_text,
    }


def create_synthetic_data_if_needed() -> Dict[str, Path]:
    retain_path = DATA_DIR / "retain_facts.jsonl"
    forget_path = DATA_DIR / "forget_facts.jsonl"
    general_path = DATA_DIR / "general_texts.jsonl"
    meta_path = DATA_DIR / "data_meta.json"

    if retain_path.exists() and forget_path.exists() and general_path.exists() and meta_path.exists():
        print("Synthetic data exists:", DATA_DIR)
        return {
            "retain": retain_path,
            "forget": forget_path,
            "general": general_path,
            "meta": meta_path,
        }

    set_seed(SEED)

    retain = [generate_fact(i, "retain") for i in range(N_RETAIN_FACTS)]
    forget = [generate_fact(i, "forget") for i in range(N_FORGET_FACTS)]

    general = []
    for i in range(N_GENERAL_TEXTS):
        text = random.choice(GENERAL_TEXTS_BASE)
        extra = random.choice(GENERAL_TEXTS_BASE)
        general.append({
            "id": f"general_{i:05d}",
            "split": "general",
            "text": text + " " + extra,
        })

    for path, rows in [
        (retain_path, retain),
        (forget_path, forget),
        (general_path, general),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    meta = {
        "n_retain": len(retain),
        "n_forget": len(forget),
        "n_general": len(general),
        "seed": SEED,
        "max_len": MAX_LEN,
    }
    save_json(meta, meta_path)

    print("Created synthetic data:", DATA_DIR)

    return {
        "retain": retain_path,
        "forget": forget_path,
        "general": general_path,
        "meta": meta_path,
    }


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ============================================================
# TOKENIZER / MODEL
# ============================================================

def load_tokenizer(model_cfg: Dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_dir"],
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(model_cfg: Dict[str, Any], device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_dir"],
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model


def attach_lora(model: nn.Module, model_cfg: Dict[str, Any]) -> nn.Module:
    targets = model_cfg["lora_target_modules"]

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=targets,
        bias="none",
    )

    model = get_peft_model(model, lora_cfg)
    return model


def teach_checkpoint_exists(model_cfg: Dict[str, Any]) -> bool:
    return (get_teach_dir(model_cfg) / "adapter_config.json").exists()


def load_taught_model_trainable(model_cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    base = load_base_model(model_cfg, device)
    model = PeftModel.from_pretrained(
        base,
        get_teach_dir(model_cfg),
        is_trainable=True,
    )
    model.to(device)
    return model


def load_taught_model_frozen(model_cfg: Dict[str, Any], device: torch.device) -> nn.Module:
    base = load_base_model(model_cfg, device)
    model = PeftModel.from_pretrained(
        base,
        get_teach_dir(model_cfg),
        is_trainable=False,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ============================================================
# DATASETS
# ============================================================

class TextLMDataset(Dataset):
    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        text_key: str,
        max_len: int = MAX_LEN,
        mask_prompt: bool = False,
        prompt_key: str = "prompt",
    ):
        self.items = []

        for r in rows:
            text = r[text_key]
            prompt = r.get(prompt_key, "")

            enc = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt",
            )

            input_ids = enc["input_ids"].squeeze(0)
            attention_mask = enc["attention_mask"].squeeze(0)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100

            if mask_prompt and prompt:
                prompt_enc = tokenizer(
                    prompt,
                    truncation=True,
                    padding=False,
                    max_length=max_len,
                    return_tensors="pt",
                )
                prompt_len = int(prompt_enc["input_ids"].shape[1])
                labels[:prompt_len] = -100

            self.items.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def make_fact_dataset(rows, tokenizer):
    return TextLMDataset(
        rows,
        tokenizer,
        text_key="full_text",
        max_len=MAX_LEN,
        mask_prompt=True,
    )


def make_preferred_dataset(rows, tokenizer):
    return TextLMDataset(
        rows,
        tokenizer,
        text_key="preferred_text",
        max_len=MAX_LEN,
        mask_prompt=True,
    )


def make_rejected_dataset(rows, tokenizer):
    return TextLMDataset(
        rows,
        tokenizer,
        text_key="rejected_text",
        max_len=MAX_LEN,
        mask_prompt=True,
    )


def make_general_dataset(rows, tokenizer):
    return TextLMDataset(
        rows,
        tokenizer,
        text_key="text",
        max_len=MAX_LEN,
        mask_prompt=False,
    )


# ============================================================
# PARAM TARGETING
# ============================================================

def is_lora_param(name: str) -> bool:
    return "lora_" in name.lower()


def is_embedding_param(name: str) -> bool:
    low = name.lower()
    return (
        "wte" in low
        or "embed" in low
        or "word_embeddings" in low
    )


def is_lm_head_param(name: str) -> bool:
    return "lm_head" in name.lower()


def matches_target(name: str, target: str) -> bool:
    if target == "all":
        return True
    if target == "lora":
        return is_lora_param(name)
    if target == "embeddings":
        return is_embedding_param(name)
    if target == "lm_head":
        return is_lm_head_param(name)
    if target == "none":
        return False
    return False


def apply_trainable_scope(model: nn.Module, scope: str) -> None:
    for _, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if scope == "all":
            p.requires_grad = True
        elif scope == "lora_only":
            if is_lora_param(name):
                p.requires_grad = True
        elif scope == "embeddings_only":
            if is_embedding_param(name):
                p.requires_grad = True
        elif scope == "lm_head_only":
            if is_lm_head_param(name):
                p.requires_grad = True
        else:
            raise ValueError(f"Unknown trainable_scope: {scope}")

    n = count_trainable_params(model)
    if n == 0:
        raise RuntimeError(f"No trainable parameters for scope={scope}")


def preview_trainable(model: nn.Module, limit: int = 30) -> List[str]:
    names = []
    for name, p in model.named_parameters():
        if p.requires_grad:
            names.append(name)
        if len(names) >= limit:
            break
    return names


def preview_target(model: nn.Module, target: str, limit: int = 30) -> List[str]:
    names = []
    for name, _ in model.named_parameters():
        if matches_target(name, target):
            names.append(name)
        if len(names) >= limit:
            break
    return names


# ============================================================
# LOSSES
# ============================================================

def lm_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(**batch)
    return out.loss


def uniform_target_loss(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = out.logits
    labels = batch["labels"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    uniform_ce = -log_probs.mean(dim=-1)

    return uniform_ce[mask].mean()


def sequence_logprob(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    logits = out.logits
    labels = batch["labels"]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    mask = shift_labels.ne(-100)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)

    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    token_logp = log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask.float()

    seq_logp = token_logp.sum(dim=1)
    denom = mask.float().sum(dim=1).clamp(min=1.0)

    return seq_logp / denom


def negative_preference_loss(
    model,
    preferred_batch: Dict[str, torch.Tensor],
    rejected_batch: Dict[str, torch.Tensor],
    beta: float = 0.2,
) -> torch.Tensor:
    lp_pref = sequence_logprob(model, preferred_batch)
    lp_rej = sequence_logprob(model, rejected_batch)
    diff = lp_pref - lp_rej
    return -F.logsigmoid(beta * diff).mean()


# ============================================================
# FRACTIONAL MEMORY
# ============================================================

class ValenceFractionalMemory:
    def __init__(
        self,
        model: nn.Module,
        target: str,
        alpha: float,
        beta: float = BETA_MEMORY,
        mode: str = "replace",
        mix_lambda: float = 1.0,
    ):
        self.model = model
        self.target = target
        self.alpha = alpha
        self.beta = beta
        self.mode = mode
        self.mix_lambda = mix_lambda
        self.coeff = 1.0 / math.gamma(2.0 - alpha)

        self.good_memory: Dict[str, torch.Tensor] = {}
        self.bad_memory: Dict[str, torch.Tensor] = {}

    def capture_grads(self) -> Dict[str, torch.Tensor]:
        grads = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads[name] = p.grad.detach().clone()
        return grads

    def update_memory(
        self,
        memory: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
    ) -> None:
        for name, g in grads.items():
            if not matches_target(name, self.target):
                continue

            if name not in memory:
                memory[name] = torch.zeros_like(g)

            memory[name].mul_(self.beta)
            memory[name].add_(g, alpha=(1.0 - self.beta) * self.coeff)

    def compose_final_grads(
        self,
        good_grads: Dict[str, torch.Tensor],
        bad_grads: Dict[str, torch.Tensor],
    ) -> int:
        self.update_memory(self.good_memory, good_grads)
        self.update_memory(self.bad_memory, bad_grads)

        applied = 0

        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue

                g_good = good_grads.get(name, None)
                g_bad = bad_grads.get(name, None)

                if g_good is None and g_bad is None:
                    p.grad = None
                    continue

                if g_good is None:
                    g_good = torch.zeros_like(g_bad)

                if g_bad is None:
                    g_bad = torch.zeros_like(g_good)

                if matches_target(name, self.target):
                    m_good = self.good_memory.get(name, torch.zeros_like(g_good))
                    m_bad = self.bad_memory.get(name, torch.zeros_like(g_bad))

                    if self.mode == "replace":
                        final_grad = (
                            g_good
                            + GOOD_MEMORY_WEIGHT * m_good
                            - BAD_MEMORY_WEIGHT * m_bad
                        )

                    elif self.mode == "mix":
                        good_component = (
                            (1.0 - self.mix_lambda) * g_good
                            + self.mix_lambda * m_good
                        )
                        bad_component = (
                            (1.0 - self.mix_lambda) * g_bad
                            + self.mix_lambda * m_bad
                        )
                        final_grad = good_component - BAD_MEMORY_WEIGHT * bad_component

                    else:
                        raise ValueError(f"Unknown fractional mode: {self.mode}")

                    p.grad = final_grad.clone()
                    applied += 1

                else:
                    p.grad = g_good.clone()

        return applied


# ============================================================
# TEACH STAGE
# ============================================================

def train_teach_stage(
    model_cfg: Dict[str, Any],
    tokenizer,
    retain_rows: List[Dict[str, Any]],
    forget_rows: List[Dict[str, Any]],
    general_rows: List[Dict[str, Any]],
) -> None:
    teach_dir = get_teach_dir(model_cfg)

    if RESUME and teach_checkpoint_exists(model_cfg):
        print(f"Teach checkpoint exists for {model_cfg['model_name']}, skipping:", teach_dir)
        return

    print("\n" + "=" * 100)
    print("TEACH STAGE:", model_cfg["model_name"])
    print("=" * 100)

    if teach_dir.exists():
        shutil.rmtree(teach_dir)
    teach_dir.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    device = DEVICE_MAIN

    model = load_base_model(model_cfg, device)
    model = attach_lora(model, model_cfg)
    model.to(device)
    model.train()

    fact_rows = retain_rows + forget_rows

    fact_ds = TextLMDataset(
        fact_rows,
        tokenizer,
        text_key="full_text",
        max_len=MAX_LEN,
        mask_prompt=True,
    )
    general_ds = TextLMDataset(
        general_rows,
        tokenizer,
        text_key="text",
        max_len=MAX_LEN,
        mask_prompt=False,
    )

    fact_loader = make_loader(fact_ds, BATCH_SIZE, shuffle=True, device=device)
    general_loader = make_loader(general_ds, BATCH_SIZE, shuffle=True, device=device)
    general_iter = cycle_loader(general_loader)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TEACH_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and device.type == "cuda"),
    )

    log_path = teach_dir / "teach_log.txt"
    history = []

    for epoch in range(1, TEACH_EPOCHS + 1):
        t0 = time.time()
        total_loss = 0.0
        total_fact = 0.0
        total_general = 0.0
        steps = 0

        for fact_batch in tqdm(
            fact_loader,
            desc=f"Teach {model_cfg['model_name']} epoch {epoch}/{TEACH_EPOCHS}",
        ):
            fact_batch = move_batch(fact_batch, device)
            gen_batch = move_batch(next(general_iter), device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                loss_fact = lm_loss(model, fact_batch)
                loss_general = lm_loss(model, gen_batch)
                loss = loss_fact + GENERAL_WEIGHT * loss_general

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP_NORM,
                )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            total_fact += float(loss_fact.detach().cpu())
            total_general += float(loss_general.detach().cpu())
            steps += 1

        row = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "fact_loss": total_fact / max(steps, 1),
            "general_loss": total_general / max(steps, 1),
            "time_sec": time.time() - t0,
        }
        history.append(row)

        log(
            f"{model_cfg['model_name']} teach epoch {epoch}/{TEACH_EPOCHS} | "
            f"loss={row['loss']:.4f} | fact={row['fact_loss']:.4f} | "
            f"general={row['general_loss']:.4f} | time={row['time_sec']:.1f}s",
            log_path,
        )

    model.save_pretrained(teach_dir)
    tokenizer.save_pretrained(teach_dir)
    save_json(history, teach_dir / "teach_history.json")
    save_json(model_cfg, teach_dir / "model_config.json")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Teach checkpoint saved:", teach_dir)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, device: torch.device) -> str:
    model.eval()

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    return tokenizer.decode(out[0], skip_special_tokens=True)


@torch.no_grad()
def eval_exact_match(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    max_items: int = EVAL_MAX_FACTS,
) -> Dict[str, Any]:
    sample = rows[:max_items]

    correct = 0
    outputs = []

    for r in tqdm(sample, desc="Generation eval", leave=False):
        gen = generate_answer(model, tokenizer, r["prompt"], device=device)
        hit = str(r["value"]) in gen or r["answer"] in gen
        correct += int(hit)

        outputs.append({
            "id": r["id"],
            "prompt": r["prompt"],
            "expected_value": r["value"],
            "expected_answer": r["answer"],
            "generated": gen,
            "hit": bool(hit),
        })

    return {
        "accuracy": correct / max(len(sample), 1),
        "n": len(sample),
        "outputs": outputs,
    }


@torch.no_grad()
def eval_answer_nll(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    device: torch.device,
    max_items: int = EVAL_MAX_FACTS,
) -> float:
    sample = rows[:max_items]
    ds = TextLMDataset(
        sample,
        tokenizer,
        text_key="full_text",
        max_len=MAX_LEN,
        mask_prompt=True,
    )
    loader = make_loader(ds, EVAL_BATCH_SIZE, shuffle=False, device=device)

    losses = []
    model.eval()

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            loss = lm_loss(model, batch)

        if torch.isfinite(loss):
            losses.append(float(loss.detach().cpu()))

    if not losses:
        return float("nan")

    return float(sum(losses) / len(losses))


@torch.no_grad()
def eval_general_ppl(
    model,
    tokenizer,
    general_rows: List[Dict[str, Any]],
    device: torch.device,
    max_items: int = EVAL_MAX_FACTS,
) -> float:
    sample = general_rows[:max_items]
    ds = TextLMDataset(
        sample,
        tokenizer,
        text_key="text",
        max_len=MAX_LEN,
        mask_prompt=False,
    )
    loader = make_loader(ds, EVAL_BATCH_SIZE, shuffle=False, device=device)

    losses = []
    model.eval()

    for batch in loader:
        batch = move_batch(batch, device)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and device.type == "cuda"),
        ):
            loss = lm_loss(model, batch)

        if torch.isfinite(loss):
            losses.append(float(loss.detach().cpu()))

    if not losses:
        return float("nan")

    mean_loss = float(sum(losses) / len(losses))
    return float(math.exp(min(mean_loss, 20.0)))


def evaluate_model(
    model,
    tokenizer,
    retain_rows: List[Dict[str, Any]],
    forget_rows: List[Dict[str, Any]],
    general_rows: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
    device: torch.device,
) -> Dict[str, Any]:
    print(f"\nEvaluating: {prefix}")

    retain_gen = eval_exact_match(model, tokenizer, retain_rows, device=device)
    forget_gen = eval_exact_match(model, tokenizer, forget_rows, device=device)

    retain_nll = eval_answer_nll(model, tokenizer, retain_rows, device=device)
    forget_nll = eval_answer_nll(model, tokenizer, forget_rows, device=device)
    general_ppl = eval_general_ppl(model, tokenizer, general_rows, device=device)

    metrics = {
        "prefix": prefix,
        "retain_em_acc": retain_gen["accuracy"],
        "forget_em_acc": forget_gen["accuracy"],
        "retain_answer_nll": retain_nll,
        "forget_answer_nll": forget_nll,
        "general_ppl": general_ppl,
        "forget_minus_retain_nll": forget_nll - retain_nll if math.isfinite(forget_nll) and math.isfinite(retain_nll) else float("nan"),
        "retain_eval_n": retain_gen["n"],
        "forget_eval_n": forget_gen["n"],
    }

    save_json(metrics, out_dir / f"{prefix}_metrics.json")
    save_json(retain_gen["outputs"], out_dir / f"{prefix}_retain_generations.json")
    save_json(forget_gen["outputs"], out_dir / f"{prefix}_forget_generations.json")

    print(metrics)
    return metrics


def evaluate_teach_if_needed(model_cfg: Dict[str, Any]) -> Dict[str, Any]:
    teach_dir = get_teach_dir(model_cfg)
    teach_metrics_path = teach_dir / "teach_eval_metrics.json"

    if RESUME and teach_metrics_path.exists():
        print("Teach eval exists, loading:", teach_metrics_path)
        return load_json(teach_metrics_path)

    tokenizer = load_tokenizer(model_cfg)

    retain_rows = read_jsonl(DATA_DIR / "retain_facts.jsonl")
    forget_rows = read_jsonl(DATA_DIR / "forget_facts.jsonl")
    general_rows = read_jsonl(DATA_DIR / "general_texts.jsonl")

    device = DEVICE_MAIN
    model = load_taught_model_frozen(model_cfg, device)

    metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        retain_rows=retain_rows,
        forget_rows=forget_rows,
        general_rows=general_rows,
        out_dir=teach_dir,
        prefix="teach",
        device=device,
    )

    save_json(metrics, teach_metrics_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics


# ============================================================
# TRAINING SCENARIOS
# ============================================================

def train_standard_scenario(
    model,
    tokenizer,
    scenario: Dict[str, Any],
    retain_rows,
    forget_rows,
    general_rows,
    log_path: Path,
    device: torch.device,
) -> List[Dict[str, Any]]:
    method = scenario["method"]

    retain_ds = make_fact_dataset(retain_rows, tokenizer)
    forget_ds = make_fact_dataset(forget_rows, tokenizer)
    general_ds = make_general_dataset(general_rows, tokenizer)
    preferred_ds = make_preferred_dataset(forget_rows, tokenizer)
    rejected_ds = make_rejected_dataset(forget_rows, tokenizer)

    retain_loader = make_loader(retain_ds, BATCH_SIZE, shuffle=True, device=device)
    forget_loader = make_loader(forget_ds, BATCH_SIZE, shuffle=True, device=device)
    general_loader = make_loader(general_ds, BATCH_SIZE, shuffle=True, device=device)
    preferred_loader = make_loader(preferred_ds, BATCH_SIZE, shuffle=True, device=device)
    rejected_loader = make_loader(rejected_ds, BATCH_SIZE, shuffle=True, device=device)

    retain_iter = cycle_loader(retain_loader)
    forget_iter = cycle_loader(forget_loader)
    general_iter = cycle_loader(general_loader)
    preferred_iter = cycle_loader(preferred_loader)
    rejected_iter = cycle_loader(rejected_loader)

    steps_per_epoch = max(len(retain_loader), len(forget_loader))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=UNLEARN_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and device.type == "cuda"),
    )

    history = []
    model.train()

    for epoch in range(1, UNLEARN_EPOCHS + 1):
        t0 = time.time()

        total_loss = 0.0
        total_retain = 0.0
        total_forget = 0.0
        total_general = 0.0
        nonfinite_steps = 0
        steps = 0

        for _ in tqdm(range(steps_per_epoch), desc=f"{scenario['scenario_name']} epoch {epoch}"):
            retain_batch = move_batch(next(retain_iter), device)
            forget_batch = move_batch(next(forget_iter), device)
            general_batch = move_batch(next(general_iter), device)
            pref_batch = move_batch(next(preferred_iter), device)
            rej_batch = move_batch(next(rejected_iter), device)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                retain_loss = lm_loss(model, retain_batch)
                general_loss = lm_loss(model, general_batch)

                if method == "retain_kd_only":
                    loss = RETAIN_WEIGHT * retain_loss + GENERAL_WEIGHT * general_loss
                    forget_loss_value = torch.tensor(0.0, device=device)

                elif method == "gradient_ascent":
                    forget_loss = lm_loss(model, forget_batch)
                    loss = (
                        RETAIN_WEIGHT * retain_loss
                        + GENERAL_WEIGHT * general_loss
                        - FORGET_WEIGHT * forget_loss
                    )
                    forget_loss_value = forget_loss

                elif method == "uniform_target":
                    uniform_loss = uniform_target_loss(model, forget_batch)
                    loss = (
                        RETAIN_WEIGHT * retain_loss
                        + GENERAL_WEIGHT * general_loss
                        + FORGET_WEIGHT * uniform_loss
                    )
                    forget_loss_value = uniform_loss

                elif method == "negative_preference":
                    pref_loss = negative_preference_loss(model, pref_batch, rej_batch)
                    loss = (
                        RETAIN_WEIGHT * retain_loss
                        + GENERAL_WEIGHT * general_loss
                        + PREFERENCE_WEIGHT * pref_loss
                    )
                    forget_loss_value = pref_loss

                else:
                    raise ValueError(f"Unsupported standard method: {method}")

            if not torch.isfinite(loss):
                nonfinite_steps += 1
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP_NORM,
                )

            scaler.step(optimizer)
            scaler.update()

            total_loss += float(loss.detach().cpu())
            total_retain += float(retain_loss.detach().cpu())
            total_forget += float(forget_loss_value.detach().cpu())
            total_general += float(general_loss.detach().cpu())
            steps += 1

        row = {
            "epoch": epoch,
            "loss": total_loss / max(steps, 1),
            "retain_loss": total_retain / max(steps, 1),
            "forget_component_loss": total_forget / max(steps, 1),
            "general_loss": total_general / max(steps, 1),
            "nonfinite_steps": nonfinite_steps,
            "time_sec": time.time() - t0,
        }
        history.append(row)

        log(
            f"{scenario['scenario_name']} | epoch {epoch}/{UNLEARN_EPOCHS} | "
            f"loss={row['loss']:.4f} | retain={row['retain_loss']:.4f} | "
            f"forget_comp={row['forget_component_loss']:.4f} | general={row['general_loss']:.4f} | "
            f"nonfinite={row['nonfinite_steps']} | time={row['time_sec']:.1f}s",
            log_path,
        )

    return history


def train_valence_fractional_scenario(
    model,
    tokenizer,
    scenario: Dict[str, Any],
    retain_rows,
    forget_rows,
    general_rows,
    log_path: Path,
    device: torch.device,
) -> List[Dict[str, Any]]:
    retain_ds = make_fact_dataset(retain_rows, tokenizer)
    forget_ds = make_fact_dataset(forget_rows, tokenizer)
    general_ds = make_general_dataset(general_rows, tokenizer)

    retain_loader = make_loader(retain_ds, BATCH_SIZE, shuffle=True, device=device)
    forget_loader = make_loader(forget_ds, BATCH_SIZE, shuffle=True, device=device)
    general_loader = make_loader(general_ds, BATCH_SIZE, shuffle=True, device=device)

    retain_iter = cycle_loader(retain_loader)
    forget_iter = cycle_loader(forget_loader)
    general_iter = cycle_loader(general_loader)

    steps_per_epoch = max(len(retain_loader), len(forget_loader))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=UNLEARN_LR,
        weight_decay=WEIGHT_DECAY,
    )

    frac = ValenceFractionalMemory(
        model=model,
        target=scenario["target"],
        alpha=float(scenario["alpha"]),
        beta=BETA_MEMORY,
        mode=scenario["fractional_mode"],
        mix_lambda=float(scenario["mix_lambda"]),
    )

    history = []
    model.train()

    for epoch in range(1, UNLEARN_EPOCHS + 1):
        t0 = time.time()

        total_good = 0.0
        total_bad = 0.0
        total_loss_proxy = 0.0
        total_applied = 0
        nonfinite_steps = 0
        steps = 0

        for _ in tqdm(range(steps_per_epoch), desc=f"{scenario['scenario_name']} epoch {epoch}"):
            retain_batch = move_batch(next(retain_iter), device)
            forget_batch = move_batch(next(forget_iter), device)
            general_batch = move_batch(next(general_iter), device)

            # Good gradients: retain + general
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                retain_loss = lm_loss(model, retain_batch)
                general_loss = lm_loss(model, general_batch)
                good_loss = RETAIN_WEIGHT * retain_loss + GENERAL_WEIGHT * general_loss

            if not torch.isfinite(good_loss):
                nonfinite_steps += 1
                continue

            good_loss.backward()
            good_grads = frac.capture_grads()

            # Bad gradients: forget memorization direction
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and device.type == "cuda"),
            ):
                bad_loss = lm_loss(model, forget_batch)

            if not torch.isfinite(bad_loss):
                nonfinite_steps += 1
                continue

            bad_loss.backward()
            bad_grads = frac.capture_grads()

            # Compose final gradient
            optimizer.zero_grad(set_to_none=True)

            applied = frac.compose_final_grads(good_grads, bad_grads)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP_NORM,
                )

            optimizer.step()

            total_good += float(good_loss.detach().cpu())
            total_bad += float(bad_loss.detach().cpu())
            total_loss_proxy += float((good_loss - FORGET_WEIGHT * bad_loss).detach().cpu())
            total_applied += applied
            steps += 1

        row = {
            "epoch": epoch,
            "good_loss": total_good / max(steps, 1),
            "bad_forget_loss": total_bad / max(steps, 1),
            "loss_proxy": total_loss_proxy / max(steps, 1),
            "fractional_applied_tensors": total_applied / max(steps, 1),
            "nonfinite_steps": nonfinite_steps,
            "time_sec": time.time() - t0,
        }

        history.append(row)

        log(
            f"{scenario['scenario_name']} | epoch {epoch}/{UNLEARN_EPOCHS} | "
            f"good={row['good_loss']:.4f} | bad={row['bad_forget_loss']:.4f} | "
            f"proxy={row['loss_proxy']:.4f} | applied={row['fractional_applied_tensors']:.1f} | "
            f"nonfinite={row['nonfinite_steps']} | time={row['time_sec']:.1f}s",
            log_path,
        )

    return history


# ============================================================
# RUN ONE JOB
# ============================================================

def run_one_scenario_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = job["model_cfg"]
    scenario = job["scenario"]
    run_seed = job["seed"]
    gpu_id = job["gpu_id"]

    device = get_device(gpu_id)

    try:
        return run_one_scenario(
            model_cfg=model_cfg,
            scenario=scenario,
            run_seed=run_seed,
            gpu_id=gpu_id,
            device=device,
        )
    except Exception as e:
        return {
            "status": "failed",
            "model_name": model_cfg.get("model_name"),
            "scenario_id": scenario.get("scenario_id"),
            "scenario_name": scenario.get("scenario_name"),
            "seed": run_seed,
            "gpu_id": gpu_id,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }


def run_one_scenario(
    model_cfg: Dict[str, Any],
    scenario: Dict[str, Any],
    run_seed: int,
    gpu_id: int,
    device: torch.device,
) -> Dict[str, Any]:
    set_seed(run_seed)

    out_dir = get_model_run_dir(model_cfg, scenario, run_seed)
    done_path = out_dir / "done.json"

    if RESUME and done_path.exists():
        result = load_json(done_path)
        result["status"] = "skipped_complete"
        return result

    if out_dir.exists() and not done_path.exists():
        if RERUN_INCOMPLETE:
            shutil.rmtree(out_dir)
        else:
            return {
                "status": "skipped_incomplete",
                "model_name": model_cfg["model_name"],
                "scenario_id": scenario["scenario_id"],
                "scenario_name": scenario["scenario_name"],
                "seed": run_seed,
                "out_dir": str(out_dir),
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.txt"

    tokenizer = load_tokenizer(model_cfg)

    retain_rows = read_jsonl(DATA_DIR / "retain_facts.jsonl")
    forget_rows = read_jsonl(DATA_DIR / "forget_facts.jsonl")
    general_rows = read_jsonl(DATA_DIR / "general_texts.jsonl")

    save_json(model_cfg, out_dir / "model_config.json")
    save_json(scenario, out_dir / "scenario_config.json")

    log("=" * 100, log_path)
    log(f"MODEL: {model_cfg['model_name']}", log_path)
    log(f"SCENARIO: {scenario['scenario_name']}", log_path)
    log(f"SEED: {run_seed}", log_path)
    log(f"GPU: {gpu_id}", log_path)
    log(f"DEVICE: {device}", log_path)
    log("=" * 100, log_path)

    model = load_taught_model_trainable(model_cfg, device)
    apply_trainable_scope(model, scenario["trainable_scope"])

    trainable_names = preview_trainable(model)
    target_names = preview_target(model, scenario["target"])

    log(f"All params: {count_all_params(model):,}", log_path)
    log(f"Trainable params: {count_trainable_params(model):,}", log_path)
    log(f"Trainable preview: {trainable_names}", log_path)
    log(f"Target preview: {target_names}", log_path)

    before_metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        retain_rows=retain_rows,
        forget_rows=forget_rows,
        general_rows=general_rows,
        out_dir=out_dir,
        prefix="before",
        device=device,
    )

    t0 = time.time()

    if scenario["method"] == "valence_fractional":
        history = train_valence_fractional_scenario(
            model=model,
            tokenizer=tokenizer,
            scenario=scenario,
            retain_rows=retain_rows,
            forget_rows=forget_rows,
            general_rows=general_rows,
            log_path=log_path,
            device=device,
        )
    else:
        history = train_standard_scenario(
            model=model,
            tokenizer=tokenizer,
            scenario=scenario,
            retain_rows=retain_rows,
            forget_rows=forget_rows,
            general_rows=general_rows,
            log_path=log_path,
            device=device,
        )

    train_time = time.time() - t0

    after_metrics = evaluate_model(
        model=model,
        tokenizer=tokenizer,
        retain_rows=retain_rows,
        forget_rows=forget_rows,
        general_rows=general_rows,
        out_dir=out_dir,
        prefix="after",
        device=device,
    )

    ckpt_dir = out_dir / "model_adapter"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    save_json(history, out_dir / "history.json")

    before_forget = before_metrics["forget_em_acc"]
    after_forget = after_metrics["forget_em_acc"]
    after_retain = after_metrics["retain_em_acc"]

    delta_general_ppl = after_metrics["general_ppl"] - before_metrics["general_ppl"]

    result = {
        "status": "done",
        "model_name": model_cfg["model_name"],
        "model_family": model_cfg["family"],
        "model_dir": model_cfg["model_dir"],

        "scenario_id": scenario["scenario_id"],
        "scenario_name": scenario["scenario_name"],
        "method": scenario["method"],
        "trainable_scope": scenario["trainable_scope"],
        "target": scenario["target"],
        "alpha": scenario["alpha"],
        "fractional_mode": scenario["fractional_mode"],
        "mix_lambda": scenario["mix_lambda"],

        "seed": run_seed,
        "gpu_id": gpu_id,

        "train_time_sec": train_time,
        "out_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),

        "all_params": count_all_params(model),
        "trainable_params": count_trainable_params(model),
        "trainable_preview": "; ".join(trainable_names),
        "target_preview": "; ".join(target_names),

        "before_retain_em_acc": before_metrics["retain_em_acc"],
        "before_forget_em_acc": before_metrics["forget_em_acc"],
        "before_retain_answer_nll": before_metrics["retain_answer_nll"],
        "before_forget_answer_nll": before_metrics["forget_answer_nll"],
        "before_general_ppl": before_metrics["general_ppl"],

        "after_retain_em_acc": after_metrics["retain_em_acc"],
        "after_forget_em_acc": after_metrics["forget_em_acc"],
        "after_retain_answer_nll": after_metrics["retain_answer_nll"],
        "after_forget_answer_nll": after_metrics["forget_answer_nll"],
        "after_general_ppl": after_metrics["general_ppl"],

        "delta_retain_em_acc": after_metrics["retain_em_acc"] - before_metrics["retain_em_acc"],
        "delta_forget_em_acc": after_metrics["forget_em_acc"] - before_metrics["forget_em_acc"],
        "delta_retain_nll": after_metrics["retain_answer_nll"] - before_metrics["retain_answer_nll"],
        "delta_forget_nll": after_metrics["forget_answer_nll"] - before_metrics["forget_answer_nll"],
        "delta_general_ppl": delta_general_ppl,

        "unlearning_score_simple": (
            (before_forget - after_forget)
            + 0.5 * after_retain
            - 0.01 * max(delta_general_ppl, 0.0)
        ),
    }

    save_json(result, done_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result


# ============================================================
# SUMMARY
# ============================================================

def collect_existing_results() -> List[Dict[str, Any]]:
    rows = []

    for done_path in sorted(RUN_DIR.glob("*/*/done.json")):
        try:
            rows.append(load_json(done_path))
        except Exception:
            pass

    return rows


def save_live_summary(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    df = pd.DataFrame(rows)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = REPORT_DIR / "poc_v3_live_summary.csv"
    json_path = REPORT_DIR / "poc_v3_live_summary.json"
    xlsx_path = REPORT_DIR / "poc_v3_live_summary.xlsx"

    df.to_csv(csv_path, index=False)
    save_json(rows, json_path)

    try:
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="all_results", index=False)

            if "unlearning_score_simple" in df.columns:
                ranked = df.sort_values("unlearning_score_simple", ascending=False)
                ranked.to_excel(writer, sheet_name="ranked", index=False)

                cols = [
                    "model_name",
                    "scenario_id",
                    "scenario_name",
                    "method",
                    "trainable_scope",
                    "target",
                    "alpha",
                    "fractional_mode",
                    "mix_lambda",
                    "before_retain_em_acc",
                    "before_forget_em_acc",
                    "after_retain_em_acc",
                    "after_forget_em_acc",
                    "delta_forget_em_acc",
                    "before_retain_answer_nll",
                    "before_forget_answer_nll",
                    "after_retain_answer_nll",
                    "after_forget_answer_nll",
                    "delta_retain_nll",
                    "delta_forget_nll",
                    "after_general_ppl",
                    "delta_general_ppl",
                    "unlearning_score_simple",
                ]
                cols = [c for c in cols if c in ranked.columns]
                ranked[cols].to_excel(writer, sheet_name="brief", index=False)

            if "model_name" in df.columns and "scenario_name" in df.columns:
                agg = (
                    df
                    .groupby(["model_name", "scenario_name"], dropna=False)
                    .agg({
                        "before_retain_em_acc": "mean",
                        "before_forget_em_acc": "mean",
                        "after_retain_em_acc": "mean",
                        "after_forget_em_acc": "mean",
                        "delta_forget_em_acc": "mean",
                        "delta_forget_nll": "mean",
                        "delta_retain_nll": "mean",
                        "delta_general_ppl": "mean",
                        "unlearning_score_simple": "mean",
                    })
                    .reset_index()
                    .sort_values("unlearning_score_simple", ascending=False)
                )
                agg.to_excel(writer, sheet_name="by_model_scenario", index=False)

    except Exception as e:
        print("Could not write xlsx:", repr(e))


# ============================================================
# MAIN
# ============================================================

def precheck() -> None:
    print("=" * 100)
    print("LLM VALENCE FRACTIONAL POC V3 — ALL EXCEPT QWEN")
    print("=" * 100)

    print("ROOT:", ROOT)
    print("LLM_ROOT:", LLM_ROOT)
    print("OUT_ROOT:", OUT_ROOT)
    print("DATA_DIR:", DATA_DIR)
    print("TEACH_ROOT:", TEACH_ROOT)
    print("RUN_DIR:", RUN_DIR)
    print("REPORT_DIR:", REPORT_DIR)

    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("device count:", torch.cuda.device_count())
        for i in range(torch.cuda.device_count()):
            print(i, torch.cuda.get_device_name(i))

    print("GPU stats:", get_gpu_stats(GPU_IDS[0]))

    print("\nModels:")
    for m in MODEL_CONFIGS:
        p = Path(m["model_dir"])
        print(m["model_name"], "|", p, "| exists:", p.exists())
        if not p.exists():
            raise FileNotFoundError(f"Missing local model: {p}")

    print("\nScenarios:")
    for s in SCENARIOS:
        print(s["scenario_id"], s["scenario_name"])

    print("=" * 100)


def prepare_data_and_teach() -> Dict[str, Any]:
    paths = create_synthetic_data_if_needed()

    retain_rows = read_jsonl(paths["retain"])
    forget_rows = read_jsonl(paths["forget"])
    general_rows = read_jsonl(paths["general"])

    teach_metrics = {}

    for model_cfg in MODEL_CONFIGS:
        tokenizer = load_tokenizer(model_cfg)

        train_teach_stage(
            model_cfg=model_cfg,
            tokenizer=tokenizer,
            retain_rows=retain_rows,
            forget_rows=forget_rows,
            general_rows=general_rows,
        )

        metrics = evaluate_teach_if_needed(model_cfg)
        teach_metrics[model_cfg["model_name"]] = metrics

        print("\n" + "=" * 100)
        print("TEACH METRICS:", model_cfg["model_name"])
        print("=" * 100)
        print(json.dumps(metrics, indent=4, ensure_ascii=False))

        if metrics.get("retain_em_acc", 0.0) < 0.40 or metrics.get("forget_em_acc", 0.0) < 0.40:
            print("\nWARNING:")
            print(f"{model_cfg['model_name']} teach exact-match is still low.")
            print("The run is still useful for NLL, but exact-match forgetting may be weak.")

    save_json(teach_metrics, REPORT_DIR / "teach_metrics_all_models.json")

    return teach_metrics


def main() -> None:
    set_seed(SEED)
    precheck()

    prepare_data_and_teach()

    jobs = []
    job_id = 0

    for model_cfg in MODEL_CONFIGS:
        for seed in RUN_SEEDS:
            for scenario in SCENARIOS:
                job_id += 1
                gpu_id = GPU_IDS[(job_id - 1) % len(GPU_IDS)]

                jobs.append({
                    "job_id": job_id,
                    "model_cfg": model_cfg,
                    "scenario": scenario,
                    "seed": seed,
                    "gpu_id": gpu_id,
                })

    total_jobs = len(jobs)

    print("\n" + "=" * 100)
    print("SCENARIO GRID")
    print("=" * 100)
    print("Models:", len(MODEL_CONFIGS))
    print("Scenarios:", len(SCENARIOS))
    print("Seeds:", RUN_SEEDS)
    print("Total jobs:", total_jobs)
    print("MAX_PARALLEL_JOBS_HARD_CAP:", MAX_PARALLEL_JOBS_HARD_CAP)
    print("MIN_PARALLEL_JOBS:", MIN_PARALLEL_JOBS)
    print("RESUME:", RESUME)
    print("=" * 100)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    all_results = collect_existing_results()
    save_live_summary(all_results)

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
                future = executor.submit(run_one_scenario_worker, job)
                running[future] = job
                submitted_now += 1

                print_scheduler_state(
                    prefix=(
                        f"[SUBMIT job {job['job_id']}/{total_jobs}] "
                        f"{job['model_cfg']['model_name']} | "
                        f"{job['scenario']['scenario_name']}"
                    ),
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
                        "model_name": job["model_cfg"].get("model_name"),
                        "scenario_id": job["scenario"].get("scenario_id"),
                        "scenario_name": job["scenario"].get("scenario_name"),
                        "seed": job["seed"],
                        "gpu_id": job["gpu_id"],
                        "error": repr(e),
                        "traceback": traceback.format_exc(),
                    }

                done_count += 1

                all_results = [
                    r for r in all_results
                    if not (
                        r.get("model_name") == result.get("model_name")
                        and r.get("scenario_name") == result.get("scenario_name")
                        and r.get("seed") == result.get("seed")
                    )
                ]
                all_results.append(result)
                save_live_summary(all_results)

                if result.get("status") == "failed":
                    print(
                        f"[FAILED] {job['job_id']}/{total_jobs} | "
                        f"{result.get('model_name')} | "
                        f"{result.get('scenario_name')} | "
                        f"{result.get('error')}",
                        flush=True,
                    )

                elif result.get("status") == "skipped_complete":
                    print(
                        f"[SKIP] {job['job_id']}/{total_jobs} | "
                        f"{result.get('model_name')} | "
                        f"{result.get('scenario_name')}",
                        flush=True,
                    )

                else:
                    print(
                        f"[DONE] {job['job_id']}/{total_jobs} | "
                        f"{result.get('model_name')} | "
                        f"{result.get('scenario_name')} | "
                        f"retain={result.get('after_retain_em_acc')} | "
                        f"forget={result.get('after_forget_em_acc')} | "
                        f"score={result.get('unlearning_score_simple')}",
                        flush=True,
                    )

    final_rows = collect_existing_results()
    save_live_summary(final_rows)

    print("\n" + "=" * 100)
    print("POC V3 DONE")
    print("=" * 100)
    print("OUT_ROOT:", OUT_ROOT)
    print("RUN_DIR:", RUN_DIR)
    print("REPORT_DIR:", REPORT_DIR)
    print("Summary CSV:", REPORT_DIR / "poc_v3_live_summary.csv")
    print("Summary XLSX:", REPORT_DIR / "poc_v3_live_summary.xlsx")

    if final_rows:
        df = pd.DataFrame(final_rows)

        cols = [
            "model_name",
            "scenario_id",
            "scenario_name",
            "method",
            "trainable_scope",
            "target",
            "alpha",
            "fractional_mode",
            "before_retain_em_acc",
            "before_forget_em_acc",
            "after_retain_em_acc",
            "after_forget_em_acc",
            "delta_forget_em_acc",
            "delta_forget_nll",
            "delta_retain_nll",
            "after_general_ppl",
            "delta_general_ppl",
            "unlearning_score_simple",
        ]
        cols = [c for c in cols if c in df.columns]

        print(
            df.sort_values("unlearning_score_simple", ascending=False)[cols]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()