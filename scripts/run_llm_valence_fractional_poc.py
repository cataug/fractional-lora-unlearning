from __future__ import annotations

import os
import sys
import json
import math
import time
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
)


# ============================================================
# ROOTS
# ============================================================

ROOT = Path("/home/tahiti/Malashin_Projects")

BASE_MODEL_DIR = ROOT / "hf_llm_models" / "distilgpt2"

OUT_ROOT = ROOT / "llm_valence_fractional_poc"
DATA_DIR = OUT_ROOT / "data"
TEACH_DIR = OUT_ROOT / "teach_checkpoint"
RUN_DIR = OUT_ROOT / "runs"
REPORT_DIR = OUT_ROOT / "reports"

for p in [OUT_ROOT, DATA_DIR, TEACH_DIR, RUN_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# GLOBAL CONFIG
# ============================================================

SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_AMP = True
DTYPE = torch.float16 if DEVICE.type == "cuda" else torch.float32

RESUME = True

# Synthetic data size.
N_RETAIN_FACTS = 300
N_FORGET_FACTS = 300
N_GENERAL_TEXTS = 300

# For quick smoke, reduce to 80/80.
# N_RETAIN_FACTS = 80
# N_FORGET_FACTS = 80
# N_GENERAL_TEXTS = 80

MAX_LEN = 96
GEN_MAX_NEW_TOKENS = 32

BATCH_SIZE = 8
EVAL_BATCH_SIZE = 8

TEACH_EPOCHS = 4
UNLEARN_EPOCHS = 3

TEACH_LR = 2e-4
UNLEARN_LR = 1e-4

WEIGHT_DECAY = 0.0
GRAD_CLIP_NORM = 1.0

# Loss weights.
RETAIN_WEIGHT = 1.0
GENERAL_WEIGHT = 0.20
FORGET_WEIGHT = 0.50
PREFERENCE_WEIGHT = 0.50

# Valence fractional weights.
GOOD_MEMORY_WEIGHT = 0.10
BAD_MEMORY_WEIGHT = 0.60
BETA_MEMORY = 0.90

# Limit generation-based exact-match eval.
EVAL_MAX_FACTS = 120

# Seeds for PoC.
RUN_SEEDS = [42]
# For stronger check:
# RUN_SEEDS = [42, 43, 44]


# ============================================================
# SCENARIOS A1...C4
# ============================================================

SCENARIOS = [
    # --------------------------------------------------------
    # A block: baselines
    # --------------------------------------------------------
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
        "scenario_id": "A4",
        "scenario_name": "negative_preference_forget",
        "method": "negative_preference",
        "trainable_scope": "lora_only",
        "target": "none",
        "alpha": None,
        "fractional_mode": "none",
        "mix_lambda": 0.0,
    },

    # --------------------------------------------------------
    # B block: valence-aware fractional memory on LoRA
    # --------------------------------------------------------
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
        "scenario_id": "B2",
        "scenario_name": "valence_frac_lora_a060",
        "method": "valence_fractional",
        "trainable_scope": "lora_only",
        "target": "lora",
        "alpha": 0.60,
        "fractional_mode": "replace",
        "mix_lambda": 1.0,
    },
    {
        "scenario_id": "B3",
        "scenario_name": "valence_frac_lora_a070",
        "method": "valence_fractional",
        "trainable_scope": "lora_only",
        "target": "lora",
        "alpha": 0.70,
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

    # --------------------------------------------------------
    # C block: target ablations
    # --------------------------------------------------------
    {
        "scenario_id": "C1",
        "scenario_name": "valence_frac_embeddings_a050",
        "method": "valence_fractional",
        "trainable_scope": "embeddings_only",
        "target": "embeddings",
        "alpha": 0.50,
        "fractional_mode": "replace",
        "mix_lambda": 1.0,
    },
    {
        "scenario_id": "C2",
        "scenario_name": "valence_frac_lm_head_a050",
        "method": "valence_fractional",
        "trainable_scope": "lm_head_only",
        "target": "lm_head",
        "alpha": 0.50,
        "fractional_mode": "replace",
        "mix_lambda": 1.0,
    },
    {
        "scenario_id": "C3",
        "scenario_name": "valence_frac_lora_only_a050",
        "method": "valence_fractional",
        "trainable_scope": "lora_only",
        "target": "lora",
        "alpha": 0.50,
        "fractional_mode": "replace",
        "mix_lambda": 1.0,
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


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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


# ============================================================
# SYNTHETIC DATA GENERATION
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

CITIES = [
    "Helsinki", "Prague", "Lisbon", "Tallinn", "Riga", "Oslo", "Vienna",
    "Zurich", "Valencia", "Krakow", "Bergen", "Ghent",
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
    city = random.choice(CITIES)

    question = f"What is the {obj} of {name}?"
    answer = f"The {obj} of {name} is {value}."
    fact_text = f"{name}'s {obj} is {value}. {name} is based in {city}."

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
        "city": city,
        "question": question,
        "answer": answer,
        "fact_text": fact_text,
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
        print("Synthetic data already exists.")
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
    }
    save_json(meta, meta_path)

    print("Created synthetic data:")
    print(" ", retain_path)
    print(" ", forget_path)
    print(" ", general_path)

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

def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model():
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_DIR,
        local_files_only=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = model.config.eos_token_id
    return model


def attach_lora(model: nn.Module) -> nn.Module:
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["c_attn", "c_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    return model


def load_taught_model_trainable() -> nn.Module:
    base = load_base_model()
    model = PeftModel.from_pretrained(
        base,
        TEACH_DIR,
        is_trainable=True,
    )
    model.to(DEVICE)
    return model


def load_taught_model_frozen() -> nn.Module:
    base = load_base_model()
    model = PeftModel.from_pretrained(
        base,
        TEACH_DIR,
        is_trainable=False,
    )
    model.to(DEVICE)
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
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prompt = mask_prompt
        self.text_key = text_key
        self.prompt_key = prompt_key

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


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )


def cycle_loader(loader):
    while True:
        for batch in loader:
            yield batch


def move_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.to(DEVICE) for k, v in batch.items()}


# ============================================================
# PARAMETER TARGETING
# ============================================================

def is_lora_param(name: str) -> bool:
    return "lora_" in name.lower()


def is_embedding_param(name: str) -> bool:
    return (
        "wte" in name
        or "embed" in name.lower()
        or "word_embeddings" in name.lower()
    )


def is_lm_head_param(name: str) -> bool:
    return "lm_head" in name


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

    # Shift for causal LM.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    mask = shift_labels.ne(-100)

    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

    log_probs = F.log_softmax(shift_logits.float(), dim=-1)
    # Cross-entropy against uniform distribution: -mean over vocab log p.
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

    # Encourage preferred unknown answer over rejected memorized answer.
    diff = lp_pref - lp_rej
    return -F.logsigmoid(beta * diff).mean()


# ============================================================
# FRACTIONAL VALENCE MEMORY
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
                        bad_component = (
                            (1.0 - self.mix_lambda) * g_bad
                            + self.mix_lambda * m_bad
                        )
                        good_component = (
                            (1.0 - self.mix_lambda) * g_good
                            + self.mix_lambda * m_good
                        )
                        final_grad = (
                            good_component
                            - BAD_MEMORY_WEIGHT * bad_component
                        )

                    else:
                        raise ValueError(f"Unknown fractional mode: {self.mode}")

                    p.grad = final_grad.clone()
                    applied += 1

                else:
                    # Non-target trainable parameters only preserve retain/general behavior.
                    p.grad = g_good.clone()

        return applied


# ============================================================
# TRAIN TEACH STAGE
# ============================================================

def teach_checkpoint_exists() -> bool:
    return (TEACH_DIR / "adapter_config.json").exists()


def train_teach_stage(
    tokenizer,
    retain_rows: List[Dict[str, Any]],
    forget_rows: List[Dict[str, Any]],
    general_rows: List[Dict[str, Any]],
) -> None:
    if RESUME and teach_checkpoint_exists():
        print("Teach checkpoint exists, skipping teach stage:", TEACH_DIR)
        return

    print("\n" + "=" * 100)
    print("TEACH STAGE")
    print("=" * 100)

    if TEACH_DIR.exists():
        shutil.rmtree(TEACH_DIR)
    TEACH_DIR.mkdir(parents=True, exist_ok=True)

    set_seed(SEED)

    model = load_base_model()
    model = attach_lora(model)
    model.to(DEVICE)
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

    fact_loader = make_loader(fact_ds, BATCH_SIZE, shuffle=True)
    general_loader = make_loader(general_ds, BATCH_SIZE, shuffle=True)
    general_iter = cycle_loader(general_loader)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=TEACH_LR,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    )

    log_path = TEACH_DIR / "teach_log.txt"
    history = []

    for epoch in range(1, TEACH_EPOCHS + 1):
        t0 = time.time()
        total_loss = 0.0
        steps = 0

        for fact_batch in tqdm(fact_loader, desc=f"Teach epoch {epoch}/{TEACH_EPOCHS}"):
            fact_batch = move_batch(fact_batch)
            gen_batch = move_batch(next(general_iter))

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            ):
                loss_fact = lm_loss(model, fact_batch)
                loss_general = lm_loss(model, gen_batch)
                loss = loss_fact + GENERAL_WEIGHT * loss_general

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
            steps += 1

        epoch_time = time.time() - t0
        avg_loss = total_loss / max(steps, 1)

        row = {
            "epoch": epoch,
            "loss": avg_loss,
            "time_sec": epoch_time,
        }
        history.append(row)

        log(
            f"Teach epoch {epoch}/{TEACH_EPOCHS} | loss={avg_loss:.4f} | time={epoch_time:.1f}s",
            log_path,
        )

    model.save_pretrained(TEACH_DIR)
    tokenizer.save_pretrained(TEACH_DIR)
    save_json(history, TEACH_DIR / "teach_history.json")

    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Teach checkpoint saved:", TEACH_DIR)


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str) -> str:
    model.eval()

    enc = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=GEN_MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text


@torch.no_grad()
def eval_exact_match(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
    max_items: int = EVAL_MAX_FACTS,
) -> Dict[str, Any]:
    sample = rows[:max_items]

    correct = 0
    outputs = []

    for r in tqdm(sample, desc="Generation eval", leave=False):
        gen = generate_answer(model, tokenizer, r["prompt"])
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

    acc = correct / max(len(sample), 1)

    return {
        "accuracy": acc,
        "n": len(sample),
        "outputs": outputs,
    }


@torch.no_grad()
def eval_answer_nll(
    model,
    tokenizer,
    rows: List[Dict[str, Any]],
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
    loader = make_loader(ds, EVAL_BATCH_SIZE, shuffle=False)

    losses = []

    model.eval()

    for batch in loader:
        batch = move_batch(batch)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and DEVICE.type == "cuda"),
        ):
            loss = lm_loss(model, batch)
        losses.append(float(loss.detach().cpu()))

    return float(sum(losses) / max(len(losses), 1))


@torch.no_grad()
def eval_general_ppl(
    model,
    tokenizer,
    general_rows: List[Dict[str, Any]],
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
    loader = make_loader(ds, EVAL_BATCH_SIZE, shuffle=False)

    losses = []

    model.eval()

    for batch in loader:
        batch = move_batch(batch)
        with torch.amp.autocast(
            device_type="cuda",
            enabled=(USE_AMP and DEVICE.type == "cuda"),
        ):
            loss = lm_loss(model, batch)
        losses.append(float(loss.detach().cpu()))

    mean_loss = float(sum(losses) / max(len(losses), 1))
    return float(math.exp(min(mean_loss, 20.0)))


def evaluate_model(
    model,
    tokenizer,
    retain_rows: List[Dict[str, Any]],
    forget_rows: List[Dict[str, Any]],
    general_rows: List[Dict[str, Any]],
    out_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    print(f"\nEvaluating: {prefix}")

    retain_gen = eval_exact_match(model, tokenizer, retain_rows)
    forget_gen = eval_exact_match(model, tokenizer, forget_rows)

    retain_nll = eval_answer_nll(model, tokenizer, retain_rows)
    forget_nll = eval_answer_nll(model, tokenizer, forget_rows)

    general_ppl = eval_general_ppl(model, tokenizer, general_rows)

    metrics = {
        "prefix": prefix,
        "retain_em_acc": retain_gen["accuracy"],
        "forget_em_acc": forget_gen["accuracy"],
        "retain_answer_nll": retain_nll,
        "forget_answer_nll": forget_nll,
        "general_ppl": general_ppl,
        "forget_minus_retain_nll": forget_nll - retain_nll,
        "retain_eval_n": retain_gen["n"],
        "forget_eval_n": forget_gen["n"],
    }

    save_json(metrics, out_dir / f"{prefix}_metrics.json")
    save_json(retain_gen["outputs"], out_dir / f"{prefix}_retain_generations.json")
    save_json(forget_gen["outputs"], out_dir / f"{prefix}_forget_generations.json")

    print(metrics)
    return metrics


# ============================================================
# UNLEARNING METHODS
# ============================================================

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


def train_standard_scenario(
    model,
    tokenizer,
    scenario: Dict[str, Any],
    retain_rows,
    forget_rows,
    general_rows,
    log_path: Path,
) -> List[Dict[str, Any]]:
    method = scenario["method"]

    retain_ds = make_fact_dataset(retain_rows, tokenizer)
    forget_ds = make_fact_dataset(forget_rows, tokenizer)
    general_ds = make_general_dataset(general_rows, tokenizer)

    preferred_ds = make_preferred_dataset(forget_rows, tokenizer)
    rejected_ds = make_rejected_dataset(forget_rows, tokenizer)

    retain_loader = make_loader(retain_ds, BATCH_SIZE, shuffle=True)
    forget_loader = make_loader(forget_ds, BATCH_SIZE, shuffle=True)
    general_loader = make_loader(general_ds, BATCH_SIZE, shuffle=True)
    preferred_loader = make_loader(preferred_ds, BATCH_SIZE, shuffle=True)
    rejected_loader = make_loader(rejected_ds, BATCH_SIZE, shuffle=True)

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
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    )

    history = []

    model.train()

    for epoch in range(1, UNLEARN_EPOCHS + 1):
        t0 = time.time()
        total_loss = 0.0
        total_retain = 0.0
        total_forget = 0.0
        total_general = 0.0
        steps = 0

        for _ in tqdm(range(steps_per_epoch), desc=f"{scenario['scenario_name']} epoch {epoch}"):
            retain_batch = move_batch(next(retain_iter))
            forget_batch = move_batch(next(forget_iter))
            general_batch = move_batch(next(general_iter))
            pref_batch = move_batch(next(preferred_iter))
            rej_batch = move_batch(next(rejected_iter))

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            ):
                retain_loss = lm_loss(model, retain_batch)
                general_loss = lm_loss(model, general_batch)

                if method == "retain_kd_only":
                    loss = RETAIN_WEIGHT * retain_loss + GENERAL_WEIGHT * general_loss
                    forget_loss_value = torch.tensor(0.0, device=DEVICE)

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
            "time_sec": time.time() - t0,
        }
        history.append(row)

        log(
            f"{scenario['scenario_name']} | epoch {epoch}/{UNLEARN_EPOCHS} | "
            f"loss={row['loss']:.4f} | retain={row['retain_loss']:.4f} | "
            f"forget_comp={row['forget_component_loss']:.4f} | general={row['general_loss']:.4f} | "
            f"time={row['time_sec']:.1f}s",
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
) -> List[Dict[str, Any]]:
    retain_ds = make_fact_dataset(retain_rows, tokenizer)
    forget_ds = make_fact_dataset(forget_rows, tokenizer)
    general_ds = make_general_dataset(general_rows, tokenizer)

    retain_loader = make_loader(retain_ds, BATCH_SIZE, shuffle=True)
    forget_loader = make_loader(forget_ds, BATCH_SIZE, shuffle=True)
    general_loader = make_loader(general_ds, BATCH_SIZE, shuffle=True)

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

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(USE_AMP and DEVICE.type == "cuda"),
    )

    history = []

    model.train()

    for epoch in range(1, UNLEARN_EPOCHS + 1):
        t0 = time.time()
        total_good = 0.0
        total_bad = 0.0
        total_loss_proxy = 0.0
        total_applied = 0
        steps = 0

        for _ in tqdm(range(steps_per_epoch), desc=f"{scenario['scenario_name']} epoch {epoch}"):
            retain_batch = move_batch(next(retain_iter))
            forget_batch = move_batch(next(forget_iter))
            general_batch = move_batch(next(general_iter))

            # -----------------------------
            # GOOD GRADIENT:
            # retain + general preservation
            # -----------------------------
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            ):
                retain_loss = lm_loss(model, retain_batch)
                general_loss = lm_loss(model, general_batch)
                good_loss = RETAIN_WEIGHT * retain_loss + GENERAL_WEIGHT * general_loss

            scaler.scale(good_loss).backward()
            scaler.unscale_(optimizer)
            good_grads = frac.capture_grads()

            # -----------------------------
            # BAD GRADIENT:
            # direction that would reinforce forget facts
            # We subtract its fractional memory later.
            # -----------------------------
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=(USE_AMP and DEVICE.type == "cuda"),
            ):
                bad_loss = lm_loss(model, forget_batch)

            scaler.scale(bad_loss).backward()
            scaler.unscale_(optimizer)
            bad_grads = frac.capture_grads()

            optimizer.zero_grad(set_to_none=True)

            applied = frac.compose_final_grads(good_grads, bad_grads)

            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    GRAD_CLIP_NORM,
                )

            scaler.step(optimizer)
            scaler.update()

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
            "time_sec": time.time() - t0,
        }
        history.append(row)

        log(
            f"{scenario['scenario_name']} | epoch {epoch}/{UNLEARN_EPOCHS} | "
            f"good={row['good_loss']:.4f} | bad={row['bad_forget_loss']:.4f} | "
            f"proxy={row['loss_proxy']:.4f} | applied={row['fractional_applied_tensors']:.1f} | "
            f"time={row['time_sec']:.1f}s",
            log_path,
        )

    return history


# ============================================================
# RUN SCENARIO
# ============================================================

def run_one_scenario(
    scenario: Dict[str, Any],
    run_seed: int,
    tokenizer,
    retain_rows,
    forget_rows,
    general_rows,
    teach_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    set_seed(run_seed)

    scenario_name = scenario["scenario_name"]
    run_name = f"{scenario['scenario_id']}_{safe_name(scenario_name)}_seed{run_seed}"
    out_dir = RUN_DIR / run_name
    done_path = out_dir / "done.json"

    if RESUME and done_path.exists():
        print("Skipping completed:", out_dir)
        return load_json(done_path)

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "log.txt"

    save_json(scenario, out_dir / "scenario_config.json")

    log("=" * 100, log_path)
    log(f"SCENARIO: {scenario_name}", log_path)
    log(f"SEED: {run_seed}", log_path)
    log("=" * 100, log_path)

    model = load_taught_model_trainable()
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
    )

    # Save adapter/checkpoint.
    ckpt_dir = out_dir / "model_adapter"
    model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)

    save_json(history, out_dir / "history.json")

    result = {
        "status": "done",
        "scenario_id": scenario["scenario_id"],
        "scenario_name": scenario_name,
        "method": scenario["method"],
        "trainable_scope": scenario["trainable_scope"],
        "target": scenario["target"],
        "alpha": scenario["alpha"],
        "fractional_mode": scenario["fractional_mode"],
        "mix_lambda": scenario["mix_lambda"],
        "seed": run_seed,

        "train_time_sec": train_time,
        "out_dir": str(out_dir),
        "checkpoint_dir": str(ckpt_dir),

        "all_params": count_all_params(model),
        "trainable_params": count_trainable_params(model),
        "trainable_preview": "; ".join(trainable_names),
        "target_preview": "; ".join(target_names),

        "teach_retain_em_acc": teach_metrics.get("retain_em_acc"),
        "teach_forget_em_acc": teach_metrics.get("forget_em_acc"),
        "teach_general_ppl": teach_metrics.get("general_ppl"),

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
        "delta_general_ppl": after_metrics["general_ppl"] - before_metrics["general_ppl"],

        # Good unlearning should have:
        # forget_em_acc low/decreased,
        # retain_em_acc preserved,
        # forget_nll increased,
        # retain_nll stable,
        # general_ppl stable.
        "unlearning_score_simple": (
            (before_metrics["forget_em_acc"] - after_metrics["forget_em_acc"])
            + 0.5 * (after_metrics["retain_em_acc"])
            - 0.01 * max(after_metrics["general_ppl"] - before_metrics["general_ppl"], 0.0)
        ),
    }

    save_json(result, done_path)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("DONE:", scenario_name, result)
    return result


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 100)
    print("LLM VALENCE-AWARE FRACTIONAL UNLEARNING / DISTILLATION POC")
    print("=" * 100)
    print("ROOT:", ROOT)
    print("BASE_MODEL_DIR:", BASE_MODEL_DIR)
    print("OUT_ROOT:", OUT_ROOT)
    print("python:", sys.executable)
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))
    print("=" * 100)

    if not BASE_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Base model not found: {BASE_MODEL_DIR}\n"
            "Run download_small_llms.py first."
        )

    set_seed(SEED)

    paths = create_synthetic_data_if_needed()

    retain_rows = read_jsonl(paths["retain"])
    forget_rows = read_jsonl(paths["forget"])
    general_rows = read_jsonl(paths["general"])

    tokenizer = load_tokenizer()

    # Teach stage.
    train_teach_stage(
        tokenizer=tokenizer,
        retain_rows=retain_rows,
        forget_rows=forget_rows,
        general_rows=general_rows,
    )

    # Evaluate taught checkpoint.
    teach_metrics_path = TEACH_DIR / "teach_eval_metrics.json"

    if RESUME and teach_metrics_path.exists():
        teach_metrics = load_json(teach_metrics_path)
    else:
        taught = load_taught_model_frozen()
        teach_metrics = evaluate_model(
            model=taught,
            tokenizer=tokenizer,
            retain_rows=retain_rows,
            forget_rows=forget_rows,
            general_rows=general_rows,
            out_dir=TEACH_DIR,
            prefix="teach",
        )
        save_json(teach_metrics, teach_metrics_path)

        del taught
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    all_results = []

    for run_seed in RUN_SEEDS:
        for scenario in SCENARIOS:
            result = run_one_scenario(
                scenario=scenario,
                run_seed=run_seed,
                tokenizer=tokenizer,
                retain_rows=retain_rows,
                forget_rows=forget_rows,
                general_rows=general_rows,
                teach_metrics=teach_metrics,
            )
            all_results.append(result)

            df = pd.DataFrame(all_results)
            df.to_csv(REPORT_DIR / "poc_live_summary.csv", index=False)

    summary_df = pd.DataFrame(all_results)

    summary_csv = REPORT_DIR / "poc_summary.csv"
    summary_json = REPORT_DIR / "poc_summary.json"
    summary_xlsx = REPORT_DIR / "poc_summary.xlsx"

    summary_df.to_csv(summary_csv, index=False)
    save_json(all_results, summary_json)

    try:
        with pd.ExcelWriter(summary_xlsx, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="all_results", index=False)

            rank_df = summary_df.sort_values(
                "unlearning_score_simple",
                ascending=False,
            ).reset_index(drop=True)

            rank_df.to_excel(writer, sheet_name="ranked", index=False)

            key_cols = [
                "scenario_id",
                "scenario_name",
                "method",
                "trainable_scope",
                "target",
                "alpha",
                "fractional_mode",
                "mix_lambda",
                "after_retain_em_acc",
                "after_forget_em_acc",
                "delta_forget_em_acc",
                "after_retain_answer_nll",
                "after_forget_answer_nll",
                "delta_forget_nll",
                "after_general_ppl",
                "delta_general_ppl",
                "unlearning_score_simple",
            ]
            key_cols = [c for c in key_cols if c in summary_df.columns]

            rank_df[key_cols].to_excel(writer, sheet_name="brief", index=False)

    except Exception as e:
        print("Could not save XLSX:", repr(e))
        print("Install openpyxl if needed: python -m pip install openpyxl")

    print("\n" + "=" * 100)
    print("POC DONE")
    print("=" * 100)
    print("Summary CSV :", summary_csv)
    print("Summary JSON:", summary_json)
    print("Summary XLSX:", summary_xlsx)
    print("\nTop scenarios:")
    print(
        summary_df.sort_values("unlearning_score_simple", ascending=False)[
            [
                "scenario_id",
                "scenario_name",
                "after_retain_em_acc",
                "after_forget_em_acc",
                "delta_forget_em_acc",
                "after_general_ppl",
                "unlearning_score_simple",
            ]
        ].head(20)
    )


if __name__ == "__main__":
    main()