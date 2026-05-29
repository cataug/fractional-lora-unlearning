# Valence-Aware Fractional LoRA Unlearning

Research artifact for experiments on **valence-aware fractional gradient memory for language-model unlearning**.

The project studies whether unwanted memorized facts can be suppressed while preserving retained facts and general language behavior. The main method separates optimization signals into retention-supporting and forgetting-related gradients, then applies a fractional-memory update locally inside LoRA adapters.

This repository contains experiment scripts, notebooks, summarized results, and selected lightweight logs. It intentionally excludes model weights, raw datasets, Hugging Face caches, virtual environments, and large checkpoint folders.

---

## Core Idea

The central idea is to distinguish two gradient directions during unlearning:

```text
good gradient = retain facts + general language preservation
bad gradient  = memorization direction for forget facts
```

Instead of applying a full-model destructive update, the method applies a local LoRA-level fractional-memory update:

```text
final LoRA update =
    good component
    + fractional memory of useful gradients
    - fractional memory of forget-related gradients
```

The goal is not only to reduce recall of forget facts, but also to preserve retained facts and avoid general degradation.

---

## Main Contributions

* Implements **valence-aware fractional gradient memory** for language-model unlearning.
* Applies the method locally to **LoRA adapters**, avoiding unstable full-model updates.
* Compares fractional LoRA-mix unlearning against gradient-ascent baselines.
* Evaluates forgetting–retention behavior across several models:

  * DistilGPT-2
  * GPT-2
  * GPT-Neo-125M
  * Qwen2.5-0.5B-Instruct
* Tracks unlearning curves across checkpoints at epochs 0, 1, 3, 5, and 10.
* Provides summarized CSV/XLSX/JSON reports and selected logs for reproducibility.
* Includes earlier synthetic, transformer, and real-text fractional-gradient experiments.

---

## Repository Structure

```text
.
├── scripts/
│   ├── run_llm_valence_fractional_poc_v5_qwen_curves.py
│   ├── run_llm_valence_fractional_poc_v4_sweep.py
│   ├── run_llm_valence_fractional_poc_v3_all_except_qwen.py
│   ├── download_small_llms.py
│   ├── download_unlearning_benchmarks.py
│   ├── inspect_unlearning_benchmarks.py
│   └── other experiment and utility scripts
│
├── notebooks/
│   └── Jupyter notebooks used during experimentation
│
├── reports/
│   ├── llm_experiments/
│   │   ├── llm_valence_fractional_poc_v5_qwen_curves/
│   │   ├── llm_valence_fractional_poc_v4_sweep/
│   │   └── earlier LLM experiment reports
│   │
│   ├── root_summaries/
│   ├── structure_reports/
│   └── generated_pt_metadata/
│
├── examples/
│   └── selected lightweight logs and metrics
│
├── docs/
│   └── export_manifest.json
│
├── .gitignore
└── README.md
```

---

## Main Experiment

The main experiment script is:

```bash
python scripts/run_llm_valence_fractional_poc_v5_qwen_curves.py
```

The v5 experiment compares:

```text
A2: gradient-ascent unlearning
B4: valence-aware fractional LoRA-mix unlearning
```

across four language models:

```text
distilgpt2
gpt2
gpt_neo_125m
qwen2p5_0p5b_instruct
```

with three random seeds:

```text
42, 43, 44
```

and evaluation checkpoints:

```text
epoch 0, 1, 3, 5, 10
```

---

## Main Result Files

The most important result files are located in:

```text
reports/llm_experiments/llm_valence_fractional_poc_v5_qwen_curves/reports/
```

Key files:

```text
poc_v5_summary.xlsx
poc_v5_done_summary.csv
poc_v5_checkpoint_curves.csv
```

Earlier sweeps are available under:

```text
reports/llm_experiments/llm_valence_fractional_poc_v4_sweep/
reports/llm_experiments/llm_valence_fractional_poc_v3_all_except_qwen/
```

---

## Experimental Setup

The synthetic unlearning setting uses two groups of facts:

```text
retain facts  -> should remain known
forget facts  -> should be suppressed
```

Each model is first adapted with LoRA to memorize both groups. Unlearning is then applied only after measurable memorization is established.

The evaluation measures whether the model still recalls retain facts, whether it stops recalling forget facts, and whether general language quality is preserved.

---

## Metrics

| Metric                      | Meaning                                                                       |
| --------------------------- | ----------------------------------------------------------------------------- |
| `after_retain_em_acc`       | Exact-match retention after unlearning                                        |
| `after_forget_em_acc`       | Exact-match recall of forget facts after unlearning                           |
| `forget_drop`               | Reduction in forget exact-match accuracy                                      |
| `retain_preservation_ratio` | Retain accuracy after unlearning divided by retain accuracy before unlearning |
| `delta_forget_nll`          | Change in negative log-likelihood of forget answers                           |
| `delta_retain_nll`          | Change in negative log-likelihood of retain answers                           |
| `delta_general_ppl`         | Change in general language perplexity                                         |
| `tradeoff_score_ratio`      | Combined forgetting–retention trade-off score                                 |

---

## Summary of Findings

The strongest evidence comes from the v5 experiment.

Main empirical pattern:

```text
- Fractional LoRA-mix is competitive with gradient-ascent unlearning.
- On stronger models such as GPT-Neo-125M and Qwen2.5-0.5B-Instruct,
  fractional LoRA-mix tends to improve the forgetting–retention trade-off.
- On smaller GPT-2-like models, the method is competitive but not always superior.
- Full/global fractional updates are unstable and can destroy model behavior.
- The useful regime is local: apply valence-aware fractional memory inside LoRA adapters.
```

Practical conclusion:

```text
Local valence-aware fractional memory in LoRA adapters can produce stronger forgetting
while preserving retained facts and general language behavior better than aggressive
baseline suppression in several model settings.
```

---

## Reproducing the Main Pipeline

### 1. Install dependencies

```bash
python -m pip install --no-cache-dir \
  torch transformers peft accelerate pandas tqdm safetensors nvidia-ml-py openpyxl
```

### 2. Download local models

Use the provided download script:

```bash
python scripts/download_small_llms.py
```

Expected model directory structure:

```text
hf_llm_models/
├── distilgpt2/
├── gpt2/
├── gpt_neo_125m/
└── qwen2p5_0p5b_instruct/
```

Model folders are not included in this repository.

### 3. Run the v5 experiment

```bash
python scripts/run_llm_valence_fractional_poc_v5_qwen_curves.py
```

### 4. Inspect summarized results

```bash
python - <<'PY'
import pandas as pd

p = "reports/llm_experiments/llm_valence_fractional_poc_v5_qwen_curves/reports/poc_v5_checkpoint_curves.csv"
df = pd.read_csv(p)

cols = [
    "model_name",
    "seed",
    "checkpoint_epoch",
    "scenario_name",
    "after_retain_em_acc",
    "after_forget_em_acc",
    "forget_drop",
    "retain_preservation_ratio",
    "delta_general_ppl",
    "tradeoff_score_ratio",
]

print(df[cols].sort_values("tradeoff_score_ratio", ascending=False).head(50).to_string(index=False))
PY
```

---

## What Is Not Included

The following files and folders are intentionally excluded:

```text
virtual environments
Hugging Face caches
downloaded model weights
raw datasets
large checkpoint folders
LoRA adapter weights
.pt / .bin / .safetensors files
full heavy run directories
```

This keeps the repository lightweight and suitable for code and results review.

---

## Notes on Reproducibility

Selected logs and checkpoint metrics are included under:

```text
examples/
```

These files illustrate the structure of raw experimental outputs without including large model artifacts.

For complete reproduction:

1. install dependencies,
2. download the required local models,
3. run the main v5 script,
4. compare generated reports with the provided summarized reports.

---

## Earlier Experiments

In addition to the main v5 LLM unlearning experiment, the repository includes scripts and summaries for earlier experiments:

```text
synthetic transformer fractional-gradient experiments
real-text fractional-gradient experiments
fractional optimizer scenario sweeps
LLM proof-of-concept unlearning experiments v1-v4
```

These earlier experiments are retained as development history and supporting evidence for the final v5 setup.

---

## Limitations

This repository is a research artifact, not a production unlearning framework.

Current limitations include:

```text
- Synthetic fact data are used for the controlled unlearning setup.
- Exact-match generation can be brittle and should be interpreted alongside NLL metrics.
- Results depend on model size, LoRA configuration, and teach-stage memorization quality.
- Full-model fractional updates were unstable in these experiments.
- Larger models and real-world unlearning benchmarks require additional evaluation.
```

---

## Recommended Reading of Results

For a quick review, start with:

```text
reports/llm_experiments/llm_valence_fractional_poc_v5_qwen_curves/reports/poc_v5_summary.xlsx
```

Then inspect:

```text
poc_v5_checkpoint_curves.csv
```

to compare forgetting–retention curves over epochs.

The most relevant comparison is:

```text
A2 gradient_ascent
vs
B4 valence_frac_lora_mix
```

especially on:

```text
gpt_neo_125m
qwen2p5_0p5b_instruct
```

---


## Citation

If this repository supports a paper or preprint, add the final citation here.

```bibtex
@misc{fractional_lora_unlearning,
  title  = {Valence-Aware Fractional LoRA Unlearning},
  author = {Anonymous},
  year   = {2026},
  note   = {Research artifact}
}
```
