# Valence-Aware Fractional Gradient Memory for LoRA-Based Unlearning

This repository contains code, summaries, and selected logs for experiments on selective fractional-gradient memory. The experimental campaign includes synthetic transformer optimization, real-text classification with pretrained encoders, and LoRA-based LLM unlearning.

Large artifacts are intentionally excluded, including model weights, checkpoints, Hugging Face caches, raw datasets, and virtual environments.

## Main run directories

- `run_logs/runs_transformer_fractional`
- `run_logs/runs_fractional_scenarios`
- `run_logs/runs_fractional_hypothesis_v2`
- `run_logs/runs_fractional_focused_final`
- `run_logs/runs_real_text_fractional_dynamic`
- `run_logs/runs_real_text_fractional_concept`
- `run_logs/llm_valence_fractional_poc`
- `run_logs/llm_valence_fractional_poc_v2`
- `run_logs/llm_valence_fractional_poc_v3_all_except_qwen`
- `run_logs/llm_valence_fractional_poc_v4_sweep`
- `run_logs/llm_valence_fractional_poc_v5_qwen_curves`

## Reports

Aggregated CSV, JSON, TXT, and XLSX summaries are stored in `reports/root_summaries`.

## Notes

The repository is a lightweight research artifact. It is intended to preserve scripts, run configurations, logs, histories, and summarized results, not full model checkpoints.
