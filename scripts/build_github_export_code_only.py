from __future__ import annotations

import os
import shutil
import json
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/home/user/fractional_unlearning")
EXPORT = ROOT / "github_export_fractional_unlearning_full"

RESET_EXPORT = True

MAX_FILES_PER_EXAMPLE_SOURCE = 200

EXCLUDE_BIG_SUFFIXES = {
    ".pt",
    ".pth",
    ".bin",
    ".safetensors",
    ".tar",
    ".gz",
    ".zip",
    ".7z",
}

EXCLUDE_DIR_NAMES = {
    "__pycache__",
    ".ipynb_checkpoints",
    ".git",
    ".venv",
    ".venv_a100",
    "venv",
    "env",
    "hf_llm_models",
    "hf_models",
    "hf_cache",
    "hf_cache_llm",
    "real_text_datasets",
    "teach_checkpoints",
    "model_adapter",
    "model_adapter_final",
}


# ============================================================
# OUTPUT DIRS
# ============================================================

if EXPORT.exists() and RESET_EXPORT:
    shutil.rmtree(EXPORT)

DIRS = {
    "scripts": EXPORT / "scripts",
    "notebooks": EXPORT / "notebooks",
    "reports_root": EXPORT / "reports" / "root_summaries",
    "reports_llm": EXPORT / "reports" / "llm_experiments",
    "reports_real_text": EXPORT / "reports" / "real_text_experiments",
    "reports_transformer": EXPORT / "reports" / "transformer_fractional",
    "reports_structure": EXPORT / "reports" / "structure_reports",
    "examples": EXPORT / "examples",
    "docs": EXPORT / "docs",
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

def is_excluded_path(path: Path) -> bool:
    parts = set(path.parts)

    if parts & EXCLUDE_DIR_NAMES:
        return True

    if path.suffix.lower() in EXCLUDE_BIG_SUFFIXES:
        return True

    return False


def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024**2
    except Exception:
        return 0.0


def copy_file(src: Path, dst: Path, max_mb: float | None = 50.0) -> bool:
    if not src.exists() or not src.is_file():
        return False

    if is_excluded_path(src):
        return False

    if max_mb is not None and file_size_mb(src) > max_mb:
        print(f"SKIP large file > {max_mb} MB: {src} ({file_size_mb(src):.2f} MB)")
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def copy_top_level_by_patterns(patterns: list[str], dst_root: Path, max_mb: float | None = 50.0) -> int:
    copied = 0

    for pat in patterns:
        for src in sorted(ROOT.glob(pat)):
            if src.is_file():
                ok = copy_file(src, dst_root / src.name, max_mb=max_mb)
                copied += int(ok)

    return copied


def copy_tree_files(
    src_root: Path,
    dst_root: Path,
    patterns: list[str],
    max_files: int | None = None,
    max_mb: float | None = 50.0,
) -> int:
    if not src_root.exists():
        return 0

    copied = 0

    for pat in patterns:
        for src in sorted(src_root.rglob(pat)):
            if not src.is_file():
                continue

            if is_excluded_path(src):
                continue

            rel = src.relative_to(src_root)
            dst = dst_root / rel

            ok = copy_file(src, dst, max_mb=max_mb)

            if ok:
                copied += 1

            if max_files is not None and copied >= max_files:
                return copied

    return copied


def copy_tree_all_light_files(
    src_root: Path,
    dst_root: Path,
    allowed_suffixes: set[str],
    max_files: int | None = None,
    max_mb: float | None = 50.0,
) -> int:
    if not src_root.exists():
        return 0

    copied = 0

    for src in sorted(src_root.rglob("*")):
        if not src.is_file():
            continue

        if is_excluded_path(src):
            continue

        if src.suffix.lower() not in allowed_suffixes:
            continue

        rel = src.relative_to(src_root)
        dst = dst_root / rel

        ok = copy_file(src, dst, max_mb=max_mb)

        if ok:
            copied += 1

        if max_files is not None and copied >= max_files:
            return copied

    return copied


def dir_size_bytes(path: Path) -> int:
    total = 0

    if not path.exists():
        return 0

    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass

    return total


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)

    for u in units:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024

    return f"{x:.2f} PB"


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "created": datetime.now().isoformat(timespec="seconds"),
    "root": str(ROOT),
    "export": str(EXPORT),
    "sections": {},
    "notes": [
        "This export intentionally excludes model weights, virtual environments, raw datasets, full checkpoints, and heavy run folders.",
        "README is intentionally not generated in this version.",
    ],
}


# ============================================================
# 1. SCRIPTS
# ============================================================

script_patterns = [
    "*.py",
]

copied_scripts = copy_top_level_by_patterns(
    patterns=script_patterns,
    dst_root=DIRS["scripts"],
    max_mb=20.0,
)

manifest["sections"]["scripts"] = copied_scripts


# ============================================================
# 2. NOTEBOOKS
# ============================================================

copied_notebooks = copy_top_level_by_patterns(
    patterns=["*.ipynb"],
    dst_root=DIRS["notebooks"],
    max_mb=50.0,
)

manifest["sections"]["notebooks"] = copied_notebooks


# ============================================================
# 3. ROOT SUMMARY / REPORT FILES
# ============================================================

root_report_patterns = [
    "*summary*.csv",
    "*summary*.json",
    "*summary*.xlsx",
    "*statistics*.csv",
    "*statistics*.json",
    "*statistics*.xlsx",
    "*results*.csv",
    "*results*.json",
    "*results*.xlsx",
    "*report*.txt",
    "*report*.csv",
    "*report*.json",
    "*report*.xlsx",
    "FINAL_ANALYSIS_*",
    "concept_*",
    "downloaded_*_report.*",
    "encoder_models_final_report.*",
    "smoke_test_*_report.*",
    "gpu_diagnostics_report.txt",
    "transformer_fractional_results.xlsx",
    "ALL_fractional_experiments_all_runs.csv",
    "ALL_fractional_experiments_statistics.xlsx",
    "fractional_scenarios_all_jobs.csv",
    "fractional_scenarios_statistics.xlsx",
    "experiment_statistics.xlsx",
    "experiment_statistics_summary.csv",
]

copied_root_reports = copy_top_level_by_patterns(
    patterns=root_report_patterns,
    dst_root=DIRS["reports_root"],
    max_mb=50.0,
)

manifest["sections"]["root_reports"] = copied_root_reports


# ============================================================
# 4. LLM EXPERIMENT REPORTS
# ============================================================

llm_dirs = [
    "llm_valence_fractional_poc",
    "llm_valence_fractional_poc_v2",
    "llm_valence_fractional_poc_v3_all_except_qwen",
    "llm_valence_fractional_poc_v4_sweep",
    "llm_valence_fractional_poc_v5_qwen_curves",
]

llm_report_count = 0

for name in llm_dirs:
    src = ROOT / name / "reports"
    dst = DIRS["reports_llm"] / name / "reports"

    if src.exists():
        copied = copy_tree_all_light_files(
            src_root=src,
            dst_root=dst,
            allowed_suffixes={".csv", ".json", ".xlsx", ".txt"},
            max_files=None,
            max_mb=50.0,
        )
        llm_report_count += copied

manifest["sections"]["llm_reports"] = llm_report_count


# ============================================================
# 5. STRUCTURE / INSPECTION REPORTS
# ============================================================

structure_sources = [
    ROOT / "unlearning_benchmarks" / "structure_reports",
    ROOT / "_folder_check",
]

structure_count = 0

for src in structure_sources:
    if src.exists():
        copied = copy_tree_all_light_files(
            src_root=src,
            dst_root=DIRS["reports_structure"] / src.name,
            allowed_suffixes={".csv", ".json", ".xlsx", ".txt"},
            max_files=None,
            max_mb=50.0,
        )
        structure_count += copied

manifest["sections"]["structure_reports"] = structure_count


# ============================================================
# 6. SELECTED EXAMPLES FROM RUNS
# ============================================================

example_patterns = [
    "log.txt",
    "history.json",
    "done.json",
    "checkpoint_metrics.json",
    "*_metrics.json",
]

example_sources = [
    ROOT / "llm_valence_fractional_poc_v5_qwen_curves" / "runs",
    ROOT / "llm_valence_fractional_poc_v4_sweep" / "runs",
    ROOT / "llm_valence_fractional_poc_v3_all_except_qwen" / "runs",
    ROOT / "llm_valence_fractional_poc_v2" / "runs",
    ROOT / "llm_valence_fractional_poc" / "runs",

    ROOT / "runs_real_text_fractional_concept",
    ROOT / "runs_real_text_fractional_dynamic",
    ROOT / "runs_fractional_focused_final",
    ROOT / "runs_fractional_hypothesis_v2",
    ROOT / "runs_fractional_scenarios",
    ROOT / "runs_transformer_fractional",
]

examples_total = 0

for src in example_sources:
    if not src.exists():
        continue

    dst = DIRS["examples"] / src.name

    copied = copy_tree_files(
        src_root=src,
        dst_root=dst,
        patterns=example_patterns,
        max_files=MAX_FILES_PER_EXAMPLE_SOURCE,
        max_mb=10.0,
    )

    examples_total += copied

manifest["sections"]["selected_examples"] = examples_total


# ============================================================
# 7. GENERATED_PT JSON METADATA ONLY
# ============================================================

generated_pt = ROOT / "generated_pt"

generated_pt_meta_count = 0

if generated_pt.exists():
    dst = EXPORT / "reports" / "generated_pt_metadata"
    dst.mkdir(parents=True, exist_ok=True)

    for src in sorted(generated_pt.glob("*.json")):
        ok = copy_file(src, dst / src.name, max_mb=10.0)
        generated_pt_meta_count += int(ok)

manifest["sections"]["generated_pt_json_metadata"] = generated_pt_meta_count


# ============================================================
# 8. GITIGNORE
# ============================================================

gitignore = """\
__pycache__/
.ipynb_checkpoints/
.venv*/
env/
venv/

hf_llm_models/
hf_models/
hf_cache/
hf_cache_llm/

real_text_datasets/
unlearning_benchmarks/raw/
unlearning_benchmarks/repos/

teach_checkpoints/
**/teach_checkpoints/
**/model_adapter/
**/model_adapter_final/
**/adapter_model.safetensors

*.safetensors
*.bin
*.pt
*.pth
*.tar.gz
*.zip
*.7z

# Full heavy run folders should not be committed.
runs_real_text_fractional_dynamic/
runs_real_text_fractional_concept/
runs_fractional_focused_final/
runs_fractional_hypothesis_v2/
runs_fractional_scenarios/
runs_transformer_fractional/
"""

(EXPORT / ".gitignore").write_text(gitignore, encoding="utf-8")


# ============================================================
# 9. MANIFEST
# ============================================================

manifest["export_size_bytes"] = dir_size_bytes(EXPORT)
manifest["export_size_human"] = human_size(manifest["export_size_bytes"])

manifest_path = DIRS["docs"] / "export_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=4, ensure_ascii=False),
    encoding="utf-8",
)


# ============================================================
# 10. PRINT SUMMARY
# ============================================================

print("=" * 100)
print("EXPORT DONE")
print("=" * 100)
print("Export:", EXPORT)
print("Size:", manifest["export_size_human"])
print()
print("Sections:")
print(json.dumps(manifest["sections"], indent=4, ensure_ascii=False))
print()
print("Manifest:", manifest_path)
print()
print("Folder size:")
os.system(f"du -h --max-depth=2 {EXPORT} | sort -hr | head -80")
print()
print("Large files check:")
os.system(f"find {EXPORT} -type f -size +50M -print")
print()
print("Next:")
print(f"cd {EXPORT}")
print("git init")
print("git add .")
print("git status")