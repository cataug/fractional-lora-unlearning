from __future__ import annotations

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict


# ============================================================
# ROOTS / ENV
# ============================================================

ROOT = Path("/home/user/fractional_unlearning")

OUT_ROOT = ROOT / "unlearning_benchmarks"
HF_CACHE = OUT_ROOT / "hf_cache"
RAW_DIR = OUT_ROOT / "raw"
EXPORT_DIR = OUT_ROOT / "exported_jsonl_csv"
REPO_DIR = OUT_ROOT / "repos"
REPORT_DIR = OUT_ROOT / "reports"

for p in [OUT_ROOT, HF_CACHE, RAW_DIR, EXPORT_DIR, REPO_DIR, REPORT_DIR]:
    p.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(HF_CACHE / "hf_home")
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE / "datasets")
os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ============================================================
# CONFIG
# ============================================================

FORCE_REDOWNLOAD_DATASETS = False
FORCE_RECLONE_REPOS = False

# Core benchmark datasets.
# Some may change configs/splits over time, so the script is defensive.
HF_DATASETS = [
    {
        "alias": "tofu",
        "hf_id": "locuslab/TOFU",
        "configs": [None],
        "required": True,
        "note": "TOFU benchmark: fictitious author QA pairs for LLM unlearning.",
    },

    # WMDP public MCQ benchmark.
    {
        "alias": "wmdp",
        "hf_id": "cais/wmdp",
        "configs": [None],
        "required": False,
        "note": "WMDP public MCQ benchmark.",
    },

    # Sometimes useful mirrored/converted version if cais/wmdp changes.
    {
        "alias": "wmdp_mirror_joschka",
        "hf_id": "Joschka/wmdp",
        "configs": [None],
        "required": False,
        "note": "Mirror/converted WMDP dataset fallback.",
    },

    # WMDP corpora may be useful but can have access/format constraints.
    {
        "alias": "wmdp_corpora",
        "hf_id": "cais/wmdp-corpora",
        "configs": [None],
        "required": False,
        "note": "WMDP corpora if accessible.",
    },
    {
        "alias": "wmdp_bio_forget_corpus",
        "hf_id": "cais/wmdp-bio-forget-corpus",
        "configs": [None],
        "required": False,
        "note": "May be gated/restricted. Failure is acceptable.",
    },
    {
        "alias": "wmdp_cyber_forget_corpus",
        "hf_id": "cais/wmdp-cyber-forget-corpus",
        "configs": [None],
        "required": False,
        "note": "May be gated/restricted. Failure is acceptable.",
    },
]

# Repositories: code/reference/evaluation scripts.
REPOS = [
    {
        "alias": "open_unlearning",
        "url": "https://github.com/locuslab/open-unlearning.git",
        "note": "Unified framework for TOFU/MUSE/WMDP.",
    },
    {
        "alias": "tofu_repo",
        "url": "https://github.com/locuslab/tofu.git",
        "note": "Original TOFU repository.",
    },
    {
        "alias": "muse_bench",
        "url": "https://github.com/jaechan-repo/muse_bench.git",
        "note": "MUSE benchmark repository.",
    },
    {
        "alias": "wmdp_repo",
        "url": "https://github.com/centerforaisafety/wmdp.git",
        "note": "WMDP benchmark repository.",
    },
]


# ============================================================
# UTILS
# ============================================================

def print_env() -> None:
    print("=" * 100)
    print("ENVIRONMENT")
    print("=" * 100)
    print("ROOT:", ROOT)
    print("python:", sys.executable)
    print("OUT_ROOT:", OUT_ROOT)
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
    print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))

    if ".venv_a100" not in sys.executable:
        print("\nWARNING: this does not look like .venv_a100 Python.")
        print("Expected:")
        print("  /home/user/fractional_unlearning/.venv_a100/bin/python")
        print("Current:")
        print(" ", sys.executable)

    print("=" * 100)


def dataset_to_dict_of_splits(ds: Any) -> Dict[str, Dataset]:
    if isinstance(ds, DatasetDict):
        return dict(ds)

    if isinstance(ds, Dataset):
        return {"train": ds}

    # Some datasets may return dict-like DatasetDict subclass.
    try:
        return {k: v for k, v in ds.items()}
    except Exception:
        raise TypeError(f"Unsupported dataset object: {type(ds)}")


def save_split_jsonl(split: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for row in split:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_split_csv(split: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = split.to_pandas()
    df.to_csv(path, index=False)


def save_dataset_exports(alias: str, config_name: Optional[str], ds: Any) -> Dict[str, Any]:
    config_safe = config_name if config_name is not None else "default"

    raw_path = RAW_DIR / alias / config_safe / "datasetdict"
    export_path = EXPORT_DIR / alias / config_safe

    if raw_path.exists() and FORCE_REDOWNLOAD_DATASETS:
        shutil.rmtree(raw_path)

    if export_path.exists() and FORCE_REDOWNLOAD_DATASETS:
        shutil.rmtree(export_path)

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.mkdir(parents=True, exist_ok=True)

    print("Saving dataset to disk:", raw_path)
    ds.save_to_disk(str(raw_path))

    splits = dataset_to_dict_of_splits(ds)

    split_infos = []

    for split_name, split in splits.items():
        jsonl_path = export_path / f"{split_name}.jsonl"
        csv_path = export_path / f"{split_name}.csv"

        print(f"Exporting split={split_name} rows={len(split)}")
        save_split_jsonl(split, jsonl_path)
        save_split_csv(split, csv_path)

        split_infos.append({
            "split": split_name,
            "num_rows": len(split),
            "columns": split.column_names,
            "jsonl_path": str(jsonl_path),
            "csv_path": str(csv_path),
        })

    info = {
        "alias": alias,
        "config": config_safe,
        "raw_path": str(raw_path),
        "export_path": str(export_path),
        "splits": split_infos,
    }

    with open(export_path / "dataset_export_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4, ensure_ascii=False)

    return info


def load_hf_dataset_one(spec: Dict[str, Any], config_name: Optional[str]) -> Dict[str, Any]:
    alias = spec["alias"]
    hf_id = spec["hf_id"]

    result = {
        "type": "dataset",
        "alias": alias,
        "hf_id": hf_id,
        "config": config_name,
        "status": "failed",
        "required": spec.get("required", False),
        "note": spec.get("note"),
        "error": None,
        "export_info": None,
    }

    print("\n" + "=" * 100)
    print("DATASET:", alias)
    print("HF ID:", hf_id)
    print("CONFIG:", config_name)
    print("=" * 100)

    try:
        kwargs = {
            "path": hf_id,
            "cache_dir": str(HF_CACHE / "datasets"),
            "trust_remote_code": False,
        }

        if config_name is not None:
            ds = load_dataset(config_name=config_name, **kwargs)
        else:
            ds = load_dataset(**kwargs)

        export_info = save_dataset_exports(alias, config_name, ds)

        result["status"] = "ok"
        result["export_info"] = export_info

    except Exception as e:
        result["error"] = repr(e)
        print("FAILED DATASET:", alias, hf_id, config_name)
        print("ERROR:", repr(e))

    return result


def clone_repo(spec: Dict[str, Any]) -> Dict[str, Any]:
    alias = spec["alias"]
    url = spec["url"]
    target = REPO_DIR / alias

    result = {
        "type": "repo",
        "alias": alias,
        "url": url,
        "target": str(target),
        "status": "failed",
        "note": spec.get("note"),
        "error": None,
    }

    print("\n" + "=" * 100)
    print("REPO:", alias)
    print("URL:", url)
    print("TARGET:", target)
    print("=" * 100)

    try:
        if target.exists() and FORCE_RECLONE_REPOS:
            shutil.rmtree(target)

        if target.exists() and (target / ".git").exists():
            print("Repo exists, pulling latest...")
            subprocess.run(
                ["git", "-C", str(target), "pull", "--ff-only"],
                check=False,
            )
        elif target.exists():
            print("Target exists but is not a git repo, skipping:", target)
        else:
            subprocess.run(
                ["git", "clone", "--depth", "1", url, str(target)],
                check=True,
            )

        result["status"] = "ok"

    except Exception as e:
        result["error"] = repr(e)
        print("FAILED REPO:", alias)
        print("ERROR:", repr(e))

    return result


# ============================================================
# QUICK INSPECTION
# ============================================================

def inspect_exports() -> List[Dict[str, Any]]:
    rows = []

    for info_path in EXPORT_DIR.rglob("dataset_export_info.json"):
        try:
            info = json.loads(info_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for s in info.get("splits", []):
            rows.append({
                "alias": info.get("alias"),
                "config": info.get("config"),
                "split": s.get("split"),
                "num_rows": s.get("num_rows"),
                "columns": ";".join(s.get("columns", [])),
                "jsonl_path": s.get("jsonl_path"),
                "csv_path": s.get("csv_path"),
            })

    return rows


def create_tofu_normalized_if_available() -> None:
    """
    Creates a simple normalized TOFU jsonl if locuslab/TOFU downloaded.
    This helps with later PoC code.
    """

    tofu_root = EXPORT_DIR / "tofu"
    if not tofu_root.exists():
        print("No TOFU export found, skipping normalized TOFU.")
        return

    out_dir = OUT_ROOT / "normalized" / "tofu"
    out_dir.mkdir(parents=True, exist_ok=True)

    normalized_rows = []

    for jsonl_path in tofu_root.rglob("*.jsonl"):
        split_name = jsonl_path.stem

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)

                    question = (
                        row.get("question")
                        or row.get("prompt")
                        or row.get("Question")
                        or row.get("input")
                    )

                    answer = (
                        row.get("answer")
                        or row.get("completion")
                        or row.get("Answer")
                        or row.get("output")
                    )

                    if question is None and answer is None:
                        # Preserve unknown format.
                        text = json.dumps(row, ensure_ascii=False)
                        question = text
                        answer = ""

                    normalized_rows.append({
                        "source": "TOFU",
                        "original_split": split_name,
                        "question": question,
                        "answer": answer,
                        "text_lm": f"Question: {question}\nAnswer: {answer}",
                        "raw": row,
                    })

        except Exception as e:
            print("Failed reading TOFU jsonl:", jsonl_path, repr(e))

    out_path = out_dir / "tofu_normalized_all.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for row in normalized_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    pd.DataFrame([
        {
            "source": r["source"],
            "original_split": r["original_split"],
            "question": r["question"],
            "answer": r["answer"],
            "text_lm": r["text_lm"],
        }
        for r in normalized_rows
    ]).to_csv(out_dir / "tofu_normalized_all.csv", index=False)

    print("Created normalized TOFU:")
    print(" ", out_path)
    print("Rows:", len(normalized_rows))


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print_env()

    results = []

    print("\n" + "#" * 100)
    print("STEP 1: DOWNLOAD HF DATASETS")
    print("#" * 100)

    for spec in HF_DATASETS:
        for config_name in spec.get("configs", [None]):
            result = load_hf_dataset_one(spec, config_name)
            results.append(result)

    print("\n" + "#" * 100)
    print("STEP 2: CLONE REPOSITORIES")
    print("#" * 100)

    for spec in REPOS:
        result = clone_repo(spec)
        results.append(result)

    print("\n" + "#" * 100)
    print("STEP 3: INSPECT EXPORTS")
    print("#" * 100)

    export_rows = inspect_exports()
    export_df = pd.DataFrame(export_rows)

    export_report_csv = REPORT_DIR / "benchmark_exports_index.csv"
    export_report_json = REPORT_DIR / "benchmark_exports_index.json"

    export_df.to_csv(export_report_csv, index=False)

    with open(export_report_json, "w", encoding="utf-8") as f:
        json.dump(export_rows, f, indent=4, ensure_ascii=False)

    print(export_df)

    print("\n" + "#" * 100)
    print("STEP 4: CREATE NORMALIZED TOFU")
    print("#" * 100)

    create_tofu_normalized_if_available()

    report_json = REPORT_DIR / "download_unlearning_benchmarks_report.json"
    report_csv = REPORT_DIR / "download_unlearning_benchmarks_report.csv"

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    pd.DataFrame(results).to_csv(report_csv, index=False)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)

    print("OUT_ROOT:", OUT_ROOT)
    print("RAW_DIR:", RAW_DIR)
    print("EXPORT_DIR:", EXPORT_DIR)
    print("REPO_DIR:", REPO_DIR)
    print("REPORT_DIR:", REPORT_DIR)

    print("\nReports:")
    print(" ", report_json)
    print(" ", report_csv)
    print(" ", export_report_csv)
    print(" ", export_report_json)

    print("\nStatus summary:")
    for r in results:
        print(
            f"{r['type']:8s} | "
            f"{r['alias']:30s} | "
            f"{r['status']:8s} | "
            f"{r.get('hf_id', r.get('url'))} | "
            f"error={r.get('error')}"
        )


if __name__ == "__main__":
    main()