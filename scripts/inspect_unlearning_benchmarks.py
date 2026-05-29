from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# CONFIG
# ============================================================

ROOT = Path("/home/tahiti/Malashin_Projects")

BENCH_ROOT = ROOT / "unlearning_benchmarks"

RAW_DIR = BENCH_ROOT / "raw"
EXPORT_DIR = BENCH_ROOT / "exported_jsonl_csv"
NORMALIZED_DIR = BENCH_ROOT / "normalized"
REPO_DIR = BENCH_ROOT / "repos"
REPORT_DIR = BENCH_ROOT / "reports"

OUT_DIR = BENCH_ROOT / "structure_reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TXT = OUT_DIR / "unlearning_benchmarks_structure_report.txt"
OUT_JSON = OUT_DIR / "unlearning_benchmarks_structure_report.json"
OUT_CSV_FILES = OUT_DIR / "files_index.csv"
OUT_CSV_DATASETS = OUT_DIR / "datasets_index.csv"
OUT_CSV_COLUMNS = OUT_DIR / "columns_index.csv"
OUT_CSV_EXAMPLES = OUT_DIR / "examples_index.csv"
OUT_CSV_REPOS = OUT_DIR / "repos_index.csv"
OUT_XLSX = OUT_DIR / "unlearning_benchmarks_structure_report.xlsx"

MAX_EXAMPLES_PER_FILE = 3
MAX_TEXT_PREVIEW = 500

print("ROOT:", ROOT)
print("BENCH_ROOT:", BENCH_ROOT)


# ============================================================
# HELPERS
# ============================================================

def file_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / 1024**2
    except Exception:
        return 0.0


def dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0

    total = 0

    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except Exception:
                pass

    return total / 1024**2


def safe_read_json(path: Path) -> Optional[Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def preview_value(x: Any, max_len: int = MAX_TEXT_PREVIEW) -> str:
    if isinstance(x, (dict, list)):
        s = json.dumps(x, ensure_ascii=False)
    else:
        s = str(x)

    s = s.replace("\n", "\\n")

    if len(s) > max_len:
        return s[:max_len] + "..."

    return s


def read_jsonl_examples(path: Path, max_examples: int = MAX_EXAMPLES_PER_FILE) -> List[Dict[str, Any]]:
    rows = []

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i >= max_examples:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    rows.append(json.loads(line))
                except Exception:
                    rows.append({"__raw_line__": line})

    except Exception as e:
        rows.append({"__read_error__": repr(e)})

    return rows


def count_jsonl_lines(path: Path) -> int:
    n = 0

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for _ in f:
                n += 1
    except Exception:
        return -1

    return n


def read_csv_head(path: Path, max_examples: int = MAX_EXAMPLES_PER_FILE) -> tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        df = pd.read_csv(path, nrows=max_examples)
        return df, None
    except Exception as e:
        return None, repr(e)


def get_csv_row_count(path: Path) -> int:
    try:
        # minus header
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return -1


def flatten_example_rows(
    source_file: Path,
    dataset_alias: str,
    config: str,
    split: str,
    examples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    out = []

    for idx, row in enumerate(examples):
        if not isinstance(row, dict):
            row = {"value": row}

        keys = list(row.keys())

        compact = {
            k: preview_value(row.get(k))
            for k in keys[:20]
        }

        out.append({
            "dataset_alias": dataset_alias,
            "config": config,
            "split": split,
            "source_file": str(source_file),
            "example_idx": idx,
            "keys": ";".join(keys),
            "example_json_preview": json.dumps(compact, ensure_ascii=False),
        })

    return out


def infer_alias_config_split_from_export_path(path: Path) -> tuple[str, str, str]:
    """
    Expected:
    exported_jsonl_csv / alias / config / split.jsonl
    """
    try:
        rel = path.relative_to(EXPORT_DIR)
        parts = rel.parts

        alias = parts[0] if len(parts) > 0 else ""
        config = parts[1] if len(parts) > 1 else ""
        split = path.stem

        return alias, config, split

    except Exception:
        return "", "", path.stem


# ============================================================
# ENV REPORT
# ============================================================

env_info = {
    "python": sys.executable,
    "cwd": str(Path.cwd()),
    "root": str(ROOT),
    "bench_root": str(BENCH_ROOT),
    "bench_root_exists": BENCH_ROOT.exists(),
    "raw_dir_exists": RAW_DIR.exists(),
    "export_dir_exists": EXPORT_DIR.exists(),
    "normalized_dir_exists": NORMALIZED_DIR.exists(),
    "repo_dir_exists": REPO_DIR.exists(),
    "report_dir_exists": REPORT_DIR.exists(),
    "bench_root_size_mb": dir_size_mb(BENCH_ROOT),
}

print("ENV:")
print(json.dumps(env_info, indent=4, ensure_ascii=False))


# ============================================================
# FILE INDEX
# ============================================================

file_rows = []

if BENCH_ROOT.exists():
    for p in sorted(BENCH_ROOT.rglob("*")):
        if p.is_file():
            rel = p.relative_to(BENCH_ROOT)

            file_rows.append({
                "relative_path": str(rel),
                "absolute_path": str(p),
                "suffix": p.suffix.lower(),
                "size_mb": file_size_mb(p),
                "parent": str(p.parent),
            })

files_df = pd.DataFrame(file_rows)

print("Files found:", len(files_df))


# ============================================================
# DATASET EXPORT INDEX
# ============================================================

dataset_rows = []
column_rows = []
example_rows = []

jsonl_files = sorted(EXPORT_DIR.rglob("*.jsonl")) if EXPORT_DIR.exists() else []

for jsonl_path in jsonl_files:
    alias, config, split = infer_alias_config_split_from_export_path(jsonl_path)

    n_rows = count_jsonl_lines(jsonl_path)
    examples = read_jsonl_examples(jsonl_path)

    keys = []
    for ex in examples:
        if isinstance(ex, dict):
            for k in ex.keys():
                if k not in keys:
                    keys.append(k)

    dataset_rows.append({
        "dataset_alias": alias,
        "config": config,
        "split": split,
        "file_type": "jsonl",
        "num_rows": n_rows,
        "num_columns_in_examples": len(keys),
        "columns_in_examples": ";".join(keys),
        "path": str(jsonl_path),
        "size_mb": file_size_mb(jsonl_path),
    })

    for k in keys:
        values_preview = []

        for ex in examples:
            if isinstance(ex, dict) and k in ex:
                values_preview.append(preview_value(ex[k], max_len=200))

        column_rows.append({
            "dataset_alias": alias,
            "config": config,
            "split": split,
            "column": k,
            "source_file": str(jsonl_path),
            "preview_values": " || ".join(values_preview[:3]),
        })

    example_rows.extend(
        flatten_example_rows(
            source_file=jsonl_path,
            dataset_alias=alias,
            config=config,
            split=split,
            examples=examples,
        )
    )

csv_files = sorted(EXPORT_DIR.rglob("*.csv")) if EXPORT_DIR.exists() else []

for csv_path in csv_files:
    alias, config, split = infer_alias_config_split_from_export_path(csv_path)

    df_head, err = read_csv_head(csv_path)

    if df_head is not None:
        cols = list(df_head.columns)
        n_rows = get_csv_row_count(csv_path)

        dataset_rows.append({
            "dataset_alias": alias,
            "config": config,
            "split": split,
            "file_type": "csv",
            "num_rows": n_rows,
            "num_columns_in_examples": len(cols),
            "columns_in_examples": ";".join(cols),
            "path": str(csv_path),
            "size_mb": file_size_mb(csv_path),
            "read_error": None,
        })

        for c in cols:
            previews = [
                preview_value(v, max_len=200)
                for v in df_head[c].tolist()
            ]

            column_rows.append({
                "dataset_alias": alias,
                "config": config,
                "split": split,
                "column": c,
                "source_file": str(csv_path),
                "preview_values": " || ".join(previews),
            })

        for i, row in df_head.iterrows():
            compact = {
                c: preview_value(row[c])
                for c in cols[:20]
            }

            example_rows.append({
                "dataset_alias": alias,
                "config": config,
                "split": split,
                "source_file": str(csv_path),
                "example_idx": int(i),
                "keys": ";".join(cols),
                "example_json_preview": json.dumps(compact, ensure_ascii=False),
            })

    else:
        dataset_rows.append({
            "dataset_alias": alias,
            "config": config,
            "split": split,
            "file_type": "csv",
            "num_rows": -1,
            "num_columns_in_examples": 0,
            "columns_in_examples": "",
            "path": str(csv_path),
            "size_mb": file_size_mb(csv_path),
            "read_error": err,
        })


datasets_df = pd.DataFrame(dataset_rows)
columns_df = pd.DataFrame(column_rows)
examples_df = pd.DataFrame(example_rows)

print("Dataset export rows:", datasets_df.shape)
print("Columns rows:", columns_df.shape)
print("Examples rows:", examples_df.shape)


# ============================================================
# RAW DATASET INFO
# ============================================================

raw_info_rows = []

for info_path in sorted(BENCH_ROOT.rglob("dataset_export_info.json")):
    info = safe_read_json(info_path)

    if not isinstance(info, dict):
        continue

    for split_info in info.get("splits", []):
        raw_info_rows.append({
            "info_path": str(info_path),
            "alias": info.get("alias"),
            "config": info.get("config"),
            "raw_path": info.get("raw_path"),
            "export_path": info.get("export_path"),
            "split": split_info.get("split"),
            "num_rows": split_info.get("num_rows"),
            "columns": ";".join(split_info.get("columns", [])),
            "jsonl_path": split_info.get("jsonl_path"),
            "csv_path": split_info.get("csv_path"),
        })

raw_info_df = pd.DataFrame(raw_info_rows)


# ============================================================
# NORMALIZED INDEX
# ============================================================

normalized_rows = []
normalized_example_rows = []

if NORMALIZED_DIR.exists():
    for p in sorted(NORMALIZED_DIR.rglob("*")):
        if not p.is_file():
            continue

        if p.suffix.lower() == ".jsonl":
            n_rows = count_jsonl_lines(p)
            examples = read_jsonl_examples(p)

            keys = []
            for ex in examples:
                if isinstance(ex, dict):
                    for k in ex.keys():
                        if k not in keys:
                            keys.append(k)

            normalized_rows.append({
                "relative_path": str(p.relative_to(NORMALIZED_DIR)),
                "absolute_path": str(p),
                "file_type": "jsonl",
                "num_rows": n_rows,
                "columns_in_examples": ";".join(keys),
                "size_mb": file_size_mb(p),
            })

            normalized_example_rows.extend(
                flatten_example_rows(
                    source_file=p,
                    dataset_alias="normalized",
                    config=str(p.relative_to(NORMALIZED_DIR).parent),
                    split=p.stem,
                    examples=examples,
                )
            )

        elif p.suffix.lower() == ".csv":
            df_head, err = read_csv_head(p)
            cols = list(df_head.columns) if df_head is not None else []

            normalized_rows.append({
                "relative_path": str(p.relative_to(NORMALIZED_DIR)),
                "absolute_path": str(p),
                "file_type": "csv",
                "num_rows": get_csv_row_count(p) if df_head is not None else -1,
                "columns_in_examples": ";".join(cols),
                "size_mb": file_size_mb(p),
                "read_error": err,
            })

normalized_df = pd.DataFrame(normalized_rows)
normalized_examples_df = pd.DataFrame(normalized_example_rows)


# ============================================================
# REPO INDEX
# ============================================================

repo_rows = []

if REPO_DIR.exists():
    for repo in sorted(REPO_DIR.iterdir()):
        if not repo.is_dir():
            continue

        git_exists = (repo / ".git").exists()

        top_files = []
        for p in sorted(repo.iterdir()):
            if p.name == ".git":
                continue
            top_files.append(p.name)

        readme_candidates = [
            repo / "README.md",
            repo / "readme.md",
            repo / "README.rst",
        ]

        readme_preview = ""

        for rp in readme_candidates:
            if rp.exists():
                try:
                    readme_preview = rp.read_text(encoding="utf-8", errors="ignore")[:2000]
                except Exception:
                    readme_preview = ""
                break

        repo_rows.append({
            "repo_name": repo.name,
            "path": str(repo),
            "is_git_repo": git_exists,
            "size_mb": dir_size_mb(repo),
            "top_level_files": ";".join(top_files[:100]),
            "readme_preview": preview_value(readme_preview, max_len=2000),
        })

repos_df = pd.DataFrame(repo_rows)


# ============================================================
# SPECIAL TOFU INSPECTION
# ============================================================

tofu_rows = []

possible_tofu_files = []

if EXPORT_DIR.exists():
    possible_tofu_files.extend(sorted((EXPORT_DIR / "tofu").rglob("*.jsonl")))

if NORMALIZED_DIR.exists():
    possible_tofu_files.extend(sorted((NORMALIZED_DIR / "tofu").rglob("*.jsonl")))

for p in possible_tofu_files:
    examples = read_jsonl_examples(p, max_examples=10)

    for i, ex in enumerate(examples):
        if not isinstance(ex, dict):
            continue

        tofu_rows.append({
            "file": str(p),
            "example_idx": i,
            "keys": ";".join(ex.keys()),
            "question": preview_value(
                ex.get("question")
                or ex.get("Question")
                or ex.get("prompt")
                or ex.get("input")
                or "",
                max_len=300,
            ),
            "answer": preview_value(
                ex.get("answer")
                or ex.get("Answer")
                or ex.get("completion")
                or ex.get("output")
                or "",
                max_len=300,
            ),
            "text_lm": preview_value(ex.get("text_lm", ""), max_len=500),
            "raw_preview": preview_value(ex, max_len=700),
        })

tofu_df = pd.DataFrame(tofu_rows)


# ============================================================
# TEXT REPORT
# ============================================================

lines = []

lines.append("=" * 100)
lines.append("UNLEARNING BENCHMARKS STRUCTURE REPORT")
lines.append("=" * 100)
lines.append("")
lines.append("ENVIRONMENT")
lines.append("-" * 100)
for k, v in env_info.items():
    lines.append(f"{k}: {v}")

lines.append("")
lines.append("DIRECTORIES")
lines.append("-" * 100)
for d in [RAW_DIR, EXPORT_DIR, NORMALIZED_DIR, REPO_DIR, REPORT_DIR]:
    lines.append(f"{d}: exists={d.exists()} size_mb={dir_size_mb(d):.2f}")

lines.append("")
lines.append("DATASET EXPORTS")
lines.append("-" * 100)

if not datasets_df.empty:
    grouped = (
        datasets_df
        .groupby(["dataset_alias", "config", "split", "file_type"], dropna=False)
        .agg(
            files=("path", "count"),
            rows=("num_rows", "max"),
            size_mb=("size_mb", "sum"),
            columns=("columns_in_examples", "first"),
        )
        .reset_index()
        .sort_values(["dataset_alias", "config", "split", "file_type"])
    )

    for _, r in grouped.iterrows():
        lines.append(
            f"{r['dataset_alias']} | config={r['config']} | split={r['split']} | "
            f"type={r['file_type']} | rows={r['rows']} | "
            f"files={r['files']} | size={r['size_mb']:.2f} MB | "
            f"cols={r['columns']}"
        )
else:
    lines.append("No exported datasets found.")

lines.append("")
lines.append("NORMALIZED DATA")
lines.append("-" * 100)

if not normalized_df.empty:
    for _, r in normalized_df.iterrows():
        lines.append(
            f"{r['relative_path']} | type={r['file_type']} | "
            f"rows={r['num_rows']} | size={r['size_mb']:.2f} MB | "
            f"cols={r.get('columns_in_examples', '')}"
        )
else:
    lines.append("No normalized data found.")

lines.append("")
lines.append("REPOSITORIES")
lines.append("-" * 100)

if not repos_df.empty:
    for _, r in repos_df.iterrows():
        lines.append(
            f"{r['repo_name']} | git={r['is_git_repo']} | "
            f"size={r['size_mb']:.2f} MB | files={r['top_level_files']}"
        )
else:
    lines.append("No repos found.")

lines.append("")
lines.append("TOFU EXAMPLES")
lines.append("-" * 100)

if not tofu_df.empty:
    for _, r in tofu_df.head(10).iterrows():
        lines.append(f"file: {r['file']}")
        lines.append(f"keys: {r['keys']}")
        lines.append(f"question: {r['question']}")
        lines.append(f"answer: {r['answer']}")
        lines.append("")
else:
    lines.append("No TOFU examples found.")

report_text = "\n".join(lines)
OUT_TXT.write_text(report_text, encoding="utf-8")

print(report_text)


# ============================================================
# SAVE CSV
# ============================================================

files_df.to_csv(OUT_CSV_FILES, index=False)
datasets_df.to_csv(OUT_CSV_DATASETS, index=False)
columns_df.to_csv(OUT_CSV_COLUMNS, index=False)
examples_df.to_csv(OUT_CSV_EXAMPLES, index=False)
repos_df.to_csv(OUT_CSV_REPOS, index=False)

raw_info_df.to_csv(OUT_DIR / "raw_dataset_export_info.csv", index=False)
normalized_df.to_csv(OUT_DIR / "normalized_index.csv", index=False)
normalized_examples_df.to_csv(OUT_DIR / "normalized_examples.csv", index=False)
tofu_df.to_csv(OUT_DIR / "tofu_examples.csv", index=False)


# ============================================================
# SAVE JSON
# ============================================================

json_report = {
    "env": env_info,
    "files_count": len(files_df),
    "datasets_count": len(datasets_df),
    "columns_count": len(columns_df),
    "examples_count": len(examples_df),
    "repos_count": len(repos_df),
    "normalized_count": len(normalized_df),
    "tofu_examples_count": len(tofu_df),
}

OUT_JSON.write_text(
    json.dumps(json_report, indent=4, ensure_ascii=False),
    encoding="utf-8",
)


# ============================================================
# SAVE XLSX
# ============================================================

try:
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        pd.DataFrame([env_info]).to_excel(writer, sheet_name="env", index=False)
        files_df.to_excel(writer, sheet_name="files_index", index=False)
        datasets_df.to_excel(writer, sheet_name="datasets_index", index=False)
        columns_df.to_excel(writer, sheet_name="columns_index", index=False)
        examples_df.to_excel(writer, sheet_name="examples_index", index=False)
        raw_info_df.to_excel(writer, sheet_name="raw_export_info", index=False)
        normalized_df.to_excel(writer, sheet_name="normalized_index", index=False)
        normalized_examples_df.to_excel(writer, sheet_name="normalized_examples", index=False)
        tofu_df.to_excel(writer, sheet_name="tofu_examples", index=False)
        repos_df.to_excel(writer, sheet_name="repos_index", index=False)

    print("Saved XLSX:", OUT_XLSX)

except Exception as e:
    print("Could not save XLSX:", repr(e))
    print("Install if needed:")
    print("  python -m pip install openpyxl")


print("\nDONE")
print("Text report:", OUT_TXT)
print("JSON report:", OUT_JSON)
print("XLSX report:", OUT_XLSX)
print("CSV files:")
print(" ", OUT_CSV_FILES)
print(" ", OUT_CSV_DATASETS)
print(" ", OUT_CSV_COLUMNS)
print(" ", OUT_CSV_EXAMPLES)
print(" ", OUT_CSV_REPOS)