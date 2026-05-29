from __future__ import annotations

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd


ROOT = Path("/home/tahiti/Malashin_Projects")
OUT_DIR = ROOT / "_folder_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_TXT = OUT_DIR / "malashin_projects_folder_report.txt"
OUT_CSV = OUT_DIR / "malashin_projects_file_index.csv"
OUT_XLSX = OUT_DIR / "malashin_projects_folder_report.xlsx"
OUT_JSON = OUT_DIR / "malashin_projects_folder_report.json"


IMPORTANT_SUFFIXES = {
    ".py", ".ipynb", ".csv", ".xlsx", ".json", ".jsonl",
    ".txt", ".log", ".pt", ".pth", ".bin", ".safetensors",
    ".tar", ".gz", ".zip", ".pdf"
}


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024:
            return f"{x:.2f} {u}"
        x /= 1024
    return f"{x:.2f} PB"


def run_cmd(cmd: list[str]) -> str:
    try:
        r = subprocess.run(
            cmd,
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return r.stdout
    except Exception as e:
        return repr(e)


def dir_size(path: Path) -> int:
    total = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
    except Exception:
        pass
    return total


def count_files(path: Path) -> int:
    n = 0
    try:
        for p in path.rglob("*"):
            if p.is_file():
                n += 1
    except Exception:
        pass
    return n


def safe_stat(path: Path):
    try:
        st = path.stat()
        return st
    except Exception:
        return None


# ============================================================
# TOP LEVEL
# ============================================================

top_rows = []

for p in sorted(ROOT.iterdir()):
    st = safe_stat(p)
    if st is None:
        continue

    if p.is_dir():
        size = dir_size(p)
        files = count_files(p)
        kind = "dir"
    else:
        size = st.st_size
        files = 1
        kind = "file"

    top_rows.append({
        "name": p.name,
        "path": str(p),
        "kind": kind,
        "size_bytes": size,
        "size_human": human_size(size),
        "files_count": files,
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
    })

top_df = pd.DataFrame(top_rows).sort_values("size_bytes", ascending=False)


# ============================================================
# FULL FILE INDEX
# ============================================================

file_rows = []

for p in ROOT.rglob("*"):
    if not p.is_file():
        continue

    if "_folder_check" in p.parts:
        continue

    st = safe_stat(p)
    if st is None:
        continue

    rel = p.relative_to(ROOT)

    file_rows.append({
        "relative_path": str(rel),
        "absolute_path": str(p),
        "parent": str(p.parent),
        "suffix": p.suffix.lower(),
        "name": p.name,
        "size_bytes": st.st_size,
        "size_human": human_size(st.st_size),
        "modified": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
        "is_important_suffix": p.suffix.lower() in IMPORTANT_SUFFIXES,
    })

files_df = pd.DataFrame(file_rows)

if not files_df.empty:
    files_df = files_df.sort_values("size_bytes", ascending=False)


# ============================================================
# REPORTS / RESULTS DETECTION
# ============================================================

interesting_patterns = [
    "poc_v5",
    "poc_v4",
    "poc_v3",
    "summary",
    "results",
    "report",
    "checkpoint_metrics",
    "done.json",
    "history.json",
    "log.txt",
]

interesting_rows = []

for _, r in files_df.iterrows():
    rel = str(r["relative_path"]).lower()
    if any(pat in rel for pat in interesting_patterns):
        interesting_rows.append(r.to_dict())

interesting_df = pd.DataFrame(interesting_rows)

if not interesting_df.empty:
    interesting_df = interesting_df.sort_values("modified", ascending=False)


# ============================================================
# LARGE FILES
# ============================================================

large_df = files_df[files_df["size_bytes"] >= 100 * 1024 * 1024].copy()
large_df = large_df.sort_values("size_bytes", ascending=False)


# ============================================================
# REPORT TEXT
# ============================================================

lines = []

lines.append("=" * 100)
lines.append("MALASHIN_PROJECTS FOLDER CHECK")
lines.append("=" * 100)
lines.append(f"ROOT: {ROOT}")
lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
lines.append("")

lines.append("=" * 100)
lines.append("DISK INFO")
lines.append("=" * 100)
lines.append(run_cmd(["df", "-h", str(ROOT)]))
lines.append("")

lines.append("=" * 100)
lines.append("TOP LEVEL SIZE")
lines.append("=" * 100)

for _, r in top_df.iterrows():
    lines.append(
        f"{r['size_human']:>12} | files={r['files_count']:<8} | "
        f"{r['kind']:<4} | {r['name']}"
    )

lines.append("")
lines.append("=" * 100)
lines.append("TOP 50 LARGEST FILES")
lines.append("=" * 100)

for _, r in files_df.head(50).iterrows():
    lines.append(
        f"{r['size_human']:>12} | {r['modified']} | {r['relative_path']}"
    )

lines.append("")
lines.append("=" * 100)
lines.append("LARGE FILES >= 100 MB")
lines.append("=" * 100)

if large_df.empty:
    lines.append("No files >= 100 MB")
else:
    for _, r in large_df.iterrows():
        lines.append(
            f"{r['size_human']:>12} | {r['modified']} | {r['relative_path']}"
        )

lines.append("")
lines.append("=" * 100)
lines.append("INTERESTING RESULT / LOG FILES")
lines.append("=" * 100)

if interesting_df.empty:
    lines.append("No obvious result/log files found.")
else:
    for _, r in interesting_df.head(120).iterrows():
        lines.append(
            f"{r['size_human']:>12} | {r['modified']} | {r['relative_path']}"
        )

lines.append("")
lines.append("=" * 100)
lines.append("FILE COUNTS BY SUFFIX")
lines.append("=" * 100)

suffix_df = (
    files_df
    .groupby("suffix", dropna=False)
    .agg(
        files=("relative_path", "count"),
        size_bytes=("size_bytes", "sum"),
    )
    .reset_index()
    .sort_values("size_bytes", ascending=False)
)

suffix_df["size_human"] = suffix_df["size_bytes"].apply(human_size)

for _, r in suffix_df.iterrows():
    lines.append(
        f"{str(r['suffix']):>15} | files={int(r['files']):<8} | {r['size_human']}"
    )

report_text = "\n".join(lines)
OUT_TXT.write_text(report_text, encoding="utf-8")

files_df.to_csv(OUT_CSV, index=False)

summary = {
    "root": str(ROOT),
    "generated": datetime.now().isoformat(timespec="seconds"),
    "top_level": top_rows,
    "num_files": int(len(files_df)),
    "num_large_files_100mb": int(len(large_df)),
    "outputs": {
        "txt": str(OUT_TXT),
        "csv": str(OUT_CSV),
        "xlsx": str(OUT_XLSX),
        "json": str(OUT_JSON),
    },
}

OUT_JSON.write_text(json.dumps(summary, indent=4, ensure_ascii=False), encoding="utf-8")

try:
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        top_df.to_excel(writer, sheet_name="top_level", index=False)
        files_df.to_excel(writer, sheet_name="all_files", index=False)
        large_df.to_excel(writer, sheet_name="large_files", index=False)
        interesting_df.to_excel(writer, sheet_name="interesting", index=False)
        suffix_df.to_excel(writer, sheet_name="suffix_counts", index=False)
except Exception as e:
    print("Could not write xlsx:", repr(e))

print(report_text)
print("\nSaved:")
print(" ", OUT_TXT)
print(" ", OUT_CSV)
print(" ", OUT_XLSX)
print(" ", OUT_JSON)