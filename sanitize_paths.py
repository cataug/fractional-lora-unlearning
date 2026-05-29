from pathlib import Path

ROOT = Path(".").resolve()

REPLACEMENTS = {
    "/home/user/fractional_unlearning": "/home/user/fractional_unlearning",
    "home/user/fractional_unlearning": "home/user/fractional_unlearning",
    "fractional_unlearning": "fractional_unlearning",
    "Fractional Unlearning": "Fractional Unlearning",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
}

SKIP_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".png",
    ".jpg",
    ".jpeg",
    ".pdf",
    ".zip",
    ".gz",
    ".tar",
    ".7z",
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
}

TEXT_SUFFIXES = {
    ".py",
    ".ipynb",
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".jsonl",
    ".log",
    ".yaml",
    ".yml",
    ".toml",
    ".sh",
    ".gitignore",
}

changed = []
failed = []

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    if any(part in SKIP_DIRS for part in path.parts):
        continue

    if path.suffix.lower() in SKIP_SUFFIXES:
        continue

    # Allow .gitignore and extensionless light text files.
    if path.suffix.lower() not in TEXT_SUFFIXES and path.name != ".gitignore":
        continue

    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        failed.append((str(path), repr(e)))
        continue

    new_text = text

    for old, new in REPLACEMENTS.items():
        new_text = new_text.replace(old, new)

    if new_text != text:
        path.write_text(new_text, encoding="utf-8")
        changed.append(str(path.relative_to(ROOT)))

print("=" * 80)
print("Sanitize paths done")
print("=" * 80)
print("Changed files:", len(changed))

for p in changed[:200]:
    print("CHANGED:", p)

if len(changed) > 200:
    print("...")

print("Failed files:", len(failed))

for p, err in failed[:50]:
    print("FAILED:", p, err)