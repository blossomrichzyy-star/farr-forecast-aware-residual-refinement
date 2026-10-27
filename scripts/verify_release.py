"""Reject runtime artifacts, configuration files, and machine-specific content."""

import ast
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
ROOT_ENTRIES = {"src", "scripts", "docs", "README.md", "LICENSE", "requirements.txt",
                ".gitignore", "THIRD_PARTY_NOTICES.md", ".git"}
DENIED = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".csv", ".json", ".jsonl",
          ".yaml", ".yml", ".log", ".ipynb", ".pdf", ".zip", ".pyc"}
PRIVATE_PATH = re.compile(r"(?i)(?:[a-z]:[/\\](?:users|home)[/\\]|/root/|/home/[^/]+/)")


def main():
    errors = []
    for path in ROOT.iterdir():
        if path.name not in ROOT_ENTRIES:
            errors.append(f"Unexpected release entry: {path.name}")
    for path in ROOT.rglob("*"):
        if ".git" in path.relative_to(ROOT).parts:
            continue
        if path.is_dir():
            if path.name in {"__pycache__", ".venv"}:
                errors.append(f"Runtime directory: {path.relative_to(ROOT)}")
            continue
        if path.suffix.lower() in DENIED:
            errors.append(f"Artifact/configuration file: {path.relative_to(ROOT)}")
        if path.suffix in {".py", ".md", ".txt"} or path.name in {"LICENSE", ".gitignore"}:
            content = path.read_text(encoding="utf-8")
            if path.resolve() != Path(__file__).resolve() and PRIVATE_PATH.search(content):
                errors.append(f"Private path: {path.relative_to(ROOT)}")
            if path.suffix == ".py":
                ast.parse(content, filename=str(path))
    if errors:
        raise SystemExit("\n".join(errors))
    print("PASS: source-only release; no configuration or experiment artifacts.")


if __name__ == "__main__":
    main()
