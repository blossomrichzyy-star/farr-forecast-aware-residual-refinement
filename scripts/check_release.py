from __future__ import annotations

import ast
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = [
    re.compile(r"/root/"),
    re.compile(r"C:\\\\Users\\\\"),
    re.compile(r"ssh\\.zw1\\.paratera\\.com"),
    re.compile(r"ssh_private_key", re.IGNORECASE),
]
FORBIDDEN_EXTENSIONS = {".pt", ".pth", ".ckpt", ".npy", ".npz", ".log"}


def main() -> None:
    errors: list[str] = []
    required = [
        "README.md", "LICENSE", "requirements.txt", ".gitignore",
        "THIRD_PARTY_NOTICES.md", "configs/default.yaml", "configs/etth1.yaml",
        "configs/weather.yaml", "configs/traffic.yaml", "configs/ecl.yaml",
        "src/farr", "scripts/run_main.py", "scripts/generate_endpoints.py",
        "docs/data.md", "docs/models.md", "docs/reproduction.md",
    ]
    allowed_root = {".git", "src", "configs", "scripts", "docs", "README.md",
                    "requirements.txt", ".gitignore", "LICENSE", "THIRD_PARTY_NOTICES.md"}
    for child in ROOT.iterdir():
        if child.name not in allowed_root:
            errors.append(f"unexpected root entry: {child.name}")
    for relative in required:
        if not (ROOT / relative).exists():
            errors.append(f"missing required path: {relative}")
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in {".git", "__pycache__"} for part in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix in FORBIDDEN_EXTENSIONS and path.name != ".gitkeep":
            errors.append(f"generated artifact in release tree: {path.relative_to(ROOT)}")
        if path.suffix in {".py", ".yaml", ".md", ".toml", ".cff"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in FORBIDDEN:
                if pattern.search(text):
                    errors.append(f"forbidden private content in {path.relative_to(ROOT)}: {pattern.pattern}")
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=str(path))
                except SyntaxError as exc:
                    errors.append(f"syntax error in {path.relative_to(ROOT)}: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print("Release tree passed static checks.")


if __name__ == "__main__":
    main()
