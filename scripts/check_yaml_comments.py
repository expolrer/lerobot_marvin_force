#!/usr/bin/env python3
"""Require a Chinese explanatory comment immediately before every YAML key."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


KEY = re.compile(r"^\s*(?:-\s+)?[A-Za-z0-9_.-]+\s*:")
CHINESE = re.compile(r"[\u3400-\u9fff]")


def check(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    errors: list[str] = []
    for index, line in enumerate(lines):
        if not KEY.match(line):
            continue
        previous = index - 1
        while previous >= 0 and not lines[previous].strip():
            previous -= 1
        if previous < 0 or not lines[previous].lstrip().startswith("#"):
            errors.append(f"{path}:{index + 1}: 参数前缺少注释")
        elif not CHINESE.search(lines[previous]):
            errors.append(f"{path}:{index + 1}: 参数注释不含中文说明")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*", type=Path)
    args = parser.parse_args()
    paths = args.paths or sorted(Path("config").rglob("*.yaml")) + sorted(
        Path("workspaces").rglob("*.yaml")
    )
    errors = [error for path in paths for error in check(path)]
    if errors:
        print("\n".join(errors))
        return 1
    print(f"YAML comments OK: {len(paths)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
