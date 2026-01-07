#!/usr/bin/env python3
"""
Cross-platform source file discovery for Meson.

Why:
  - On Windows, 'find' resolves to find.exe (not GNU find), breaking Meson builds.
  - We prefer a deterministic, portable way to enumerate sources.

Usage:
  python dev/meson/list_sources.py --base <dir> --glob "<pattern>"

Notes:
  - Output paths are *relative to --base* (Meson subdir meson.build expects that).
  - Paths are normalized to forward slashes.
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Iterable


def _norm_rel(base: str, path: str) -> str:
    rel = os.path.relpath(path, base)
    return rel.replace("\\", "/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base directory to search from")
    ap.add_argument("--glob", required=True, help="Glob pattern (relative to base)")
    ap.add_argument(
        "--exclude-substr",
        action="append",
        default=[],
        help="Exclude any path whose normalized form contains this substring (repeatable)",
    )
    args = ap.parse_args()

    base = os.path.abspath(args.base)
    pattern = os.path.join(base, args.glob)

    files: Iterable[str] = glob.glob(pattern, recursive=True)

    out: list[str] = []
    for p in files:
        if not os.path.isfile(p):
            continue
        rel = _norm_rel(base, p)
        rel_norm = rel  # already forward-slashed
        if any(excl in rel_norm for excl in args.exclude_substr):
            continue
        out.append(rel_norm)

    out.sort()
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
