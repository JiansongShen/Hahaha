#!/usr/bin/env python3
"""
Fetch Meson WrapDB .wrap dependencies into ./subprojects WITHOUT using meson.

Why:
  - Reuse repo-tracked .wrap versions to keep CMake builds reproducible.
  - Avoid introducing a hard dependency on Meson when building with CMake.

What it does:
  - Parses ./subprojects/<name>.wrap
  - Downloads source tarball and verifies sha256
  - Extracts into ./subprojects/<directory>
  - Downloads WrapDB patch zip and verifies sha256
  - Extracts patch zip over the extracted source tree (overlay)

Usage:
  python3 dev/fetch_wrap_deps.py gtest imgui
  python3 dev/fetch_wrap_deps.py --all
  python3 dev/fetch_wrap_deps.py --clean --all
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WrapInfo:
    name: str
    directory: str
    source_url: str
    source_filename: str
    source_hash: str
    patch_url: str
    patch_filename: str
    patch_hash: str


def _parse_wrap(path: Path) -> WrapInfo:
    data: dict[str, str] = {}
    name = path.stem
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("["):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip()

    required = [
        "directory",
        "source_url",
        "source_filename",
        "source_hash",
        "patch_url",
        "patch_filename",
        "patch_hash",
    ]
    missing = [k for k in required if k not in data]
    if missing:
        raise ValueError(f"{path}: missing keys: {', '.join(missing)}")

    return WrapInfo(
        name=name,
        directory=data["directory"],
        source_url=data["source_url"],
        source_filename=data["source_filename"],
        source_hash=data["source_hash"],
        patch_url=data["patch_url"],
        patch_filename=data["patch_filename"],
        patch_hash=data["patch_hash"],
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def _extract_tar(archive: Path, out_dir: Path) -> None:
    # WrapDB tarballs are typically safe, but still ensure extraction stays in out_dir.
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tf:
        for m in tf.getmembers():
            target = out_dir / m.name
            if not str(target.resolve()).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe tar member path: {m.name}")
        tf.extractall(out_dir)


def _extract_zip_overlay(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = out_dir / name
            if not str(target.resolve()).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"Unsafe zip member path: {name}")
        zf.extractall(out_dir)


def _fetch_one(root: Path, wrap: WrapInfo, clean: bool) -> None:
    subprojects = root / "subprojects"
    packagecache = subprojects / "packagecache"
    wrap_path = subprojects / f"{wrap.name}.wrap"

    if not wrap_path.exists():
        raise FileNotFoundError(f"wrap file not found: {wrap_path}")

    out_dir = subprojects / wrap.directory
    if clean and out_dir.exists():
        shutil.rmtree(out_dir)

    src_cache = packagecache / wrap.source_filename
    patch_cache = packagecache / wrap.patch_filename

    if not src_cache.exists():
        print(f"[{wrap.name}] downloading source: {wrap.source_url}")
        _download(wrap.source_url, src_cache)
    got = _sha256(src_cache)
    if got != wrap.source_hash:
        raise RuntimeError(f"[{wrap.name}] source sha256 mismatch: expected {wrap.source_hash}, got {got}")

    if not out_dir.exists():
        print(f"[{wrap.name}] extracting source -> {out_dir}")
        _extract_tar(src_cache, subprojects)
        if not out_dir.exists():
            raise RuntimeError(f"[{wrap.name}] expected directory not created after extract: {out_dir}")
    else:
        print(f"[{wrap.name}] source dir exists, skip extract: {out_dir}")

    if not patch_cache.exists():
        print(f"[{wrap.name}] downloading patch: {wrap.patch_url}")
        _download(wrap.patch_url, patch_cache)
    gotp = _sha256(patch_cache)
    if gotp != wrap.patch_hash:
        raise RuntimeError(f"[{wrap.name}] patch sha256 mismatch: expected {wrap.patch_hash}, got {gotp}")

    print(f"[{wrap.name}] applying patch overlay")
    _extract_zip_overlay(patch_cache, out_dir)

    print(f"[{wrap.name}] done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="Wrap names to fetch (e.g. gtest imgui)")
    ap.add_argument("--all", action="store_true", help="Fetch all known wraps used by this repo (gtest, imgui)")
    ap.add_argument("--clean", action="store_true", help="Delete extracted subproject dirs before fetching")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    subprojects = root / "subprojects"

    known = ["gtest", "imgui"]
    names = known if args.all else args.names
    if not names:
        ap.error("Provide wrap names, or use --all")

    for name in names:
        if name not in known:
            raise SystemExit(f"Unknown wrap '{name}'. Known: {', '.join(known)}")
        wrap_path = subprojects / f"{name}.wrap"
        wrap = _parse_wrap(wrap_path)
        _fetch_one(root, wrap, clean=args.clean)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
