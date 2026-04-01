"""Shared IO helpers for the benchmark pipeline."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: Path, default=None):
    """Load a JSON file if it exists, else return the supplied default."""

    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    """Write JSON with stable formatting."""

    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_id_csv(path: Path) -> List[str]:
    """Load a one-column CSV of ids."""

    frame = pd.read_csv(path)
    if "id" not in frame.columns:
        raise ValueError(f"Expected an id column in {path}")
    return frame["id"].astype(str).tolist()


def write_id_csv(path: Path, ids: Sequence[str]) -> None:
    """Write a deterministic one-column CSV of ids."""

    ensure_dir(path.parent)
    pd.DataFrame({"id": list(ids)}).to_csv(path, index=False)


def stable_sha1(text: str) -> str:
    """Return a stable sha1 digest for a text key."""

    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def stable_int(text: str) -> int:
    """Map text to a deterministic positive integer."""

    return int(stable_sha1(text)[:12], 16)


def require_dependency(module_name: str, package_name: Optional[str] = None):
    """Import an optional dependency with a clear error message."""

    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        package = package_name or module_name
        raise ModuleNotFoundError(
            f"Missing optional dependency '{package}'. Install it with `pip install {package}`."
        ) from exc


def upsert_csv_records(csv_path: Path, records: Sequence[Dict], key_columns: Sequence[str]) -> None:
    """Upsert records into a CSV keyed by one or more columns."""

    ensure_dir(csv_path.parent)
    new_frame = pd.DataFrame(records)
    if new_frame.empty:
        return

    if csv_path.exists():
        old_frame = pd.read_csv(csv_path)
        combined = pd.concat([old_frame, new_frame], ignore_index=True, sort=False)
    else:
        combined = new_frame

    combined = combined.drop_duplicates(subset=list(key_columns), keep="last")
    combined = combined.sort_values(list(key_columns)).reset_index(drop=True)
    combined.to_csv(csv_path, index=False)


def load_dataframe_records(csv_path: Path) -> pd.DataFrame:
    """Load a CSV if present, else return an empty frame."""

    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def save_numpy(path: Path, array: np.ndarray) -> None:
    """Persist a numpy array with parent creation."""

    ensure_dir(path.parent)
    np.save(path, array)


def load_numpy(path: Path, mmap_mode: Optional[str] = None) -> np.ndarray:
    """Load a saved numpy array."""

    return np.load(path, allow_pickle=False, mmap_mode=mmap_mode)


def link_or_copy_file(source: Path, destination: Path) -> None:
    """Create a hard link when possible, else fall back to copying."""

    ensure_dir(destination.parent)
    destination.unlink(missing_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


@contextmanager
def timer() -> Iterator[List[float]]:
    """Context manager that exposes elapsed time through a mutable cell."""

    start = time.perf_counter()
    cell = [0.0]
    try:
        yield cell
    finally:
        cell[0] = time.perf_counter() - start
