"""Deterministic benchmark subset creation for local-first experiments."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, read_id_csv, stable_int, write_id_csv, write_json


@dataclass
class StratifiedReservoirSampler:
    """Per-category reservoir sampler with deterministic replacement."""

    subset_name: str
    quotas: Dict[str, int]
    seed: int

    def __post_init__(self) -> None:
        self.rng = random.Random(self.seed)
        self.seen_counts = Counter()
        self.rows = defaultdict(list)

    def process(self, row: Dict) -> None:
        category = row["primary_category"]
        quota = self.quotas.get(category, 0)
        if quota <= 0:
            return

        self.seen_counts[category] += 1
        bucket = self.rows[category]
        if len(bucket) < quota:
            bucket.append(dict(row))
            return

        j = self.rng.randrange(self.seen_counts[category])
        if j < quota:
            bucket[j] = dict(row)

    def to_frame(self) -> pd.DataFrame:
        values: List[Dict] = []
        for category in sorted(self.rows):
            values.extend(self.rows[category])
        frame = pd.DataFrame(values)
        if not frame.empty:
            frame = frame.sort_values(["primary_category", "id"]).reset_index(drop=True)
        return frame


def _iter_clean_batches(clean_path: Path, batch_size: int = 100_000):
    """Yield parquet batches from the cleaned modeling dataset."""

    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    parquet_file = pq.ParquetFile(clean_path)
    yield from parquet_file.iter_batches(batch_size=batch_size)


def _count_categories(clean_path: Path) -> Counter:
    """Count primary categories in the cleaned parquet dataset."""

    counts = Counter()
    for batch in _iter_clean_batches(clean_path):
        frame = batch.to_pandas()
        counts.update(frame["primary_category"].astype(str))
    return counts


def allocate_category_quotas(
    category_counts: Dict[str, int],
    target_size: int,
    min_category_count: int,
    min_subset_per_category: int,
) -> Dict[str, int]:
    """Allocate a deterministic stratified sample budget across categories."""

    eligible = {cat: count for cat, count in category_counts.items() if count >= min_category_count}
    if not eligible:
        return {}

    categories = sorted(eligible)
    minimum = min_subset_per_category
    if target_size < minimum * len(categories):
        minimum = max(1, target_size // len(categories))

    quotas = {cat: min(minimum, eligible[cat]) for cat in categories}
    remaining = target_size - sum(quotas.values())
    capacity = {cat: max(0, eligible[cat] - quotas[cat]) for cat in categories}
    total_capacity = sum(capacity.values())
    if remaining <= 0 or total_capacity <= 0:
        return quotas

    raw_extras = {cat: remaining * capacity[cat] / total_capacity for cat in categories}
    extra_floor = {cat: min(capacity[cat], int(math.floor(raw_extras[cat]))) for cat in categories}
    quotas = {cat: quotas[cat] + extra_floor[cat] for cat in categories}

    used = sum(quotas.values())
    remainder = target_size - used
    ranked = sorted(
        categories,
        key=lambda cat: (raw_extras[cat] - math.floor(raw_extras[cat]), capacity[cat], cat),
        reverse=True,
    )
    for category in ranked:
        if remainder <= 0:
            break
        if quotas[category] < eligible[category]:
            quotas[category] += 1
            remainder -= 1

    return quotas


def _shuffle_ids(ids: Sequence[str], seed: int, category: str) -> List[str]:
    """Return a deterministic shuffled copy of a category id list."""

    rng = random.Random(seed + stable_int(category))
    values = list(ids)
    rng.shuffle(values)
    return values


def deterministic_stratified_split(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
) -> Dict[str, List[str]]:
    """Split a subset into deterministic per-category train/val/test ids."""

    if not math.isclose(train_fraction + val_fraction + test_fraction, 1.0, abs_tol=1e-6):
        raise ValueError("Split fractions must sum to 1.")

    train_ids: List[str] = []
    val_ids: List[str] = []
    test_ids: List[str] = []

    for category, group in frame.groupby("primary_category", sort=True):
        ids = _shuffle_ids(group["id"].astype(str).tolist(), seed, category)
        n = len(ids)
        if n < 3:
            raise ValueError(f"Category {category} has too few rows ({n}) for a 70/10/20 split.")

        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        n_test = n - n_train - n_val

        while n_test < 1 and n_train > 1:
            n_train -= 1
            n_test += 1
        while n_test < 1 and n_val > 1:
            n_val -= 1
            n_test += 1
        while n_val < 1 and n_train > 1:
            n_train -= 1
            n_val += 1

        if n_train + n_val + n_test != n:
            n_test = n - n_train - n_val

        train_ids.extend(ids[:n_train])
        val_ids.extend(ids[n_train : n_train + n_val])
        test_ids.extend(ids[n_train + n_val :])

    return {"train": train_ids, "val": val_ids, "test": test_ids}


def _subset_path(profile: ProfileConfig, subset_name: str) -> Path:
    return profile.splits_dir() / f"{subset_name}.parquet"


def _split_csv_path(profile: ProfileConfig, subset_name: str, split_name: str) -> Path:
    return profile.splits_dir() / f"{subset_name}_{split_name}_ids.csv"


def subset_exists(profile: ProfileConfig, subset_name: str) -> bool:
    """Check whether a subset and its split files already exist."""

    return (
        _subset_path(profile, subset_name).exists()
        and _split_csv_path(profile, subset_name, "train").exists()
        and _split_csv_path(profile, subset_name, "val").exists()
        and _split_csv_path(profile, subset_name, "test").exists()
    )


def create_benchmark_subsets(
    profile: ProfileConfig,
    force: bool = False,
    batch_size: int = 100_000,
) -> Dict:
    """Create deterministic stratified subsets and split files."""

    ensure_dir(profile.splits_dir())
    summary_path = profile.split_summary_path()
    if summary_path.exists() and not force:
        return load_json(summary_path, default={}) or {}

    if force:
        for path in profile.splits_dir().glob("*"):
            if path.is_file():
                path.unlink()

    category_counts = _count_categories(profile.clean_dataset_path())
    specs = profile.subset_configs()
    samplers = {}
    for subset_name, spec in specs.items():
        quotas = allocate_category_quotas(
            category_counts=category_counts,
            target_size=spec.size,
            min_category_count=spec.min_category_count,
            min_subset_per_category=spec.min_subset_per_category,
        )
        samplers[subset_name] = StratifiedReservoirSampler(
            subset_name=subset_name,
            quotas=quotas,
            seed=profile.seed + stable_int(subset_name),
        )

    for batch in _iter_clean_batches(profile.clean_dataset_path(), batch_size=batch_size):
        frame = batch.to_pandas()
        for row in frame.to_dict(orient="records"):
            for sampler in samplers.values():
                sampler.process(row)

    split_summary = {
        "cleaned_row_count": int(sum(category_counts.values())),
        "cleaned_category_count": int(len(category_counts)),
        "subset_summaries": {},
    }

    for subset_name, sampler in samplers.items():
        subset_frame = sampler.to_frame()
        subset_path = _subset_path(profile, subset_name)
        subset_frame.to_parquet(subset_path, index=False)

        splits = deterministic_stratified_split(
            subset_frame,
            seed=profile.seed + stable_int(subset_name),
            train_fraction=profile.train_fraction,
            val_fraction=profile.val_fraction,
            test_fraction=profile.test_fraction,
        )
        for split_name, ids in splits.items():
            write_id_csv(_split_csv_path(profile, subset_name, split_name), ids)

        split_summary["subset_summaries"][subset_name] = {
            "target_size": specs[subset_name].size,
            "actual_size": int(len(subset_frame)),
            "category_count": int(subset_frame["primary_category"].nunique()),
            "train_size": int(len(splits["train"])),
            "val_size": int(len(splits["val"])),
            "test_size": int(len(splits["test"])),
        }

    write_json(summary_path, split_summary)
    return split_summary


def load_subset(profile: ProfileConfig, subset_name: str) -> pd.DataFrame:
    """Load a saved subset parquet."""

    return pd.read_parquet(_subset_path(profile, subset_name))


def load_subset_split_frames(profile: ProfileConfig, subset_name: str) -> Dict[str, pd.DataFrame]:
    """Load a subset and partition it into split dataframes."""

    frame = load_subset(profile, subset_name)
    split_frames = {}
    for split_name in ("train", "val", "test"):
        ids = set(read_id_csv(_split_csv_path(profile, subset_name, split_name)))
        split_frames[split_name] = frame[frame["id"].astype(str).isin(ids)].copy().reset_index(drop=True)
    return split_frames
