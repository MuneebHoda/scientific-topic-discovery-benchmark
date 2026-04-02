"""Deterministic benchmark subset creation for local and full-corpus runs."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

from src.config import ProfileConfig, SubsetConfig
from src.io_utils import ensure_dir, load_json, stable_int, write_json


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


def _iter_parquet_batches(path: Path, batch_size: int = 100_000, columns=None):
    """Yield parquet batches from the supplied path."""

    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    parquet_file = pq.ParquetFile(path)
    yield from parquet_file.iter_batches(batch_size=batch_size, columns=columns)


def _count_categories(clean_path: Path) -> Counter:
    """Count primary categories in the cleaned parquet dataset."""

    counts = Counter()
    for batch in _iter_parquet_batches(clean_path, columns=["primary_category"]):
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


def split_parquet_path(profile: ProfileConfig, subset_name: str, split_name: str) -> Path:
    return profile.splits_dir() / f"{subset_name}_{split_name}.parquet"


def split_row_count(profile: ProfileConfig, subset_name: str, split_name: str) -> int:
    """Return the number of rows in a saved split parquet."""

    pq = __import__("pyarrow.parquet", fromlist=["parquet"])
    return int(pq.ParquetFile(split_parquet_path(profile, subset_name, split_name)).metadata.num_rows)


def subset_exists(profile: ProfileConfig, subset_name: str) -> bool:
    """Check whether a subset and its split files already exist."""

    return (
        split_parquet_path(profile, subset_name, "train").exists()
        and split_parquet_path(profile, subset_name, "val").exists()
        and split_parquet_path(profile, subset_name, "test").exists()
    )


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    frame.to_parquet(path, index=False)


def _write_split_parquets_from_frame(
    profile: ProfileConfig,
    subset_name: str,
    subset_frame: pd.DataFrame,
    split_ids: Dict[str, Sequence[str]],
) -> None:
    """Write split parquet files from an in-memory subset frame."""

    id_sets = {split_name: set(ids) for split_name, ids in split_ids.items()}
    for split_name, ids in split_ids.items():
        frame = subset_frame[subset_frame["id"].astype(str).isin(id_sets[split_name])].copy()
        frame = frame.reset_index(drop=True)
        _write_parquet(split_parquet_path(profile, subset_name, split_name), frame)


def _write_split_parquets_from_clean_dataset(
    profile: ProfileConfig,
    subset_name: str,
    split_ids: Dict[str, Sequence[str]],
    batch_size: int = 100_000,
) -> None:
    """Write split parquet files for a full-corpus subset by streaming the clean dataset."""

    pyarrow = __import__("pyarrow")
    pq = __import__("pyarrow.parquet", fromlist=["parquet"])

    id_to_split = {}
    for split_name, ids in split_ids.items():
        for paper_id in ids:
            id_to_split[str(paper_id)] = split_name

    writers = {}
    try:
        for batch in _iter_parquet_batches(profile.clean_dataset_path(), batch_size=batch_size):
            frame = batch.to_pandas()
            frame["split_name"] = frame["id"].astype(str).map(id_to_split)
            frame = frame[frame["split_name"].notna()].copy()
            if frame.empty:
                continue

            for split_name in ("train", "val", "test"):
                split_frame = frame[frame["split_name"] == split_name].drop(columns=["split_name"])
                if split_frame.empty:
                    continue
                split_path = split_parquet_path(profile, subset_name, split_name)
                table = pyarrow.Table.from_pandas(split_frame, preserve_index=False)
                writer = writers.get(split_name)
                if writer is None:
                    ensure_dir(split_path.parent)
                    writer = pq.ParquetWriter(split_path, table.schema, compression="zstd")
                    writers[split_name] = writer
                writer.write_table(table)
    finally:
        for writer in writers.values():
            writer.close()


def _create_full_dataset_subset(
    profile: ProfileConfig,
    spec: SubsetConfig,
    batch_size: int,
) -> Dict:
    """Create split files for a subset that intentionally covers the whole cleaned dataset."""

    id_frame = pd.read_parquet(profile.clean_dataset_path(), columns=["id", "primary_category"])
    splits = deterministic_stratified_split(
        id_frame,
        seed=profile.seed + stable_int(spec.name),
        train_fraction=profile.train_fraction,
        val_fraction=profile.val_fraction,
        test_fraction=profile.test_fraction,
    )
    _write_split_parquets_from_clean_dataset(profile, spec.name, splits, batch_size=batch_size)

    return {
        "target_size": None,
        "actual_size": int(len(id_frame)),
        "category_count": int(id_frame["primary_category"].nunique()),
        "train_size": int(len(splits["train"])),
        "val_size": int(len(splits["val"])),
        "test_size": int(len(splits["test"])),
        "source": "clean_dataset",
        "use_full_dataset": True,
    }


def create_benchmark_subsets(
    profile: ProfileConfig,
    force: bool = False,
    batch_size: int = 100_000,
) -> Dict:
    """Create deterministic stratified subsets and split parquet files."""

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

    split_summary = {
        "cleaned_row_count": int(sum(category_counts.values())),
        "cleaned_category_count": int(len(category_counts)),
        "subset_summaries": {},
    }

    sampled_specs = {}
    for subset_name, spec in specs.items():
        if spec.use_full_dataset:
            split_summary["subset_summaries"][subset_name] = _create_full_dataset_subset(profile, spec, batch_size=batch_size)
            continue

        quotas = allocate_category_quotas(
            category_counts=category_counts,
            target_size=int(spec.size or 0),
            min_category_count=spec.min_category_count,
            min_subset_per_category=spec.min_subset_per_category,
        )
        sampled_specs[subset_name] = {
            "spec": spec,
            "sampler": StratifiedReservoirSampler(
                subset_name=subset_name,
                quotas=quotas,
                seed=profile.seed + stable_int(subset_name),
            ),
        }

    if sampled_specs:
        for batch in _iter_parquet_batches(profile.clean_dataset_path(), batch_size=batch_size):
            frame = batch.to_pandas()
            for row in frame.to_dict(orient="records"):
                for payload in sampled_specs.values():
                    payload["sampler"].process(row)

    for subset_name, payload in sampled_specs.items():
        spec = payload["spec"]
        subset_frame = payload["sampler"].to_frame()
        if subset_frame.empty:
            raise ValueError(
                f"Subset '{subset_name}' is empty. Reduce `min_category_count` or use a larger cleaned dataset."
            )
        subset_path = _subset_path(profile, subset_name)
        _write_parquet(subset_path, subset_frame)

        splits = deterministic_stratified_split(
            subset_frame,
            seed=profile.seed + stable_int(subset_name),
            train_fraction=profile.train_fraction,
            val_fraction=profile.val_fraction,
            test_fraction=profile.test_fraction,
        )
        _write_split_parquets_from_frame(profile, subset_name, subset_frame, splits)

        split_summary["subset_summaries"][subset_name] = {
            "target_size": spec.size,
            "actual_size": int(len(subset_frame)),
            "category_count": int(subset_frame["primary_category"].nunique()),
            "train_size": int(len(splits["train"])),
            "val_size": int(len(splits["val"])),
            "test_size": int(len(splits["test"])),
            "source": "sampled_subset",
            "use_full_dataset": False,
        }

    write_json(summary_path, split_summary)
    return split_summary


def load_split_frame(profile: ProfileConfig, subset_name: str, split_name: str, columns=None) -> pd.DataFrame:
    """Load a single split parquet."""

    return pd.read_parquet(split_parquet_path(profile, subset_name, split_name), columns=columns)


def iter_split_batches(profile: ProfileConfig, subset_name: str, split_name: str, batch_size: int = 50_000, columns=None):
    """Yield parquet batches for a saved split."""

    yield from _iter_parquet_batches(split_parquet_path(profile, subset_name, split_name), batch_size=batch_size, columns=columns)
