"""Lightweight UMAP search and cached dimensionality reduction."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, load_numpy, save_numpy, upsert_csv_records, write_json


def choose_proxy_k(train_label_count: int) -> int:
    """Choose a simple KMeans proxy for UMAP tuning."""

    if train_label_count >= 172:
        return 172
    if train_label_count >= 100:
        return 100
    return max(10, train_label_count)


def _silhouette_score_sample(embeddings: np.ndarray, labels: np.ndarray, sample_size: int, seed: int) -> float:
    """Compute a deterministic silhouette score on a bounded sample."""

    from sklearn.metrics import silhouette_score

    labels = np.asarray(labels)
    valid = labels >= 0
    embeddings = np.asarray(embeddings)[valid]
    labels = labels[valid]
    unique_labels = np.unique(labels)
    if embeddings.shape[0] < 2 or unique_labels.size < 2 or unique_labels.size >= embeddings.shape[0]:
        return float("nan")

    if embeddings.shape[0] > sample_size:
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(embeddings.shape[0], size=sample_size, replace=False)
        embeddings = embeddings[sample_indices]
        labels = labels[sample_indices]

    if np.unique(labels).size < 2 or np.unique(labels).size >= embeddings.shape[0]:
        return float("nan")
    return float(silhouette_score(embeddings, labels, metric="euclidean"))


def run_umap_search(
    profile: ProfileConfig,
    embedding_name: str,
    subset_name: str,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    test_embeddings: np.ndarray,
    train_primary_categories,
    cache_dir: Path,
    metric: str = "cosine",
) -> Dict:
    """Run a tiny UMAP search, cache the best reducer output, and return arrays."""

    ensure_dir(cache_dir)
    best_config_path = cache_dir / "best_umap_config.json"
    train_out = cache_dir / "train_reduced.npy"
    val_out = cache_dir / "val_reduced.npy"
    test_out = cache_dir / "test_reduced.npy"

    if best_config_path.exists() and train_out.exists() and val_out.exists() and test_out.exists():
        best = load_json(best_config_path, default={}) or {}
        return {
            "best_config": best,
            "train": load_numpy(train_out),
            "val": load_numpy(val_out),
            "test": load_numpy(test_out),
            "reused_cache": True,
        }

    umap = __import__("umap")
    search_rows = []
    best_result = None
    proxy_k = choose_proxy_k(int(pd.Series(train_primary_categories).nunique()))
    proxy_k = min(proxy_k, max(2, int(train_embeddings.shape[0] - 1)))

    for n_neighbors in profile.umap_neighbors:
        for min_dist in profile.umap_min_dists:
            reducer = umap.UMAP(
                n_components=profile.umap_components,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=profile.seed,
                transform_seed=profile.seed,
                low_memory=True,
            )
            reduced_train = reducer.fit_transform(train_embeddings)
            reduced_val = reducer.transform(val_embeddings)
            reduced_test = reducer.transform(test_embeddings)

            kmeans = KMeans(n_clusters=proxy_k, random_state=profile.seed, n_init=10)
            kmeans.fit(reduced_train)
            val_labels = kmeans.predict(reduced_val)
            silhouette = _silhouette_score_sample(
                reduced_val,
                val_labels,
                sample_size=profile.silhouette_eval_sample_size,
                seed=profile.seed,
            )

            row = {
                "embedding": embedding_name,
                "subset_name": subset_name,
                "n_neighbors": n_neighbors,
                "min_dist": min_dist,
                "proxy_k": proxy_k,
                "validation_silhouette": silhouette,
            }
            search_rows.append(row)

            if best_result is None or row["validation_silhouette"] > best_result["score"]:
                best_result = {
                    "score": row["validation_silhouette"],
                    "config": row,
                    "train": reduced_train.astype(np.float32),
                    "val": reduced_val.astype(np.float32),
                    "test": reduced_test.astype(np.float32),
                }

    if best_result is None:
        raise RuntimeError(f"No UMAP result produced for {embedding_name}/{subset_name}")

    save_numpy(train_out, best_result["train"])
    save_numpy(val_out, best_result["val"])
    save_numpy(test_out, best_result["test"])
    write_json(best_config_path, best_result["config"])

    upsert_csv_records(
        profile.reduction_dir() / "umap_search_results.csv",
        records=search_rows,
        key_columns=("embedding", "subset_name", "n_neighbors", "min_dist"),
    )
    upsert_csv_records(
        profile.reduction_dir() / "best_umap_config.csv",
        records=[best_result["config"]],
        key_columns=("embedding", "subset_name"),
    )

    return {
        "best_config": best_result["config"],
        "train": best_result["train"],
        "val": best_result["val"],
        "test": best_result["test"],
        "reused_cache": False,
    }
