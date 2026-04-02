"""HDBSCAN clustering with repaired noise labels."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances_argmin

from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, load_numpy, save_numpy, write_json
from src.reduce_umap import _silhouette_score_sample


def _output_dir(profile: ProfileConfig, embedding_name: str, subset_name: str) -> Path:
    return ensure_dir(profile.clustering_dir() / f"{embedding_name}_hdbscan" / subset_name)


def _repair_noise_labels(train_embeddings: np.ndarray, train_labels: np.ndarray, split_embeddings: np.ndarray, split_labels: np.ndarray) -> np.ndarray:
    """Assign HDBSCAN noise points to the nearest non-noise centroid for evaluation."""

    repaired = split_labels.astype(np.int32).copy()
    non_noise_labels = np.array(sorted(label for label in np.unique(train_labels) if label >= 0), dtype=np.int32)
    if non_noise_labels.size == 0:
        return repaired

    centroids = np.vstack(
        [train_embeddings[train_labels == label].mean(axis=0) for label in non_noise_labels]
    ).astype(np.float32)
    noise_mask = repaired < 0
    if noise_mask.any():
        nearest = pairwise_distances_argmin(split_embeddings[noise_mask], centroids, metric="euclidean")
        repaired[noise_mask] = non_noise_labels[nearest]
    return repaired


def run_hdbscan_clustering(profile: ProfileConfig, embedding_name: str, subset_name: str) -> Dict:
    """Tune HDBSCAN on the configured split set and cache raw plus repaired labels."""

    hdbscan = __import__("hdbscan")
    prediction = __import__("hdbscan.prediction", fromlist=["approximate_predict"])

    output_dir = _output_dir(profile, embedding_name, subset_name)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and (output_dir / "test_repaired_labels.npy").exists():
        metadata = load_json(metadata_path, default={}) or {}
        metadata["reused_cache"] = True
        return metadata

    reduced_dir = profile.embeddings_dir() / embedding_name / subset_name / "reduced"
    train_embeddings = load_numpy(reduced_dir / "train_reduced.npy", mmap_mode="r")
    val_embeddings = load_numpy(reduced_dir / "val_reduced.npy", mmap_mode="r")
    test_embeddings = load_numpy(reduced_dir / "test_reduced.npy", mmap_mode="r")

    search_rows = []
    best = None
    start = time.perf_counter()
    for min_cluster_size in profile.hdbscan_min_cluster_sizes:
        model = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            metric="euclidean",
            prediction_data=True,
        )
        train_raw = model.fit_predict(train_embeddings).astype(np.int32)
        val_raw, _ = prediction.approximate_predict(model, val_embeddings)
        test_raw, _ = prediction.approximate_predict(model, test_embeddings)
        val_raw = val_raw.astype(np.int32)
        test_raw = test_raw.astype(np.int32)

        train_repaired = _repair_noise_labels(train_embeddings, train_raw, train_embeddings, train_raw)
        val_repaired = _repair_noise_labels(train_embeddings, train_raw, val_embeddings, val_raw)
        test_repaired = _repair_noise_labels(train_embeddings, train_raw, test_embeddings, test_raw)
        score = _silhouette_score_sample(
            val_embeddings,
            val_repaired,
            sample_size=profile.silhouette_eval_sample_size,
            seed=profile.seed,
        )
        row = {
            "embedding": embedding_name,
            "subset_name": subset_name,
            "min_cluster_size": int(min_cluster_size),
            "validation_silhouette": score,
            "train_noise_rate": float(np.mean(train_raw < 0)),
            "val_noise_rate": float(np.mean(val_raw < 0)),
            "test_noise_rate": float(np.mean(test_raw < 0)),
        }
        search_rows.append(row)
        if best is None or row["validation_silhouette"] > best["validation_silhouette"]:
            best = {
                "min_cluster_size": int(min_cluster_size),
                "validation_silhouette": score,
                "train_raw": train_raw,
                "val_raw": val_raw,
                "test_raw": test_raw,
                "train_repaired": train_repaired,
                "val_repaired": val_repaired,
                "test_repaired": test_repaired,
            }

    if best is None:
        raise RuntimeError(f"HDBSCAN search failed for {embedding_name}/{subset_name}")

    pd.DataFrame(search_rows).to_csv(output_dir / "search_results.csv", index=False)
    save_numpy(output_dir / "train_raw_labels.npy", best["train_raw"])
    save_numpy(output_dir / "val_raw_labels.npy", best["val_raw"])
    save_numpy(output_dir / "test_raw_labels.npy", best["test_raw"])
    save_numpy(output_dir / "train_repaired_labels.npy", best["train_repaired"])
    save_numpy(output_dir / "val_repaired_labels.npy", best["val_repaired"])
    save_numpy(output_dir / "test_repaired_labels.npy", best["test_repaired"])

    metadata = {
        "embedding": embedding_name,
        "clustering": "hdbscan",
        "subset_name": subset_name,
        "best_param": f"min_cluster_size={best['min_cluster_size']}",
        "validation_silhouette": best["validation_silhouette"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
        "notes": "Noise labels repaired by nearest non-noise centroid for evaluation.",
    }
    write_json(metadata_path, metadata)
    return metadata
