"""Agglomerative clustering with centroid assignment to held-out splits."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances_argmin

from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, load_numpy, save_numpy, write_json
from src.reduce_umap import _silhouette_score_sample


def _output_dir(profile: ProfileConfig, embedding_name: str, subset_name: str) -> Path:
    return ensure_dir(profile.clustering_dir() / f"{embedding_name}_agglomerative" / subset_name)


def _assign_to_centroids(embeddings: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    nearest = pairwise_distances_argmin(embeddings, centroids, metric="euclidean")
    return nearest.astype(np.int32)


def run_agglomerative_clustering(profile: ProfileConfig, embedding_name: str, subset_name: str) -> Dict:
    """Tune Ward agglomerative clustering on the configured split set."""

    output_dir = _output_dir(profile, embedding_name, subset_name)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and (output_dir / "test_labels.npy").exists():
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
    valid_k_values = [int(k) for k in profile.agglomerative_k_values if 2 <= int(k) <= int(train_embeddings.shape[0] - 1)]
    notes = ""
    if not valid_k_values:
        fallback_k = max(2, min(20, int(train_embeddings.shape[0] - 1)))
        valid_k_values = [fallback_k]
        notes = f"Default K grid exceeded train size; used fallback k={fallback_k}."
    for k in valid_k_values:
        model = AgglomerativeClustering(n_clusters=int(k), linkage="ward")
        train_labels = model.fit_predict(train_embeddings).astype(np.int32)
        centroids = np.vstack(
            [train_embeddings[train_labels == label].mean(axis=0) for label in sorted(np.unique(train_labels))]
        ).astype(np.float32)
        val_labels = _assign_to_centroids(val_embeddings, centroids)
        test_labels = _assign_to_centroids(test_embeddings, centroids)
        score = _silhouette_score_sample(
            val_embeddings,
            val_labels,
            sample_size=profile.silhouette_eval_sample_size,
            seed=profile.seed,
        )
        row = {"embedding": embedding_name, "subset_name": subset_name, "k": int(k), "validation_silhouette": score}
        search_rows.append(row)
        if best is None or row["validation_silhouette"] > best["validation_silhouette"]:
            best = {
                "k": int(k),
                "validation_silhouette": score,
                "train_labels": train_labels,
                "val_labels": val_labels,
                "test_labels": test_labels,
                "centroids": centroids,
            }

    if best is None:
        raise RuntimeError(f"Agglomerative search failed for {embedding_name}/{subset_name}")

    pd.DataFrame(search_rows).to_csv(output_dir / "search_results.csv", index=False)
    save_numpy(output_dir / "train_labels.npy", best["train_labels"])
    save_numpy(output_dir / "val_labels.npy", best["val_labels"])
    save_numpy(output_dir / "test_labels.npy", best["test_labels"])
    save_numpy(output_dir / "centroids.npy", best["centroids"])

    metadata = {
        "embedding": embedding_name,
        "clustering": "agglomerative",
        "subset_name": subset_name,
        "best_param": f"k={best['k']}",
        "validation_silhouette": best["validation_silhouette"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
        "notes": (
            "Ward linkage trained on the configured train split; val/test labels assigned by nearest train centroid. "
            + notes
        ).strip(),
    }
    write_json(metadata_path, metadata)
    return metadata
