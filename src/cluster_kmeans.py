"""Primary KMeans clustering runner for reduced embeddings."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, load_numpy, save_numpy, write_json
from src.reduce_umap import _silhouette_score_sample


def _output_dir(profile: ProfileConfig, embedding_name: str, subset_name: str) -> Path:
    return ensure_dir(profile.clustering_dir() / f"{embedding_name}_kmeans" / subset_name)


def run_kmeans_clustering(profile: ProfileConfig, embedding_name: str, subset_name: str) -> Dict:
    """Tune KMeans by validation silhouette and cache label assignments."""

    output_dir = _output_dir(profile, embedding_name, subset_name)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists() and (output_dir / "test_labels.npy").exists():
        metadata = load_json(metadata_path, default={}) or {}
        metadata["reused_cache"] = True
        return metadata

    reduced_dir = profile.embeddings_dir() / embedding_name / subset_name / "reduced"
    train_embeddings = load_numpy(reduced_dir / "train_reduced.npy")
    val_embeddings = load_numpy(reduced_dir / "val_reduced.npy")
    test_embeddings = load_numpy(reduced_dir / "test_reduced.npy")

    search_rows = []
    best = None
    start = time.perf_counter()
    valid_k_values = [int(k) for k in profile.kmeans_k_values if 2 <= int(k) <= int(train_embeddings.shape[0] - 1)]
    notes = ""
    if not valid_k_values:
        fallback_k = max(2, min(20, int(train_embeddings.shape[0] - 1)))
        valid_k_values = [fallback_k]
        notes = f"Default K grid exceeded train size; used fallback k={fallback_k}."
    for k in valid_k_values:
        model = KMeans(n_clusters=int(k), random_state=profile.seed, n_init=10)
        model.fit(train_embeddings)
        val_labels = model.predict(val_embeddings).astype(np.int32)
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
                "model": model,
                "k": int(k),
                "validation_silhouette": score,
                "val_labels": val_labels,
            }

    if best is None:
        raise RuntimeError(f"KMeans search failed for {embedding_name}/{subset_name}")

    train_labels = best["model"].predict(train_embeddings).astype(np.int32)
    test_labels = best["model"].predict(test_embeddings).astype(np.int32)
    centroids = best["model"].cluster_centers_.astype(np.float32)
    pd.DataFrame(search_rows).to_csv(output_dir / "search_results.csv", index=False)
    save_numpy(output_dir / "train_labels.npy", train_labels)
    save_numpy(output_dir / "val_labels.npy", best["val_labels"])
    save_numpy(output_dir / "test_labels.npy", test_labels)
    save_numpy(output_dir / "centroids.npy", centroids)

    metadata = {
        "embedding": embedding_name,
        "clustering": "kmeans",
        "subset_name": subset_name,
        "best_param": f"k={best['k']}",
        "validation_silhouette": best["validation_silhouette"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
        "notes": notes,
    }
    write_json(metadata_path, metadata)
    return metadata
