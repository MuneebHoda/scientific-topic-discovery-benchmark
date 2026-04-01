"""Dense retrieval evaluation with sklearn and optional FAISS backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.benchmark_splits import load_subset_split_frames
from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_numpy, write_json


@dataclass
class SklearnCosineIndex:
    """Simple cosine nearest-neighbor index for local mode."""

    embeddings: Optional[np.ndarray] = None
    ids: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    model: Optional[NearestNeighbors] = None

    def fit(self, embeddings: np.ndarray, ids: List[str], labels: List[str]) -> None:
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.ids = np.asarray(ids)
        self.labels = np.asarray(labels)
        self.model = NearestNeighbors(metric="cosine", algorithm="brute")
        self.model.fit(self.embeddings)

    def query(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise RuntimeError("Index has not been fit.")
        distances, indices = self.model.kneighbors(np.asarray(queries, dtype=np.float32), n_neighbors=int(top_k))
        return distances, indices


@dataclass
class FaissInnerProductIndex:
    """Optional FAISS index for Colab/server profiles."""

    ids: Optional[np.ndarray] = None
    labels: Optional[np.ndarray] = None
    index = None

    def fit(self, embeddings: np.ndarray, ids: List[str], labels: List[str]) -> None:
        faiss = __import__("faiss")
        normalized = np.asarray(embeddings, dtype=np.float32).copy()
        faiss.normalize_L2(normalized)
        self.ids = np.asarray(ids)
        self.labels = np.asarray(labels)
        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)

    def query(self, queries: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise RuntimeError("Index has not been fit.")
        faiss = __import__("faiss")
        normalized = np.asarray(queries, dtype=np.float32).copy()
        faiss.normalize_L2(normalized)
        scores, indices = self.index.search(normalized, int(top_k))
        return scores, indices


def _build_index(backend: str):
    """Create the requested retrieval backend, with graceful FAISS fallback."""

    if backend == "faiss":
        try:
            __import__("faiss")
            return FaissInnerProductIndex(), "faiss"
        except ModuleNotFoundError:
            return SklearnCosineIndex(), "sklearn"
    return SklearnCosineIndex(), "sklearn"


def evaluate_mpnet_retrieval(profile: ProfileConfig, subset_name: str = "mpnet_main") -> pd.DataFrame:
    """Evaluate centroid-based dense retrieval over the MPNET train split."""

    output_dir = ensure_dir(profile.retrieval_dir())
    metrics_path = output_dir / "retrieval_metrics.csv"
    examples_path = output_dir / "retrieval_examples.json"
    metadata_path = output_dir / "index_metadata.json"
    if metrics_path.exists() and examples_path.exists() and metadata_path.exists():
        return pd.read_csv(metrics_path)

    split_frames = load_subset_split_frames(profile, subset_name)
    train_embeddings = load_numpy(profile.embeddings_dir() / "mpnet" / subset_name / "train_raw.npy")
    train_ids = split_frames["train"]["id"].astype(str).tolist()
    train_labels = split_frames["train"]["primary_category"].astype(str).tolist()

    index, actual_backend = _build_index(profile.retrieval_backend)
    index.fit(train_embeddings, train_ids, train_labels)

    train_frame = split_frames["train"].copy()
    train_frame["embedding_index"] = np.arange(len(train_frame))
    test_counts = split_frames["test"]["primary_category"].value_counts().to_dict()

    category_centroids = {}
    for category, group in train_frame.groupby("primary_category", sort=True):
        idx = group["embedding_index"].to_numpy()
        category_centroids[category] = train_embeddings[idx].mean(axis=0).astype(np.float32)

    max_k = min(20, len(train_embeddings))
    total_weight = 0
    weighted_sums = {5: {"MRR": 0.0, "Recall": 0.0}, 10: {"MRR": 0.0, "Recall": 0.0}, 20: {"MRR": 0.0, "Recall": 0.0}}
    examples = []

    for category in sorted(set(category_centroids).intersection(test_counts)):
        query = category_centroids[category][None, :]
        _, indices = index.query(query, top_k=max_k)
        retrieved_indices = indices[0]
        retrieved_labels = index.labels[retrieved_indices].tolist()
        retrieved_ids = index.ids[retrieved_indices].tolist()
        weight = int(test_counts[category])
        total_weight += weight

        first_relevant_rank = None
        for rank, retrieved_label in enumerate(retrieved_labels, start=1):
            if retrieved_label == category:
                first_relevant_rank = rank
                break

        for k in (5, 10, 20):
            prefix = retrieved_labels[:k]
            weighted_sums[k]["Recall"] += weight * float(category in prefix)
            if first_relevant_rank is not None and first_relevant_rank <= k:
                weighted_sums[k]["MRR"] += weight * (1.0 / first_relevant_rank)

        if len(examples) < 10:
            examples.append(
                {
                    "category": category,
                    "test_support": weight,
                    "retrieved_ids_top10": retrieved_ids[:10],
                    "retrieved_labels_top10": retrieved_labels[:10],
                }
            )

    rows = []
    for k in (5, 10, 20):
        rows.append(
            {
                "K": k,
                "MRR": weighted_sums[k]["MRR"] / max(total_weight, 1),
                "Recall@K": weighted_sums[k]["Recall"] / max(total_weight, 1),
                "query_categories": len(category_centroids),
                "weighted_test_docs": total_weight,
            }
        )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(metrics_path, index=False)
    write_json(examples_path, examples)
    write_json(
        metadata_path,
        {
            "backend": actual_backend,
            "subset_name": subset_name,
            "train_size": int(len(train_frame)),
            "test_size": int(len(split_frames["test"])),
            "category_query_count": int(len(category_centroids)),
            "note": "Category centroid pseudo-queries are weighted by held-out test support.",
        },
    )
    return metrics
