"""TF-IDF baseline embedding pipeline with cache-aware full-corpus support."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.benchmark_splits import iter_split_batches, split_row_count
from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, save_numpy, write_json
from src.preprocess import tfidf_clean
from src.reduce_umap import run_umap_search


def _tfidf_output_dir(profile: ProfileConfig, subset_name: str) -> Path:
    return ensure_dir(profile.embeddings_dir() / "tfidf" / subset_name)


def run_tfidf_pipeline(profile: ProfileConfig, subset_name: str) -> Dict:
    """Fit TF-IDF + SVD + reduction for a benchmark subset or full corpus."""

    output_dir = _tfidf_output_dir(profile, subset_name)
    metadata_path = output_dir / "metadata.json"
    vectorizer_path = output_dir / "vectorizer.joblib"
    svd_path = output_dir / "svd.joblib"
    reduced_dir = ensure_dir(output_dir / "reduced")

    if metadata_path.exists() and vectorizer_path.exists() and svd_path.exists():
        metadata = load_json(metadata_path, default={}) or {}
        metadata["reused_cache"] = True
        return metadata

    scipy_sparse = __import__("scipy.sparse", fromlist=["sparse"])

    def iter_clean_texts(split_name: str):
        for batch in iter_split_batches(profile, subset_name, split_name, columns=["text_input"]):
            frame = batch.to_pandas()
            for text in frame["text_input"].astype(str):
                yield tfidf_clean(text)

    train_label_frame = pd.read_parquet(
        profile.splits_dir() / f"{subset_name}_train.parquet",
        columns=["primary_category"],
    )

    start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=profile.tfidf_max_features,
        min_df=profile.tfidf_min_df,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(iter_clean_texts("train"))
    val_matrix = vectorizer.transform(iter_clean_texts("val"))
    test_matrix = vectorizer.transform(iter_clean_texts("test"))
    if profile.save_sparse_tfidf_matrices:
        scipy_sparse.save_npz(output_dir / "train_tfidf.npz", train_matrix)
        scipy_sparse.save_npz(output_dir / "val_tfidf.npz", val_matrix)
        scipy_sparse.save_npz(output_dir / "test_tfidf.npz", test_matrix)
    joblib.dump(vectorizer, vectorizer_path)

    n_components = min(profile.svd_components, max(2, min(train_matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=profile.seed)
    train_svd = svd.fit_transform(train_matrix).astype(np.float32)
    val_svd = svd.transform(val_matrix).astype(np.float32)
    test_svd = svd.transform(test_matrix).astype(np.float32)
    joblib.dump(svd, svd_path)
    save_numpy(output_dir / "train_svd.npy", train_svd)
    save_numpy(output_dir / "val_svd.npy", val_svd)
    save_numpy(output_dir / "test_svd.npy", test_svd)

    umap_result = run_umap_search(
        profile=profile,
        embedding_name="tfidf",
        subset_name=subset_name,
        train_embeddings=train_svd,
        val_embeddings=val_svd,
        test_embeddings=test_svd,
        train_primary_categories=train_label_frame["primary_category"].tolist(),
        cache_dir=reduced_dir,
        metric="cosine",
        source_array_paths={
            "train": output_dir / "train_svd.npy",
            "val": output_dir / "val_svd.npy",
            "test": output_dir / "test_svd.npy",
        },
    )

    metadata = {
        "embedding": "tfidf",
        "subset_name": subset_name,
        "n_train": int(split_row_count(profile, subset_name, "train")),
        "n_val": int(split_row_count(profile, subset_name, "val")),
        "n_test": int(split_row_count(profile, subset_name, "test")),
        "n_features": int(train_matrix.shape[1]),
        "svd_components": int(n_components),
        "best_umap_config": umap_result["best_config"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
    }
    write_json(metadata_path, metadata)
    return metadata
