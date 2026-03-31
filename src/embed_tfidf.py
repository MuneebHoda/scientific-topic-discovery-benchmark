"""TF-IDF baseline embedding pipeline with cached reductions."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.benchmark_splits import load_subset_split_frames
from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, save_numpy, write_json
from src.preprocess import tfidf_clean
from src.reduce_umap import run_umap_search


def _tfidf_output_dir(profile: ProfileConfig, subset_name: str) -> Path:
    return ensure_dir(profile.embeddings_dir() / "tfidf" / subset_name)


def run_tfidf_pipeline(profile: ProfileConfig, subset_name: str) -> Dict:
    """Fit TF-IDF + SVD + UMAP for a benchmark subset."""

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
    split_frames = load_subset_split_frames(profile, subset_name)
    train_texts = split_frames["train"]["text_input"].map(tfidf_clean).tolist()
    val_texts = split_frames["val"]["text_input"].map(tfidf_clean).tolist()
    test_texts = split_frames["test"]["text_input"].map(tfidf_clean).tolist()

    start = time.perf_counter()
    vectorizer = TfidfVectorizer(
        max_features=profile.tfidf_max_features,
        min_df=profile.tfidf_min_df,
        sublinear_tf=True,
        norm="l2",
        dtype=np.float32,
    )
    train_matrix = vectorizer.fit_transform(train_texts)
    val_matrix = vectorizer.transform(val_texts)
    test_matrix = vectorizer.transform(test_texts)
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
        train_primary_categories=split_frames["train"]["primary_category"].tolist(),
        cache_dir=reduced_dir,
        metric="cosine",
    )

    metadata = {
        "embedding": "tfidf",
        "subset_name": subset_name,
        "n_train": int(train_matrix.shape[0]),
        "n_val": int(val_matrix.shape[0]),
        "n_test": int(test_matrix.shape[0]),
        "n_features": int(train_matrix.shape[1]),
        "svd_components": int(n_components),
        "best_umap_config": umap_result["best_config"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
    }
    write_json(metadata_path, metadata)
    return metadata

