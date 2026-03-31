"""MPNET dense embedding pipeline with chunked cached outputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.lib.format import open_memmap

from src.benchmark_splits import load_subset_split_frames
from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, write_json
from src.preprocess import dense_clean
from src.reduce_umap import run_umap_search


def _mpnet_output_dir(profile: ProfileConfig, subset_name: str) -> Path:
    return ensure_dir(profile.embeddings_dir() / "mpnet" / subset_name)


def _detect_torch_device(torch_module) -> str:
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_texts_to_npy(
    model,
    texts: List[str],
    batch_size: int,
    output_path: Path,
) -> np.ndarray:
    """Encode texts in batches directly into an on-disk .npy array."""

    if output_path.exists():
        return np.load(output_path, mmap_mode="r")

    embedding_dim = model.get_sentence_embedding_dimension()
    array = open_memmap(output_path, mode="w+", dtype="float32", shape=(len(texts), embedding_dim))
    for start in range(0, len(texts), batch_size):
        stop = min(len(texts), start + batch_size)
        batch_embeddings = model.encode(
            texts[start:stop],
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)
        array[start:stop] = batch_embeddings
    del array
    return np.load(output_path, mmap_mode="r")


def run_mpnet_pipeline(profile: ProfileConfig, subset_name: str) -> Dict:
    """Encode a subset with all-mpnet-base-v2 and reduce it for clustering."""

    sentence_transformers = __import__("sentence_transformers", fromlist=["SentenceTransformer"])
    torch = __import__("torch")

    output_dir = _mpnet_output_dir(profile, subset_name)
    metadata_path = output_dir / "metadata.json"
    reduced_dir = ensure_dir(output_dir / "reduced")
    if metadata_path.exists() and (reduced_dir / "train_reduced.npy").exists():
        metadata = load_json(metadata_path, default={}) or {}
        metadata["reused_cache"] = True
        return metadata

    split_frames = load_subset_split_frames(profile, subset_name)
    train_texts = split_frames["train"]["text_input"].map(dense_clean).tolist()
    val_texts = split_frames["val"]["text_input"].map(dense_clean).tolist()
    test_texts = split_frames["test"]["text_input"].map(dense_clean).tolist()

    device = _detect_torch_device(torch)
    start = time.perf_counter()
    model = sentence_transformers.SentenceTransformer(profile.mpnet_model_name, device=device)

    train_embeddings = _encode_texts_to_npy(
        model=model,
        texts=train_texts,
        batch_size=profile.mpnet_batch_size,
        output_path=output_dir / "train_raw.npy",
    )
    val_embeddings = _encode_texts_to_npy(
        model=model,
        texts=val_texts,
        batch_size=profile.mpnet_batch_size,
        output_path=output_dir / "val_raw.npy",
    )
    test_embeddings = _encode_texts_to_npy(
        model=model,
        texts=test_texts,
        batch_size=profile.mpnet_batch_size,
        output_path=output_dir / "test_raw.npy",
    )

    umap_result = run_umap_search(
        profile=profile,
        embedding_name="mpnet",
        subset_name=subset_name,
        train_embeddings=np.asarray(train_embeddings, dtype=np.float32),
        val_embeddings=np.asarray(val_embeddings, dtype=np.float32),
        test_embeddings=np.asarray(test_embeddings, dtype=np.float32),
        train_primary_categories=split_frames["train"]["primary_category"].tolist(),
        cache_dir=reduced_dir,
        metric="cosine",
    )

    metadata = {
        "embedding": "mpnet",
        "subset_name": subset_name,
        "device": device,
        "model_name": profile.mpnet_model_name,
        "n_train": int(len(train_texts)),
        "n_val": int(len(val_texts)),
        "n_test": int(len(test_texts)),
        "best_umap_config": umap_result["best_config"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
    }
    write_json(metadata_path, metadata)
    return metadata

