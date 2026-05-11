"""MPNET dense embedding pipeline with CUDA-aware chunked outputs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

from src.benchmark_splits import iter_split_batches, split_row_count
from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, write_json
from src.preprocess import dense_clean
from src.reduce_umap import run_umap_search


def _mpnet_output_dir(profile: ProfileConfig, subset_name: str) -> Path:
    return ensure_dir(profile.embeddings_dir() / "mpnet" / subset_name)


def _detect_torch_device(torch_module) -> str:
    if torch_module.cuda.is_available():
        return "cuda"
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_split_to_npy(
    profile: ProfileConfig,
    subset_name: str,
    split_name: str,
    model,
    batch_size: int,
    output_path: Path,
) -> np.ndarray:
    """Encode a split directly into an on-disk numpy array."""

    total_rows = split_row_count(profile, subset_name, split_name)
    embedding_dim = model.get_sentence_embedding_dimension()
    expected_shape = (total_rows, embedding_dim)

    if output_path.exists():
        try:
            cached = np.load(output_path, mmap_mode="r")
            if tuple(cached.shape) == expected_shape:
                return cached
        except Exception:
            pass
        output_path.unlink(missing_ok=True)

    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    temp_path.unlink(missing_ok=True)
    array = open_memmap(temp_path, mode="w+", dtype="float32", shape=expected_shape)

    cursor = 0
    progress_every = 50_000
    next_progress = progress_every
    print(f"[mpnet] encoding {subset_name}/{split_name}: rows={total_rows:,} dim={embedding_dim}", flush=True)
    for batch in iter_split_batches(profile, subset_name, split_name, columns=["text_input"]):
        frame = batch.to_pandas()
        texts = frame["text_input"].astype(str).map(dense_clean).tolist()
        for start in range(0, len(texts), batch_size):
            stop = min(len(texts), start + batch_size)
            batch_embeddings = model.encode(
                texts[start:stop],
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)
            next_cursor = cursor + len(batch_embeddings)
            array[cursor:next_cursor] = batch_embeddings
            cursor = next_cursor
            if cursor >= next_progress or cursor == total_rows:
                print(f"[mpnet] {subset_name}/{split_name} encoded={cursor:,}/{total_rows:,}", flush=True)
                while next_progress <= cursor:
                    next_progress += progress_every

    del array
    temp_path.replace(output_path)
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

    device = _detect_torch_device(torch)
    start = time.perf_counter()
    model = sentence_transformers.SentenceTransformer(profile.mpnet_model_name, device=device)

    train_embeddings = _encode_split_to_npy(
        profile=profile,
        subset_name=subset_name,
        split_name="train",
        model=model,
        batch_size=profile.mpnet_batch_size,
        output_path=output_dir / "train_raw.npy",
    )
    val_embeddings = _encode_split_to_npy(
        profile=profile,
        subset_name=subset_name,
        split_name="val",
        model=model,
        batch_size=profile.mpnet_batch_size,
        output_path=output_dir / "val_raw.npy",
    )
    test_embeddings = _encode_split_to_npy(
        profile=profile,
        subset_name=subset_name,
        split_name="test",
        model=model,
        batch_size=profile.mpnet_batch_size,
        output_path=output_dir / "test_raw.npy",
    )

    train_labels = pd.read_parquet(
        profile.splits_dir() / f"{subset_name}_train.parquet",
        columns=["primary_category"],
    )["primary_category"].tolist()

    umap_result = run_umap_search(
        profile=profile,
        embedding_name="mpnet",
        subset_name=subset_name,
        train_embeddings=train_embeddings,
        val_embeddings=val_embeddings,
        test_embeddings=test_embeddings,
        train_primary_categories=train_labels,
        cache_dir=reduced_dir,
        metric="cosine",
        source_array_paths={
            "train": output_dir / "train_raw.npy",
            "val": output_dir / "val_raw.npy",
            "test": output_dir / "test_raw.npy",
        },
    )

    metadata = {
        "embedding": "mpnet",
        "subset_name": subset_name,
        "device": device,
        "model_name": profile.mpnet_model_name,
        "n_train": int(split_row_count(profile, subset_name, "train")),
        "n_val": int(split_row_count(profile, subset_name, "val")),
        "n_test": int(split_row_count(profile, subset_name, "test")),
        "best_umap_config": umap_result["best_config"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
    }
    write_json(metadata_path, metadata)
    return metadata
