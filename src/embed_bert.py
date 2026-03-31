"""Optional BERT embedding pipeline for small local subsets."""

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


def _bert_output_dir(profile: ProfileConfig, subset_name: str) -> Path:
    return ensure_dir(profile.embeddings_dir() / "bert" / subset_name)


def _detect_torch_device(torch_module) -> str:
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return "mps"
    return "cpu"


def _encode_bert_cls(
    tokenizer,
    model,
    torch,
    texts: List[str],
    batch_size: int,
    max_length: int,
    output_path: Path,
    device: str,
) -> np.ndarray:
    """Encode [CLS] embeddings into a persisted .npy array."""

    if output_path.exists():
        return np.load(output_path, mmap_mode="r")

    hidden_size = int(model.config.hidden_size)
    array = open_memmap(output_path, mode="w+", dtype="float32", shape=(len(texts), hidden_size))
    for start in range(0, len(texts), batch_size):
        stop = min(len(texts), start + batch_size)
        encoded = tokenizer(
            texts[start:stop],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().float().numpy()
        array[start:stop] = cls_embeddings.astype(np.float32)
    del array
    return np.load(output_path, mmap_mode="r")


def run_bert_pipeline(profile: ProfileConfig, subset_name: str) -> Dict:
    """Encode a subset with bert-base-uncased using [CLS] representations."""

    torch = __import__("torch")
    transformers = __import__("transformers", fromlist=["AutoModel", "AutoTokenizer"])

    output_dir = _bert_output_dir(profile, subset_name)
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
    tokenizer = transformers.AutoTokenizer.from_pretrained(profile.bert_model_name)
    model = transformers.AutoModel.from_pretrained(profile.bert_model_name)
    model = model.to(device)
    model.eval()

    train_embeddings = _encode_bert_cls(
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        texts=train_texts,
        batch_size=profile.bert_batch_size,
        max_length=profile.bert_max_length,
        output_path=output_dir / "train_raw.npy",
        device=device,
    )
    val_embeddings = _encode_bert_cls(
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        texts=val_texts,
        batch_size=profile.bert_batch_size,
        max_length=profile.bert_max_length,
        output_path=output_dir / "val_raw.npy",
        device=device,
    )
    test_embeddings = _encode_bert_cls(
        tokenizer=tokenizer,
        model=model,
        torch=torch,
        texts=test_texts,
        batch_size=profile.bert_batch_size,
        max_length=profile.bert_max_length,
        output_path=output_dir / "test_raw.npy",
        device=device,
    )

    umap_result = run_umap_search(
        profile=profile,
        embedding_name="bert",
        subset_name=subset_name,
        train_embeddings=np.asarray(train_embeddings, dtype=np.float32),
        val_embeddings=np.asarray(val_embeddings, dtype=np.float32),
        test_embeddings=np.asarray(test_embeddings, dtype=np.float32),
        train_primary_categories=split_frames["train"]["primary_category"].tolist(),
        cache_dir=reduced_dir,
        metric="cosine",
    )

    metadata = {
        "embedding": "bert",
        "subset_name": subset_name,
        "device": device,
        "model_name": profile.bert_model_name,
        "n_train": int(len(train_texts)),
        "n_val": int(len(val_texts)),
        "n_test": int(len(test_texts)),
        "best_umap_config": umap_result["best_config"],
        "runtime_seconds": time.perf_counter() - start,
        "reused_cache": False,
    }
    write_json(metadata_path, metadata)
    return metadata

