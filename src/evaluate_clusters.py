"""Cluster-quality metrics and report-table updates."""

from __future__ import annotations

import itertools
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import normalized_mutual_info_score

from src.benchmark_splits import iter_split_batches, load_split_frame, split_row_count
from src.config import ProfileConfig
from src.io_utils import ensure_dir, load_json, load_numpy, upsert_csv_records, write_json
from src.preprocess import tokenize_for_npmi
from src.reduce_umap import _silhouette_score_sample


def _token_cache_path(profile: ProfileConfig, subset_name: str, split_name: str) -> Path:
    return ensure_dir(profile.metrics_dir() / "token_cache") / f"{subset_name}_{split_name}_{profile.npmi_sample_size}_tokens.pkl"


def _load_or_build_token_cache(profile: ProfileConfig, subset_name: str, split_name: str, texts: Sequence[str]) -> List[List[str]]:
    """Cache tokenized documents so NPMI can be reused across pipelines."""

    path = _token_cache_path(profile, subset_name, split_name)
    if path.exists():
        with path.open("rb") as handle:
            return pickle.load(handle)

    tokens = tokenize_for_npmi(texts)
    with path.open("wb") as handle:
        pickle.dump(tokens, handle)
    return tokens


def compute_npmi(texts: Sequence[str], labels: np.ndarray, tokenized_docs: Sequence[Sequence[str]], top_n: int = 10) -> float:
    """Compute a stable document-level NPMI score for cluster top terms."""

    labels = np.asarray(labels)
    valid_mask = labels >= 0
    filtered_texts = [text for keep, text in zip(valid_mask, texts) if keep]
    filtered_labels = labels[valid_mask]
    filtered_tokens = [tokens for keep, tokens in zip(valid_mask, tokenized_docs) if keep]
    if len(filtered_texts) < 2 or np.unique(filtered_labels).size < 2:
        return float("nan")

    cleaned_docs = [" ".join(tokens) for tokens in filtered_tokens]
    vectorizer = CountVectorizer(max_features=20_000, min_df=2)
    matrix = vectorizer.fit_transform(cleaned_docs)
    if matrix.shape[1] == 0:
        return float("nan")

    feature_names = np.asarray(vectorizer.get_feature_names_out())
    cluster_terms = {}
    candidate_terms = set()
    for cluster_id in sorted(np.unique(filtered_labels)):
        cluster_indices = np.where(filtered_labels == cluster_id)[0]
        if cluster_indices.size < 5:
            continue
        aggregated = np.asarray(matrix[cluster_indices].sum(axis=0)).ravel()
        top_indices = aggregated.argsort()[::-1]
        terms = [feature_names[idx] for idx in top_indices[:top_n] if aggregated[idx] > 0]
        if len(terms) >= 2:
            cluster_terms[int(cluster_id)] = terms
            candidate_terms.update(terms)

    if not cluster_terms:
        return float("nan")

    postings = {term: set() for term in candidate_terms}
    for doc_idx, doc_tokens in enumerate(filtered_tokens):
        unique_tokens = set(doc_tokens)
        for term in unique_tokens.intersection(candidate_terms):
            postings[term].add(doc_idx)

    n_docs = len(filtered_tokens)
    cluster_scores = []
    for terms in cluster_terms.values():
        pair_scores = []
        for term_a, term_b in itertools.combinations(terms, 2):
            docs_a = postings.get(term_a, set())
            docs_b = postings.get(term_b, set())
            if not docs_a or not docs_b:
                continue
            cooccur = len(docs_a.intersection(docs_b))
            if cooccur == 0:
                continue
            p_a = len(docs_a) / n_docs
            p_b = len(docs_b) / n_docs
            p_ab = cooccur / n_docs
            numerator = np.log(p_ab / (p_a * p_b))
            denominator = -np.log(p_ab)
            if denominator == 0:
                continue
            pair_scores.append(float(numerator / denominator))
        if pair_scores:
            cluster_scores.append(float(np.mean(pair_scores)))

    if not cluster_scores:
        return float("nan")
    return float(np.mean(cluster_scores))


def _load_labels_for_clustering(profile: ProfileConfig, embedding_name: str, clustering_name: str, subset_name: str):
    """Load the test/validation labels for the requested clustering run."""

    output_dir = profile.clustering_dir() / f"{embedding_name}_{clustering_name}" / subset_name
    if clustering_name == "hdbscan":
        return {
            "val": load_numpy(output_dir / "val_repaired_labels.npy"),
            "test": load_numpy(output_dir / "test_repaired_labels.npy"),
        }
    return {
        "val": load_numpy(output_dir / "val_labels.npy"),
        "test": load_numpy(output_dir / "test_labels.npy"),
    }


def evaluate_clustering_run(
    profile: ProfileConfig,
    embedding_name: str,
    clustering_name: str,
    subset_name: str,
) -> Dict:
    """Evaluate a cached clustering run and append it to the unified metrics table."""

    reduced_dir = profile.embeddings_dir() / embedding_name / subset_name / "reduced"
    metadata_path = profile.clustering_dir() / f"{embedding_name}_{clustering_name}" / subset_name / "metadata.json"
    metadata = load_json(metadata_path, default={}) or {}
    labels = _load_labels_for_clustering(profile, embedding_name, clustering_name, subset_name)
    val_embeddings = load_numpy(reduced_dir / "val_reduced.npy", mmap_mode="r")
    test_embeddings = load_numpy(reduced_dir / "test_reduced.npy", mmap_mode="r")

    val_true = load_split_frame(profile, subset_name, "val", columns=["primary_category"])["primary_category"].astype(str).to_numpy()
    test_true = load_split_frame(profile, subset_name, "test", columns=["primary_category"])["primary_category"].astype(str).to_numpy()

    test_count = split_row_count(profile, subset_name, "test")
    if test_count > profile.npmi_sample_size:
        rng = np.random.default_rng(profile.seed)
        sampled_indices = np.sort(rng.choice(test_count, size=profile.npmi_sample_size, replace=False))
    else:
        sampled_indices = np.arange(test_count, dtype=np.int64)

    sampled_texts = []
    cursor = 0
    for batch in iter_split_batches(profile, subset_name, "test", columns=["text_input"]):
        frame = batch.to_pandas()
        batch_start = cursor
        batch_stop = cursor + len(frame)
        batch_indices = sampled_indices[(sampled_indices >= batch_start) & (sampled_indices < batch_stop)] - batch_start
        if len(batch_indices) > 0:
            sampled_texts.extend(frame.iloc[batch_indices]["text_input"].astype(str).tolist())
        cursor = batch_stop
    sampled_pred_labels = labels["test"][sampled_indices]
    token_cache = _load_or_build_token_cache(profile, subset_name, "test", sampled_texts)

    val_silhouette = _silhouette_score_sample(
        val_embeddings,
        labels["val"],
        sample_size=profile.silhouette_eval_sample_size,
        seed=profile.seed,
    )
    test_silhouette = _silhouette_score_sample(
        test_embeddings,
        labels["test"],
        sample_size=profile.silhouette_eval_sample_size,
        seed=profile.seed,
    )
    val_nmi = float(normalized_mutual_info_score(val_true, labels["val"]))
    test_nmi = float(normalized_mutual_info_score(test_true, labels["test"]))
    test_npmi = compute_npmi(sampled_texts, sampled_pred_labels, token_cache)

    row = {
        "embedding": embedding_name,
        "clustering": clustering_name,
        "subset_name": subset_name,
        "subset_size": int(test_count),
        "best_param": metadata.get("best_param", ""),
        "silhouette": test_silhouette,
        "npmi": test_npmi,
        "nmi": test_nmi,
        "validation_silhouette": val_silhouette,
        "validation_nmi": val_nmi,
        "runtime_seconds": metadata.get("runtime_seconds"),
        "ran_successfully": True,
        "notes": (
            metadata.get("notes", "")
            + (f" NPMI computed on a deterministic sample of {len(sampled_texts):,} test documents." if len(sampled_texts) < test_count else "")
        ).strip(),
    }
    upsert_csv_records(
        profile.clustering_metrics_path(),
        records=[row],
        key_columns=("embedding", "clustering", "subset_name"),
    )

    details = {
        **row,
        "validation_label_count": int(pd.Series(labels["val"]).nunique()),
        "test_label_count": int(pd.Series(labels["test"]).nunique()),
    }
    detail_path = ensure_dir(profile.metrics_dir() / "details") / f"{embedding_name}_{clustering_name}_{subset_name}.json"
    write_json(detail_path, details)
    return row
