"""Orchestrate the benchmark pipeline across local and Colab profiles."""

from __future__ import annotations

import argparse
import traceback
from typing import Callable, Dict, Optional, Set, Tuple

from src.benchmark_splits import create_benchmark_subsets
from src.build_results_notebook import build_results_notebook
from src.cluster_agglomerative import run_agglomerative_clustering
from src.cluster_hdbscan import run_hdbscan_clustering
from src.cluster_kmeans import run_kmeans_clustering
from src.config import ProfileConfig, ensure_artifact_dirs, get_profile
from src.dataset_builder import build_clean_dataset
from src.embed_bert import run_bert_pipeline
from src.embed_mpnet import run_mpnet_pipeline
from src.embed_tfidf import run_tfidf_pipeline
from src.evaluate_clusters import evaluate_clustering_run
from src.make_report_tables import refresh_report_artifacts
from src.retrieval_dense import evaluate_mpnet_retrieval


EmbeddingKey = Tuple[str, str]
ClusteringKey = Tuple[str, str, str]


def _run_step(name: str, fn: Callable[[], object]):
    """Run a pipeline step with concise logging and non-fatal failure handling."""

    print(f"[run_pipeline] starting {name}")
    try:
        result = fn()
        print(f"[run_pipeline] finished {name}")
        return result
    except Exception as exc:
        print(f"[run_pipeline] step failed: {name}")
        print(f"[run_pipeline] {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return None


def _embedding_runner(embedding_name: str) -> Callable[[ProfileConfig, str], Dict]:
    runners = {
        "tfidf": run_tfidf_pipeline,
        "mpnet": run_mpnet_pipeline,
        "bert": run_bert_pipeline,
    }
    if embedding_name not in runners:
        raise ValueError(f"Unsupported embedding: {embedding_name}")
    return runners[embedding_name]


def _clustering_runner(clustering_name: str) -> Callable[[ProfileConfig, str, str], Dict]:
    runners = {
        "kmeans": run_kmeans_clustering,
        "hdbscan": run_hdbscan_clustering,
        "agglomerative": run_agglomerative_clustering,
    }
    if clustering_name not in runners:
        raise ValueError(f"Unsupported clustering algorithm: {clustering_name}")
    return runners[clustering_name]


def _ensure_embedding(
    profile: ProfileConfig,
    embedding_name: str,
    subset_name: str,
    completed: Set[EmbeddingKey],
    failed: Set[EmbeddingKey],
) -> Optional[Dict]:
    """Run an embedding stage once per embedding/subset pair."""

    key = (embedding_name, subset_name)
    if key in completed:
        print(f"[run_pipeline] reusing scheduled embedding stage {embedding_name}/{subset_name}")
        return {}
    if key in failed:
        print(f"[run_pipeline] skipping embedding stage after earlier failure {embedding_name}/{subset_name}")
        return None

    result = _run_step(
        f"{embedding_name}_embedding_{subset_name}",
        lambda: _embedding_runner(embedding_name)(profile, subset_name),
    )
    if result is not None:
        completed.add(key)
    else:
        failed.add(key)
    return result


def _ensure_clustering(
    profile: ProfileConfig,
    embedding_name: str,
    clustering_name: str,
    subset_name: str,
    completed: Set[ClusteringKey],
    failed: Set[ClusteringKey],
) -> Optional[Dict]:
    """Run a clustering stage once per embedding/clustering/subset combination."""

    key = (embedding_name, clustering_name, subset_name)
    if key in completed:
        print(f"[run_pipeline] reusing scheduled clustering stage {embedding_name}/{clustering_name}/{subset_name}")
        return {}
    if key in failed:
        print(f"[run_pipeline] skipping clustering stage after earlier failure {embedding_name}/{clustering_name}/{subset_name}")
        return None

    result = _run_step(
        f"{embedding_name}_{clustering_name}_{subset_name}",
        lambda: _clustering_runner(clustering_name)(profile, embedding_name, subset_name),
    )
    if result is not None:
        completed.add(key)
    else:
        failed.add(key)
    return result


def _ensure_clustering_metrics(
    profile: ProfileConfig,
    embedding_name: str,
    clustering_name: str,
    subset_name: str,
    completed: Set[ClusteringKey],
    failed: Set[ClusteringKey],
) -> Optional[Dict]:
    """Run the evaluation stage once per embedding/clustering/subset combination."""

    key = (embedding_name, clustering_name, subset_name)
    if key in completed:
        print(f"[run_pipeline] reusing scheduled metrics stage {embedding_name}/{clustering_name}/{subset_name}")
        return {}
    if key in failed:
        print(f"[run_pipeline] skipping metrics stage after earlier failure {embedding_name}/{clustering_name}/{subset_name}")
        return None

    result = _run_step(
        f"{embedding_name}_{clustering_name}_metrics_{subset_name}",
        lambda: evaluate_clustering_run(profile, embedding_name, clustering_name, subset_name),
    )
    if result is not None:
        completed.add(key)
    else:
        failed.add(key)
    return result


def _run_clustering_bundle(
    profile: ProfileConfig,
    embedding_name: str,
    clustering_name: str,
    subset_name: str,
    embedding_stages: Set[EmbeddingKey],
    failed_embedding_stages: Set[EmbeddingKey],
    clustering_stages: Set[ClusteringKey],
    failed_clustering_stages: Set[ClusteringKey],
    metric_stages: Set[ClusteringKey],
    failed_metric_stages: Set[ClusteringKey],
) -> None:
    """Run embedding, clustering, and evaluation for one configured pipeline bundle."""

    if _ensure_embedding(profile, embedding_name, subset_name, embedding_stages, failed_embedding_stages) is None:
        return
    if _ensure_clustering(profile, embedding_name, clustering_name, subset_name, clustering_stages, failed_clustering_stages) is None:
        return
    _ensure_clustering_metrics(profile, embedding_name, clustering_name, subset_name, metric_stages, failed_metric_stages)


def _print_profile_notes(profile: ProfileConfig) -> None:
    """Log important execution notes for very large full-corpus profiles."""

    if profile.hdbscan_runs_full_corpus() and profile.run_hdbscan:
        print("[run_pipeline] note: HDBSCAN is configured on full_corpus and remains CPU/memory heavy at this scale.")
    if profile.agglomerative_runs_full_corpus() and profile.run_agglomerative:
        print("[run_pipeline] note: Agglomerative clustering is configured on full_corpus and can become prohibitively expensive.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the benchmark pipeline with cache reuse.")
    parser.add_argument(
        "--profile",
        default="local_profile",
        choices=("local_profile", "full_profile", "colab_a100_profile", "colab_h100_full_profile"),
    )
    bert_group = parser.add_mutually_exclusive_group()
    bert_group.add_argument("--run-bert", action="store_true", help="Force-enable the BERT pipelines.")
    bert_group.add_argument("--skip-bert", action="store_true", help="Force-disable the BERT pipelines.")
    args = parser.parse_args()

    run_bert_override = True if args.run_bert else False if args.skip_bert else None
    profile = get_profile(args.profile, run_bert_override=run_bert_override)
    ensure_artifact_dirs(profile)
    _print_profile_notes(profile)

    _run_step("build_clean_dataset", lambda: build_clean_dataset(profile))
    _run_step("create_benchmark_subsets", lambda: create_benchmark_subsets(profile))

    embedding_stages: Set[EmbeddingKey] = set()
    failed_embedding_stages: Set[EmbeddingKey] = set()
    clustering_stages: Set[ClusteringKey] = set()
    failed_clustering_stages: Set[ClusteringKey] = set()
    metric_stages: Set[ClusteringKey] = set()
    failed_metric_stages: Set[ClusteringKey] = set()

    pipeline_bundles = [
        ("tfidf", "kmeans", profile.tfidf_main_subset_name, True),
        ("mpnet", "kmeans", profile.mpnet_main_subset_name, bool(profile.run_mpnet)),
        ("tfidf", "hdbscan", profile.hdbscan_subset_name, bool(profile.run_hdbscan)),
        ("mpnet", "hdbscan", profile.hdbscan_subset_name, bool(profile.run_mpnet and profile.run_hdbscan)),
        ("tfidf", "agglomerative", profile.agglomerative_subset_name, bool(profile.run_agglomerative)),
        ("mpnet", "agglomerative", profile.agglomerative_subset_name, bool(profile.run_mpnet and profile.run_agglomerative)),
        ("bert", "kmeans", profile.bert_main_subset_name, bool(profile.run_bert)),
        ("bert", "hdbscan", profile.hdbscan_subset_name, bool(profile.run_bert and profile.run_hdbscan)),
        ("bert", "agglomerative", profile.agglomerative_subset_name, bool(profile.run_bert and profile.run_agglomerative)),
    ]

    for embedding_name, clustering_name, subset_name, enabled in pipeline_bundles:
        if not enabled:
            continue
        _run_clustering_bundle(
            profile=profile,
            embedding_name=embedding_name,
            clustering_name=clustering_name,
            subset_name=subset_name,
            embedding_stages=embedding_stages,
            failed_embedding_stages=failed_embedding_stages,
            clustering_stages=clustering_stages,
            failed_clustering_stages=failed_clustering_stages,
            metric_stages=metric_stages,
            failed_metric_stages=failed_metric_stages,
        )

    if profile.run_mpnet:
        if _ensure_embedding(
            profile,
            "mpnet",
            profile.mpnet_main_subset_name,
            embedding_stages,
            failed_embedding_stages,
        ) is not None:
            _run_step("mpnet_retrieval", lambda: evaluate_mpnet_retrieval(profile, profile.mpnet_main_subset_name))

    refresh_report_artifacts(profile)
    notebook_path = build_results_notebook(profile)
    print(f"[run_pipeline] notebook written to {notebook_path}")


if __name__ == "__main__":
    main()
