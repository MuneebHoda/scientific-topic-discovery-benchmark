"""Orchestrate the benchmark pipeline across local and Colab profiles."""

from __future__ import annotations

import argparse
import traceback

from src.benchmark_splits import create_benchmark_subsets
from src.build_results_notebook import build_results_notebook
from src.cluster_agglomerative import run_agglomerative_clustering
from src.cluster_hdbscan import run_hdbscan_clustering
from src.cluster_kmeans import run_kmeans_clustering
from src.config import ensure_artifact_dirs, get_profile
from src.dataset_builder import build_clean_dataset
from src.embed_bert import run_bert_pipeline
from src.embed_mpnet import run_mpnet_pipeline
from src.embed_tfidf import run_tfidf_pipeline
from src.evaluate_clusters import evaluate_clustering_run
from src.make_report_tables import refresh_report_artifacts
from src.retrieval_dense import evaluate_mpnet_retrieval


def _run_step(name: str, fn):
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

    _run_step("build_clean_dataset", lambda: build_clean_dataset(profile))
    _run_step("create_benchmark_subsets", lambda: create_benchmark_subsets(profile))

    # Phase 1: TF-IDF baseline
    tfidf_subset = profile.tfidf_main_subset_name
    if _run_step("tfidf_main_embedding", lambda: run_tfidf_pipeline(profile, tfidf_subset)) is not None:
        if _run_step("tfidf_main_kmeans", lambda: run_kmeans_clustering(profile, "tfidf", tfidf_subset)) is not None:
            _run_step("tfidf_main_kmeans_metrics", lambda: evaluate_clustering_run(profile, "tfidf", "kmeans", tfidf_subset))
    refresh_report_artifacts(profile)

    # Phase 2: main dense baseline and retrieval
    mpnet_subset = profile.mpnet_main_subset_name
    if profile.run_mpnet:
        if _run_step("mpnet_main_embedding", lambda: run_mpnet_pipeline(profile, mpnet_subset)) is not None:
            if _run_step("mpnet_main_kmeans", lambda: run_kmeans_clustering(profile, "mpnet", mpnet_subset)) is not None:
                _run_step("mpnet_main_kmeans_metrics", lambda: evaluate_clustering_run(profile, "mpnet", "kmeans", mpnet_subset))
            _run_step("mpnet_retrieval", lambda: evaluate_mpnet_retrieval(profile, mpnet_subset))
    refresh_report_artifacts(profile)

    # Phase 3: reduced subset clustering families
    hdbscan_subset = profile.hdbscan_subset_name
    if profile.run_hdbscan:
        if _run_step("tfidf_hdbscan_embedding", lambda: run_tfidf_pipeline(profile, hdbscan_subset)) is not None:
            if _run_step("tfidf_hdbscan", lambda: run_hdbscan_clustering(profile, "tfidf", hdbscan_subset)) is not None:
                _run_step("tfidf_hdbscan_metrics", lambda: evaluate_clustering_run(profile, "tfidf", "hdbscan", hdbscan_subset))
        if profile.run_mpnet:
            if _run_step("mpnet_hdbscan_embedding", lambda: run_mpnet_pipeline(profile, hdbscan_subset)) is not None:
                if _run_step("mpnet_hdbscan", lambda: run_hdbscan_clustering(profile, "mpnet", hdbscan_subset)) is not None:
                    _run_step("mpnet_hdbscan_metrics", lambda: evaluate_clustering_run(profile, "mpnet", "hdbscan", hdbscan_subset))

    agglomerative_subset = profile.agglomerative_subset_name
    if profile.run_agglomerative:
        if _run_step("tfidf_agglomerative_embedding", lambda: run_tfidf_pipeline(profile, agglomerative_subset)) is not None:
            if _run_step("tfidf_agglomerative", lambda: run_agglomerative_clustering(profile, "tfidf", agglomerative_subset)) is not None:
                _run_step("tfidf_agglomerative_metrics", lambda: evaluate_clustering_run(profile, "tfidf", "agglomerative", agglomerative_subset))
        if profile.run_mpnet:
            if _run_step("mpnet_agglomerative_embedding", lambda: run_mpnet_pipeline(profile, agglomerative_subset)) is not None:
                if _run_step("mpnet_agglomerative", lambda: run_agglomerative_clustering(profile, "mpnet", agglomerative_subset)) is not None:
                    _run_step("mpnet_agglomerative_metrics", lambda: evaluate_clustering_run(profile, "mpnet", "agglomerative", agglomerative_subset))
    refresh_report_artifacts(profile)

    # Phase 4: optional BERT
    bert_subset = profile.bert_main_subset_name
    if profile.run_bert:
        if _run_step("bert_main_embedding", lambda: run_bert_pipeline(profile, bert_subset)) is not None:
            if _run_step("bert_main_kmeans", lambda: run_kmeans_clustering(profile, "bert", bert_subset)) is not None:
                _run_step("bert_main_kmeans_metrics", lambda: evaluate_clustering_run(profile, "bert", "kmeans", bert_subset))
        if profile.run_hdbscan:
            if _run_step("bert_hdbscan_embedding", lambda: run_bert_pipeline(profile, hdbscan_subset)) is not None:
                if _run_step("bert_hdbscan", lambda: run_hdbscan_clustering(profile, "bert", hdbscan_subset)) is not None:
                    _run_step("bert_hdbscan_metrics", lambda: evaluate_clustering_run(profile, "bert", "hdbscan", hdbscan_subset))
        if profile.run_agglomerative:
            if _run_step("bert_agglomerative_embedding", lambda: run_bert_pipeline(profile, agglomerative_subset)) is not None:
                if _run_step("bert_agglomerative", lambda: run_agglomerative_clustering(profile, "bert", agglomerative_subset)) is not None:
                    _run_step("bert_agglomerative_metrics", lambda: evaluate_clustering_run(profile, "bert", "agglomerative", agglomerative_subset))
        refresh_report_artifacts(profile)

    notebook_path = build_results_notebook(profile)
    print(f"[run_pipeline] notebook written to {notebook_path}")


if __name__ == "__main__":
    main()
