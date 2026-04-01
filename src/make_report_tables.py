"""Utilities for report-ready CSVs and pipeline status tables."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import ProfileConfig
from src.io_utils import load_json, write_json


def _expected_pipeline_rows(profile: ProfileConfig) -> List[Dict]:
    return [
        {"pipeline_name": "tfidf_kmeans_main", "enabled_by_default": True, "embedding": "tfidf", "clustering": "kmeans", "subset_name": profile.tfidf_main_subset_name},
        {"pipeline_name": "tfidf_hdbscan_small", "enabled_by_default": bool(profile.run_hdbscan), "embedding": "tfidf", "clustering": "hdbscan", "subset_name": profile.hdbscan_subset_name},
        {"pipeline_name": "mpnet_kmeans_main", "enabled_by_default": bool(profile.run_mpnet), "embedding": "mpnet", "clustering": "kmeans", "subset_name": profile.mpnet_main_subset_name},
        {"pipeline_name": "mpnet_hdbscan_small", "enabled_by_default": bool(profile.run_mpnet and profile.run_hdbscan), "embedding": "mpnet", "clustering": "hdbscan", "subset_name": profile.hdbscan_subset_name},
        {"pipeline_name": "tfidf_agglomerative_small", "enabled_by_default": bool(profile.run_agglomerative), "embedding": "tfidf", "clustering": "agglomerative", "subset_name": profile.agglomerative_subset_name},
        {"pipeline_name": "mpnet_agglomerative_small", "enabled_by_default": bool(profile.run_mpnet and profile.run_agglomerative), "embedding": "mpnet", "clustering": "agglomerative", "subset_name": profile.agglomerative_subset_name},
        {"pipeline_name": "bert_kmeans_main", "enabled_by_default": bool(profile.run_bert), "embedding": "bert", "clustering": "kmeans", "subset_name": profile.bert_main_subset_name},
        {"pipeline_name": "bert_hdbscan_small", "enabled_by_default": bool(profile.run_bert and profile.run_hdbscan), "embedding": "bert", "clustering": "hdbscan", "subset_name": profile.hdbscan_subset_name},
        {"pipeline_name": "bert_agglomerative_small", "enabled_by_default": bool(profile.run_bert and profile.run_agglomerative), "embedding": "bert", "clustering": "agglomerative", "subset_name": profile.agglomerative_subset_name},
        {"pipeline_name": "mpnet_retrieval_main", "enabled_by_default": bool(profile.run_mpnet), "embedding": "mpnet", "clustering": "retrieval", "subset_name": profile.mpnet_main_subset_name},
    ]


def refresh_report_artifacts(profile: ProfileConfig) -> Dict[str, str]:
    """Build compact report CSVs from the saved modeling artifacts."""

    results_dir = profile.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    clustering_metrics_path = profile.clustering_metrics_path()
    if clustering_metrics_path.exists():
        clustering = pd.read_csv(clustering_metrics_path)
    else:
        clustering = pd.DataFrame(
            columns=["embedding", "clustering", "subset_name", "silhouette", "npmi", "nmi", "ran_successfully"]
        )
    if "ran_successfully" in clustering.columns:
        clustering["ran_successfully"] = (
            clustering["ran_successfully"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )

    clustering_report = clustering.copy()
    if not clustering_report.empty:
        clustering_report = clustering_report[clustering_report["ran_successfully"] == True]
        clustering_report = clustering_report.rename(
            columns={
                "embedding": "Embedding",
                "clustering": "Clustering",
                "silhouette": "Silhouette",
                "npmi": "NPMI",
                "nmi": "NMI",
            }
        )
        clustering_report = clustering_report[["Embedding", "Clustering", "Silhouette", "NPMI", "NMI"]]
    clustering_report_path = results_dir / "clustering_results_report.csv"
    clustering_report.to_csv(clustering_report_path, index=False)

    retrieval_path = profile.retrieval_metrics_path()
    if retrieval_path.exists():
        retrieval = pd.read_csv(retrieval_path)
    else:
        retrieval = pd.DataFrame(columns=["K", "MRR", "Recall@K"])
    retrieval_report = retrieval[["K", "MRR", "Recall@K"]] if not retrieval.empty else retrieval
    retrieval_report_path = results_dir / "retrieval_results_report.csv"
    retrieval_report.to_csv(retrieval_report_path, index=False)

    split_summary = load_json(profile.split_summary_path(), default={}) or {}
    split_report = {
        "cleaned_row_count": split_summary.get("cleaned_row_count"),
        "category_count_used_by_subset": {
            subset_name: details.get("category_count")
            for subset_name, details in split_summary.get("subset_summaries", {}).items()
        },
        "split_sizes": {
            subset_name: {
                "train": details.get("train_size"),
                "val": details.get("val_size"),
                "test": details.get("test_size"),
            }
            for subset_name, details in split_summary.get("subset_summaries", {}).items()
        },
        "notes": [
            "local_profile uses deterministic subsets sized for a laptop-friendly pipeline.",
            "The H100 Colab profile can run TF-IDF and BERT over the full cleaned corpus.",
            "Full-corpus runs use scalable approximations where needed, including MiniBatchKMeans and UMAP bypass on very large splits.",
        ],
    }
    split_report_path = results_dir / "split_report.json"
    write_json(split_report_path, split_report)

    status_rows = []
    for spec in _expected_pipeline_rows(profile):
        if spec["clustering"] == "retrieval":
            ran = retrieval_path.exists()
            success = ran and not retrieval.empty
            reason = "" if success else ("disabled_by_default" if not spec["enabled_by_default"] else "not_run_or_skipped")
        elif not clustering.empty:
            row = clustering[
                (clustering["embedding"] == spec["embedding"])
                & (clustering["clustering"] == spec["clustering"])
                & (clustering["subset_name"] == spec["subset_name"])
            ]
            ran = not row.empty
            success = bool(ran and row["ran_successfully"].fillna(False).iloc[-1])
            reason = "" if success else ("disabled_by_default" if not spec["enabled_by_default"] else "not_run_or_skipped")
        else:
            ran = False
            success = False
            reason = "disabled_by_default" if not spec["enabled_by_default"] else "not_run_or_skipped"

        status_rows.append(
            {
                "pipeline_name": spec["pipeline_name"],
                "enabled_by_default": spec["enabled_by_default"],
                "ran": ran,
                "success": success,
                "reason_skipped": reason,
            }
        )

    pipeline_status = pd.DataFrame(status_rows)
    pipeline_status_path = profile.pipeline_status_path()
    pipeline_status.to_csv(pipeline_status_path, index=False)

    return {
        "clustering_report": str(clustering_report_path),
        "retrieval_report": str(retrieval_report_path),
        "split_report": str(split_report_path),
        "pipeline_status": str(pipeline_status_path),
    }
