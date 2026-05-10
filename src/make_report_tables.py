"""Utilities for report-ready CSVs and pipeline status tables."""

from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.config import ProfileConfig
from src.io_utils import load_json, write_json


def _pipeline_name(embedding: str, clustering: str, subset_name: str) -> str:
    return f"{embedding}_{clustering}_{subset_name}"


def _expected_pipeline_rows(profile: ProfileConfig) -> List[Dict]:
    return [
        {
            "pipeline_name": _pipeline_name("tfidf", "kmeans", profile.tfidf_main_subset_name),
            "enabled_by_default": bool(profile.run_tfidf and profile.run_kmeans),
            "embedding": "tfidf",
            "clustering": "kmeans",
            "subset_name": profile.tfidf_main_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("tfidf", "hdbscan", profile.hdbscan_subset_name),
            "enabled_by_default": bool(profile.run_tfidf and profile.run_hdbscan),
            "embedding": "tfidf",
            "clustering": "hdbscan",
            "subset_name": profile.hdbscan_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("mpnet", "kmeans", profile.mpnet_main_subset_name),
            "enabled_by_default": bool(profile.run_mpnet and profile.run_kmeans),
            "embedding": "mpnet",
            "clustering": "kmeans",
            "subset_name": profile.mpnet_main_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("mpnet", "hdbscan", profile.hdbscan_subset_name),
            "enabled_by_default": bool(profile.run_mpnet and profile.run_hdbscan),
            "embedding": "mpnet",
            "clustering": "hdbscan",
            "subset_name": profile.hdbscan_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("tfidf", "agglomerative", profile.agglomerative_subset_name),
            "enabled_by_default": bool(profile.run_tfidf and profile.run_agglomerative),
            "embedding": "tfidf",
            "clustering": "agglomerative",
            "subset_name": profile.agglomerative_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("mpnet", "agglomerative", profile.agglomerative_subset_name),
            "enabled_by_default": bool(profile.run_mpnet and profile.run_agglomerative),
            "embedding": "mpnet",
            "clustering": "agglomerative",
            "subset_name": profile.agglomerative_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("bert", "kmeans", profile.bert_main_subset_name),
            "enabled_by_default": bool(profile.run_bert and profile.run_kmeans),
            "embedding": "bert",
            "clustering": "kmeans",
            "subset_name": profile.bert_main_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("bert", "hdbscan", profile.hdbscan_subset_name),
            "enabled_by_default": bool(profile.run_bert and profile.run_hdbscan),
            "embedding": "bert",
            "clustering": "hdbscan",
            "subset_name": profile.hdbscan_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("bert", "agglomerative", profile.agglomerative_subset_name),
            "enabled_by_default": bool(profile.run_bert and profile.run_agglomerative),
            "embedding": "bert",
            "clustering": "agglomerative",
            "subset_name": profile.agglomerative_subset_name,
        },
        {
            "pipeline_name": _pipeline_name("mpnet", "retrieval", profile.mpnet_main_subset_name),
            "enabled_by_default": bool(profile.run_mpnet),
            "embedding": "mpnet",
            "clustering": "retrieval",
            "subset_name": profile.mpnet_main_subset_name,
        },
    ]


def _profile_summary(profile: ProfileConfig, split_summary: Dict) -> Dict:
    subset_configs = profile.subset_configs()
    return {
        "profile_name": profile.name,
        "raw_data_path": str(profile.raw_data_path),
        "artifacts_dir": str(profile.artifacts_dir),
        "subset_configs": {
            subset_name: {
                "size": spec.size,
                "min_category_count": spec.min_category_count,
                "min_subset_per_category": spec.min_subset_per_category,
                "use_full_dataset": spec.use_full_dataset,
            }
            for subset_name, spec in subset_configs.items()
        },
        "full_corpus_subsets": sorted(subset_name for subset_name, spec in subset_configs.items() if spec.use_full_dataset),
        "enabled_components": {
            "tfidf": bool(profile.run_tfidf),
            "mpnet": bool(profile.run_mpnet),
            "bert": bool(profile.run_bert),
            "kmeans": bool(profile.run_kmeans),
            "hdbscan": bool(profile.run_hdbscan),
            "agglomerative": bool(profile.run_agglomerative),
            "retrieval": bool(profile.run_mpnet),
        },
        "evaluation_sampling": {
            "silhouette_eval_sample_size": int(profile.silhouette_eval_sample_size),
            "npmi_sample_size": int(profile.npmi_sample_size),
        },
        "split_fractions": {
            "train": float(profile.train_fraction),
            "val": float(profile.val_fraction),
            "test": float(profile.test_fraction),
        },
        "split_summary": split_summary,
    }


def refresh_report_artifacts(profile: ProfileConfig) -> Dict[str, str]:
    """Build compact report CSVs from the saved modeling artifacts."""

    results_dir = profile.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)

    clustering_metrics_path = profile.clustering_metrics_path()
    if clustering_metrics_path.exists():
        clustering = pd.read_csv(clustering_metrics_path)
    else:
        clustering = pd.DataFrame(
            columns=[
                "embedding",
                "clustering",
                "subset_name",
                "subset_size",
                "best_param",
                "silhouette",
                "npmi",
                "nmi",
                "validation_silhouette",
                "validation_nmi",
                "runtime_seconds",
                "ran_successfully",
                "notes",
            ]
        )
    if "ran_successfully" in clustering.columns:
        clustering["ran_successfully"] = (
            clustering["ran_successfully"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False})
            .fillna(False)
        )
    if not clustering.empty:
        clustering = clustering.sort_values(["nmi", "npmi", "silhouette"], ascending=[False, False, False]).reset_index(drop=True)

    clustering_full_path = profile.results_clustering_full_path()
    clustering.to_csv(clustering_full_path, index=False)

    clustering_report = clustering.copy()
    if not clustering_report.empty:
        clustering_report = clustering_report[clustering_report["ran_successfully"] == True]
        clustering_report["Pipeline"] = (
            clustering_report["embedding"].str.upper()
            + " + "
            + clustering_report["clustering"].str.title()
            + " ("
            + clustering_report["subset_name"].astype(str)
            + ")"
        )
        clustering_report = clustering_report.rename(
            columns={
                "embedding": "Embedding",
                "clustering": "Clustering",
                "subset_name": "Subset",
                "best_param": "BestParam",
                "silhouette": "Silhouette",
                "npmi": "NPMI",
                "nmi": "NMI",
                "validation_silhouette": "ValidationSilhouette",
                "validation_nmi": "ValidationNMI",
                "runtime_seconds": "RuntimeSeconds",
                "notes": "Notes",
            }
        )
        clustering_report = clustering_report[
            [
                "Pipeline",
                "Embedding",
                "Clustering",
                "Subset",
                "BestParam",
                "Silhouette",
                "NPMI",
                "NMI",
                "ValidationSilhouette",
                "ValidationNMI",
                "RuntimeSeconds",
                "Notes",
            ]
        ]
    clustering_report_path = profile.results_clustering_report_path()
    clustering_report.to_csv(clustering_report_path, index=False)

    retrieval_path = profile.retrieval_metrics_path()
    if retrieval_path.exists():
        retrieval = pd.read_csv(retrieval_path)
    else:
        retrieval = pd.DataFrame(columns=["K", "MRR", "Recall@K", "query_categories", "weighted_test_docs"])
    retrieval_full_path = profile.results_retrieval_full_path()
    retrieval.to_csv(retrieval_full_path, index=False)
    retrieval_report = retrieval[["K", "MRR", "Recall@K", "query_categories", "weighted_test_docs"]] if not retrieval.empty else retrieval
    retrieval_report_path = profile.results_retrieval_report_path()
    retrieval_report.to_csv(retrieval_report_path, index=False)

    split_summary = load_json(profile.split_summary_path(), default={}) or {}
    split_report = {
        "cleaned_row_count": split_summary.get("cleaned_row_count"),
        "cleaned_category_count": split_summary.get("cleaned_category_count"),
        "subset_summaries": split_summary.get("subset_summaries", {}),
        "notes": [
            "The report notebook reads saved artifacts only; it does not rerun model training or embedding generation.",
            "Subset size None is treated as a full-corpus run in the active profile configuration.",
            "Silhouette and NPMI remain sampled in large runs to keep evaluation tractable.",
        ],
    }
    split_report_path = profile.results_split_report_path()
    write_json(split_report_path, split_report)

    profile_summary_path = profile.results_profile_summary_path()
    write_json(profile_summary_path, _profile_summary(profile, split_report))

    status_rows = []
    for spec in _expected_pipeline_rows(profile):
        if not spec["enabled_by_default"]:
            ran = False
            success = False
            reason = "disabled_by_default"
        elif spec["clustering"] == "retrieval":
            ran = retrieval_path.exists()
            success = ran and not retrieval.empty
            reason = "" if success else "not_run_or_skipped"
        elif not clustering.empty:
            row = clustering[
                (clustering["embedding"] == spec["embedding"])
                & (clustering["clustering"] == spec["clustering"])
                & (clustering["subset_name"] == spec["subset_name"])
            ]
            ran = not row.empty
            success = bool(ran and row["ran_successfully"].fillna(False).iloc[-1])
            reason = "" if success else "not_run_or_skipped"
        else:
            ran = False
            success = False
            reason = "not_run_or_skipped"

        status_rows.append(
            {
                "pipeline_name": spec["pipeline_name"],
                "enabled_by_default": spec["enabled_by_default"],
                "embedding": spec["embedding"],
                "clustering": spec["clustering"],
                "subset_name": spec["subset_name"],
                "ran": ran,
                "success": success,
                "reason_skipped": reason,
            }
        )

    pipeline_status = pd.DataFrame(status_rows)
    pipeline_status_path = profile.pipeline_status_path()
    pipeline_status.to_csv(pipeline_status_path, index=False)

    return {
        "clustering_full": str(clustering_full_path),
        "clustering_report": str(clustering_report_path),
        "retrieval_full": str(retrieval_full_path),
        "retrieval_report": str(retrieval_report_path),
        "split_report": str(split_report_path),
        "profile_summary": str(profile_summary_path),
        "pipeline_status": str(pipeline_status_path),
    }
