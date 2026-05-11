"""Configuration for the benchmark pipeline across local and Colab profiles."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parent.parent


def _resolve_raw_data_path() -> Path:
    env_path = os.environ.get("ARXIV_DATA_PATH")
    if env_path:
        return Path(env_path).expanduser()
    candidates = [
        ROOT_DIR / "arxiv-metadata-oai-snapshot.json",
        Path("/content/arxiv-metadata-oai-snapshot.json"),
        Path("/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json"),
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return ROOT_DIR / "arxiv-metadata-oai-snapshot.json"


def _resolve_artifacts_dir() -> Path:
    env_path = os.environ.get("ARXIV_ARTIFACTS_DIR")
    if env_path:
        return Path(env_path).expanduser()
    return ROOT_DIR / "artifacts" / "modeling"


RAW_DATA_PATH = _resolve_raw_data_path()
EDA_DIR = ROOT_DIR / "artifacts" / "eda_full"
MODELING_DIR = _resolve_artifacts_dir()


def missing_raw_data_message(path: Path) -> str:
    """Return an actionable message for a missing raw arXiv metadata file."""

    return (
        f"Raw arXiv metadata file not found at: {path}\n"
        "Keep arxiv-metadata-oai-snapshot.json out of git and place it on a data filesystem, then either:\n"
        "  export ARXIV_DATA_PATH=/path/to/arxiv-metadata-oai-snapshot.json\n"
        "or pass:\n"
        "  python -m src.run_pipeline --data-path /path/to/arxiv-metadata-oai-snapshot.json\n"
        "On Delta, a typical location is:\n"
        "  $WORK/arxiv/data/arxiv-metadata-oai-snapshot.json"
    )


@dataclass(frozen=True)
class SubsetConfig:
    """Definition of a benchmark subset."""

    name: str
    size: Optional[int]
    min_category_count: int = 25
    min_subset_per_category: int = 5
    use_full_dataset: bool = False


@dataclass(frozen=True)
class ProfileConfig:
    """Execution profile tuned for a specific hardware budget."""

    name: str
    raw_data_path: Path = RAW_DATA_PATH
    eda_dir: Path = EDA_DIR
    artifacts_dir: Path = MODELING_DIR
    seed: int = RANDOM_SEED
    main_tf_idf_subset_size: Optional[int] = 40_000
    main_mpnet_subset_size: Optional[int] = 12_000
    main_bert_subset_size: Optional[int] = 5_000
    hdbscan_subset_size: Optional[int] = 8_000
    agglomerative_subset_size: Optional[int] = 2_500
    tfidf_main_subset_name: str = "tfidf_main"
    mpnet_main_subset_name: str = "mpnet_main"
    bert_main_subset_name: str = "bert_main"
    hdbscan_subset_name: str = "hdbscan_shared"
    agglomerative_subset_name: str = "agglomerative_small"
    tfidf_main_use_full_dataset: bool = False
    mpnet_main_use_full_dataset: bool = False
    bert_main_use_full_dataset: bool = False
    hdbscan_use_full_dataset: bool = False
    agglomerative_use_full_dataset: bool = False
    silhouette_eval_sample_size: int = 5_000
    npmi_sample_size: int = 25_000
    tfidf_max_features: int = 50_000
    tfidf_min_df: int = 5
    save_sparse_tfidf_matrices: bool = True
    svd_components: int = 100
    umap_components: int = 20
    umap_neighbors: Tuple[int, ...] = (15, 30)
    umap_min_dists: Tuple[float, ...] = (0.0, 0.1)
    skip_umap_above_rows: int = 200_000
    kmeans_k_values: Tuple[int, ...] = (50, 100, 172)
    kmeans_mode: str = "auto"
    kmeans_minibatch_threshold: int = 100_000
    kmeans_batch_size: int = 4_096
    kmeans_max_iter: int = 100
    hdbscan_min_cluster_sizes: Tuple[int, ...] = (30, 50, 100)
    agglomerative_k_values: Tuple[int, ...] = (50, 100, 172)
    mpnet_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    bert_model_name: str = "bert-base-uncased"
    mpnet_batch_size: int = 16
    bert_batch_size: int = 8
    bert_max_length: int = 256
    run_tfidf: bool = True
    run_bert: bool = False
    run_mpnet: bool = True
    run_kmeans: bool = True
    run_hdbscan: bool = True
    run_agglomerative: bool = True
    retrieval_backend: str = "sklearn"
    train_fraction: float = 0.7
    val_fraction: float = 0.1
    test_fraction: float = 0.2

    def tfidf_main_runs_full_corpus(self) -> bool:
        return bool(self.tfidf_main_use_full_dataset or self.main_tf_idf_subset_size is None)

    def mpnet_main_runs_full_corpus(self) -> bool:
        return bool(self.mpnet_main_use_full_dataset or self.main_mpnet_subset_size is None)

    def bert_main_runs_full_corpus(self) -> bool:
        return bool(self.bert_main_use_full_dataset or self.main_bert_subset_size is None)

    def hdbscan_runs_full_corpus(self) -> bool:
        return bool(self.hdbscan_use_full_dataset or self.hdbscan_subset_size is None)

    def agglomerative_runs_full_corpus(self) -> bool:
        return bool(self.agglomerative_use_full_dataset or self.agglomerative_subset_size is None)

    def subset_configs(self) -> Dict[str, SubsetConfig]:
        """Return the configured benchmark subsets."""

        tfidf_full = self.tfidf_main_runs_full_corpus()
        mpnet_full = self.mpnet_main_runs_full_corpus()
        bert_full = self.bert_main_runs_full_corpus()
        hdbscan_full = self.hdbscan_runs_full_corpus()
        agglomerative_full = self.agglomerative_runs_full_corpus()

        requested = []
        if self.run_tfidf:
            if not tfidf_full and self.main_tf_idf_subset_size is None:
                raise ValueError("`main_tf_idf_subset_size` must be set when `run_tfidf=True` and not using the full dataset.")
            requested.append(
                SubsetConfig(
                    name=self.tfidf_main_subset_name,
                    size=None if tfidf_full else self.main_tf_idf_subset_size,
                    min_category_count=1 if tfidf_full else 25,
                    min_subset_per_category=1 if tfidf_full else 5,
                    use_full_dataset=tfidf_full,
                )
            )
        if self.run_mpnet:
            if not mpnet_full and self.main_mpnet_subset_size is None:
                raise ValueError("`main_mpnet_subset_size` must be set when `run_mpnet=True` and not using the full dataset.")
            requested.append(
                SubsetConfig(
                    name=self.mpnet_main_subset_name,
                    size=None if mpnet_full else self.main_mpnet_subset_size,
                    min_category_count=1 if mpnet_full else 25,
                    min_subset_per_category=1 if mpnet_full else 5,
                    use_full_dataset=mpnet_full,
                )
            )
        if self.run_bert:
            if not bert_full and self.main_bert_subset_size is None:
                raise ValueError("`main_bert_subset_size` must be set when `run_bert=True` and not using the full dataset.")
            requested.append(
                SubsetConfig(
                    name=self.bert_main_subset_name,
                    size=None if bert_full else self.main_bert_subset_size,
                    min_category_count=1 if bert_full else 25,
                    min_subset_per_category=1 if bert_full else 5,
                    use_full_dataset=bert_full,
                )
            )
        if self.run_hdbscan:
            if not hdbscan_full and self.hdbscan_subset_size is None:
                raise ValueError("`hdbscan_subset_size` must be set when `run_hdbscan=True` and not using the full dataset.")
            requested.append(
                SubsetConfig(
                    name=self.hdbscan_subset_name,
                    size=None if hdbscan_full else self.hdbscan_subset_size,
                    min_category_count=1 if hdbscan_full else 25,
                    min_subset_per_category=1 if hdbscan_full else 5,
                    use_full_dataset=hdbscan_full,
                )
            )
        if self.run_agglomerative:
            if not agglomerative_full and self.agglomerative_subset_size is None:
                raise ValueError("`agglomerative_subset_size` must be set when `run_agglomerative=True` and not using the full dataset.")
            requested.append(
                SubsetConfig(
                    name=self.agglomerative_subset_name,
                    size=None if agglomerative_full else self.agglomerative_subset_size,
                    min_category_count=1 if agglomerative_full else 25,
                    min_subset_per_category=1 if agglomerative_full else 5,
                    use_full_dataset=agglomerative_full,
                )
            )

        unique: Dict[str, SubsetConfig] = {}
        for spec in requested:
            existing = unique.get(spec.name)
            if existing is None:
                unique[spec.name] = spec
                continue

            if (
                existing.size != spec.size
                or existing.min_category_count != spec.min_category_count
                or existing.min_subset_per_category != spec.min_subset_per_category
                or existing.use_full_dataset != spec.use_full_dataset
            ):
                raise ValueError(f"Conflicting subset definitions for shared subset name '{spec.name}'.")
        return unique

    def clean_dir(self) -> Path:
        return self.artifacts_dir / "clean"

    def splits_dir(self) -> Path:
        return self.artifacts_dir / "splits"

    def embeddings_dir(self) -> Path:
        return self.artifacts_dir / "embeddings"

    def reduction_dir(self) -> Path:
        return self.artifacts_dir / "reduction"

    def clustering_dir(self) -> Path:
        return self.artifacts_dir / "clustering"

    def metrics_dir(self) -> Path:
        return self.artifacts_dir / "metrics"

    def retrieval_dir(self) -> Path:
        return self.artifacts_dir / "retrieval"

    def results_dir(self) -> Path:
        return self.artifacts_dir / "results"

    def clean_dataset_path(self) -> Path:
        return self.clean_dir() / "clean_papers.parquet"

    def cleaning_manifest_path(self) -> Path:
        return self.clean_dir() / "cleaning_manifest.json"

    def split_summary_path(self) -> Path:
        return self.splits_dir() / "split_summary.json"

    def clustering_metrics_path(self) -> Path:
        return self.metrics_dir() / "clustering_metrics.csv"

    def retrieval_metrics_path(self) -> Path:
        return self.retrieval_dir() / "retrieval_metrics.csv"

    def pipeline_status_path(self) -> Path:
        return self.results_dir() / f"{self.name}_pipeline_status.csv"

    def results_clustering_full_path(self) -> Path:
        return self.results_dir() / f"{self.name}_clustering_results_full.csv"

    def results_clustering_report_path(self) -> Path:
        return self.results_dir() / f"{self.name}_clustering_results_report.csv"

    def results_retrieval_full_path(self) -> Path:
        return self.results_dir() / f"{self.name}_retrieval_results_full.csv"

    def results_retrieval_report_path(self) -> Path:
        return self.results_dir() / f"{self.name}_retrieval_results_report.csv"

    def results_split_report_path(self) -> Path:
        return self.results_dir() / f"{self.name}_split_report.json"

    def results_profile_summary_path(self) -> Path:
        return self.results_dir() / f"{self.name}_profile_summary.json"

    def to_jsonable(self) -> Dict[str, Any]:
        """Convert the profile to a JSON-serializable dictionary."""

        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data


def get_profile(name: str = "local_profile", run_bert_override: Optional[bool] = None) -> ProfileConfig:
    """Build the requested execution profile."""

    if name == "local_profile":
        profile = ProfileConfig(name="local_profile")
    elif name == "full_profile":
        profile = ProfileConfig(
            name="full_profile",
            main_tf_idf_subset_size=120_000,
            main_mpnet_subset_size=30_000,
            main_bert_subset_size=12_000,
            hdbscan_subset_size=20_000,
            agglomerative_subset_size=5_000,
            silhouette_eval_sample_size=10_000,
            svd_components=150,
            umap_components=35,
            umap_neighbors=(15, 30, 50),
            umap_min_dists=(0.0, 0.1),
            mpnet_batch_size=32,
            bert_batch_size=16,
            bert_max_length=320,
            run_bert=True,
            retrieval_backend="faiss",
        )
    elif name == "colab_a100_profile":
        profile = ProfileConfig(
            name="colab_a100_profile",
            main_tf_idf_subset_size=200_000,
            main_mpnet_subset_size=80_000,
            main_bert_subset_size=30_000,
            hdbscan_subset_size=30_000,
            agglomerative_subset_size=10_000,
            silhouette_eval_sample_size=15_000,
            tfidf_max_features=100_000,
            svd_components=200,
            umap_components=50,
            umap_neighbors=(15, 30, 50),
            umap_min_dists=(0.0, 0.05, 0.1),
            mpnet_batch_size=64,
            bert_batch_size=32,
            bert_max_length=384,
            run_bert=True,
            retrieval_backend="faiss",
        )
    elif name == "colab_h100_full_profile":
        profile = ProfileConfig(
            name="colab_h100_full_profile",
            main_tf_idf_subset_size=None,
            main_mpnet_subset_size=None,
            main_bert_subset_size=None,
            hdbscan_subset_size=None,
            agglomerative_subset_size=None,
            tfidf_main_subset_name="full_corpus",
            mpnet_main_subset_name="full_corpus",
            bert_main_subset_name="full_corpus",
            hdbscan_subset_name="full_corpus",
            agglomerative_subset_name="full_corpus",
            tfidf_main_use_full_dataset=True,
            mpnet_main_use_full_dataset=True,
            bert_main_use_full_dataset=True,
            hdbscan_use_full_dataset=True,
            agglomerative_use_full_dataset=True,
            silhouette_eval_sample_size=20_000,
            npmi_sample_size=50_000,
            tfidf_max_features=250_000,
            tfidf_min_df=10,
            save_sparse_tfidf_matrices=False,
            svd_components=256,
            umap_components=50,
            umap_neighbors=(15, 30, 50),
            umap_min_dists=(0.0, 0.05, 0.1),
            skip_umap_above_rows=120_000,
            kmeans_mode="auto",
            kmeans_minibatch_threshold=100_000,
            kmeans_batch_size=8_192,
            kmeans_max_iter=200,
            bert_model_name="allenai/scibert_scivocab_uncased",
            bert_batch_size=64,
            bert_max_length=384,
            run_mpnet=True,
            run_bert=True,
            run_hdbscan=True,
            run_agglomerative=True,
            retrieval_backend="faiss",
        )
    elif name == "delta_retrieval_full_profile":
        profile = ProfileConfig(
            name="delta_retrieval_full_profile",
            main_tf_idf_subset_size=None,
            main_mpnet_subset_size=None,
            main_bert_subset_size=None,
            hdbscan_subset_size=None,
            agglomerative_subset_size=None,
            tfidf_main_subset_name="full_corpus",
            mpnet_main_subset_name="full_corpus",
            bert_main_subset_name="full_corpus",
            hdbscan_subset_name="full_corpus",
            agglomerative_subset_name="full_corpus",
            tfidf_main_use_full_dataset=True,
            mpnet_main_use_full_dataset=True,
            bert_main_use_full_dataset=True,
            hdbscan_use_full_dataset=True,
            agglomerative_use_full_dataset=True,
            skip_umap_above_rows=120_000,
            mpnet_batch_size=64,
            run_tfidf=False,
            run_mpnet=True,
            run_bert=False,
            run_kmeans=False,
            run_hdbscan=False,
            run_agglomerative=False,
            retrieval_backend="faiss",
        )
    elif name == "delta_retrieval_sample_profile":
        profile = ProfileConfig(
            name="delta_retrieval_sample_profile",
            main_tf_idf_subset_size=None,
            main_mpnet_subset_size=120_000,
            main_bert_subset_size=None,
            hdbscan_subset_size=None,
            agglomerative_subset_size=None,
            tfidf_main_subset_name="delta_retrieval_sample",
            mpnet_main_subset_name="delta_retrieval_sample",
            bert_main_subset_name="delta_retrieval_sample",
            hdbscan_subset_name="delta_retrieval_sample",
            agglomerative_subset_name="delta_retrieval_sample",
            mpnet_batch_size=64,
            skip_umap_above_rows=50_000,
            run_tfidf=False,
            run_mpnet=True,
            run_bert=False,
            run_kmeans=False,
            run_hdbscan=False,
            run_agglomerative=False,
            retrieval_backend="faiss",
        )
    else:
        raise ValueError(f"Unknown profile: {name}")

    if run_bert_override is not None:
        profile = replace(profile, run_bert=run_bert_override)

    return profile


def ensure_artifact_dirs(profile: ProfileConfig) -> None:
    """Create all pipeline artifact directories."""

    for path in (
        profile.artifacts_dir,
        profile.clean_dir(),
        profile.splits_dir(),
        profile.embeddings_dir(),
        profile.reduction_dir(),
        profile.clustering_dir(),
        profile.metrics_dir(),
        profile.retrieval_dir(),
        profile.results_dir(),
    ):
        path.mkdir(parents=True, exist_ok=True)
