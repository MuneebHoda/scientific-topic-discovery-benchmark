"""Configuration for the local-first benchmark pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Tuple


RANDOM_SEED = 42
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = ROOT_DIR / "arxiv-metadata-oai-snapshot.json"
EDA_DIR = ROOT_DIR / "artifacts" / "eda"
MODELING_DIR = ROOT_DIR / "artifacts" / "modeling"


@dataclass(frozen=True)
class SubsetConfig:
    """Definition of a benchmark subset."""

    name: str
    size: int
    min_category_count: int = 25
    min_subset_per_category: int = 5


@dataclass(frozen=True)
class ProfileConfig:
    """Execution profile tuned for a specific hardware budget."""

    name: str
    raw_data_path: Path = RAW_DATA_PATH
    eda_dir: Path = EDA_DIR
    artifacts_dir: Path = MODELING_DIR
    seed: int = RANDOM_SEED
    main_tf_idf_subset_size: int = 40_000
    main_mpnet_subset_size: int = 12_000
    main_bert_subset_size: int = 5_000
    hdbscan_subset_size: int = 8_000
    agglomerative_subset_size: int = 2_500
    silhouette_eval_sample_size: int = 5_000
    tfidf_max_features: int = 50_000
    tfidf_min_df: int = 5
    svd_components: int = 100
    umap_components: int = 20
    umap_neighbors: Tuple[int, ...] = (15, 30)
    umap_min_dists: Tuple[float, ...] = (0.0, 0.1)
    kmeans_k_values: Tuple[int, ...] = (50, 100, 172)
    hdbscan_min_cluster_sizes: Tuple[int, ...] = (30, 50, 100)
    agglomerative_k_values: Tuple[int, ...] = (50, 100, 172)
    mpnet_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    bert_model_name: str = "bert-base-uncased"
    mpnet_batch_size: int = 16
    bert_batch_size: int = 8
    bert_max_length: int = 256
    run_bert: bool = False
    run_mpnet: bool = True
    run_hdbscan: bool = True
    run_agglomerative: bool = True
    retrieval_backend: str = "sklearn"
    train_fraction: float = 0.7
    val_fraction: float = 0.1
    test_fraction: float = 0.2

    def subset_configs(self) -> Dict[str, SubsetConfig]:
        """Return the configured benchmark subsets."""

        return {
            "tfidf_main": SubsetConfig("tfidf_main", self.main_tf_idf_subset_size),
            "mpnet_main": SubsetConfig("mpnet_main", self.main_mpnet_subset_size),
            "bert_main": SubsetConfig("bert_main", self.main_bert_subset_size),
            "hdbscan_shared": SubsetConfig("hdbscan_shared", self.hdbscan_subset_size),
            "agglomerative_small": SubsetConfig(
                "agglomerative_small",
                self.agglomerative_subset_size,
            ),
        }

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
        return self.results_dir() / "pipeline_status.csv"

    def to_jsonable(self) -> Dict[str, Any]:
        """Convert the profile to a JSON-serializable dictionary."""

        data = asdict(self)
        for key, value in list(data.items()):
            if isinstance(value, Path):
                data[key] = str(value)
            elif isinstance(value, tuple):
                data[key] = list(value)
        return data


def get_profile(name: str = "local_profile", run_bert_override: bool | None = None) -> ProfileConfig:
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
