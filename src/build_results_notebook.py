"""Generate a polished benchmark results notebook from saved artifacts."""

from __future__ import annotations

import json
from textwrap import dedent

import nbformat as nbf

from src.config import ProfileConfig


def build_results_notebook(profile: ProfileConfig) -> str:
    """Create a presentation-quality benchmark summary notebook."""

    output_path = profile.results_dir() / f"{profile.name}_Benchmark_Summary.ipynb"
    profile.results_dir().mkdir(parents=True, exist_ok=True)

    profile_payload = json.dumps(profile.to_jsonable(), indent=2)

    nb = nbf.v4.new_notebook()
    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))

    def code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

    md(
        f"""
        # Benchmark Results Review

        This notebook summarizes saved artifacts for the **`{profile.name}`** profile. It is designed
        as a report-facing review notebook rather than a training notebook: it loads cached outputs,
        checks execution coverage, compares pipelines, and interprets the results in the context of
        scientific topic discovery on the cleaned arXiv-style corpus.
        """
    )

    md(
        """
        ## Review Goals

        This notebook focuses on four questions:

        1. Did the configured profile actually run the intended pipelines and data scope?
        2. Which embedding and clustering combinations perform best under different metrics?
        3. Do the results suggest fine-grained category recovery or broader macro-topic structure?
        4. Which conclusions are strong enough to carry into the final report?
        """
    )

    code(
        f"""
        from pathlib import Path
        import json
        import math
        import re

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display

        PROFILE = json.loads({profile_payload!r})
        ARTIFACTS_DIR = Path(PROFILE["artifacts_dir"])
        RESULTS_DIR = ARTIFACTS_DIR / "results"
        CLUSTERING_DIR = ARTIFACTS_DIR / "clustering"
        EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
        RETRIEVAL_DIR = ARTIFACTS_DIR / "retrieval"
        METRICS_DIR = ARTIFACTS_DIR / "metrics"
        PROFILE_NAME = PROFILE["name"]

        sns.set_theme(style="whitegrid", context="talk")
        pd.set_option("display.max_columns", 100)
        pd.set_option("display.max_colwidth", 120)
        pd.options.display.float_format = lambda value: f"{{value:,.4f}}"


        def load_json(path: Path, default=None):
            if not path.exists():
                return default
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)


        def read_csv(path: Path) -> pd.DataFrame:
            return pd.read_csv(path) if path.exists() else pd.DataFrame()


        def pipeline_label(embedding: str, clustering: str, subset_name: str) -> str:
            return f"{{embedding.upper()}} + {{clustering.title()}}\\n({{subset_name}})"


        def search_param_label(row: pd.Series) -> str:
            for column in ("k", "min_cluster_size", "n_neighbors", "min_dist"):
                if column in row and pd.notna(row[column]):
                    return f"{{column}}={{row[column]}}"
            return "default"


        def collect_detail_metrics() -> pd.DataFrame:
            records = []
            details_dir = METRICS_DIR / "details"
            if not details_dir.exists():
                return pd.DataFrame()
            for path in sorted(details_dir.glob("*.json")):
                payload = load_json(path, default={{}}) or {{}}
                payload["detail_file"] = path.name
                records.append(payload)
            return pd.DataFrame(records)


        def collect_search_results() -> pd.DataFrame:
            records = []
            if not CLUSTERING_DIR.exists():
                return pd.DataFrame()
            for path in sorted(CLUSTERING_DIR.glob("*/*/search_results.csv")):
                stage_name = path.parent.parent.name
                if "_" not in stage_name:
                    continue
                embedding, clustering = stage_name.split("_", 1)
                frame = pd.read_csv(path)
                if frame.empty:
                    continue
                frame["embedding"] = embedding
                frame["clustering"] = clustering
                frame["subset_name"] = path.parent.name
                frame["pipeline"] = frame.apply(
                    lambda row: pipeline_label(row["embedding"], row["clustering"], row["subset_name"]),
                    axis=1,
                )
                frame["parameter"] = frame.apply(search_param_label, axis=1)
                records.append(frame)
            return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


        def collect_stage_metadata() -> pd.DataFrame:
            records = []

            for path in sorted(EMBEDDINGS_DIR.glob("*/*/metadata.json")):
                payload = load_json(path, default={{}}) or {{}}
                subset_name = path.parent.name
                embedding_name = path.parent.parent.name
                best_umap_config = payload.get("best_umap_config", {{}}) or {{}}
                records.append(
                    {{
                        "stage_type": "embedding",
                        "stage_name": embedding_name,
                        "subset_name": subset_name,
                        "runtime_seconds": payload.get("runtime_seconds"),
                        "device": payload.get("device", ""),
                        "model_name": payload.get("model_name", ""),
                        "reducer": best_umap_config.get("reducer", "umap"),
                        "reducer_reason": best_umap_config.get("reason", ""),
                        "notes": payload.get("notes", ""),
                    }}
                )

            for path in sorted(CLUSTERING_DIR.glob("*/*/metadata.json")):
                payload = load_json(path, default={{}}) or {{}}
                subset_name = path.parent.name
                stage_name = path.parent.parent.name
                records.append(
                    {{
                        "stage_type": "clustering",
                        "stage_name": stage_name,
                        "subset_name": subset_name,
                        "runtime_seconds": payload.get("runtime_seconds"),
                        "device": "",
                        "model_name": "",
                        "reducer": "",
                        "reducer_reason": "",
                        "notes": payload.get("notes", ""),
                    }}
                )

            retrieval_meta = load_json(RETRIEVAL_DIR / "index_metadata.json", default={{}})
            if retrieval_meta:
                records.append(
                    {{
                        "stage_type": "retrieval",
                        "stage_name": "mpnet_retrieval",
                        "subset_name": retrieval_meta.get("subset_name", ""),
                        "runtime_seconds": retrieval_meta.get("runtime_seconds"),
                        "device": "",
                        "model_name": "",
                        "reducer": "",
                        "reducer_reason": "",
                        "notes": retrieval_meta.get("note", ""),
                    }}
                )

            return pd.DataFrame(records)


        profile_summary = load_json(RESULTS_DIR / f"{{PROFILE_NAME}}_profile_summary.json", default={{}}) or {{}}
        split_report = load_json(RESULTS_DIR / f"{{PROFILE_NAME}}_split_report.json", default={{}}) or {{}}
        status = read_csv(RESULTS_DIR / f"{{PROFILE_NAME}}_pipeline_status.csv")
        clustering = read_csv(RESULTS_DIR / f"{{PROFILE_NAME}}_clustering_results_full.csv")
        retrieval = read_csv(RESULTS_DIR / f"{{PROFILE_NAME}}_retrieval_results_full.csv")
        retrieval_examples = load_json(RETRIEVAL_DIR / "retrieval_examples.json", default=[]) or []

        detail_metrics = collect_detail_metrics()
        search_results = collect_search_results()
        stage_metadata = collect_stage_metadata()

        if not status.empty:
            for column in ("enabled_by_default", "ran", "success"):
                if column in status.columns:
                    status[column] = (
                        status[column]
                        .astype(str)
                        .str.lower()
                        .map({{"true": True, "false": False}})
                        .fillna(False)
                    )

        if not clustering.empty and "ran_successfully" in clustering.columns:
            clustering["ran_successfully"] = (
                clustering["ran_successfully"]
                .astype(str)
                .str.lower()
                .map({{"true": True, "false": False}})
                .fillna(False)
            )
            clustering = clustering[clustering["ran_successfully"] == True].reset_index(drop=True)

        if not clustering.empty:
            clustering["pipeline"] = clustering.apply(
                lambda row: pipeline_label(row["embedding"], row["clustering"], row["subset_name"]),
                axis=1,
            )

        if not detail_metrics.empty:
            detail_metrics = detail_metrics.drop_duplicates(
                subset=["embedding", "clustering", "subset_name"],
                keep="last",
            )

        if not clustering.empty and not detail_metrics.empty:
            clustering = clustering.merge(
                detail_metrics[
                    [
                        "embedding",
                        "clustering",
                        "subset_name",
                        "validation_label_count",
                        "test_label_count",
                    ]
                ],
                on=["embedding", "clustering", "subset_name"],
                how="left",
            )

        subset_summaries = split_report.get("subset_summaries", {{}})
        reference_category_map = {{
            subset_name: details.get("category_count")
            for subset_name, details in subset_summaries.items()
        }}
        actual_size_map = {{
            subset_name: details.get("actual_size")
            for subset_name, details in subset_summaries.items()
        }}

        if not clustering.empty:
            clustering["reference_category_count"] = clustering["subset_name"].map(reference_category_map)
            clustering["subset_actual_size"] = clustering["subset_name"].map(actual_size_map)
            if "test_label_count" in clustering.columns:
                clustering["cluster_to_category_ratio"] = (
                    clustering["test_label_count"] / clustering["reference_category_count"]
                )
            clustering = clustering.sort_values(["nmi", "npmi", "silhouette"], ascending=[False, False, False]).reset_index(drop=True)

        print("Profile:", PROFILE["name"])
        print("Artifacts directory:", ARTIFACTS_DIR)
        print("Results directory:", RESULTS_DIR)
        """
    )

    md(
        """
        ## 1. Execution Scope

        We start by verifying the intended run scope. For this project, that means checking whether
        the active profile points the modeling stages at deterministic subsets or at the full cleaned
        corpus, and whether the evaluation metrics still rely on bounded sampling.
        """
    )

    code(
        """
        subset_rows = []
        for subset_name, details in subset_summaries.items():
            subset_config = profile_summary.get("subset_configs", {}).get(subset_name, {})
            subset_rows.append(
                {
                    "Subset": subset_name,
                    "UseFullDataset": subset_config.get("use_full_dataset"),
                    "ConfiguredSize": subset_config.get("size"),
                    "ActualRows": details.get("actual_size"),
                    "ReferenceCategories": details.get("category_count"),
                    "TrainRows": details.get("train_size"),
                    "ValRows": details.get("val_size"),
                    "TestRows": details.get("test_size"),
                    "Source": details.get("source"),
                }
            )

        scope_df = pd.DataFrame(subset_rows)
        display(scope_df if not scope_df.empty else pd.DataFrame([{"Message": "No split summary available yet."}]))

        sampling_df = pd.DataFrame(
            [
                {
                    "Metric": "Silhouette",
                    "EvaluationScope": "Sampled",
                    "SampleSize": profile_summary.get("evaluation_sampling", {}).get("silhouette_eval_sample_size"),
                },
                {
                    "Metric": "NPMI",
                    "EvaluationScope": "Sampled",
                    "SampleSize": profile_summary.get("evaluation_sampling", {}).get("npmi_sample_size"),
                },
            ]
        )
        display(sampling_df)

        if not scope_df.empty:
            split_plot = scope_df.melt(
                id_vars=["Subset"],
                value_vars=["TrainRows", "ValRows", "TestRows"],
                var_name="Split",
                value_name="Rows",
            )
            plt.figure(figsize=(12, 5))
            sns.barplot(data=split_plot, x="Subset", y="Rows", hue="Split")
            plt.title("Split Sizes by Benchmark Subset")
            plt.xlabel("Subset")
            plt.ylabel("Rows")
            plt.xticks(rotation=20, ha="right")
            plt.tight_layout()
            plt.show()
        """
    )

    md(
        """
        **Interpretation.** If `UseFullDataset` is `True`, the corresponding pipeline stages are fed by
        the full cleaned corpus rather than a sampled subset. The two remaining sampled components in
        large runs are silhouette and NPMI evaluation, which are intentionally bounded to keep
        full-corpus reporting tractable.
        """
    )

    md(
        """
        ## 2. Pipeline Coverage

        Before interpreting any metric, we need to check what actually completed. This section makes
        it easy to distinguish between pipelines that were disabled by configuration, pipelines that
        were attempted but did not finish, and pipelines that completed successfully.
        """
    )

    code(
        """
        display(status if not status.empty else pd.DataFrame([{"Message": "No pipeline status table found."}]))

        if not status.empty:
            outcome = status.copy()
            outcome["Outcome"] = np.where(
                outcome["success"],
                "Succeeded",
                np.where(outcome["enabled_by_default"], "Missing/Failed", "Disabled"),
            )
            plt.figure(figsize=(10, 4))
            sns.countplot(data=outcome, x="Outcome", order=["Succeeded", "Missing/Failed", "Disabled"])
            plt.title("Pipeline Coverage Summary")
            plt.xlabel("Pipeline Outcome")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()
        """
    )

    md(
        """
        **Interpretation.** For a serious benchmark, missing runs matter as much as successful runs.
        If a full-corpus profile leaves `HDBSCAN` or `Agglomerative` unfinished, that is itself an
        important scalability result and should be documented rather than hidden.
        """
    )

    md(
        """
        ## 3. Clustering Results Overview

        We now compare completed clustering pipelines across the three main metrics used in this
        project:

        - **NMI** for alignment with held-out category labels,
        - **NPMI** for topic coherence,
        - **Silhouette** for geometric separation in embedding space.

        No single metric is sufficient on its own, so the goal here is to compare trade-offs rather
        than crown a winner from one number alone.
        """
    )

    code(
        """
        clustering_view_columns = [
            "pipeline",
            "embedding",
            "clustering",
            "subset_name",
            "best_param",
            "nmi",
            "npmi",
            "silhouette",
            "validation_nmi",
            "validation_silhouette",
            "test_label_count",
            "reference_category_count",
            "cluster_to_category_ratio",
            "runtime_seconds",
            "notes",
        ]
        available_columns = [column for column in clustering_view_columns if column in clustering.columns]
        display(clustering[available_columns] if not clustering.empty else pd.DataFrame([{"Message": "No completed clustering metrics found."}]))

        if not clustering.empty:
            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            ranking = clustering.sort_values("nmi", ascending=False)
            sns.barplot(data=ranking, x="pipeline", y="nmi", hue="embedding", dodge=False, ax=axes[0])
            axes[0].set_title("Test NMI by Pipeline")
            axes[0].set_xlabel("Pipeline")
            axes[0].set_ylabel("NMI")
            axes[0].tick_params(axis="x", rotation=25)

            sns.scatterplot(
                data=clustering,
                x="nmi",
                y="npmi",
                hue="embedding",
                style="clustering",
                size="silhouette",
                sizes=(80, 280),
                ax=axes[1],
            )
            axes[1].set_title("Label Alignment vs Topic Coherence")
            axes[1].set_xlabel("NMI")
            axes[1].set_ylabel("NPMI")

            plt.tight_layout()
            plt.show()
        """
    )

    md(
        """
        **Interpretation.** Strong topic-discovery pipelines should ideally score well on both NMI and
        NPMI. If a pipeline scores well on one but not the other, that usually means the discovered
        clusters are either label-aligned but semantically noisy, or semantically coherent but too
        coarse or too idiosyncratic relative to the reference categories.
        """
    )

    md(
        """
        ## 4. Metric Trade-offs and Generalization

        This section digs deeper into model behavior. We compare metric leaders, inspect
        validation-to-test gaps, and look at the number of discovered clusters relative to the number
        of reference categories used for evaluation.
        """
    )

    code(
        """
        if not clustering.empty:
            metric_leaders = []
            for metric in ("nmi", "npmi", "silhouette"):
                leader = clustering.sort_values(metric, ascending=False).iloc[0]
                metric_leaders.append(
                    {
                        "Metric": metric.upper(),
                        "BestPipeline": leader["pipeline"],
                        "Value": leader[metric],
                        "BestParam": leader.get("best_param", ""),
                    }
                )
            display(pd.DataFrame(metric_leaders))

            gap_df = clustering.copy()
            gap_df["nmi_gap"] = gap_df["validation_nmi"] - gap_df["nmi"]
            gap_df["silhouette_gap"] = gap_df["validation_silhouette"] - gap_df["silhouette"]
            gap_columns = [
                "pipeline",
                "nmi",
                "validation_nmi",
                "nmi_gap",
                "silhouette",
                "validation_silhouette",
                "silhouette_gap",
            ]
            display(gap_df[gap_columns].sort_values("nmi_gap", key=lambda series: series.abs(), ascending=False))

            fig, axes = plt.subplots(1, 2, figsize=(16, 5))

            comparison = gap_df.melt(
                id_vars=["pipeline"],
                value_vars=["validation_nmi", "nmi"],
                var_name="MetricSplit",
                value_name="Value",
            )
            sns.barplot(data=comparison, x="pipeline", y="Value", hue="MetricSplit", ax=axes[0])
            axes[0].set_title("Validation vs Test NMI")
            axes[0].set_xlabel("Pipeline")
            axes[0].set_ylabel("NMI")
            axes[0].tick_params(axis="x", rotation=25)

            if "cluster_to_category_ratio" in gap_df.columns:
                sns.barplot(data=gap_df, x="pipeline", y="cluster_to_category_ratio", hue="embedding", dodge=False, ax=axes[1])
                axes[1].axhline(1.0, linestyle="--", color="black", linewidth=1)
                axes[1].set_title("Recovered Cluster Count Relative to Reference Categories")
                axes[1].set_xlabel("Pipeline")
                axes[1].set_ylabel("Test Clusters / Reference Categories")
                axes[1].tick_params(axis="x", rotation=25)
            else:
                axes[1].axis("off")

            plt.tight_layout()
            plt.show()
        """
    )

    md(
        """
        **Interpretation.** If the best-performing configuration uses substantially fewer discovered
        clusters than the number of reference categories, that suggests the unsupervised structure is
        favoring broader macro-topics rather than the full label granularity. For this project, that
        is a meaningful finding rather than a failure: topic discovery does not have to reproduce the
        entire arXiv taxonomy one-for-one.
        """
    )

    md(
        """
        ## 5. Hyperparameter Search Review

        Good benchmark reporting should show not only the winning configuration, but also whether the
        choice was stable. Here we review the saved search grids and compare the best setting to the
        runner-up for each completed pipeline.
        """
    )

    code(
        """
        if not search_results.empty:
            search_summary_rows = []
            for (embedding, clustering_name, subset_name), frame in search_results.groupby(["embedding", "clustering", "subset_name"]):
                ranked = frame.sort_values("validation_silhouette", ascending=False).reset_index(drop=True)
                best = ranked.iloc[0]
                runner_up_score = ranked.iloc[1]["validation_silhouette"] if len(ranked) > 1 else np.nan
                search_summary_rows.append(
                    {
                        "Pipeline": pipeline_label(embedding, clustering_name, subset_name),
                        "BestParameter": best["parameter"],
                        "BestValidationSilhouette": best["validation_silhouette"],
                        "RunnerUpValidationSilhouette": runner_up_score,
                        "MarginVsRunnerUp": best["validation_silhouette"] - runner_up_score if pd.notna(runner_up_score) else np.nan,
                    }
                )
            search_summary = pd.DataFrame(search_summary_rows).sort_values("BestValidationSilhouette", ascending=False)
            display(search_summary)

            plt.figure(figsize=(12, 5))
            sns.barplot(data=search_results, x="pipeline", y="validation_silhouette", hue="parameter")
            plt.title("Saved Search Grid Scores")
            plt.xlabel("Pipeline")
            plt.ylabel("Validation Silhouette")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.show()
        else:
            display(pd.DataFrame([{"Message": "No search result grids were found."}]))
        """
    )

    md(
        """
        **Interpretation.** Small margins between the top two settings indicate a relatively stable
        hyperparameter region; large margins imply that the pipeline is much more sensitive to the
        chosen clustering resolution.
        """
    )

    md(
        """
        ## 6. Runtime and Scalability Notes

        This section turns the cached stage metadata into an operational view of the benchmark. It is
        especially important for Colab/H100 runs, where some stages are GPU-friendly while others are
        still dominated by CPU time, memory pressure, or algorithmic scaling.
        """
    )

    code(
        """
        display(stage_metadata.sort_values(["stage_type", "runtime_seconds"], ascending=[True, False]) if not stage_metadata.empty else pd.DataFrame([{"Message": "No stage metadata found."}]))

        if not stage_metadata.empty and stage_metadata["runtime_seconds"].notna().any():
            runtime_view = stage_metadata.dropna(subset=["runtime_seconds"]).copy()
            runtime_view["Stage"] = runtime_view["stage_name"] + "\\n(" + runtime_view["subset_name"].astype(str) + ")"
            plt.figure(figsize=(12, 5))
            sns.barplot(data=runtime_view, x="Stage", y="runtime_seconds", hue="stage_type")
            plt.title("Saved Runtime by Stage")
            plt.xlabel("Stage")
            plt.ylabel("Runtime (seconds)")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.show()

        if not stage_metadata.empty:
            identity_rows = stage_metadata[(stage_metadata["stage_type"] == "embedding") & (stage_metadata["reducer"] == "identity")]
            if not identity_rows.empty:
                display(identity_rows[["stage_name", "subset_name", "reducer", "reducer_reason"]])
        """
    )

    md(
        """
        **Interpretation.** For very large runs, identity reduction or other scale-oriented shortcuts
        are not mistakes; they are part of the experimental design. They should be documented clearly
        so the final report distinguishes between modeling choices made for quality and choices made
        for tractability.
        """
    )

    md(
        """
        ## 7. Retrieval Evaluation

        Retrieval is evaluated here through category-centroid pseudo-queries against the MPNET index.
        These results do not replace paper-level retrieval testing, but they do provide a useful
        signal about whether the dense space preserves label-local neighborhoods.
        """
    )

    code(
        """
        display(retrieval if not retrieval.empty else pd.DataFrame([{"Message": "No retrieval metrics found."}]))

        if not retrieval.empty:
            plt.figure(figsize=(8, 5))
            plt.plot(retrieval["K"], retrieval["Recall@K"], marker="o", label="Recall@K")
            plt.plot(retrieval["K"], retrieval["MRR"], marker="o", label="MRR")
            plt.title("Dense Retrieval Metrics")
            plt.xlabel("K")
            plt.ylabel("Score")
            plt.xticks(retrieval["K"])
            plt.ylim(0, min(1.05, max(1.0, retrieval[["Recall@K", "MRR"]].to_numpy().max() + 0.05)))
            plt.legend()
            plt.tight_layout()
            plt.show()

        if retrieval_examples:
            examples_df = pd.DataFrame(retrieval_examples)[:5]
            display(examples_df)
        """
    )

    md(
        """
        **Interpretation.** High centroid-based retrieval scores suggest that papers from the same
        category tend to remain close in the dense embedding space. That is encouraging for downstream
        search and exploration, but it should still be treated as a coarse retrieval proxy rather
        than a substitute for query-level relevance evaluation.
        """
    )

    md(
        """
        ## 8. Key Findings and Recommendations

        The final section below is written to be close to report-ready. It summarizes the strongest
        empirical findings from the saved artifacts, while also calling out scope limits that should
        be mentioned explicitly in the project write-up.
        """
    )

    code(
        """
        findings = []

        full_corpus_subsets = profile_summary.get("full_corpus_subsets", [])
        if full_corpus_subsets:
            findings.append(
                f"- The active profile routes the following subsets to the full cleaned corpus: {', '.join(full_corpus_subsets)}."
            )

        sampling = profile_summary.get("evaluation_sampling", {})
        if sampling:
            sil_sample = sampling.get("silhouette_eval_sample_size")
            npmi_sample = sampling.get("npmi_sample_size")
            sil_text = f"{int(sil_sample):,}" if sil_sample is not None else "N/A"
            npmi_text = f"{int(npmi_sample):,}" if npmi_sample is not None else "N/A"
            findings.append(
                f"- Silhouette and NPMI remain sampled for reporting at {sil_text} and {npmi_text} documents respectively."
            )
        else:
            findings.append("- Evaluation sampling metadata was not found.")

        if not clustering.empty:
            best_nmi = clustering.sort_values("nmi", ascending=False).iloc[0]
            best_npmi = clustering.sort_values("npmi", ascending=False).iloc[0]
            best_silhouette = clustering.sort_values("silhouette", ascending=False).iloc[0]
            findings.append(
                f"- The strongest label-alignment result is **{best_nmi['pipeline']}** with test NMI = {best_nmi['nmi']:.4f}."
            )
            findings.append(
                f"- The strongest topic-coherence result is **{best_npmi['pipeline']}** with NPMI = {best_npmi['npmi']:.4f}."
            )
            findings.append(
                f"- The strongest geometric separation result is **{best_silhouette['pipeline']}** with silhouette = {best_silhouette['silhouette']:.4f}."
            )

            if "cluster_to_category_ratio" in clustering.columns and clustering["cluster_to_category_ratio"].notna().any():
                coarse = clustering.sort_values("nmi", ascending=False).iloc[0]["cluster_to_category_ratio"]
                findings.append(
                    f"- The best-NMI pipeline recovers about {coarse:.2f} discovered clusters per reference category, which helps indicate whether the benchmark is finding macro-topics or near-label-level structure."
                )

        incomplete = status[(status["enabled_by_default"] == True) & (status["success"] == False)] if not status.empty else pd.DataFrame()
        if not incomplete.empty:
            pending_names = ", ".join(incomplete["pipeline_name"].tolist())
            findings.append(
                f"- Some enabled pipelines did not finish successfully: {pending_names}. These gaps should be treated as part of the scalability story, not ignored."
            )

        if not retrieval.empty:
            best_retrieval = retrieval.sort_values(["Recall@K", "MRR"], ascending=False).iloc[0]
            findings.append(
                f"- Dense retrieval is strongest at K={int(best_retrieval['K'])}, with Recall@K = {best_retrieval['Recall@K']:.4f} and MRR = {best_retrieval['MRR']:.4f}."
            )

        recommendation_text = "\\n".join(findings) if findings else "- No completed benchmark artifacts were available to summarize."
        display(Markdown("### Report-Ready Summary\\n" + recommendation_text))
        """
    )

    nb["cells"] = cells
    with output_path.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)
    return str(output_path)


if __name__ == "__main__":
    from src.config import get_profile

    print(build_results_notebook(get_profile()))
