"""Generate a lightweight presentation notebook from saved artifacts only."""

from __future__ import annotations

from textwrap import dedent

import nbformat as nbf

from src.config import ProfileConfig


def build_results_notebook(profile: ProfileConfig) -> str:
    """Create the local benchmark summary notebook."""

    output_path = profile.results_dir() / "Local_Benchmark_Summary.ipynb"
    profile.results_dir().mkdir(parents=True, exist_ok=True)

    nb = nbf.v4.new_notebook()
    cells = []

    def md(text: str) -> None:
        cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))

    def code(text: str) -> None:
        cells.append(nbf.v4.new_code_cell(dedent(text).strip()))

    md(
        """
        # Local Benchmark Summary

        This notebook is intentionally lightweight. It only loads saved modeling artifacts from
        `artifacts/modeling/` and renders compact report tables and figures for the laptop-safe
        benchmark pipeline.
        """
    )

    code(
        """
        from pathlib import Path
        import json

        import matplotlib.pyplot as plt
        import pandas as pd

        ROOT = Path("artifacts/modeling/results")
        clustering_path = ROOT / "clustering_results_report.csv"
        retrieval_path = ROOT / "retrieval_results_report.csv"
        split_report_path = ROOT / "split_report.json"
        status_path = ROOT / "pipeline_status.csv"

        clustering = pd.read_csv(clustering_path) if clustering_path.exists() else pd.DataFrame()
        retrieval = pd.read_csv(retrieval_path) if retrieval_path.exists() else pd.DataFrame()
        status = pd.read_csv(status_path) if status_path.exists() else pd.DataFrame()
        split_report = json.loads(split_report_path.read_text()) if split_report_path.exists() else {}

        print("Loaded artifacts from", ROOT.resolve())
        """
    )

    md("## Cleaning and Split Summary")
    code(
        """
        pd.json_normalize(split_report).T
        """
    )

    md(
        """
        The local profile uses the full raw corpus only for the already-computed EDA artifacts.
        All modeling experiments below run on deterministic cached subsets sized for a 16GB laptop.
        """
    )

    md("## Clustering Results")
    code(
        """
        clustering
        """
    )
    code(
        """
        if not clustering.empty:
            chart = clustering.copy()
            chart["Pipeline"] = chart["Embedding"].str.upper() + " + " + chart["Clustering"].str.title()
            chart = chart.sort_values("NMI", ascending=False)
            plt.figure(figsize=(10, 4))
            plt.bar(chart["Pipeline"], chart["NMI"])
            plt.xticks(rotation=30, ha="right")
            plt.ylabel("NMI")
            plt.title("Clustering Alignment by Pipeline")
            plt.tight_layout()
            plt.show()
        else:
            print("No clustering metrics available yet.")
        """
    )

    md("## Retrieval Results")
    code(
        """
        retrieval
        """
    )
    code(
        """
        if not retrieval.empty:
            plt.figure(figsize=(6, 4))
            plt.plot(retrieval["K"], retrieval["Recall@K"], marker="o", label="Recall@K")
            plt.plot(retrieval["K"], retrieval["MRR"], marker="o", label="MRR")
            plt.xticks(retrieval["K"])
            plt.title("MPNET Dense Retrieval")
            plt.xlabel("K")
            plt.ylabel("Metric")
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            print("No retrieval metrics available yet.")
        """
    )

    md("## Pipeline Status")
    code(
        """
        status
        """
    )

    md(
        """
        ## Interpretation Notes

        - TF-IDF is the main large local baseline and should populate first.
        - MPNET is the main dense baseline for local experiments and retrieval.
        - BERT exists in the codebase but is disabled by default to keep the local profile practical.
        """
    )

    nb["cells"] = cells
    with output_path.open("w", encoding="utf-8") as handle:
        nbf.write(nb, handle)
    return str(output_path)


if __name__ == "__main__":
    from src.config import get_profile

    print(build_results_notebook(get_profile()))

