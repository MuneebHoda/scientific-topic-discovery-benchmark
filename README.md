# A Benchmark of Text Embeddings and Clustering Algorithms for Scientific Topic Discovery

## Team

- Syed Muneeb Hoda
- Saad Sher Alam
- Yuming

## Project Purpose

This project studies how well different text representations and clustering methods work for scientific topic discovery on a large arXiv-style paper corpus.

The main goals are:

- analyze the structure and quality of the scientific paper dataset,
- compare embedding approaches for representing titles and abstracts,
- evaluate clustering algorithms for discovering topic structure,
- use arXiv category labels as a reference signal for analysis and benchmarking,
- support later retrieval and exploration of relevant scientific papers.

The primary deliverable in this repository is a polished exploratory data analysis notebook:

- [`Project.ipynb`](./Project.ipynb)

The repository also includes a modeling pipeline in `src/` for building cleaned data, creating deterministic splits, generating embeddings, clustering papers, and producing result tables.

## Repository Contents

- `Project.ipynb`
  Main exploratory data analysis notebook for the project.
- `arxiv-metadata-oai-snapshot.json`
  Raw arXiv-style metadata dataset used by the notebook.
- `requirements.txt`
  Python dependencies for running the notebook.
- `scripts/build_project_notebook.py`
  Script used to generate the notebook in a reproducible way.
- `src/run_pipeline.py`
  Main entrypoint for the modeling pipeline.
- `artifacts/`
  Cached EDA summaries created by the notebook to avoid recomputing expensive full-dataset statistics on every run.

## What the Notebook Does

The notebook performs a full EDA focused on downstream topic discovery and benchmarking. It includes:

- dataset overview and schema inspection,
- missing-value and duplicate analysis,
- category and domain imbalance analysis,
- title and abstract length analysis,
- preprocessing-oriented token and vocabulary analysis,
- temporal and metadata analysis,
- split-strategy analysis for experiments,
- key findings and recommendations for the final report.

## How to Run

### 1. Create and activate a virtual environment

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

### 4. Open the notebook

Open:

- `Project.ipynb`

### 5. Select the correct kernel

Make sure the notebook is using the Python environment from this project, ideally:

```text
.venv/bin/python
```

If you are unsure, run this in a notebook cell:

```python
import sys
print(sys.executable)
```

## Important Notes

- The dataset file is large, so the first full run can take time.
- The notebook is designed to reuse cached summaries in `artifacts/eda_full` when available.
- The first code cell checks for required packages and installs missing ones into the active notebook environment if necessary.
- If you run this in Google Colab, prefer a **high-RAM runtime**. GPU is optional, but the expensive EDA stages are mostly **CPU and memory bound**, not GPU bound.
- If package installation fails inside Jupyter, install dependencies manually in the same environment with:

```bash
python -m pip install -r requirements.txt
```

## Recommended Run Order

Run the notebook from top to bottom:

1. dependency/setup cell,
2. data loading and artifact check,
3. EDA sections in order,
4. final recommendations section.

## Running the Modeling Pipeline

For a full Google Colab run with a strong GPU, use the H100-oriented profile:

```bash
python -m src.run_pipeline --profile colab_h100_full_profile
```

This profile is designed around the following assumptions:

- `TF-IDF` can run on the full cleaned corpus with bounded vocabulary settings,
- `BERT` embeddings can run on the full cleaned corpus using CUDA,
- `KMeans` should switch to a scalable path (`MiniBatchKMeans`) on large training splits,
- `Agglomerative` and `HDBSCAN` remain disabled by default for the full-corpus H100 run because they are not the right baseline for this scale.

The pipeline looks for the raw JSON in these places, in order:

- `ARXIV_DATA_PATH` if you set it,
- the repository root,
- `/content/arxiv-metadata-oai-snapshot.json`,
- `/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json`

The older subset-oriented profiles are still available:

- `local_profile`
- `full_profile`
- `colab_a100_profile`
- `colab_h100_full_profile`

## Expected Output

After running successfully, the notebook provides:

- tables and plots describing the dataset,
- report-ready interpretations for each major section,
- insights on preprocessing and splitting,
- final recommendations for embeddings, clustering, and evaluation design.
