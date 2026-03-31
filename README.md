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

## Repository Contents

- `Project.ipynb`
  Main exploratory data analysis notebook for the project.
- `arxiv-metadata-oai-snapshot.json`
  Raw arXiv-style metadata dataset used by the notebook.
- `requirements.txt`
  Python dependencies for running the notebook.
- `scripts/build_project_notebook.py`
  Script used to generate the notebook in a reproducible way.
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
- The notebook is designed to reuse cached summaries in `artifacts/eda` when available.
- The first code cell checks for required packages and installs missing ones into the active notebook environment if necessary.
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

## Expected Output

After running successfully, the notebook provides:

- tables and plots describing the dataset,
- report-ready interpretations for each major section,
- insights on preprocessing and splitting,
- final recommendations for embeddings, clustering, and evaluation design.
