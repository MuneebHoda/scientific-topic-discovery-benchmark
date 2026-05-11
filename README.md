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
- `requirements-delta.txt`
  Delta retrieval dependencies that assume PyTorch comes from Delta's PyTorch module.
- `scripts/build_project_notebook.py`
  Script used to generate the notebook in a reproducible way.
- `scripts/run_delta_retrieval.sbatch`
  Slurm batch script for full-corpus dense retrieval on NCSA Delta.
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
- `MPNET` and `BERT` embeddings can run on the full cleaned corpus using CUDA,
- `KMeans` should switch to a scalable path (`MiniBatchKMeans`) on large training splits,
- `HDBSCAN` and `Agglomerative` are also pointed at `full_corpus` in this profile, but they remain the least practical stages at this scale and should be treated as explicit scalability stress tests.

In `colab_h100_full_profile`, the modeling inputs for `TF-IDF`, `MPNET`, `BERT`, `KMeans`, `HDBSCAN`, and `Agglomerative` are all pointed at the **full cleaned corpus**, not a sampled subset. The only remaining sampling in that profile is for expensive evaluation metrics such as:

- silhouette score computation,
- NPMI topic-coherence estimation.

Those are sampled intentionally because full-corpus versions of those metrics are disproportionately expensive and do not change the fact that the underlying embeddings and clustering runs use the full dataset.

In the profile config, a subset size of `None` is treated as a full-corpus run for that pipeline family.

For this profile, `subset_configs()` resolves to a single benchmark subset:

- `full_corpus`

That means the pipeline no longer creates separate sampled modeling subsets for the major methods in the H100 configuration.

The pipeline looks for the raw JSON in these places, in order:

- `ARXIV_DATA_PATH` if you set it,
- the repository root,
- `/content/arxiv-metadata-oai-snapshot.json`,
- `/content/drive/MyDrive/arxiv-metadata-oai-snapshot.json`

You can also pass `--data-path` directly. Large modeling artifacts default to `artifacts/modeling`, but can be moved outside the repo with `ARXIV_ARTIFACTS_DIR` or `--artifacts-dir`.

Available pipeline profiles:

- `local_profile`
- `full_profile`
- `colab_a100_profile`
- `colab_h100_full_profile`
- `delta_retrieval_full_profile`
- `delta_retrieval_sample_profile`

### Running full-corpus retrieval on Delta

The raw `arxiv-metadata-oai-snapshot.json` file is intentionally not committed to git because it is several GB. A fresh clone on Delta will therefore not contain the JSON. Put the file on Delta storage and point the pipeline to it with `ARXIV_DATA_PATH`.

Log in and clone the repo:

```bash
ssh <netid>@login.delta.ncsa.illinois.edu
accounts
git clone <repo-url>
cd <repo-name>
```

Place the JSON and large modeling artifacts outside the repo, preferably on `$WORK`:

```bash
mkdir -p $WORK/arxiv/data
# Transfer arxiv-metadata-oai-snapshot.json with Globus, rsync, or scp.
export ARXIV_DATA_PATH=$WORK/arxiv/data/arxiv-metadata-oai-snapshot.json
export ARXIV_ARTIFACTS_DIR=$WORK/arxiv/artifacts/modeling
```

Create the Python environment once:

```bash
module reset
module load pytorch-conda/2.8
python -m venv --system-site-packages $WORK/envs/arxiv-retrieval
source $WORK/envs/arxiv-retrieval/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-delta.txt
```

Submit the retrieval job:

```bash
sbatch \
  --account=<your-delta-gpu-project> \
  --partition=gpuA100x4 \
  scripts/run_delta_retrieval.sbatch
```

The Delta profile runs full-corpus MPNET retrieval and skips the clustering stress-test stages. Outputs are written under:

```text
$ARXIV_ARTIFACTS_DIR/retrieval/
$ARXIV_ARTIFACTS_DIR/results/
```

If allocation time is tight, use the sampled Delta profile. It reuses the same cleaned dataset but embeds a bounded retrieval subset:

```bash
sbatch \
  --export=ALL,PIPELINE_PROFILE=delta_retrieval_sample_profile,ARXIV_DATA_PATH=$ARXIV_DATA_PATH,ARXIV_ARTIFACTS_DIR=$ARXIV_ARTIFACTS_DIR \
  --account=<your-delta-gpu-project> \
  --partition=gpuA100x4 \
  --time=02:00:00 \
  scripts/run_delta_retrieval.sbatch
```

For a quick environment check inside an interactive GPU allocation, run:

```bash
python - <<'PY'
import torch
import sentence_transformers
print(torch.cuda.is_available())
print(sentence_transformers.__version__)
PY
```

You can also run the profile manually:

```bash
python -m src.run_pipeline \
  --profile delta_retrieval_full_profile \
  --data-path "$ARXIV_DATA_PATH" \
  --artifacts-dir "$ARXIV_ARTIFACTS_DIR"
```

## Generated Benchmark Review Notebook

After `src.run_pipeline` finishes, the pipeline also generates a polished results-review notebook in:

- `artifacts/modeling/results/<profile>_Benchmark_Summary.ipynb`

For example, the H100 profile writes:

- `artifacts/modeling/results/colab_h100_full_profile_Benchmark_Summary.ipynb`

This notebook is designed for project reporting. It summarizes:

- execution scope and split sizes,
- which pipelines completed successfully,
- clustering rankings by `NMI`, `NPMI`, and `silhouette`,
- validation-to-test gaps,
- recovered cluster granularity relative to reference categories,
- runtime and scalability notes,
- retrieval metrics and examples,
- final report-ready findings and recommendations.

The accompanying report artifacts are also written with profile-specific filenames, so runs from
different profiles do not overwrite one another inside `artifacts/modeling/results/`.

## Expected Output

After running successfully, the notebook provides:

- tables and plots describing the dataset,
- report-ready interpretations for each major section,
- insights on preprocessing and splitting,
- final recommendations for embeddings, clustering, and evaluation design.
