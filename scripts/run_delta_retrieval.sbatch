#!/bin/bash
#SBATCH --job-name=arxiv-retrieval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=verbose,per_task:1
#SBATCH --output=delta-retrieval-%j.out
#SBATCH --error=delta-retrieval-%j.err

set -euo pipefail

module reset
module load pytorch-conda/2.8
module list

if [[ -z "${WORK:-}" ]]; then
  echo "WORK is not set. Run this on Delta after 'module reset', or set WORK to your project work directory." >&2
  exit 2
fi

export ARXIV_DATA_PATH="${ARXIV_DATA_PATH:-$WORK/arxiv/data/arxiv-metadata-oai-snapshot.json}"
export ARXIV_ARTIFACTS_DIR="${ARXIV_ARTIFACTS_DIR:-$WORK/arxiv/artifacts/modeling}"
export HF_HOME="${HF_HOME:-$WORK/arxiv/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export SENTENCE_TRANSFORMERS_HOME="${SENTENCE_TRANSFORMERS_HOME:-$HF_HOME/sentence_transformers}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$WORK/arxiv/cache}"
export PIPELINE_PROFILE="${PIPELINE_PROFILE:-delta_retrieval_full_profile}"

mkdir -p "$ARXIV_ARTIFACTS_DIR" "$HF_HOME" "$HF_HUB_CACHE" "$SENTENCE_TRANSFORMERS_HOME" "$XDG_CACHE_HOME"

if [[ ! -f "$ARXIV_DATA_PATH" ]]; then
  echo "Raw JSON not found: $ARXIV_DATA_PATH" >&2
  echo "Transfer arxiv-metadata-oai-snapshot.json to Delta, or set ARXIV_DATA_PATH before sbatch." >&2
  exit 2
fi

if [[ ! -d "$WORK/envs/arxiv-retrieval" ]]; then
  echo "Missing environment: $WORK/envs/arxiv-retrieval" >&2
  echo "Create it once with the README Delta setup commands, then resubmit this job." >&2
  exit 2
fi

source "$WORK/envs/arxiv-retrieval/bin/activate"

echo "Job started on $(hostname)"
echo "ARXIV_DATA_PATH=$ARXIV_DATA_PATH"
echo "ARXIV_ARTIFACTS_DIR=$ARXIV_ARTIFACTS_DIR"
echo "PIPELINE_PROFILE=$PIPELINE_PROFILE"

python - <<'PY'
import torch

print("python torch cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("python torch cuda device:", torch.cuda.get_device_name(0))
PY

srun python -m src.run_pipeline \
  --profile "$PIPELINE_PROFILE" \
  --data-path "$ARXIV_DATA_PATH" \
  --artifacts-dir "$ARXIV_ARTIFACTS_DIR"
