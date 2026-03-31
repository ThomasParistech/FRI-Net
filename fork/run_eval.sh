#!/usr/bin/env bash
# Build and run RoomFormer polygon extraction in Docker.
#
# Usage:
#   bash fork/scripts/run_extract.sh
#
# Expects data/ to already contain the datasets and checkpoints
# (run `python fork/scripts/download_data.py` first).

set -euo pipefail

FORK_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${FORK_ROOT}/.." && pwd)"
MODULES_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"
IMAGE_NAME="frinet-infer"

echo "Building Docker image '${IMAGE_NAME}'..."
docker build -t "${IMAGE_NAME}" -f "${FORK_ROOT}/Dockerfile.infer" "${REPO_ROOT}"

echo "Running extraction for all datasets..."
docker run --rm --gpus all \
    -v "${REPO_ROOT}/checkpoints:/app/checkpoints" \
    -v "${REPO_ROOT}/data:/app/data" \
    -v "${REPO_ROOT}/s3d_floorplan_eval/montefloor_data:/app/s3d_floorplan_eval/montefloor_data" \
    -v "${MODULES_ROOT}/benchmark_helper:/app/benchmark_helper" \
    "${IMAGE_NAME}" \
    conda run --no-capture-output -n frinet python3 eval_stru3d.py --checkpoint ./checkpoints/pretrained_ckpt.pth

echo "Done. Results in ${REPO_ROOT}/checkpoints/eval_stru3d"