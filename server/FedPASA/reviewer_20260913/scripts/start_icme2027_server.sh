#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA/icme2027_workspace
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0
exec /root/autodl-tmp/miniconda3/envs/fbu/bin/python -u scripts/run_icme2027_queue.py >> outputs/icme2027/logs/queue.log 2>&1
