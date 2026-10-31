#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA/reviewer_20260913
export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
mkdir -p outputs/reviewer_20260913/logs
exec /root/autodl-tmp/miniconda3/envs/fbu/bin/python -u scripts/reviewer_20260913_queue.py >> outputs/reviewer_20260913/logs/queue.log 2>&1
