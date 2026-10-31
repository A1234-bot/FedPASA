#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA
mkdir -p outputs/server_logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0
/root/autodl-tmp/miniconda3/envs/fbu/bin/python -u scripts/run_matched_fedmpt.py \
  --datasets coco nus --rounds 200 --clients 10 --seed 7 --batch-size 32 --workers 2 \
  >> outputs/server_logs/fedmpt_resume_20260907.log 2>&1
