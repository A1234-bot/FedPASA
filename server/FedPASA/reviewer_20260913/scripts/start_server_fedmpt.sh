#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA
mkdir -p outputs/server_logs
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0
/root/autodl-tmp/miniconda3/envs/fbu/bin/python -u scripts/run_matched_fedmpt.py \
  --datasets coco nus --rounds 200 --clients 10 --seed 7 --batch-size 32 \
  --coco-workers 4 --nus-workers 8 --cpu-threads 1 \
  --no-activation-checkpointing --cache-static-text --fused-gradient-check \
  >> outputs/server_logs/fedmpt_optimized_20260907.log 2>&1
