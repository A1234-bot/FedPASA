#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA
while kill -0 473 2>/dev/null; do sleep 2; done
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
/root/autodl-tmp/miniconda3/envs/fbu/bin/python -u scripts/benchmark_server_fedmpt.py --dataset coco > tmp/server_performance/coco.log 2>&1
/root/autodl-tmp/miniconda3/envs/fbu/bin/python -u scripts/benchmark_server_fedmpt.py --dataset nus > tmp/server_performance/nus.log 2>&1
