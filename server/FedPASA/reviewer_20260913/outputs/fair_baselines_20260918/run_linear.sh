#!/bin/bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA/reviewer_20260913
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
P=/root/autodl-tmp/miniconda3/envs/fbu/bin/python
$P scripts/run_fair_linear.py --phase tune --dataset coco > outputs/fair_baselines_20260918/logs/linear_coco.log 2>&1
$P scripts/run_fair_linear.py --phase tune --dataset nus > outputs/fair_baselines_20260918/logs/linear_nus.log 2>&1
printf 'COMPLETE\n' > outputs/fair_baselines_20260918/linear_tuning_complete.txt
