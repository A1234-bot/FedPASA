#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/FedPASA/reviewer_20260913
PY=/root/autodl-tmp/miniconda3/envs/fbu/bin/python
if [ -f outputs/supplement_20261029/PAUSE ]; then
    echo 'PAUSE marker present; refusing automatic resume'
    exit 2
fi
"$PY" - <<'PY'
import fcntl
from pathlib import Path
p=Path('outputs/supplement_20261029');p.mkdir(parents=True,exist_ok=True)
with (p/'queue.lock').open('a') as f:
    fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
print('QUEUE_LOCK_AVAILABLE')
PY
mkdir -p outputs/supplement_20261029/logs
nohup env CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$PY" -u scripts/queue_supplement_20261029.py --workers 4 >> outputs/supplement_20261029/logs/queue.log 2>&1 < /dev/null &
printf 'QUEUE_PID=%s\n' "$!"
