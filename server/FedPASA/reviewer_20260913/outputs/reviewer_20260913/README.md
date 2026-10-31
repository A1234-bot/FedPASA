# Reviewer experiments — 2026-09-13

Queue: COMPLETE; 60/60 jobs complete.

- Controlled baseline: 10 clients, original KMeans partition seed7, same frozen CLIP cache; training split seed17 fixed in advance. Feature representation and solver family are explicit in the table.
- Split audit uses all three historical prediction arrays and independently recomputes each mAP, mean single-split mAP and score-ensemble mAP. No test-based headline change.
- Scaling uses original t=10/20/40/60/80/100% partitions, seed17; full-population local Top-K rebuilt at every point. Dirichlet beta=0.1/0.3/0.5 uses 10 clients, seed7.
- Full curves use locked historical calibration weights, plus actual newly fitted federated QA/Ridge; Full runtime reruns reselect from training-only validation and are labeled separately.
- Exact Full runtime includes all three training splits and full refits. It explicitly pools validation score vectors and labels for exact AP; this is NOT private statistic-only calibration.
- Symmetric packing is an actual round-trip check for ridge Gram matrices. Both FP64 and FP32 score errors are reported. QA packet packing sizes follow recorded tensor shapes.
- Memory is measured in a fresh largest-client subprocess; process RSS and CUDA memory are distinct. Single-server timings do not measure physical network latency.
- Historical prompt-adaptation timings are separately attributed, not silently mixed with these new runs.
- Existing tie-boundary evidence is retained as prior P2 evidence; no new artificial tie-rate experiment was scheduled.

## Files

controlled_baselines.csv, split_aggregation_audit.csv, performance_scalability.csv, efficiency.csv, symmetric_packing.csv, isolated_client_memory.csv, hyperparameter_sensitivity.csv, figures/, queue.json, logs/.

Missing/failed jobs remain visible in queue.json; no placeholder or literature number is substituted for a missing controlled result.
