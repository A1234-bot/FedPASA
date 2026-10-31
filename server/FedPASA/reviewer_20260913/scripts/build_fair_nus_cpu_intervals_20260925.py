"""Finish fixed-model NUS paired-image intervals on CPU, without Full refitting.

This is a reporting-only continuation of the frozen 2,000-replicate protocol.
It deliberately omits the unmeasured new-grid Full three-split operation point.
"""
from pathlib import Path
import numpy as np

from fair_baselines_common import OUT, cache, digest, metric, read, save, stamp
from build_fair_baselines_report import prepared_ap
from run_icme2027_paired import weighted_ap, check_math


def run():
    dataset = "nus"
    final = OUT / "final" / dataset
    final.mkdir(parents=True, exist_ok=True)
    ridge = OUT / "ridge" / dataset
    predictions = {
        "Full single split17": [ridge / "full_single_test_scores.npy"],
        "Three-view pooled Ridge": [ridge / "pooled_test_scores.npy"],
        "Local Ridge Avg": [ridge / "local_avg_test_scores.npy"],
        "Global fusion": [ridge / "global_fusion_test_scores.npy"],
    }
    for mode, label in (("central", "Central linear BCE"), ("fedavg", "FedAvg linear BCE")):
        root = OUT / "linear" / dataset / mode / "final"
        predictions[label] = [root / f"seed{seed}" / "test_scores.npy" for seed in (7, 17, 27)]
    for method, label in (("fedmvp", "FedMVP release multilabel adaptation"), ("fedram", "Fed-RAM verified-core adaptation")):
        root = OUT / method / dataset / "final"
        predictions[label] = [root / f"seed{seed}" / "test_scores.npy" for seed in (7, 17, 27)]

    targets_path = cache(dataset) / "test_targets.npy"
    y = np.load(targets_path)
    n, classes = y.shape
    repeats = 2000
    rng = np.random.default_rng(20260918)
    weights = np.empty((repeats, n), dtype=np.uint16)
    for i in range(repeats):
        weights[i] = np.bincount(rng.integers(0, n, n), minlength=n)
    check_math()
    rows, sources, arrays = [], [], {}
    for label, paths in predictions.items():
        bootstrap = final / (label.replace(" ", "_") + "_bootstrap.npy")
        provenance_path = bootstrap.with_suffix(".json")
        provenance = {
            "predictions": [digest(p) for p in paths],
            "targets": digest(targets_path),
            "replicates": repeats,
            "seed": 20260918,
            "metric": "per-fit mean mAP, tie grouped weighted AP v1",
        }
        already = bootstrap.exists() and provenance_path.exists() and read(provenance_path) == provenance
        total = np.load(bootstrap) if already else np.zeros(repeats)
        metrics = []
        for path in paths:
            score = np.load(path)
            assert score.shape == y.shape and np.isfinite(score).all()
            metrics.append(metric(y, score))
            sources.append({"method": label, "path": str(path.relative_to(OUT.parents[1])), "sha256": digest(path)})
            if not already:
                for j in range(classes):
                    np.testing.assert_allclose(
                        weighted_ap(y[:, j], score[:, j], np.ones((1, n), np.uint16))[0] * 100,
                        metric(y[:, j:j + 1], score[:, j:j + 1]), atol=1e-9)
                    values = prepared_ap(y[:, j], score[:, j], weights)
                    np.testing.assert_allclose(values[:3], weighted_ap(y[:, j], score[:, j], weights[:3]), atol=1e-12)
                    total += values * 100 / (classes * len(paths))
        if not already:
            np.save(bootstrap, total)
            save(provenance_path, provenance)
        arrays[label] = total
        rows.append({
            "dataset": dataset, "method": label, "mAP_mean": float(np.mean(metrics)),
            "mAP_sample_std": float(np.std(metrics, ddof=1)) if len(metrics) > 1 else None,
            "runs": len(metrics), "run_mAP": metrics,
            "uncertainty_scope": "fixed partition7; selection split17 fixed; training randomness only; deterministic methods single fit",
        })
        save(final / "progress.json", {"completed_methods": len(rows), "total_methods": len(predictions), "time": stamp()})
        print("REPORT", dataset, label, metrics, flush=True)
    baseline = arrays["Full single split17"]
    basepoint = next(x["mAP_mean"] for x in rows if x["method"] == "Full single split17")
    comparisons = []
    for row in rows:
        if row["method"] == "Full single split17":
            continue
        difference = baseline - arrays[row["method"]]
        comparisons.append({
            "comparison": "Full single split17 minus " + row["method"],
            "difference_mAP": basepoint - row["mAP_mean"],
            "paired_image_bootstrap_95_CI": np.quantile(difference, [.025, .975]).tolist(),
            "interpretation": "conditional on fitted models; average of per-run mAP differences, not averaged predictions; uncorrected percentile CI, not training-repeat inference",
        })
    save(final / "results.json", {
        "status": "COMPLETE", "scope": "CPU reporting from archived predictions; no Full fresh-cost rerun",
        "rows": rows, "comparisons": comparisons, "replicates": repeats,
        "resampling_seed": 20260918, "sources": sources,
        "test_targets_sha256": digest(targets_path), "finished": stamp(),
    })


if __name__ == "__main__":
    run()
