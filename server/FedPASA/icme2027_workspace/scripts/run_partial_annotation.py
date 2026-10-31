from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS  # noqa: E402
from fedpasa.core import mean_average_precision  # noqa: E402
from fedpasa.supervised import RidgePacket, fit_ridge_from_packets  # noqa: E402


FEATURES = ("cls", "patch_mean", "patch_max_text")


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def normalized_features(cache_root: Path, split: str) -> np.ndarray:
    arrays = [
        np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32)
        for name in FEATURES
    ]
    x = np.concatenate(arrays, axis=1)
    x /= np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x.astype(np.float64)


def packet(x: np.ndarray, y: np.ndarray, xtx: np.ndarray | None = None) -> RidgePacket:
    y64 = np.asarray(y, dtype=np.float64)
    return RidgePacket(
        count=len(x),
        sum_x=torch.from_numpy(x.sum(axis=0)),
        sum_y=torch.from_numpy(y64.sum(axis=0)),
        xtx=torch.from_numpy(x.T @ x if xtx is None else xtx),
        xty=torch.from_numpy(x.T @ y64),
    )


def predict(x: np.ndarray, adapter: object) -> np.ndarray:
    return np.asarray(x @ adapter.weight.numpy() + adapter.bias.numpy(), dtype=np.float32)


def standardize(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0)
    std = reference.std(axis=0).clip(min=1e-6)
    return (reference - mean) / std, (values - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["voc07", "voc12", "nus"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--mask-rates", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1])
    parser.add_argument("--weights", type=float, nargs="+", default=np.linspace(0.0, 1.0, 21).tolist())
    args = parser.parse_args()

    all_reports: dict[str, object] = {}
    for dataset in args.datasets:
        if dataset not in BENCHMARKS:
            raise ValueError(f"unknown dataset: {dataset}")
        started = perf_counter()
        cache_root = resolve_cache(dataset)
        dataset_root = ROOT / "outputs" / "icme" / dataset
        train_x = normalized_features(cache_root, "train")
        test_x = normalized_features(cache_root, "test")
        train_y = np.load(cache_root / "train_targets.npy")
        test_y = np.load(cache_root / "test_targets.npy")
        full_xtx = train_x.T @ train_x
        test_piaa = np.load(dataset_root / "scores" / "qafedpasa_representative_scores.npy")
        rows: list[dict[str, object]] = []

        for seed in args.seeds:
            ridge_root = dataset_root / "full_seeds" / f"seed{seed}" / "ridge"
            fusion_root = ridge_root.parent / "ridge_patch_fusion"
            fit_indices = np.load(ridge_root / "fit_indices.npy")
            validation_indices = np.load(ridge_root / "validation_indices.npy")
            fit_x = train_x[fit_indices]
            fit_xtx = fit_x.T @ fit_x
            validation_piaa = np.load(fusion_root / "training_validation_piaa_scores.npy")
            validation_piaa_z, test_piaa_z = standardize(validation_piaa, test_piaa)

            for mask_rate in args.mask_rates:
                mask_rng = np.random.default_rng(seed * 1000 + int(round(mask_rate * 100)))
                masked_y = train_y.copy()
                positive = masked_y > 0
                remove = positive & (mask_rng.random(masked_y.shape) < mask_rate)
                masked_y[remove] = 0
                fit_y = masked_y[fit_indices]
                validation_y = masked_y[validation_indices]
                fit_packet = packet(fit_x, fit_y, fit_xtx)
                tuning: list[dict[str, float]] = []
                adapters: list[object] = []
                for alpha in args.alphas:
                    adapter = fit_ridge_from_packets((fit_packet,), regularization=alpha)
                    validation_scores = predict(train_x[validation_indices], adapter)
                    value, _ = mean_average_precision(validation_y, validation_scores)
                    tuning.append({"alpha": alpha, "observed_validation_mAP": value})
                    adapters.append(adapter)
                choice = int(np.argmax([value["observed_validation_mAP"] for value in tuning]))
                alpha = float(args.alphas[choice])
                validation_adapter = adapters[choice]
                full_adapter = fit_ridge_from_packets(
                    (packet(train_x, masked_y, full_xtx),), regularization=alpha
                )
                validation_ridge = predict(train_x[validation_indices], validation_adapter)
                test_ridge = predict(test_x, full_adapter)
                ridge_map, _ = mean_average_precision(test_y, test_ridge)
                validation_ridge_z, test_ridge_z = standardize(validation_ridge, test_ridge)

                validation_per_class: list[np.ndarray] = []
                global_maps: list[float] = []
                for ridge_weight in args.weights:
                    combined = (
                        ridge_weight * validation_ridge_z
                        + (1.0 - ridge_weight) * validation_piaa_z
                    )
                    value, per_class = mean_average_precision(validation_y, combined)
                    global_maps.append(value)
                    validation_per_class.append(per_class)
                global_index = int(np.argmax(global_maps))
                global_weight = float(args.weights[global_index])
                global_scores = (
                    global_weight * test_ridge_z + (1.0 - global_weight) * test_piaa_z
                )
                global_map, _ = mean_average_precision(test_y, global_scores)
                validation_matrix = np.stack(validation_per_class)
                class_weights = np.asarray(args.weights)[validation_matrix.argmax(axis=0)]
                classwise_scores = (
                    class_weights[None, :] * test_ridge_z
                    + (1.0 - class_weights[None, :]) * test_piaa_z
                )
                classwise_map, _ = mean_average_precision(test_y, classwise_scores)
                row = {
                    "seed": seed,
                    "masked_positive_annotations_percent": 100.0 * mask_rate,
                    "retained_positive_annotations_percent": 100.0 * (1.0 - mask_rate),
                    "realized_retained_positive_percent": 100.0 * (positive & ~remove).sum() / positive.sum(),
                    "selected_alpha": alpha,
                    "ridge_mAP": ridge_map,
                    "global_fusion_weight": global_weight,
                    "global_fusion_mAP": global_map,
                    "classwise_fusion_mAP": classwise_map,
                    "test_labels_used_for_selection": False,
                }
                rows.append(row)
                print(f"PARTIAL dataset={dataset} {json.dumps(row)}", flush=True)
            del fit_x, fit_xtx

        summaries: list[dict[str, float]] = []
        for mask_rate in args.mask_rates:
            selected = [
                row for row in rows
                if row["masked_positive_annotations_percent"] == 100.0 * mask_rate
            ]
            summaries.append(
                {
                    "masked_positive_annotations_percent": 100.0 * mask_rate,
                    "ridge_mAP_mean": float(np.mean([row["ridge_mAP"] for row in selected])),
                    "ridge_mAP_std": float(np.std([row["ridge_mAP"] for row in selected], ddof=1)),
                    "global_fusion_mAP_mean": float(np.mean([row["global_fusion_mAP"] for row in selected])),
                    "classwise_fusion_mAP_mean": float(np.mean([row["classwise_fusion_mAP"] for row in selected])),
                    "classwise_fusion_mAP_std": float(np.std([row["classwise_fusion_mAP"] for row in selected], ddof=1)),
                }
            )
        all_reports[dataset] = {
            "protocol": {
                "masked_entries": "randomly selected positive training annotations only",
                "negative_and_test_annotations": "unchanged",
                "selection": "alpha and fusion weights selected using only the correspondingly masked training-side validation labels",
                "test_labels_used_for_selection": False,
            },
            "rows": rows,
            "summary": summaries,
            "runtime_seconds": perf_counter() - started,
        }

    destination = ROOT / "outputs" / "icme" / "paper" / "partial_annotation.json"
    # A dataset-specific continuation must not discard completed datasets from
    # an earlier run.  This also makes the expensive study safely resumable.
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        previous.update(all_reports)
        all_reports = previous
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps(all_reports, indent=2), encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
