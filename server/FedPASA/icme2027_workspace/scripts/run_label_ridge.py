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

from fedpasa.benchmarks import BENCHMARKS, FEDMPT_MAP  # noqa: E402
from fedpasa.core import mean_average_precision  # noqa: E402
from fedpasa.supervised import fit_ridge_from_packets, make_ridge_packet  # noqa: E402


def resolve_cache(dataset: str, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def scores(features: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x @ weight + bias


def load_features(cache_root: Path, split: str, names: list[str]) -> np.ndarray:
    arrays = [np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32) for name in names]
    return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=("cls", "patch_mean", "patch_max_text"),
        default=["cls"],
    )
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.dataset]
    cache_root = resolve_cache(args.dataset, args.cache_root)
    feature_tag = "_".join(args.features)
    default_name = "label_ridge" if args.features == ["cls"] else f"label_ridge_{feature_tag}"
    report_root = args.report_root or ROOT / "outputs" / "icme" / args.dataset / default_name
    required = (
        *[f"train_{name}.npy" for name in args.features],
        *[f"test_{name}.npy" for name in args.features],
        "train_targets.npy",
        "test_targets.npy",
    )
    missing = [name for name in required if not (cache_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing caches: {missing}")
    report_root.mkdir(parents=True, exist_ok=True)

    train_x = load_features(cache_root, "train", args.features)
    train_y = np.load(cache_root / "train_targets.npy")
    test_x = load_features(cache_root, "test", args.features)
    test_y = np.load(cache_root / "test_targets.npy")
    rng = np.random.default_rng(args.seed)
    permutation = rng.permutation(len(train_y))
    validation_size = max(1, int(round(len(train_y) * args.validation_fraction)))
    validation_indices = permutation[:validation_size]
    fit_indices = permutation[validation_size:]

    started = perf_counter()
    tuning_packet = make_ridge_packet(train_x[fit_indices], train_y[fit_indices])
    tuning: list[dict[str, float]] = []
    for alpha in args.alphas:
        adapter = fit_ridge_from_packets((tuning_packet,), regularization=alpha)
        validation_scores = scores(
            train_x[validation_indices], adapter.weight.numpy(), adapter.bias.numpy()
        )
        value, _ = mean_average_precision(train_y[validation_indices], validation_scores)
        tuning.append({"alpha": alpha, "validation_mAP": value})
        print(f"RIDGE_TUNE dataset={args.dataset} alpha={alpha:g} mAP={value:.4f}", flush=True)
    selected = max(tuning, key=lambda value: value["validation_mAP"])

    # Keep the adapter used to produce genuinely held-out validation
    # predictions.  The full-data adapter below is only for final test
    # inference and must not be reused for fusion-weight selection.
    validation_adapter = fit_ridge_from_packets(
        (tuning_packet,), regularization=float(selected["alpha"])
    )

    full_packet = make_ridge_packet(train_x, train_y)
    adapter = fit_ridge_from_packets((full_packet,), regularization=float(selected["alpha"]))
    test_scores = scores(test_x, adapter.weight.numpy(), adapter.bias.numpy())
    test_map, per_class = mean_average_precision(test_y, test_scores)
    np.save(report_root / "test_scores.npy", test_scores)
    np.save(report_root / "fit_indices.npy", fit_indices)
    np.save(report_root / "validation_indices.npy", validation_indices)
    torch.save(
        {"weight": adapter.weight, "bias": adapter.bias, "metadata": adapter.metadata},
        report_root / "adapter.pt",
    )
    torch.save(
        {
            "weight": validation_adapter.weight,
            "bias": validation_adapter.bias,
            "metadata": validation_adapter.metadata,
        },
        report_root / "validation_adapter.pt",
    )
    strongest = max(FEDMPT_MAP.get(args.dataset, (float("nan"),)))
    report = {
        "dataset": args.dataset,
        "method": adapter.metadata,
        "protocol": {
            "training_side_validation_fraction": args.validation_fraction,
            "split_seed": args.seed,
            "test_labels_used_for_selection": False,
            "features": args.features,
        },
        "tuning": tuning,
        "selection": selected,
        "test": {
            "mAP": test_map,
            "per_class_AP": per_class.tolist(),
            "strongest_fedmpt_mAP": strongest if np.isfinite(strongest) else None,
            "margin_over_strongest_fedmpt_points": test_map - strongest if np.isfinite(strongest) else None,
            "passes_strongest_fedmpt_plus_2": test_map >= strongest + 2.0 if np.isfinite(strongest) else None,
        },
        "runtime_seconds": perf_counter() - started,
    }
    temporary = report_root / ".result.json.partial"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(report_root / "result.json")
    print(json.dumps(report["test"], indent=2), flush=True)


if __name__ == "__main__":
    main()
