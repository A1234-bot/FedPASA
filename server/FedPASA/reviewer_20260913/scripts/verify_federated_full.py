from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS  # noqa: E402
from fedpasa.core import mean_average_precision  # noqa: E402
from fedpasa.supervised import fit_ridge_from_packets, make_ridge_packet  # noqa: E402


FEATURES = ("cls", "patch_mean", "patch_max_text")


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def load_features(cache_root: Path, split: str) -> np.ndarray:
    arrays = [
        np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32)
        for name in FEATURES
    ]
    return np.concatenate(arrays, axis=1)


def predict(features: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x @ weight.numpy() + bias.numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["voc07", "voc12", "nus"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--t-percent", type=int, default=60)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        benchmark = BENCHMARKS[dataset]
        clients = max(2, round(len(benchmark.classes) * args.t_percent / 100))
        output_root = ROOT / "outputs" / "icme" / dataset
        ridge_root = output_root / "full_seeds" / f"seed{args.seed}" / "ridge"
        report = json.loads((ridge_root / "result.json").read_text(encoding="utf-8"))
        alpha = float(report["selection"]["alpha"])
        cache_root = resolve_cache(dataset)
        train_x = load_features(cache_root, "train")
        train_y = np.load(cache_root / "train_targets.npy")
        assignments = np.load(
            output_root / "partitions" / f"clients{clients}_seed{args.seed}.npy"
        )
        packets = tuple(
            make_ridge_packet(train_x[assignments == client], train_y[assignments == client])
            for client in range(clients)
        )
        federated = fit_ridge_from_packets(packets, regularization=alpha)
        central_payload = torch.load(
            ridge_root / "adapter.pt", map_location="cpu", weights_only=True
        )
        weight_error = float((federated.weight - central_payload["weight"]).abs().max())
        bias_error = float((federated.bias - central_payload["bias"]).abs().max())

        test_x = load_features(cache_root, "test")
        test_y = np.load(cache_root / "test_targets.npy")
        federated_scores = predict(test_x, federated.weight, federated.bias)
        central_scores = predict(test_x, central_payload["weight"], central_payload["bias"])
        score_error = float(np.max(np.abs(federated_scores - central_scores)))
        federated_map, _ = mean_average_precision(test_y, federated_scores)
        central_map, _ = mean_average_precision(test_y, central_scores)
        main_report = json.loads((output_root / "main.json").read_text(encoding="utf-8"))
        qafed_run = next(
            run
            for run in main_report["runs"]
            if run["t_percent"] == args.t_percent and run["seed"] == args.seed
        )
        ridge_communication = float(federated.metadata["estimated_communication_MiB"])
        qafed_communication = float(qafed_run["communication_MiB"])
        row = {
            "dataset": dataset,
            "t_percent": args.t_percent,
            "clients": clients,
            "partition_seed": args.seed,
            "selected_alpha": alpha,
            "central_ridge_mAP": central_map,
            "federated_ridge_mAP": federated_map,
            "mAP_absolute_error": abs(federated_map - central_map),
            "max_abs_weight_error": weight_error,
            "max_abs_bias_error": bias_error,
            "max_abs_test_score_error": score_error,
            "ridge_one_round_communication_MiB": ridge_communication,
            "qafedpasa_communication_MiB": qafed_communication,
            "combined_communication_MiB": ridge_communication + qafed_communication,
            "individual_examples_transmitted": False,
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)

    destination = ROOT / "outputs" / "icme" / "paper" / "federated_full_verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Preserve verified rows for datasets not requested in this continuation.
    if destination.exists():
        previous = json.loads(destination.read_text(encoding="utf-8"))
        by_dataset = {str(row["dataset"]): row for row in previous.get("rows", [])}
        by_dataset.update({str(row["dataset"]): row for row in rows})
        rows = [by_dataset[name] for name in BENCHMARKS if name in by_dataset]
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    temporary.replace(destination)


if __name__ == "__main__":
    main()
