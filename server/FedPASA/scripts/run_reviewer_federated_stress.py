from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
import numpy as np
import torch


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS  # noqa: E402
from fedpasa.core import (  # noqa: E402
    PatchBank,
    fit_central_piaa,
    fit_quantile_aligned_fedpasa,
    mean_average_precision,
)
from fedpasa.supervised import fit_ridge_from_packets, make_ridge_packet  # noqa: E402
from run_label_patch_fusion import piaa_scores, truncate_bank  # noqa: E402


FEATURES = ("cls", "patch_mean", "patch_max_text")
DEFAULT_DATASETS = ("voc07", "voc12", "coco", "nus")


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def deserialize_bank(payload: dict[int, dict[str, torch.Tensor]]) -> dict[int, PatchBank]:
    return {
        int(class_id): PatchBank(
            value["features"],
            value["probabilities"],
            value["entropy"],
            value.get("image_indices"),
        )
        for class_id, value in payload.items()
    }


def load_features(cache_root: Path, split: str) -> np.ndarray:
    arrays = [
        np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32)
        for name in FEATURES
    ]
    return np.concatenate(arrays, axis=1)


def predict(features: np.ndarray, adapter: object) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return np.asarray(x @ adapter.weight.numpy() + adapter.bias.numpy(), dtype=np.float32)


def frozen_standardize(calibration: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return (scores - calibration[0]) / np.maximum(calibration[1], 1e-6)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_participation(
    datasets: list[str],
    fractions: list[float],
    seeds: list[int],
    *,
    batch_size: int,
    device: str,
) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    for dataset in datasets:
        started = perf_counter()
        benchmark = BENCHMARKS[dataset]
        cache_root = resolve_cache(dataset)
        dataset_root = ROOT / "outputs" / "icme" / dataset
        clients = max(2, round(benchmark.class_count * 0.4))
        bank_path = dataset_root / "banks" / f"local_clients{clients}_seed7_k512.pt"
        serialized = torch.load(bank_path, map_location="cpu", weights_only=True)
        banks = tuple(deserialize_bank(value) for value in serialized)
        del serialized
        assignments = np.load(dataset_root / "partitions" / f"clients{clients}_seed7.npy")
        train_x = load_features(cache_root, "train")
        train_y = np.load(cache_root / "train_targets.npy")
        test_x = load_features(cache_root, "test")
        test_y = np.load(cache_root / "test_targets.npy")
        text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
        packets = tuple(
            make_ridge_packet(train_x[assignments == client], train_y[assignments == client])
            for client in range(clients)
        )
        ridge_root = dataset_root / "full_seeds" / "seed7" / "ridge"
        fusion_root = ridge_root.parent / "ridge_patch_fusion"
        alpha = float(json.loads((ridge_root / "result.json").read_text(encoding="utf-8"))["selection"]["alpha"])
        fusion_report = json.loads((fusion_root / "result.json").read_text(encoding="utf-8"))
        class_weights = np.asarray(fusion_report["classwise_weight"]["weights"], dtype=np.float32)
        piaa_calibration = np.load(fusion_root / "piaa_calibration.npy")
        ridge_calibration = np.load(fusion_root / "ridge_calibration.npy")
        existing_piaa = np.load(dataset_root / "scores" / "qafedpasa_representative_scores.npy")
        full_ridge_payload = torch.load(ridge_root / "adapter.pt", map_location="cpu", weights_only=True)
        existing_ridge = predict(
            test_x,
            type("FrozenAdapter", (), {"weight": full_ridge_payload["weight"], "bias": full_ridge_payload["bias"]})(),
        )
        rows: list[dict[str, object]] = []
        score_root = dataset_root / "reviewer" / "client_participation"
        score_root.mkdir(parents=True, exist_ok=True)
        cache: dict[tuple[int, ...], tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
        for fraction in fractions:
            selected_count = max(1, int(round(fraction * clients)))
            for seed in seeds:
                rng = np.random.default_rng(seed + 10_000 * clients + int(round(100 * fraction)))
                selected = tuple(sorted(rng.choice(clients, size=selected_count, replace=False).tolist()))
                if selected in cache:
                    qafed, ridge, metadata = cache[selected]
                elif selected_count == clients:
                    qafed = existing_piaa
                    ridge = existing_ridge
                    metadata = {
                        "qafedpasa_communication_MiB": float(
                            fit_quantile_aligned_fedpasa(
                                banks,
                                text,
                                k=512,
                                purification_temperature=benchmark.purification_temperature,
                            ).metadata["estimated_communication_MiB"]
                        ),
                        "ridge_communication_MiB": float(
                            fit_ridge_from_packets(packets, regularization=alpha).metadata[
                                "estimated_communication_MiB"
                            ]
                        ),
                    }
                    cache[selected] = (qafed, ridge, metadata)
                else:
                    qafed_adapter = fit_quantile_aligned_fedpasa(
                        tuple(banks[index] for index in selected),
                        text,
                        k=512,
                        purification_temperature=benchmark.purification_temperature,
                    )
                    qafed = piaa_scores(
                        cache_root / "test_patch.npy",
                        cache_root / "test_cls.npy",
                        np.arange(len(test_y), dtype=np.int64),
                        text,
                        qafed_adapter.weight,
                        qafed_adapter.bias,
                        cls_scale=benchmark.cls_scale,
                        patch_scale=benchmark.patch_scale,
                        batch_images=batch_size,
                        device=device,
                    )
                    ridge_adapter = fit_ridge_from_packets(
                        tuple(packets[index] for index in selected), regularization=alpha
                    )
                    ridge = predict(test_x, ridge_adapter)
                    metadata = {
                        "qafedpasa_communication_MiB": float(
                            qafed_adapter.metadata["estimated_communication_MiB"]
                        ),
                        "ridge_communication_MiB": float(
                            ridge_adapter.metadata["estimated_communication_MiB"]
                        ),
                    }
                    cache[selected] = (qafed, ridge, metadata)
                qafed_map, _ = mean_average_precision(test_y, qafed)
                ridge_map, _ = mean_average_precision(test_y, ridge)
                full_scores = (
                    class_weights[None, :] * frozen_standardize(ridge_calibration, ridge)
                    + (1.0 - class_weights[None, :])
                    * frozen_standardize(piaa_calibration, qafed)
                )
                full_map, _ = mean_average_precision(test_y, full_scores)
                row = {
                    "dataset": dataset,
                    "available_clients_percent": 100.0 * selected_count / clients,
                    "requested_fraction_percent": 100.0 * fraction,
                    "selected_clients": "/".join(map(str, selected)),
                    "subset_seed": seed,
                    "available_clients": selected_count,
                    "total_clients": clients,
                    "retained_training_images_percent": float(
                        100.0 * np.isin(assignments, selected).sum() / len(assignments)
                    ),
                    "qafedpasa_mAP": qafed_map,
                    "ridge_mAP": ridge_map,
                    "frozen_fusion_mAP": full_map,
                    "qafedpasa_communication_MiB": metadata["qafedpasa_communication_MiB"],
                    "ridge_communication_MiB": metadata["ridge_communication_MiB"],
                    "combined_communication_MiB": metadata["qafedpasa_communication_MiB"]
                    + metadata["ridge_communication_MiB"],
                    "test_labels_used_for_selection": False,
                }
                rows.append(row)
                print(f"PARTICIPATION {json.dumps(row)}", flush=True)
        summary: list[dict[str, object]] = []
        for fraction in fractions:
            selected_rows = [row for row in rows if row["requested_fraction_percent"] == 100.0 * fraction]
            item = {
                "dataset": dataset,
                "requested_fraction_percent": 100.0 * fraction,
                "available_clients_percent": float(np.mean([row["available_clients_percent"] for row in selected_rows])),
                "retained_training_images_percent_mean": float(
                    np.mean([row["retained_training_images_percent"] for row in selected_rows])
                ),
                "qafedpasa_mAP_mean": float(np.mean([row["qafedpasa_mAP"] for row in selected_rows])),
                "qafedpasa_mAP_std": float(np.std([row["qafedpasa_mAP"] for row in selected_rows], ddof=1)),
                "ridge_mAP_mean": float(np.mean([row["ridge_mAP"] for row in selected_rows])),
                "ridge_mAP_std": float(np.std([row["ridge_mAP"] for row in selected_rows], ddof=1)),
                "frozen_fusion_mAP_mean": float(np.mean([row["frozen_fusion_mAP"] for row in selected_rows])),
                "frozen_fusion_mAP_std": float(np.std([row["frozen_fusion_mAP"] for row in selected_rows], ddof=1)),
                "combined_communication_MiB_mean": float(
                    np.mean([row["combined_communication_MiB"] for row in selected_rows])
                ),
                "test_labels_used_for_selection": False,
            }
            summary.append(item)
            csv_rows.append(item)
        report[dataset] = {
            "protocol": {
                "base_partition": "t=40% KMeans partition, seed 7",
                "availability": "uniform client subsampling without replacement before adapter aggregation/refit",
                "fusion": "classwise weights and per-class score calibration frozen from seed-7 training-side validation before the availability shock",
                "selection": "three predeclared subset seeds; no condition is chosen by test performance",
                "test_labels_used_for_selection": False,
            },
            "rows": rows,
            "summary": summary,
            "runtime_seconds": perf_counter() - started,
        }
        atomic_json(paper_root / "client_participation.json", report)
        del banks, packets, train_x, test_x
    write_csv(paper_root / "client_participation.csv", csv_rows)
    return report


def run_temperature(
    datasets: list[str], multipliers: list[float], *, batch_size: int, device: str
) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    for dataset in datasets:
        started = perf_counter()
        benchmark = BENCHMARKS[dataset]
        cache_root = resolve_cache(dataset)
        dataset_root = ROOT / "outputs" / "icme" / dataset
        text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
        bank_candidates = sorted(
            (dataset_root / "banks").glob("global_k*.pt"),
            key=lambda path: int(path.stem.split("k")[-1]),
            reverse=True,
        )
        bank = truncate_bank(
            deserialize_bank(torch.load(bank_candidates[0], map_location="cpu", weights_only=True)),
            512,
        )
        test_y = np.load(cache_root / "test_targets.npy")
        rows: list[dict[str, object]] = []
        score_root = dataset_root / "reviewer" / "temperature"
        score_root.mkdir(parents=True, exist_ok=True)
        for multiplier in multipliers:
            temperature = benchmark.purification_temperature * multiplier
            score_path = score_root / f"multiplier_{multiplier:g}.npy"
            if multiplier == 1.0:
                scores = np.load(dataset_root / "scores" / "qafedpasa_representative_scores.npy")
            elif score_path.exists():
                scores = np.load(score_path)
            else:
                adapter = fit_central_piaa(
                    bank, text, purification_temperature=temperature
                )
                scores = piaa_scores(
                    cache_root / "test_patch.npy",
                    cache_root / "test_cls.npy",
                    np.arange(len(test_y), dtype=np.int64),
                    text,
                    adapter.weight,
                    adapter.bias,
                    cls_scale=benchmark.cls_scale,
                    patch_scale=benchmark.patch_scale,
                    batch_images=batch_size,
                    device=device,
                )
                np.save(score_path, scores)
            value, _ = mean_average_precision(test_y, scores)
            row = {
                "dataset": dataset,
                "temperature_multiplier": multiplier,
                "purification_temperature": temperature,
                "mAP": value,
                "predeclared_reference_multiplier": multiplier == 1.0,
                "test_labels_used_for_selection": False,
            }
            rows.append(row)
            csv_rows.append(row)
            print(f"TEMPERATURE {json.dumps(row)}", flush=True)
        report[dataset] = {
            "protocol": {
                "grid": "multipliers fixed before evaluation and applied to each benchmark's released PIAA temperature",
                "purpose": "sensitivity reporting only; the reference multiplier remains 1.0 regardless of test outcomes",
                "test_labels_used_for_selection": False,
            },
            "rows": rows,
            "range_mAP": max(row["mAP"] for row in rows) - min(row["mAP"] for row in rows),
            "runtime_seconds": perf_counter() - started,
        }
        atomic_json(paper_root / "temperature_sensitivity.json", report)
    write_csv(paper_root / "temperature_sensitivity.csv", csv_rows)
    return report


def plot_summaries() -> None:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    figure_root = paper_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    path = paper_root / "client_participation.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        for dataset, value in payload.items():
            summary = value["summary"]
            ax.errorbar(
                [row["available_clients_percent"] for row in summary],
                [row["frozen_fusion_mAP_mean"] for row in summary],
                yerr=[row["frozen_fusion_mAP_std"] for row in summary],
                marker="o",
                capsize=3,
                label=dataset.upper(),
            )
        ax.set_xlabel("Available clients (%)")
        ax.set_ylabel("Frozen-deployment fusion mAP")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(figure_root / "client_participation.png", dpi=220)
        figure.savefig(figure_root / "client_participation.pdf", bbox_inches="tight")
        plt.close(figure)
    path = paper_root / "temperature_sensitivity.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        for dataset, value in payload.items():
            rows = value["rows"]
            ax.plot(
                [row["temperature_multiplier"] for row in rows],
                [row["mAP"] for row in rows],
                marker="o",
                label=dataset.upper(),
            )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Purification-temperature multiplier")
        ax.set_ylabel("QAFedPASA mAP")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(figure_root / "temperature_sensitivity.png", dpi=220)
        figure.savefig(figure_root / "temperature_sensitivity.pdf", bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--studies", nargs="+", choices=("participation", "temperature", "plot"), default=["participation", "temperature"]
    )
    parser.add_argument("--fractions", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--subset-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--temperature-multipliers", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0, 4.0])
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if "participation" in args.studies:
        run_participation(
            args.datasets,
            args.fractions,
            args.subset_seeds,
            batch_size=args.batch_size,
            device=args.device,
        )
    if "temperature" in args.studies:
        run_temperature(
            args.datasets,
            args.temperature_multipliers,
            batch_size=args.batch_size,
            device=args.device,
        )
    plot_summaries()


if __name__ == "__main__":
    main()
