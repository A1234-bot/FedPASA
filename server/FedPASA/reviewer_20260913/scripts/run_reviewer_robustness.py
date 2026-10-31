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
from fedpasa.core import mean_average_precision  # noqa: E402
from fedpasa.supervised import RidgePacket, fit_ridge_from_packets  # noqa: E402


FEATURES = ("cls", "patch_mean", "patch_max_text")
DEFAULT_DATASETS = ("voc07", "voc12", "coco", "nus")
ALPHAS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
FUSION_WEIGHTS = np.linspace(0.0, 1.0, 21)


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def load_features(cache_root: Path, split: str) -> np.ndarray:
    arrays = [
        np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32)
        for name in FEATURES
    ]
    result = np.concatenate(arrays, axis=1)
    result /= np.linalg.norm(result, axis=1, keepdims=True).clip(min=1e-12)
    return result.astype(np.float64)


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


def balanced_pair_noise(targets: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, dict[str, int]]:
    """Swap equal positive/negative counts per class, preserving prevalence exactly."""

    output = np.asarray(targets, dtype=np.uint8).copy()
    rng = np.random.default_rng(seed)
    positive_to_negative = 0
    negative_to_positive = 0
    for class_id in range(output.shape[1]):
        positive = np.flatnonzero(output[:, class_id] > 0)
        negative = np.flatnonzero(output[:, class_id] == 0)
        count = min(int(round(rate * len(positive))), len(negative))
        if count == 0:
            continue
        remove = rng.choice(positive, size=count, replace=False)
        add = rng.choice(negative, size=count, replace=False)
        output[remove, class_id] = 0
        output[add, class_id] = 1
        positive_to_negative += count
        negative_to_positive += count
    return output, {
        "positive_to_negative": positive_to_negative,
        "negative_to_positive": negative_to_positive,
    }


def run_label_noise(datasets: list[str], seeds: list[int], rates: list[float]) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    for dataset in datasets:
        started = perf_counter()
        cache_root = resolve_cache(dataset)
        dataset_root = ROOT / "outputs" / "icme" / dataset
        train_x = load_features(cache_root, "train")
        test_x = load_features(cache_root, "test")
        train_y = np.load(cache_root / "train_targets.npy")
        test_y = np.load(cache_root / "test_targets.npy")
        full_xtx = train_x.T @ train_x
        test_piaa = np.load(dataset_root / "scores" / "qafedpasa_representative_scores.npy")
        rows: list[dict[str, object]] = []
        for seed in seeds:
            ridge_root = dataset_root / "full_seeds" / f"seed{seed}" / "ridge"
            fusion_root = ridge_root.parent / "ridge_patch_fusion"
            fit_indices = np.load(ridge_root / "fit_indices.npy")
            validation_indices = np.load(ridge_root / "validation_indices.npy")
            fit_x = train_x[fit_indices]
            fit_xtx = fit_x.T @ fit_x
            validation_y = train_y[validation_indices]
            validation_piaa = np.load(fusion_root / "training_validation_piaa_scores.npy")
            validation_piaa_z, test_piaa_z = standardize(validation_piaa, test_piaa)
            for rate in rates:
                noisy_y, flips = balanced_pair_noise(
                    train_y, rate, seed * 100_000 + int(round(rate * 10_000))
                )
                fit_y = noisy_y[fit_indices]
                fit_packet = packet(fit_x, fit_y, fit_xtx)
                adapters: list[object] = []
                tuning: list[dict[str, float]] = []
                for alpha in ALPHAS:
                    adapter = fit_ridge_from_packets((fit_packet,), regularization=alpha)
                    value, _ = mean_average_precision(
                        validation_y, predict(train_x[validation_indices], adapter)
                    )
                    adapters.append(adapter)
                    tuning.append({"alpha": alpha, "clean_validation_mAP": value})
                selected = int(np.argmax([item["clean_validation_mAP"] for item in tuning]))
                alpha = ALPHAS[selected]
                validation_adapter = adapters[selected]
                full_adapter = fit_ridge_from_packets(
                    (packet(train_x, noisy_y, full_xtx),), regularization=alpha
                )
                validation_ridge = predict(train_x[validation_indices], validation_adapter)
                test_ridge = predict(test_x, full_adapter)
                ridge_map, _ = mean_average_precision(test_y, test_ridge)
                validation_ridge_z, test_ridge_z = standardize(validation_ridge, test_ridge)

                validation_per_class: list[np.ndarray] = []
                validation_maps: list[float] = []
                for weight in FUSION_WEIGHTS:
                    value, per_class = mean_average_precision(
                        validation_y,
                        weight * validation_ridge_z + (1.0 - weight) * validation_piaa_z,
                    )
                    validation_maps.append(value)
                    validation_per_class.append(per_class)
                global_index = int(np.argmax(validation_maps))
                global_weight = float(FUSION_WEIGHTS[global_index])
                global_map, _ = mean_average_precision(
                    test_y,
                    global_weight * test_ridge_z + (1.0 - global_weight) * test_piaa_z,
                )
                class_weights = FUSION_WEIGHTS[np.stack(validation_per_class).argmax(axis=0)]
                classwise_map, _ = mean_average_precision(
                    test_y,
                    class_weights[None, :] * test_ridge_z
                    + (1.0 - class_weights[None, :]) * test_piaa_z,
                )
                row = {
                    "dataset": dataset,
                    "seed": seed,
                    "pair_noise_percent": 100.0 * rate,
                    **flips,
                    "selected_alpha": alpha,
                    "ridge_mAP": ridge_map,
                    "global_fusion_weight": global_weight,
                    "global_fusion_mAP": global_map,
                    "classwise_fusion_mAP": classwise_map,
                    "qafedpasa_label_free_mAP": float(
                        mean_average_precision(test_y, test_piaa)[0]
                    ),
                    "test_labels_used_for_selection": False,
                }
                rows.append(row)
                print(f"LABEL_NOISE {json.dumps(row)}", flush=True)
        summary: list[dict[str, object]] = []
        for rate in rates:
            selected_rows = [row for row in rows if row["pair_noise_percent"] == 100.0 * rate]
            summary_row = {
                "dataset": dataset,
                "pair_noise_percent": 100.0 * rate,
                "ridge_mAP_mean": float(np.mean([row["ridge_mAP"] for row in selected_rows])),
                "ridge_mAP_std": float(np.std([row["ridge_mAP"] for row in selected_rows], ddof=1)),
                "global_fusion_mAP_mean": float(
                    np.mean([row["global_fusion_mAP"] for row in selected_rows])
                ),
                "global_fusion_mAP_std": float(
                    np.std([row["global_fusion_mAP"] for row in selected_rows], ddof=1)
                ),
                "classwise_fusion_mAP_mean": float(
                    np.mean([row["classwise_fusion_mAP"] for row in selected_rows])
                ),
                "classwise_fusion_mAP_std": float(
                    np.std([row["classwise_fusion_mAP"] for row in selected_rows], ddof=1)
                ),
                "test_labels_used_for_selection": False,
            }
            summary.append(summary_row)
            csv_rows.append(summary_row)
        report[dataset] = {
            "protocol": {
                "noise": "within each class, replace r% of positive labels by negatives and the same number of negatives by positives",
                "prevalence": "preserved exactly per class",
                "selection": "ridge alpha and fusion weights use the clean training-side 20% validation split only",
                "final_fit": "selected alpha is refit on all noisy training annotations",
                "test_labels_used_for_selection": False,
            },
            "rows": rows,
            "summary": summary,
            "runtime_seconds": perf_counter() - started,
        }
        atomic_json(paper_root / "label_noise.json", report)
    write_csv(paper_root / "label_noise.csv", csv_rows)
    return report


def quantize_tensor(value: torch.Tensor, codec: str) -> tuple[torch.Tensor, int]:
    value = value.double()
    if codec == "fp64":
        return value, value.numel() * 8
    if codec == "fp32":
        return value.float().double(), value.numel() * 4
    if codec == "fp16":
        return value.half().double(), value.numel() * 2
    if codec == "int8":
        maximum = float(value.abs().max())
        scale = maximum / 127.0 if maximum > 0 else 1.0
        quantized = torch.clamp(torch.round(value / scale), -127, 127).to(torch.int8)
        return quantized.double() * scale, value.numel() + 8
    raise ValueError(codec)


def compressed_aggregate(
    train_x: np.ndarray,
    train_y: np.ndarray,
    assignments: np.ndarray,
    clients: int,
    codecs: tuple[str, ...],
) -> tuple[dict[str, RidgePacket], dict[str, int]]:
    dimension = train_x.shape[1]
    classes = train_y.shape[1]
    values: dict[str, dict[str, torch.Tensor | int]] = {
        codec: {
            "count": 0,
            "sum_x": torch.zeros(dimension, dtype=torch.float64),
            "sum_y": torch.zeros(classes, dtype=torch.float64),
            "xtx": torch.zeros((dimension, dimension), dtype=torch.float64),
            "xty": torch.zeros((dimension, classes), dtype=torch.float64),
        }
        for codec in codecs
    }
    payload = {codec: 0 for codec in codecs}
    for client in range(clients):
        indices = assignments == client
        local = packet(train_x[indices], train_y[indices])
        for codec in codecs:
            values[codec]["count"] = int(values[codec]["count"]) + local.count
            payload[codec] += 8
            for field in ("sum_x", "sum_y", "xtx", "xty"):
                decoded, encoded_bytes = quantize_tensor(getattr(local, field), codec)
                values[codec][field] = values[codec][field] + decoded
                payload[codec] += encoded_bytes
        print(f"COMPRESS_PACKET client={client + 1}/{clients}", flush=True)
    packets = {
        codec: RidgePacket(
            count=int(value["count"]),
            sum_x=value["sum_x"],
            sum_y=value["sum_y"],
            xtx=value["xtx"],
            xty=value["xty"],
        )
        for codec, value in values.items()
    }
    return packets, payload


def run_compression(datasets: list[str]) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    codecs = ("fp64", "fp32", "fp16", "int8")
    report: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    for dataset in datasets:
        started = perf_counter()
        benchmark = BENCHMARKS[dataset]
        dataset_root = ROOT / "outputs" / "icme" / dataset
        cache_root = resolve_cache(dataset)
        train_x = load_features(cache_root, "train")
        test_x = load_features(cache_root, "test")
        train_y = np.load(cache_root / "train_targets.npy")
        test_y = np.load(cache_root / "test_targets.npy")
        clients = max(2, round(benchmark.class_count * 0.6))
        assignments = np.load(dataset_root / "partitions" / f"clients{clients}_seed17.npy")
        alpha = float(
            json.loads(
                (dataset_root / "full_seeds" / "seed17" / "ridge" / "result.json").read_text(
                    encoding="utf-8"
                )
            )["selection"]["alpha"]
        )
        packets, payload = compressed_aggregate(
            train_x, train_y, assignments, clients, codecs
        )
        rows: list[dict[str, object]] = []
        reference_scores: np.ndarray | None = None
        reference_map: float | None = None
        reference_bytes = payload["fp64"]
        for codec in codecs:
            adapter = fit_ridge_from_packets((packets[codec],), regularization=alpha)
            scores = predict(test_x, adapter)
            value, _ = mean_average_precision(test_y, scores)
            if codec == "fp64":
                reference_scores = scores
                reference_map = value
            assert reference_scores is not None and reference_map is not None
            row = {
                "dataset": dataset,
                "codec": codec,
                "clients": clients,
                "partition_seed": 17,
                "selected_alpha": alpha,
                "communication_MiB": payload[codec] / 1024**2,
                "compression_ratio_vs_fp64": reference_bytes / payload[codec],
                "mAP": value,
                "mAP_delta_vs_fp64": value - reference_map,
                "max_abs_score_error_vs_fp64": float(np.max(np.abs(scores - reference_scores))),
                "individual_examples_transmitted": False,
                "test_labels_used_for_selection": False,
            }
            rows.append(row)
            csv_rows.append(row)
            print(f"COMPRESSION {json.dumps(row)}", flush=True)
        report[dataset] = {
            "protocol": {
                "scope": "client-to-server ridge sufficient-statistic packets at t=60%, seed 17",
                "int8": "symmetric per-client per-tensor max-absolute scaling with one float64 scale",
                "aggregation": "all decoded packets are accumulated in float64 at the server",
                "test_labels_used_for_selection": False,
            },
            "rows": rows,
            "runtime_seconds": perf_counter() - started,
        }
        atomic_json(paper_root / "communication_compression.json", report)
    write_csv(paper_root / "communication_compression.csv", csv_rows)
    return report


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def run_tail_analysis(datasets: list[str]) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    all_rows: list[dict[str, object]] = []
    report: dict[str, object] = {}
    figure, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8), squeeze=False)
    for panel, dataset in enumerate(datasets):
        benchmark = BENCHMARKS[dataset]
        cache_root = resolve_cache(dataset)
        dataset_root = ROOT / "outputs" / "icme" / dataset
        train_y = np.load(cache_root / "train_targets.npy")
        test_y = np.load(cache_root / "test_targets.npy")
        final_scores = np.load(dataset_root / "cv_fusion" / "test_scores.npy")
        qafed_scores = np.load(dataset_root / "scores" / "qafedpasa_representative_scores.npy")
        full_map, full_ap = mean_average_precision(test_y, final_scores)
        qafed_map, qafed_ap = mean_average_precision(test_y, qafed_scores)
        positives = train_y.sum(axis=0).astype(int)
        test_positives = test_y.sum(axis=0).astype(int)
        boundaries = np.quantile(positives, (0.25, 0.5, 0.75))
        rows: list[dict[str, object]] = []
        for class_id, name in enumerate(benchmark.classes):
            quartile = int(np.searchsorted(boundaries, positives[class_id], side="right") + 1)
            row = {
                "dataset": dataset,
                "class_id": class_id,
                "class_name": name,
                "train_positives": int(positives[class_id]),
                "test_positives": int(test_positives[class_id]),
                "prevalence_quartile": quartile,
                "qafedpasa_AP": float(qafed_ap[class_id]),
                "fedpasa_full_AP": float(full_ap[class_id]),
                "full_gain_over_qafedpasa": float(full_ap[class_id] - qafed_ap[class_id]),
            }
            rows.append(row)
            all_rows.append(row)
        rare = [row for row in rows if row["prevalence_quartile"] == 1]
        common = [row for row in rows if row["prevalence_quartile"] == 4]
        report[dataset] = {
            "full_mAP": full_map,
            "qafedpasa_mAP": qafed_map,
            "spearman_log_train_prevalence_vs_full_AP": spearman(
                np.log1p(positives), full_ap
            ),
            "rare_quartile_full_AP_mean": float(np.mean([row["fedpasa_full_AP"] for row in rare])),
            "common_quartile_full_AP_mean": float(
                np.mean([row["fedpasa_full_AP"] for row in common])
            ),
            "classes_hurt_by_fusion": int(
                np.sum(np.asarray([row["full_gain_over_qafedpasa"] for row in rows]) < 0)
            ),
            "five_lowest_AP": sorted(rows, key=lambda row: row["fedpasa_full_AP"])[:5],
            "five_largest_gains": sorted(
                rows, key=lambda row: row["full_gain_over_qafedpasa"], reverse=True
            )[:5],
            "five_largest_losses": sorted(
                rows, key=lambda row: row["full_gain_over_qafedpasa"]
            )[:5],
            "test_labels_used_for_selection": False,
        }
        ax = axes[0, panel]
        scatter = ax.scatter(
            positives,
            full_ap,
            c=np.asarray([row["full_gain_over_qafedpasa"] for row in rows]),
            cmap="coolwarm",
            vmin=-max(1.0, np.max(np.abs(full_ap - qafed_ap))),
            vmax=max(1.0, np.max(np.abs(full_ap - qafed_ap))),
            s=24,
            alpha=0.8,
        )
        ax.set_xscale("log")
        ax.set_title(dataset.upper())
        ax.set_xlabel("Training positives (log scale)")
        if panel == 0:
            ax.set_ylabel("FedPASA-Full per-class AP")
        ax.grid(alpha=0.25)
        figure.colorbar(scatter, ax=ax, label="AP gain over QAFedPASA")
    figure.tight_layout()
    figure_root = paper_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    figure.savefig(figure_root / "tail_failure_analysis.png", dpi=220, bbox_inches="tight")
    figure.savefig(figure_root / "tail_failure_analysis.pdf", bbox_inches="tight")
    plt.close(figure)
    write_csv(paper_root / "per_class_failure_analysis.csv", all_rows)
    atomic_json(paper_root / "per_class_failure_analysis.json", report)
    return report


def plot_summary() -> None:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    figure_root = paper_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    noise_path = paper_root / "label_noise.json"
    compression_path = paper_root / "communication_compression.json"
    if noise_path.exists():
        report = json.loads(noise_path.read_text(encoding="utf-8"))
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        for dataset, payload in report.items():
            summary = payload["summary"]
            ax.errorbar(
                [row["pair_noise_percent"] for row in summary],
                [row["classwise_fusion_mAP_mean"] for row in summary],
                yerr=[row["classwise_fusion_mAP_std"] for row in summary],
                marker="o",
                capsize=3,
                label=dataset.upper(),
            )
        ax.set_xlabel("Balanced pair-label noise (%)")
        ax.set_ylabel("FedPASA-Full mAP")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(figure_root / "label_noise_robustness.png", dpi=220)
        figure.savefig(figure_root / "label_noise_robustness.pdf", bbox_inches="tight")
        plt.close(figure)
    if compression_path.exists():
        report = json.loads(compression_path.read_text(encoding="utf-8"))
        figure, ax = plt.subplots(figsize=(6.4, 4.2))
        codecs = ("fp64", "fp32", "fp16", "int8")
        for dataset, payload in report.items():
            by_codec = {row["codec"]: row for row in payload["rows"]}
            ax.plot(
                codecs,
                [by_codec[codec]["mAP_delta_vs_fp64"] for codec in codecs],
                marker="o",
                label=dataset.upper(),
            )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xlabel("Sufficient-statistic packet codec")
        ax.set_ylabel("mAP change vs FP64")
        ax.grid(alpha=0.25)
        ax.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(figure_root / "communication_compression.png", dpi=220)
        figure.savefig(figure_root / "communication_compression.pdf", bbox_inches="tight")
        plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--studies", nargs="+", choices=("label_noise", "compression", "tail", "plot"), default=["label_noise", "compression", "tail"]
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--noise-rates", type=float, nargs="+", default=[0.0, 0.05, 0.1, 0.2])
    args = parser.parse_args()
    for dataset in args.datasets:
        if dataset not in BENCHMARKS:
            raise ValueError(f"unknown dataset: {dataset}")
    if "label_noise" in args.studies:
        run_label_noise(args.datasets, args.seeds, args.noise_rates)
    if "compression" in args.studies:
        run_compression(args.datasets)
    if "tail" in args.studies:
        run_tail_analysis(args.datasets)
    plot_summary()


if __name__ == "__main__":
    main()
