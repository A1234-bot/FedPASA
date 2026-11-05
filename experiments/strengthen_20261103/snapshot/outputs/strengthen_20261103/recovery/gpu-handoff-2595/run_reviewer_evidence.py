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
import torch.nn.functional as F


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS  # noqa: E402
from fedpasa.core import (  # noqa: E402
    Adapter,
    PatchBank,
    _gda_from_samples,
    _global_quantile_selection,
    _stack_banks,
    build_client_banks,
    fit_central_piaa,
    fit_quantile_aligned_fedpasa,
    mean_average_precision,
)
from fedpasa.supervised import fit_ridge_from_packets, make_ridge_packet  # noqa: E402
from run_icme_experiments import (  # noqa: E402
    assignments_for,
    deserialize_bank,
    dirichlet_assignments,
    partition_features,
    truncate_bank,
)
from run_label_patch_fusion import piaa_scores  # noqa: E402


DEFAULT_DATASETS = ("voc07", "voc12", "coco", "nus")
FEATURES = ("cls", "patch_mean", "patch_max_text")


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_features(cache_root: Path, split: str) -> np.ndarray:
    arrays = [
        np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32)
        for name in FEATURES
    ]
    return np.concatenate(arrays, axis=1)


def predict(features: np.ndarray, adapter: Adapter) -> np.ndarray:
    normalized = np.asarray(features, dtype=np.float32)
    normalized = normalized / np.linalg.norm(normalized, axis=1, keepdims=True).clip(min=1e-12)
    return np.asarray(
        normalized @ adapter.weight.numpy() + adapter.bias.numpy(), dtype=np.float32
    )


def union_banks(banks: tuple[dict[int, PatchBank], ...]) -> dict[int, PatchBank]:
    output: dict[int, PatchBank] = {}
    classes = sorted({class_id for bank in banks for class_id in bank})
    for class_id in classes:
        values = [bank[class_id] for bank in banks if class_id in bank]
        origins = (
            torch.cat([value.image_indices for value in values])
            if all(value.image_indices is not None for value in values)
            else None
        )
        output[class_id] = PatchBank(
            torch.cat([value.features for value in values]),
            torch.cat([value.probabilities for value in values]),
            torch.cat([value.entropy for value in values]),
            origins,
        )
    return output


def quantile_selected_bank(
    banks: tuple[dict[int, PatchBank], ...], k: int
) -> dict[int, PatchBank]:
    selections = _global_quantile_selection(banks, k)
    output: dict[int, PatchBank] = {}
    for bank, selection in zip(banks, selections):
        for class_id, indices in selection.items():
            value = bank[class_id]
            selected = PatchBank(
                value.features[indices],
                value.probabilities[indices],
                value.entropy[indices],
                None if value.image_indices is None else value.image_indices[indices],
            )
            if class_id not in output:
                output[class_id] = selected
                continue
            current = output[class_id]
            origins = (
                torch.cat((current.image_indices, selected.image_indices))
                if current.image_indices is not None and selected.image_indices is not None
                else None
            )
            output[class_id] = PatchBank(
                torch.cat((current.features, selected.features)),
                torch.cat((current.probabilities, selected.probabilities)),
                torch.cat((current.entropy, selected.entropy)),
                origins,
            )
    return output


def bank_keys(bank: dict[int, PatchBank]) -> set[tuple[int, int, int, int, int]]:
    keys: set[tuple[int, int, int, int, int]] = set()
    for class_id, value in bank.items():
        if value.image_indices is None:
            raise ValueError("bank overlap requires source-image indices")
        origins = value.image_indices.numpy().astype(np.int64, copy=False)
        entropy = value.entropy.numpy().astype(np.float32, copy=False).view(np.uint32)
        feature_bits = (
            value.features[:, :2].numpy().astype(np.float32, copy=False).view(np.uint32).reshape(-1, 2)
        )
        keys.update(
            (
                int(class_id),
                int(origin),
                int(entropy_value),
                int(bits[0]),
                int(bits[1]),
            )
            for origin, entropy_value, bits in zip(origins, entropy, feature_bits)
        )
    return keys


def bank_overlap(
    reference: dict[int, PatchBank], candidate: dict[int, PatchBank]
) -> dict[str, float | int]:
    reference_keys = bank_keys(reference)
    candidate_keys = bank_keys(candidate)
    intersection = len(reference_keys & candidate_keys)
    union = len(reference_keys | candidate_keys)
    return {
        "reference_patches": len(reference_keys),
        "candidate_patches": len(candidate_keys),
        "intersection_patches": intersection,
        "central_topk_recall": intersection / max(len(reference_keys), 1),
        "bank_mismatch_ratio": 1.0 - intersection / max(union, 1),
    }


def average_adapters(
    adapters: list[Adapter], weights: list[float], method: str
) -> Adapter:
    normalized = torch.as_tensor(weights, dtype=torch.float64)
    normalized = normalized / normalized.sum()
    weight = sum(
        factor * adapter.weight.double() for factor, adapter in zip(normalized, adapters)
    )
    bias = sum(
        factor * adapter.bias.double() for factor, adapter in zip(normalized, adapters)
    )
    return Adapter(
        weight.float(),
        bias.float(),
        {"method": method, "clients": len(adapters), "aggregation": "sample-weighted"},
    )


def fit_tie_safe_piaa(
    bank: dict[int, PatchBank],
    text_features: torch.Tensor,
    *,
    purification_temperature: float,
) -> Adapter:
    features, probabilities, labels = _stack_banks((bank,))
    preliminary_weight, preliminary_bias, _ = _gda_from_samples(
        features, probabilities, labels, text_features
    )
    logits = features @ preliminary_weight + preliminary_bias
    refined = F.softmax(logits * purification_temperature, dim=-1)
    predictions = logits.argmax(dim=-1)
    keep_parts: list[torch.Tensor] = []
    clamps = 0
    for class_id in predictions.unique().tolist():
        indices = torch.nonzero(predictions == class_id, as_tuple=False).flatten()
        confidence = refined[indices, class_id]
        deviation = (
            confidence.std(unbiased=True)
            if confidence.numel() > 1
            else confidence.new_zeros(())
        )
        raw_threshold = confidence.mean() + deviation
        threshold = torch.minimum(raw_threshold, confidence.max())
        clamps += int(raw_threshold > confidence.max())
        keep_parts.append(indices[confidence >= threshold])
    keep = torch.cat(keep_parts)
    weight, bias, _ = _gda_from_samples(
        features[keep], refined[keep], predictions[keep], text_features
    )
    return Adapter(
        weight.float(),
        bias.float(),
        {
            "method": "central_piaa_tie_safe_for_naive_control",
            "selected_patches": int(len(keep)),
            "raw_patches": int(len(features)),
            "threshold_clamps": clamps,
        },
    )


def adapter_errors(reference: Adapter, candidate: Adapter) -> dict[str, float]:
    weight_delta = candidate.weight.double() - reference.weight.double()
    bias_delta = candidate.bias.double() - reference.bias.double()
    return {
        "max_abs_weight_error": float(weight_delta.abs().max()),
        "relative_weight_l2_error": float(
            torch.linalg.vector_norm(weight_delta)
            / torch.linalg.vector_norm(reference.weight.double()).clamp_min(1e-12)
        ),
        "max_abs_bias_error": float(bias_delta.abs().max()),
    }


def score_errors(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    return {
        "max_abs_score_error": float(np.abs(delta).max()),
        "relative_score_l2_error": float(
            np.linalg.norm(delta) / max(np.linalg.norm(reference), 1e-12)
        ),
    }


def evaluate_piaa(
    dataset: str,
    cache_root: Path,
    output_path: Path,
    text: torch.Tensor,
    adapter: Adapter,
    *,
    batch_size: int,
    device: str,
) -> tuple[np.ndarray, float]:
    if output_path.exists():
        return np.load(output_path), 0.0
    benchmark = BENCHMARKS[dataset]
    targets = np.load(cache_root / "test_targets.npy", mmap_mode="r")
    started = perf_counter()
    scores = piaa_scores(
        cache_root / "test_patch.npy",
        cache_root / "test_cls.npy",
        np.arange(len(targets), dtype=np.int64),
        text,
        adapter.weight,
        adapter.bias,
        cls_scale=benchmark.cls_scale,
        patch_scale=benchmark.patch_scale,
        batch_images=batch_size,
        device=device,
    )
    seconds = perf_counter() - started
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, scores)
    return scores, seconds


def condition_name(partition: str, beta: float | None) -> str:
    return "kmeans" if partition == "kmeans" else f"dirichlet_beta_{beta:g}"


def estimator_condition(
    dataset: str,
    partition: str,
    beta: float | None,
    seed: int,
    *,
    clients: int,
    k: int,
    batch_size: int,
    device: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    benchmark = BENCHMARKS[dataset]
    cache_root = resolve_cache(dataset)
    dataset_root = ROOT / "outputs" / "icme" / dataset
    condition = condition_name(partition, beta)
    score_root = dataset_root / "reviewer" / "estimator_preserving" / f"{condition}_seed{seed}"
    targets = np.load(cache_root / "train_targets.npy")
    test_targets = np.load(cache_root / "test_targets.npy")
    text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
    if partition == "kmeans":
        features = partition_features(
            cache_root,
            text,
            batch_images=max(batch_size, 256),
            device=device,
        )
        assignments, client_sizes, partition_seconds = assignments_for(
            features,
            dataset_root,
            clients=clients,
            seed=seed,
            n_init=10,
        )
    else:
        if beta is None:
            raise ValueError("Dirichlet partition requires beta")
        assignments, client_sizes, partition_seconds = dirichlet_assignments(
            targets,
            dataset_root,
            clients=clients,
            beta=beta,
            seed=seed,
        )
    large_global_path = dataset_root / "banks" / "global_k1024.pt"
    global_bank = truncate_bank(
        deserialize_bank(torch.load(large_global_path, map_location="cpu", weights_only=True)), k
    )
    central_piaa_started = perf_counter()
    central_piaa = fit_central_piaa(
        global_bank,
        text,
        purification_temperature=benchmark.purification_temperature,
    )
    central_piaa_seconds = perf_counter() - central_piaa_started
    central_piaa_scores = np.load(dataset_root / "scores" / "central_piaa_scores.npy")
    wall_started = perf_counter()
    local_bank_started = perf_counter()
    local_banks = build_client_banks(
        cache_root / "train_patch.npy",
        assignments,
        text,
        clients=clients,
        k=k,
        batch_images=batch_size,
        device=device,
    )
    local_bank_seconds = perf_counter() - local_bank_started
    local_union = union_banks(local_banks)
    aligned_bank = quantile_selected_bank(local_banks, k)
    union_overlap = bank_overlap(global_bank, local_union)
    aligned_overlap = bank_overlap(global_bank, aligned_bank)

    started = perf_counter()
    naive_union = fit_tie_safe_piaa(
        local_union,
        text,
        purification_temperature=benchmark.purification_temperature,
    )
    naive_union_fit_seconds = perf_counter() - started

    started = perf_counter()
    local_piaa = [
        fit_tie_safe_piaa(
            bank,
            text,
            purification_temperature=benchmark.purification_temperature,
        )
        for bank in local_banks
    ]
    naive_local_gda = average_adapters(
        local_piaa,
        [float(adapter.metadata["selected_patches"]) for adapter in local_piaa],
        "naive_local_gda_parameter_average",
    )
    naive_local_gda_fit_seconds = perf_counter() - started

    started = perf_counter()
    fedpasa = fit_quantile_aligned_fedpasa(
        local_banks,
        text,
        k=k,
        purification_temperature=benchmark.purification_temperature,
    )
    fedpasa_fit_seconds = perf_counter() - started

    train_features = load_features(cache_root, "train")
    test_features = load_features(cache_root, "test")
    ridge_alpha = float(
        json.loads(
            (dataset_root / "full_seeds" / "seed7" / "ridge" / "result.json").read_text(
                encoding="utf-8"
            )
        )["selection"]["alpha"]
    )
    started = perf_counter()
    ridge_packets = tuple(
        make_ridge_packet(train_features[assignments == client_id], targets[assignments == client_id])
        for client_id in range(clients)
    )
    ridge_packet_seconds = perf_counter() - started
    started = perf_counter()
    central_ridge = fit_ridge_from_packets(ridge_packets, regularization=ridge_alpha)
    central_ridge_fit_seconds = perf_counter() - started
    central_ridge_scores = predict(test_features, central_ridge)
    started = perf_counter()
    local_ridge = [
        fit_ridge_from_packets((packet,), regularization=ridge_alpha) for packet in ridge_packets
    ]
    naive_local_ridge = average_adapters(
        local_ridge,
        [float(packet.count) for packet in ridge_packets],
        "naive_local_ridge_parameter_average",
    )
    naive_local_ridge_fit_seconds = perf_counter() - started
    naive_local_ridge_scores = predict(test_features, naive_local_ridge)

    naive_union_scores, naive_union_eval_seconds = evaluate_piaa(
        dataset,
        cache_root,
        score_root / "naive_a_local_union.npy",
        text,
        naive_union,
        batch_size=batch_size,
        device=device,
    )
    naive_local_gda_scores, naive_local_gda_eval_seconds = evaluate_piaa(
        dataset,
        cache_root,
        score_root / "naive_b_local_gda_average.npy",
        text,
        naive_local_gda,
        batch_size=batch_size,
        device=device,
    )
    fedpasa_adapter_error = adapter_errors(central_piaa, fedpasa)
    if max(
        fedpasa_adapter_error["max_abs_weight_error"],
        fedpasa_adapter_error["max_abs_bias_error"],
    ) <= 1e-5:
        fedpasa_scores = central_piaa_scores
        fedpasa_eval_seconds = 0.0
        fedpasa_score_provenance = "reused centralized scores after adapter equality check"
    else:
        fedpasa_scores, fedpasa_eval_seconds = evaluate_piaa(
            dataset,
            cache_root,
            score_root / "fedpasa_quantile_aligned.npy",
            text,
            fedpasa,
            batch_size=batch_size,
            device=device,
        )
        fedpasa_score_provenance = "direct full-test evaluation"

    rows: list[dict[str, object]] = []

    def append_row(
        method: str,
        branch: str,
        reference_name: str,
        adapter: Adapter,
        scores: np.ndarray,
        reference_adapter: Adapter,
        reference_scores: np.ndarray,
        overlap: dict[str, float | int] | None,
        fit_seconds: float,
        evaluation_seconds: float,
        score_provenance: str,
    ) -> None:
        value, _ = mean_average_precision(test_targets, scores)
        row: dict[str, object] = {
            "dataset": dataset,
            "partition": condition,
            "seed": seed,
            "clients": clients,
            "method": method,
            "branch": branch,
            "reference_estimator": reference_name,
            "mAP": value,
            **adapter_errors(reference_adapter, adapter),
            **score_errors(reference_scores, scores),
            "fit_seconds": fit_seconds,
            "evaluation_seconds": evaluation_seconds,
            "score_provenance": score_provenance,
        }
        if overlap is None:
            row.update(
                {
                    "reference_patches": None,
                    "candidate_patches": None,
                    "central_topk_recall": None,
                    "bank_mismatch_ratio": None,
                }
            )
        else:
            row.update(overlap)
        rows.append(row)

    append_row(
        "Centralized",
        "PIAA",
        "centralized_global_topk_pooled_solve",
        central_piaa,
        central_piaa_scores,
        central_piaa,
        central_piaa_scores,
        bank_overlap(global_bank, global_bank),
        central_piaa_seconds,
        0.0,
        "existing centralized full-test scores",
    )
    append_row(
        "Naive-A local-TopK union",
        "PIAA",
        "centralized_global_topk_pooled_solve",
        naive_union,
        naive_union_scores,
        central_piaa,
        central_piaa_scores,
        union_overlap,
        naive_union_fit_seconds,
        naive_union_eval_seconds,
        "direct full-test evaluation",
    )
    append_row(
        "Naive-B local-GDA average",
        "PIAA",
        "centralized_global_topk_pooled_solve",
        naive_local_gda,
        naive_local_gda_scores,
        central_piaa,
        central_piaa_scores,
        union_overlap,
        naive_local_gda_fit_seconds,
        naive_local_gda_eval_seconds,
        "direct full-test evaluation",
    )
    append_row(
        "FedPASA quantile-aligned",
        "PIAA",
        "centralized_global_topk_pooled_solve",
        fedpasa,
        fedpasa_scores,
        central_piaa,
        central_piaa_scores,
        aligned_overlap,
        fedpasa_fit_seconds,
        fedpasa_eval_seconds,
        fedpasa_score_provenance,
    )
    append_row(
        "Centralized Ridge",
        "Ridge",
        "centralized_ridge_pooled_solve",
        central_ridge,
        central_ridge_scores,
        central_ridge,
        central_ridge_scores,
        None,
        central_ridge_fit_seconds,
        0.0,
        "direct frozen-feature evaluation",
    )
    append_row(
        "Naive-C local-Ridge average",
        "Ridge",
        "centralized_ridge_pooled_solve",
        naive_local_ridge,
        naive_local_ridge_scores,
        central_ridge,
        central_ridge_scores,
        None,
        naive_local_ridge_fit_seconds,
        0.0,
        "direct frozen-feature evaluation",
    )
    metadata = {
        "dataset": dataset,
        "partition": condition,
        "seed": seed,
        "clients": clients,
        "client_sizes": client_sizes,
        "partition_seconds": partition_seconds,
        "client_bank_seconds_sequential_simulation": local_bank_seconds,
        "client_ridge_packet_seconds_sequential_simulation": ridge_packet_seconds,
        "condition_wall_clock_seconds": perf_counter() - wall_started,
        "test_labels_used_for_selection": False,
        "ridge_alpha_source": "seed-7 training-side validation",
    }
    del local_banks, local_union, aligned_bank, train_features, test_features
    torch.cuda.empty_cache()
    return rows, metadata


def summarize_estimator(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    keys = sorted({(row["dataset"], row["partition"], row["method"]) for row in rows})
    for dataset, partition, method in keys:
        selected = [
            row
            for row in rows
            if row["dataset"] == dataset
            and row["partition"] == partition
            and row["method"] == method
        ]
        summary.append(
            {
                "dataset": dataset,
                "partition": partition,
                "method": method,
                "seeds": "/".join(str(row["seed"]) for row in selected),
                "mAP_mean": float(np.mean([row["mAP"] for row in selected])),
                "mAP_std": float(np.std([row["mAP"] for row in selected], ddof=1))
                if len(selected) > 1
                else 0.0,
                "max_abs_score_error_mean": float(
                    np.mean([row["max_abs_score_error"] for row in selected])
                ),
                "relative_weight_l2_error_mean": float(
                    np.mean([row["relative_weight_l2_error"] for row in selected])
                ),
                "central_topk_recall_mean": float(
                    np.mean(
                        [
                            row["central_topk_recall"]
                            for row in selected
                            if row["central_topk_recall"] is not None
                        ]
                    )
                )
                if selected[0]["central_topk_recall"] is not None
                else None,
                "bank_mismatch_ratio_mean": float(
                    np.mean(
                        [
                            row["bank_mismatch_ratio"]
                            for row in selected
                            if row["bank_mismatch_ratio"] is not None
                        ]
                    )
                )
                if selected[0]["bank_mismatch_ratio"] is not None
                else None,
            }
        )
    return summary


def run_estimator(args: argparse.Namespace) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    report_path = paper_root / "estimator_preserving_evidence.json"
    rows: list[dict[str, object]] = []
    conditions: list[dict[str, object]] = []
    if args.resume and report_path.exists():
        previous = json.loads(report_path.read_text(encoding="utf-8"))
        rows = list(previous.get("rows", []))
        conditions = list(previous.get("conditions", []))
    completed = {
        (row["dataset"], row["partition"], int(row["seed"]))
        for row in rows
        if row["method"] == "FedPASA quantile-aligned"
    }
    requested = [("dirichlet", beta) for beta in args.betas]
    if args.include_kmeans:
        requested.append(("kmeans", None))
    for dataset in args.datasets:
        for partition, beta in requested:
            for seed in args.seeds:
                key = (dataset, condition_name(partition, beta), seed)
                if key in completed:
                    print(f"ESTIMATOR_SKIP dataset={dataset} condition={key[1]} seed={seed}", flush=True)
                    continue
                condition_rows, metadata = estimator_condition(
                    dataset,
                    partition,
                    beta,
                    seed,
                    clients=args.clients,
                    k=args.k,
                    batch_size=args.batch_size,
                    device=args.device,
                )
                rows.extend(condition_rows)
                conditions.append(metadata)
                payload = {
                    "protocol": {
                        "backbone": "CLIP ViT-B/16 frozen caches",
                        "clients": args.clients,
                        "global_top_k_per_class": args.k,
                        "partitions": [condition_name(name, value) for name, value in requested],
                        "seeds": args.seeds,
                        "naive_parameter_averaging": "sample-weighted",
                        "naive_purification_safeguard": "mean+std threshold is clamped to the observed class maximum so saturated ties retain evidence",
                        "test_labels_used_for_selection": False,
                    },
                    "rows": rows,
                    "conditions": conditions,
                    "summary": summarize_estimator(rows),
                }
                atomic_json(report_path, payload)
                print(
                    f"ESTIMATOR_COMPLETE dataset={dataset} condition={key[1]} seed={seed}",
                    flush=True,
                )
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    write_csv(paper_root / "estimator_preserving_evidence.csv", payload["rows"])
    write_csv(paper_root / "estimator_preserving_summary.csv", payload["summary"])
    return payload


def run_complementarity(datasets: list[str], tolerance: float) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    report: dict[str, object] = {
        "protocol": {
            "full": "mean of three leakage-safe classwise-fusion score matrices",
            "ridge": "mean of the matching three ridge-only score matrices",
            "global_fusion": "mean of the matching three globally fused score matrices",
            "approximately_equal_tolerance_AP_points": tolerance,
            "test_labels_used_for_selection": False,
        },
        "datasets": {},
    }
    per_class_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for dataset in datasets:
        benchmark = BENCHMARKS[dataset]
        dataset_root = ROOT / "outputs" / "icme" / dataset
        cache_root = resolve_cache(dataset)
        targets = np.load(cache_root / "test_targets.npy")
        ridge_scores = np.mean(
            [
                np.load(dataset_root / "full_seeds" / f"seed{seed}" / "ridge" / "test_scores.npy")
                for seed in (7, 17, 27)
            ],
            axis=0,
        )
        full_scores = np.load(dataset_root / "cv_fusion" / "test_scores.npy")
        global_scores = np.mean(
            [
                np.load(
                    dataset_root
                    / "full_seeds"
                    / f"seed{seed}"
                    / "ridge_patch_fusion"
                    / "test_scores_global.npy"
                )
                for seed in (7, 17, 27)
            ],
            axis=0,
        )
        ridge_map, ridge_ap = mean_average_precision(targets, ridge_scores)
        full_map, full_ap = mean_average_precision(targets, full_scores)
        global_map, global_ap = mean_average_precision(targets, global_scores)
        differences = full_ap - ridge_ap
        categories = np.where(
            differences > tolerance,
            "Full > Ridge",
            np.where(differences < -tolerance, "Full < Ridge", "Full ≈ Ridge"),
        )
        positive = differences[differences > tolerance]
        summary = {
            "dataset": dataset,
            "classes": benchmark.class_count,
            "full_gt_ridge": int(np.sum(categories == "Full > Ridge")),
            "full_approx_ridge": int(np.sum(categories == "Full ≈ Ridge")),
            "full_lt_ridge": int(np.sum(categories == "Full < Ridge")),
            "average_positive_gain_AP": float(positive.mean()) if len(positive) else 0.0,
            "ridge_only_mAP": ridge_map,
            "global_fusion_mAP": global_map,
            "classwise_full_mAP": full_map,
            "classwise_gain_over_global_mAP": full_map - global_map,
            "classwise_gain_over_ridge_mAP": full_map - ridge_map,
        }
        summary_rows.append(summary)
        rows: list[dict[str, object]] = []
        for class_id, class_name in enumerate(benchmark.classes):
            row = {
                "dataset": dataset,
                "class_id": class_id,
                "class_name": class_name,
                "ridge_AP": float(ridge_ap[class_id]),
                "global_fusion_AP": float(global_ap[class_id]),
                "classwise_full_AP": float(full_ap[class_id]),
                "full_minus_ridge_AP": float(differences[class_id]),
                "full_minus_global_AP": float(full_ap[class_id] - global_ap[class_id]),
                "category": str(categories[class_id]),
            }
            rows.append(row)
            per_class_rows.append(row)
        report["datasets"][dataset] = {"summary": summary, "per_class": rows}
    atomic_json(paper_root / "classwise_complementarity.json", report)
    write_csv(paper_root / "classwise_complementarity.csv", per_class_rows)
    write_csv(paper_root / "classwise_complementarity_summary.csv", summary_rows)
    return report


def communication_components(
    clients: int, classes: int, ridge_dimension: int, *, k: int, piaa_dimension: int = 512
) -> dict[str, float | int]:
    quantile_bytes = 4 * clients * classes * k
    moment_floats = (
        piaa_dimension * piaa_dimension
        + 2 * classes * piaa_dimension
        + 2 * classes
    )
    preliminary_gda_bytes = 4 * clients * moment_floats
    purification_moment_bytes = 4 * clients * 3 * classes
    final_gda_bytes = 4 * clients * moment_floats
    gda_bytes = preliminary_gda_bytes + purification_moment_bytes + final_gda_bytes
    ridge_floats = (
        ridge_dimension * ridge_dimension
        + ridge_dimension * classes
        + ridge_dimension
        + classes
        + 1
    )
    ridge_bytes = 8 * clients * ridge_floats
    return {
        "quantile_MiB": quantile_bytes / 1024**2,
        "preliminary_gda_moments_MiB": preliminary_gda_bytes / 1024**2,
        "purification_moments_MiB": purification_moment_bytes / 1024**2,
        "final_gda_moments_MiB": final_gda_bytes / 1024**2,
        "gda_moments_MiB": gda_bytes / 1024**2,
        "ridge_stats_MiB": ridge_bytes / 1024**2,
        "total_MiB": (quantile_bytes + gda_bytes + ridge_bytes) / 1024**2,
        "qafedpasa_client_uploads": 4,
        "communication_scope": "client-to-server uploads only, dense FP32 upper bound",
        "ridge_upload_rounds": 1,
    }


def run_communication(
    estimator_report: dict[str, object] | None, datasets: list[str], k: int
) -> dict[str, object]:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    verification = json.loads(
        (paper_root / "federated_full_verification.json").read_text(encoding="utf-8")
    )
    runtime_lookup: dict[str, dict[str, object]] = {}
    if estimator_report is not None:
        for condition in estimator_report.get("conditions", []):
            if condition["partition"] == "kmeans" and int(condition["seed"]) == 7:
                runtime_lookup[str(condition["dataset"])] = condition
    rows: list[dict[str, object]] = []
    for source in verification["rows"]:
        dataset = source["dataset"]
        if dataset not in datasets:
            continue
        cache_root = resolve_cache(dataset)
        ridge_dimension = sum(
            np.load(cache_root / f"train_{name}.npy", mmap_mode="r").shape[1]
            for name in FEATURES
        )
        components = communication_components(
            int(source["clients"]),
            BENCHMARKS[dataset].class_count,
            ridge_dimension,
            k=k,
        )
        runtime = runtime_lookup.get(dataset, {})
        row = {
            "dataset": dataset,
            "clients": source["clients"],
            "t_percent": source["t_percent"],
            **components,
            "reported_total_MiB": source["combined_communication_MiB"],
            "breakdown_minus_reported_MiB": components["total_MiB"]
            - source["combined_communication_MiB"],
            "runtime_protocol": runtime.get("partition"),
            "runtime_clients": runtime.get("clients"),
            "client_bank_seconds_sequential": runtime.get(
                "client_bank_seconds_sequential_simulation"
            ),
            "client_ridge_packet_seconds_sequential": runtime.get(
                "client_ridge_packet_seconds_sequential_simulation"
            ),
            "total_condition_wall_clock_seconds": runtime.get(
                "condition_wall_clock_seconds"
            ),
        }
        rows.append(row)
    report = {
        "protocol": {
            "communication": "exact client-to-server payload dimensions for the t=60%, seed-17 recovery table",
            "runtime": "measured from the 10-client KMeans estimator-evidence condition when available",
            "rounds": "four QAFedPASA client uploads plus one Ridge-statistic upload; q-moments require a separate upload after the preliminary global GDA is returned",
        },
        "rows": rows,
    }
    atomic_json(paper_root / "communication_runtime_breakdown.json", report)
    write_csv(paper_root / "communication_runtime_breakdown.csv", rows)
    return report


def plot_estimator(report: dict[str, object]) -> None:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    figure_root = paper_root / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    datasets = [dataset for dataset in ("coco", "nus") if dataset in report["protocol"].get("datasets", [])]
    if not datasets:
        datasets = sorted({row["dataset"] for row in summary})[-2:]
    order = ["dirichlet_beta_0.1", "dirichlet_beta_0.3", "dirichlet_beta_0.5", "kmeans"]
    labels = [r"$\beta$=0.1", r"$\beta$=0.3", r"$\beta$=0.5", "KMeans"]
    figure, axes = plt.subplots(1, len(datasets), figsize=(7.2, 3.0), squeeze=False)
    methods = (
        ("Naive-A local-TopK union", "Naive local-union", "#C44E52", "o"),
        ("FedPASA quantile-aligned", "FedPASA", "#2F6B9A", "s"),
    )
    for axis, dataset in zip(axes[0], datasets):
        for method, label, color, marker in methods:
            lookup = {
                row["partition"]: row
                for row in summary
                if row["dataset"] == dataset and row["method"] == method
            }
            available = [name for name in order if name in lookup]
            axis.plot(
                range(len(available)),
                [lookup[name]["mAP_mean"] for name in available],
                color=color,
                marker=marker,
                linewidth=1.8,
                label=label,
            )
            if any(float(lookup[name]["mAP_std"]) > 0 for name in available):
                axis.errorbar(
                    range(len(available)),
                    [lookup[name]["mAP_mean"] for name in available],
                    yerr=[lookup[name]["mAP_std"] for name in available],
                    color=color,
                    capsize=2,
                    linestyle="none",
                )
        axis.set_xticks(range(len(available)), [labels[order.index(name)] for name in available])
        axis.set_title(dataset.upper())
        axis.set_ylabel("mAP")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figure_root / "estimator_preserving_non_iid.png", dpi=220)
    figure.savefig(figure_root / "estimator_preserving_non_iid.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_complementarity(report: dict[str, object]) -> None:
    from plot_classwise_complementarity import plot_complementarity as render

    render(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=tuple(BENCHMARKS), default=list(DEFAULT_DATASETS))
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=("estimator", "complementarity", "communication", "plot"),
        default=["estimator", "complementarity", "communication", "plot"],
    )
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--betas", type=float, nargs="+", default=[0.1, 0.3, 0.5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--include-kmeans", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--complementarity-tolerance", type=float, default=0.05)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    estimator_report = None
    estimator_path = ROOT / "outputs" / "icme" / "paper" / "estimator_preserving_evidence.json"
    if "estimator" in args.studies:
        estimator_report = run_estimator(args)
    elif estimator_path.exists():
        estimator_report = json.loads(estimator_path.read_text(encoding="utf-8"))
    complementarity_report = None
    if "complementarity" in args.studies:
        complementarity_report = run_complementarity(
            args.datasets, args.complementarity_tolerance
        )
    complementarity_path = (
        ROOT / "outputs" / "icme" / "paper" / "classwise_complementarity.json"
    )
    if complementarity_report is None and complementarity_path.exists():
        complementarity_report = json.loads(complementarity_path.read_text(encoding="utf-8"))
    if "communication" in args.studies:
        run_communication(estimator_report, args.datasets, args.k)
    if "plot" in args.studies:
        if estimator_report is not None:
            estimator_report["protocol"]["datasets"] = args.datasets
            plot_estimator(estimator_report)
        if complementarity_report is not None:
            plot_complementarity(complementarity_report)


if __name__ == "__main__":
    main()
