from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.core import (  # noqa: E402
    Adapter,
    ClientPacket,
    PatchBank,
    build_client_banks,
    fit_central_piaa,
    fit_fedpasa_from_packets,
    fit_quantile_aligned_fedpasa,
    make_client_packet,
    mean_average_precision,
)


EXTERNAL_BASELINES_VOC07 = {
    # Strongest number for each method in FedMPT Table 1 across its heterogeneity ratios.
    # FedMPT itself reaches 90.13 at the hardest comparison point for our pass criterion.
    "Fed-DualCoOp (NeurIPS 2022)": 87.67,
    "Fed-SCPNet (CVPR 2023)": 83.37,
    "Fed-MaPLe (CVPR 2023)": 87.87,
    "FedPGP (ICML 2024)": 85.47,
    "Fed-TCP (CVPR 2024)": 81.56,
    "FedTPG (ICLR 2024)": 88.11,
    "Fed-PosCoOp (WACV 2025)": 87.42,
    "FedAWA (CVPR 2025)": 84.75,
    "Fed-RAM (CVPR 2025)": 88.05,
    "FedMVP (ICCV 2025)": 87.14,
    "FedMPT (CVPR 2026)": 90.13,
}

FEDMPT_BY_CLIENT_PERCENT = {
    10: 90.13,
    20: 90.12,
    40: 90.01,
    60: 90.00,
    80: 89.61,
    100: 87.19,
}


def _serialize_banks(banks: tuple[dict[int, PatchBank], ...]) -> list[dict[int, dict[str, torch.Tensor]]]:
    return [
        {
            class_id: {
                "features": value.features,
                "probabilities": value.probabilities,
                "entropy": value.entropy,
                "image_indices": value.image_indices,
            }
            for class_id, value in bank.items()
        }
        for bank in banks
    ]


def _deserialize_banks(payload: list[dict[int, dict[str, torch.Tensor]]]) -> tuple[dict[int, PatchBank], ...]:
    return tuple(
        {
            int(class_id): PatchBank(
                value["features"],
                value["probabilities"],
                value["entropy"],
                value.get("image_indices"),
            )
            for class_id, value in bank.items()
        }
        for bank in payload
    )


def cached_banks(
    cache_path: Path,
    patch_path: Path,
    assignments: np.ndarray,
    text_features: torch.Tensor,
    *,
    clients: int,
    k: int,
    batch_images: int,
) -> tuple[dict[int, PatchBank], ...]:
    if cache_path.exists():
        print(f"BANK_REUSE {cache_path}", flush=True)
        return _deserialize_banks(torch.load(cache_path, map_location="cpu", weights_only=True))
    started = perf_counter()
    banks = build_client_banks(
        patch_path,
        assignments,
        text_features,
        clients=clients,
        k=k,
        batch_images=batch_images,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name(f".{cache_path.name}.partial")
    torch.save(_serialize_banks(banks), temporary)
    temporary.replace(cache_path)
    print(f"BANK_SAVED {cache_path} seconds={perf_counter() - started:.2f}", flush=True)
    return banks


def make_packets(
    banks: tuple[dict[int, PatchBank], ...], text_features: torch.Tensor
) -> tuple[ClientPacket, ...]:
    packets: list[ClientPacket] = []
    for client_id, bank in enumerate(banks):
        started = perf_counter()
        packets.append(make_client_packet(bank, text_features))
        print(
            f"PACKET client={client_id + 1}/{len(banks)} seconds={perf_counter() - started:.2f}",
            flush=True,
        )
    return tuple(packets)


@torch.inference_mode()
def evaluate_many(
    patch_path: Path,
    cls_path: Path,
    targets: np.ndarray,
    indices: np.ndarray,
    text_features: torch.Tensor,
    adapters: dict[str, Adapter],
    *,
    batch_images: int,
    device: str = "cuda",
) -> dict[str, dict[str, object]]:
    patches = np.load(patch_path, mmap_mode="r")
    cls_features = np.load(cls_path, mmap_mode="r")
    indices = np.asarray(indices, dtype=np.int64)
    text = F.normalize(text_features.float().to(device), dim=-1)
    gpu_adapters = {
        name: (value.weight.float().to(device), value.bias.float().to(device))
        for name, value in adapters.items()
    }
    fixed_scores = {name: [] for name in adapters}
    adaptive_scores = {name: [] for name in adapters}
    log_classes = float(np.log(text.shape[0]))
    prior_logit = float(np.log(0.9 / 0.1))

    for start in range(0, len(indices), batch_images):
        batch_indices = indices[start : start + batch_images]
        patch = torch.from_numpy(np.asarray(patches[batch_indices], dtype=np.float32)).to(device)
        patch = F.normalize(patch, dim=-1)
        cls = torch.from_numpy(np.asarray(cls_features[batch_indices], dtype=np.float32)).to(device)
        cls = F.normalize(cls, dim=-1)
        cls_probabilities = F.softmax((cls @ text.T) * 80.0, dim=-1)
        cls_confidence = 1.0 - (
            -(cls_probabilities * cls_probabilities.clamp_min(1e-10).log()).sum(-1) / log_classes
        )
        flat = patch.reshape(-1, patch.shape[-1])
        for name, (weight, bias) in gpu_adapters.items():
            logits = flat @ weight + bias
            refined = F.softmax(logits / 500.0, dim=-1)
            refined = refined.reshape(len(batch_indices), patch.shape[1], -1)
            patch_max = refined.max(dim=1).values
            patch_probabilities = F.softmax(patch_max * 200.0, dim=-1)
            fixed = 0.1 * cls_probabilities + 0.9 * patch_probabilities
            patch_confidence = 1.0 - (
                -(patch_probabilities * patch_probabilities.clamp_min(1e-10).log()).sum(-1)
                / log_classes
            )
            alpha = torch.sigmoid(
                torch.as_tensor(prior_logit, device=device)
                + 2.0 * (patch_confidence - cls_confidence)
            )[:, None]
            adaptive = (1.0 - alpha) * cls_probabilities + alpha * patch_probabilities
            fixed_scores[name].append(fixed.cpu().numpy())
            adaptive_scores[name].append(adaptive.cpu().numpy())
        if start % (batch_images * 20) == 0 or start + batch_images >= len(indices):
            print(f"EVAL images={min(start + batch_images, len(indices))}/{len(indices)}", flush=True)

    result: dict[str, dict[str, object]] = {}
    evaluated_targets = targets[indices]
    for name, adapter in adapters.items():
        result[name] = {"adapter": adapter.metadata}
        for fusion, chunks in (
            ("fixed_0.9", fixed_scores[name]),
            ("entropy_adaptive", adaptive_scores[name]),
        ):
            scores = np.concatenate(chunks, axis=0)
            map_value, per_class = mean_average_precision(evaluated_targets, scores)
            result[name][fusion] = {
                "mAP": map_value,
                "per_class_AP": per_class.tolist(),
            }
    return result


def visual_assignments(
    cls_features: np.ndarray,
    fit_indices: np.ndarray,
    *,
    clients: int,
    seed: int,
) -> tuple[np.ndarray, list[int]]:
    normalized = np.asarray(cls_features[fit_indices], dtype=np.float32)
    normalized /= np.linalg.norm(normalized, axis=1, keepdims=True).clip(min=1e-12)
    labels = KMeans(n_clusters=clients, random_state=seed, n_init=20).fit_predict(normalized)
    assignments = np.full(len(cls_features), -1, dtype=np.int64)
    assignments[fit_indices] = labels
    return assignments, np.bincount(labels, minlength=clients).tolist()


def official_train_val_indices(cache_root: Path, voc_root: Path) -> tuple[np.ndarray, np.ndarray]:
    identifiers = (cache_root / "train_ids.txt").read_text(encoding="utf-8").splitlines()
    index = {identifier: position for position, identifier in enumerate(identifiers)}
    imagesets = (
        voc_root
        / "VOCtrainval_06-Nov-2007"
        / "VOCdevkit"
        / "VOC2007"
        / "ImageSets"
        / "Main"
    )
    train_ids = (imagesets / "train.txt").read_text(encoding="utf-8").split()
    val_ids = (imagesets / "val.txt").read_text(encoding="utf-8").split()
    return (
        np.asarray([index[value] for value in train_ids], dtype=np.int64),
        np.asarray([index[value] for value in val_ids], dtype=np.int64),
    )


def tune(args, cache_root: Path, report_root: Path, text: torch.Tensor) -> dict[str, object]:
    train_targets = np.load(cache_root / "train_targets.npy")
    train_cls = np.load(cache_root / "train_cls.npy", mmap_mode="r")
    fit_indices, validation_indices = official_train_val_indices(cache_root, args.voc_root)
    assignments, client_sizes = visual_assignments(
        train_cls, fit_indices, clients=args.clients, seed=args.tune_seed
    )
    central_assignments = np.full(len(train_cls), -1, dtype=np.int64)
    central_assignments[fit_indices] = 0
    central_banks = cached_banks(
        report_root / "banks" / f"tune_central_k{args.k}.pt",
        cache_root / "train_patch.npy",
        central_assignments,
        text,
        clients=1,
        k=args.k,
        batch_images=args.bank_batch_size,
    )
    client_banks = cached_banks(
        report_root / "banks" / f"tune_clients{args.clients}_seed{args.tune_seed}_k{args.k}.pt",
        cache_root / "train_patch.npy",
        assignments,
        text,
        clients=args.clients,
        k=args.k,
        batch_images=args.bank_batch_size,
    )
    central = fit_central_piaa(central_banks[0], text)
    packets = make_packets(client_banks, text)
    adapters: dict[str, Adapter] = {"central_piaa": central}
    adapters["fedpasa_quantile_aligned"] = fit_quantile_aligned_fedpasa(
        client_banks, text, k=args.k
    )
    adapters["fedpasa_quantile_aligned"].metadata["central_recovery_max_abs_weight"] = float(
        (adapters["fedpasa_quantile_aligned"].weight - central.weight).abs().max()
    )
    adapters["fedpasa_quantile_aligned"].metadata["central_recovery_max_abs_bias"] = float(
        (adapters["fedpasa_quantile_aligned"].bias - central.bias).abs().max()
    )
    grid = [
        (entropy, consensus, prior)
        for entropy in (1.0, 2.0)
        for consensus in (4.0, 8.0)
        for prior in (0.0, 8.0)
    ]
    # Explicit no-reliability ablation.
    grid.insert(0, (0.0, 0.0, 0.0))
    for entropy, consensus, prior in grid:
        name = f"fedpasa_e{entropy:g}_c{consensus:g}_p{prior:g}"
        adapters[name] = fit_fedpasa_from_packets(
            packets,
            text,
            entropy_strength=entropy,
            consensus_strength=consensus,
            prior_strength=prior,
        )
    evaluation = evaluate_many(
        cache_root / "train_patch.npy",
        cache_root / "train_cls.npy",
        train_targets,
        validation_indices,
        text,
        adapters,
        batch_images=args.eval_batch_size,
    )
    candidates: list[tuple[float, str, str]] = []
    for name, value in evaluation.items():
        if not name.startswith("fedpasa_"):
            continue
        for fusion in ("fixed_0.9", "entropy_adaptive"):
            candidates.append((float(value[fusion]["mAP"]), name, fusion))
    best_map, best_name, best_fusion = max(candidates)
    best_adapter = adapters[best_name]
    selection = {
        "name": best_name,
        "fusion": best_fusion,
        "validation_mAP": best_map,
        "entropy_strength": best_adapter.metadata.get("entropy_strength"),
        "consensus_strength": best_adapter.metadata.get("consensus_strength"),
        "prior_strength": best_adapter.metadata.get("prior_strength"),
    }
    report = {
        "phase": "training-side model selection",
        "fit_samples": len(fit_indices),
        "validation_samples": len(validation_indices),
        "test_labels_accessed": False,
        "partition": "KMeans on normalized ViT-B/16 image embeddings",
        "client_sizes": client_sizes,
        "evaluation": evaluation,
        "selection": selection,
    }
    (report_root / "tuning.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"TUNING_SELECTED {json.dumps(selection)}", flush=True)
    return report


def run_final(
    args,
    cache_root: Path,
    report_root: Path,
    text: torch.Tensor,
    selection: dict[str, object],
) -> dict[str, object]:
    train_cls = np.load(cache_root / "train_cls.npy", mmap_mode="r")
    test_targets = np.load(cache_root / "test_targets.npy")
    all_train = np.arange(len(train_cls), dtype=np.int64)
    central_assignments = np.zeros(len(train_cls), dtype=np.int64)
    central_banks = cached_banks(
        report_root / "banks" / f"final_central_k{args.k}.pt",
        cache_root / "train_patch.npy",
        central_assignments,
        text,
        clients=1,
        k=args.k,
        batch_images=args.bank_batch_size,
    )
    central = fit_central_piaa(central_banks[0], text)
    seed_reports: list[dict[str, object]] = []
    for seed in args.seeds:
        assignments, client_sizes = visual_assignments(
            train_cls, all_train, clients=args.clients, seed=seed
        )
        banks = cached_banks(
            report_root / "banks" / f"final_clients{args.clients}_seed{seed}_k{args.k}.pt",
            cache_root / "train_patch.npy",
            assignments,
            text,
            clients=args.clients,
            k=args.k,
            batch_images=args.bank_batch_size,
        )
        packets = make_packets(banks, text)
        if selection["name"] == "fedpasa_quantile_aligned":
            fedpasa = fit_quantile_aligned_fedpasa(banks, text, k=args.k)
        else:
            fedpasa = fit_fedpasa_from_packets(
                packets,
                text,
                entropy_strength=float(selection["entropy_strength"]),
                consensus_strength=float(selection["consensus_strength"]),
                prior_strength=float(selection["prior_strength"]),
            )
        no_reliability = fit_fedpasa_from_packets(
            packets,
            text,
            entropy_strength=0.0,
            consensus_strength=0.0,
            prior_strength=0.0,
        )
        adapters = {
            "central_piaa_reproduction": central,
            "federated_equal_statistics": no_reliability,
            "fedpasa": fedpasa,
        }
        evaluation = evaluate_many(
            cache_root / "test_patch.npy",
            cache_root / "test_cls.npy",
            test_targets,
            np.arange(len(test_targets)),
            text,
            adapters,
            batch_images=args.eval_batch_size,
        )
        selected_map = float(evaluation["fedpasa"][str(selection["fusion"])]["mAP"])
        strongest_name, strongest_value = max(EXTERNAL_BASELINES_VOC07.items(), key=lambda item: item[1])
        seed_reports.append(
            {
                "seed": seed,
                "client_sizes": client_sizes,
                "evaluation": evaluation,
                "selected_fedpasa_mAP": selected_map,
                "strongest_external_baseline": {"name": strongest_name, "mAP": strongest_value},
                "margin_points": selected_map - strongest_value,
                "passes_plus_2": selected_map >= strongest_value + 2.0,
                "central_recovery_max_abs_weight": float(
                    (fedpasa.weight - central.weight).abs().max()
                ),
                "central_recovery_max_abs_bias": float(
                    (fedpasa.bias - central.bias).abs().max()
                ),
            }
        )
        print(
            f"FINAL seed={seed} mAP={selected_map:.4f} margin={selected_map - strongest_value:+.4f}",
            flush=True,
        )
    maps = np.asarray([float(value["selected_fedpasa_mAP"]) for value in seed_reports])
    margins = np.asarray([float(value["margin_points"]) for value in seed_reports])
    report = {
        "phase": "locked held-out VOC07 test",
        "selection": selection,
        "external_baselines": EXTERNAL_BASELINES_VOC07,
        "seeds": seed_reports,
        "summary": {
            "mean_mAP": float(maps.mean()),
            "std_mAP": float(maps.std(ddof=1)) if len(maps) > 1 else 0.0,
            "minimum_margin_points": float(margins.min()),
            "mean_margin_points": float(margins.mean()),
            "all_seeds_pass_plus_2": bool((margins >= 2.0).all()),
        },
    }
    (report_root / "final.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_heterogeneity_sweep(
    args,
    cache_root: Path,
    report_root: Path,
    text: torch.Tensor,
) -> dict[str, object]:
    """Verify all six FedMPT heterogeneity settings through adapter equality."""

    train_cls = np.load(cache_root / "train_cls.npy", mmap_mode="r")
    all_train = np.arange(len(train_cls), dtype=np.int64)
    central_banks = cached_banks(
        report_root / "banks" / f"final_central_k{args.k}.pt",
        cache_root / "train_patch.npy",
        np.zeros(len(train_cls), dtype=np.int64),
        text,
        clients=1,
        k=args.k,
        batch_images=args.bank_batch_size,
    )
    central = fit_central_piaa(central_banks[0], text)
    final_path = report_root / "final.json"
    if final_path.exists():
        existing_final = json.loads(final_path.read_text(encoding="utf-8"))
        central_map = float(
            existing_final["seeds"][0]["evaluation"]["central_piaa_reproduction"]["fixed_0.9"]["mAP"]
        )
    else:
        test_targets = np.load(cache_root / "test_targets.npy")
        central_result = evaluate_many(
            cache_root / "test_patch.npy",
            cache_root / "test_cls.npy",
            test_targets,
            np.arange(len(test_targets)),
            text,
            {"central": central},
            batch_images=args.eval_batch_size,
        )
        central_map = float(central_result["central"]["fixed_0.9"]["mAP"])

    runs: list[dict[str, object]] = []
    sweep_adapters: dict[str, Adapter] = {}
    for clients in args.client_counts:
        percentage = int(round(100 * clients / 20))
        if percentage not in FEDMPT_BY_CLIENT_PERCENT:
            raise ValueError(f"client count {clients} does not map to a FedMPT VOC07 setting")
        for seed in args.seeds:
            assignments, client_sizes = visual_assignments(
                train_cls, all_train, clients=clients, seed=seed
            )
            banks = cached_banks(
                report_root / "banks" / f"sweep_clients{clients}_seed{seed}_k{args.k}.pt",
                cache_root / "train_patch.npy",
                assignments,
                text,
                clients=clients,
                k=args.k,
                batch_images=args.bank_batch_size,
            )
            adapter = fit_quantile_aligned_fedpasa(banks, text, k=args.k)
            weight_error = float((adapter.weight - central.weight).abs().max())
            bias_error = float((adapter.bias - central.bias).abs().max())
            equivalent = weight_error <= 1e-5 and bias_error <= 1e-5
            baseline = FEDMPT_BY_CLIENT_PERCENT[percentage]
            adapter_key = f"clients{clients}_seed{seed}"
            sweep_adapters[adapter_key] = adapter
            run = {
                "adapter_key": adapter_key,
                "clients": clients,
                "client_percent_of_classes": percentage,
                "seed": seed,
                "client_sizes": client_sizes,
                "central_recovery_max_abs_weight": weight_error,
                "central_recovery_max_abs_bias": bias_error,
                "numerically_equivalent_adapter_at_1e-5": equivalent,
                "fedmpt_mAP": baseline,
                "communication_MiB": adapter.metadata["estimated_communication_MiB"],
            }
            runs.append(run)
            print(
                f"SWEEP clients={clients} t={percentage}% seed={seed} "
                f"weight_error={weight_error:.3e} bias_error={bias_error:.3e}",
                flush=True,
            )
    test_targets = np.load(cache_root / "test_targets.npy")
    evaluations = evaluate_many(
        cache_root / "test_patch.npy",
        cache_root / "test_cls.npy",
        test_targets,
        np.arange(len(test_targets)),
        text,
        sweep_adapters,
        batch_images=args.eval_batch_size,
    )
    for run in runs:
        test_map = float(evaluations[str(run["adapter_key"])]["fixed_0.9"]["mAP"])
        baseline = float(run["fedmpt_mAP"])
        run["fedpasa_mAP"] = test_map
        run["margin_points"] = test_map - baseline
        run["passes_plus_2"] = test_map >= baseline + 2.0
    report = {
        "protocol": "FedMPT Table 1 VOC07 client counts; KMeans on ViT-B/16 visual embeddings",
        "central_test_mAP": central_map,
        "fedmpt_by_client_percent": FEDMPT_BY_CLIENT_PERCENT,
        "runs": runs,
        "summary": {
            "configurations": len(runs),
            "all_numerically_equivalent_at_1e-5": all(
                bool(value["numerically_equivalent_adapter_at_1e-5"]) for value in runs
            ),
            "all_pass_plus_2": all(bool(value["passes_plus_2"]) for value in runs),
            "minimum_margin_points": min(float(value["margin_points"]) for value in runs),
            "maximum_communication_MiB": max(float(value["communication_MiB"]) for value in runs),
        },
    }
    (report_root / "heterogeneity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("tune", "final", "sweep", "all"), default="all")
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(r"D:\project\VScodeProject\BCP\data\raw\voc2007"),
    )
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs" / "voc07" / "cache")
    parser.add_argument("--report-root", type=Path, default=ROOT / "outputs" / "voc07")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--tune-seed", type=int, default=7)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--client-counts", type=int, nargs="+", default=[2, 4, 8, 12, 16, 20])
    parser.add_argument("--bank-batch-size", type=int, default=48)
    parser.add_argument("--eval-batch-size", type=int, default=48)
    args = parser.parse_args()

    required = ("train_patch.npy", "test_patch.npy", "train_cls.npy", "test_cls.npy")
    missing = [name for name in required if not (args.cache_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing feature caches {missing}; run scripts/extract_voc07_features.py first"
        )
    args.report_root.mkdir(parents=True, exist_ok=True)
    text = torch.from_numpy(np.load(args.cache_root / "text_features.npy")).float()
    tuning_path = args.report_root / "tuning.json"
    tuning = None
    if args.phase in {"tune", "all"}:
        tuning = tune(args, args.cache_root, args.report_root, text)
    if args.phase in {"final", "all"}:
        if tuning is None:
            if not tuning_path.exists():
                raise FileNotFoundError("tuning.json is required before final evaluation")
            tuning = json.loads(tuning_path.read_text(encoding="utf-8"))
        final = run_final(
            args, args.cache_root, args.report_root, text, tuning["selection"]
        )
        print(json.dumps(final["summary"], indent=2), flush=True)
    if args.phase in {"sweep", "all"}:
        sweep = run_heterogeneity_sweep(args, args.cache_root, args.report_root, text)
        print(json.dumps(sweep["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
