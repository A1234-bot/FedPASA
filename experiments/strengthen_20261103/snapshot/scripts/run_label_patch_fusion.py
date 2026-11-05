from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS, FEDMPT_MAP  # noqa: E402
from fedpasa.core import (  # noqa: E402
    PatchBank,
    build_client_banks,
    fit_central_piaa,
    mean_average_precision,
)


def serialize_bank(bank: dict[int, PatchBank]) -> dict[int, dict[str, torch.Tensor | None]]:
    return {
        int(class_id): {
            "features": value.features,
            "probabilities": value.probabilities,
            "entropy": value.entropy,
            "image_indices": value.image_indices,
        }
        for class_id, value in bank.items()
    }


def deserialize_bank(payload: dict[int, dict[str, torch.Tensor]]) -> dict[int, PatchBank]:
    return {
        int(class_id): PatchBank(
            value["features"], value["probabilities"], value["entropy"], value.get("image_indices")
        )
        for class_id, value in payload.items()
    }


def truncate_bank(bank: dict[int, PatchBank], k: int) -> dict[int, PatchBank]:
    output: dict[int, PatchBank] = {}
    for class_id, value in bank.items():
        keep = torch.topk(value.entropy, min(k, len(value.entropy)), largest=False).indices
        output[class_id] = PatchBank(
            value.features[keep],
            value.probabilities[keep],
            value.entropy[keep],
            None if value.image_indices is None else value.image_indices[keep],
        )
    return output


@torch.inference_mode()
def piaa_scores(
    patch_path: Path,
    cls_path: Path,
    indices: np.ndarray,
    text_features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    cls_scale: float,
    patch_scale: float,
    batch_images: int,
    device: str,
) -> np.ndarray:
    from fedpasa import cache_io
    patches = np.load(patch_path, mmap_mode="r")
    cls_cache = np.load(cls_path, mmap_mode="r")
    indices = np.asarray(indices, dtype=np.int64)
    text = F.normalize(text_features.float().to(device), dim=-1)
    weight = weight.float().to(device)
    bias = bias.float().to(device)
    output = np.empty((len(indices), text.shape[0]), dtype=np.float32)
    for start, batch_indices, (host_patch, host_cls) in cache_io.batches((patches, cls_cache), indices, batch_images):
        stop = min(start + batch_images, len(indices))
        patch = torch.from_numpy(host_patch).to(device)
        patch = F.normalize(patch, dim=-1)
        cls = torch.from_numpy(host_cls).to(device)
        cls = F.normalize(cls, dim=-1)
        cls_probability = F.softmax((cls @ text.T) * cls_scale, dim=-1)
        flat = patch.reshape(-1, patch.shape[-1])
        refined = F.softmax((flat @ weight + bias) / 500.0, dim=-1)
        refined = refined.reshape(stop - start, patch.shape[1], -1)
        patch_probability = F.softmax(refined.max(dim=1).values * patch_scale, dim=-1)
        output[start:stop] = (0.1 * cls_probability + 0.9 * patch_probability).cpu().numpy()
        if start % (batch_images * 50) == 0 or stop == len(indices):
            print(f"FUSION_PIAA images={stop}/{len(indices)}", flush=True)
    return output


def ridge_scores(features: np.ndarray, weight: torch.Tensor, bias: torch.Tensor) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    x = x / np.linalg.norm(x, axis=1, keepdims=True).clip(min=1e-12)
    return x @ weight.numpy() + bias.numpy()


def standardize(reference: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = reference.mean(axis=0)
    std = reference.std(axis=0).clip(min=1e-6)
    return (reference - mean) / std, (values - mean) / std, np.stack((mean, std))


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def load_ridge_features(cache_root: Path, split: str, names: list[str]) -> np.ndarray:
    values = [
        np.asarray(np.load(cache_root / f"{split}_{name}.npy", mmap_mode="r"), dtype=np.float32)
        for name in names
    ]
    return values[0] if len(values) == 1 else np.concatenate(values, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--weights", type=float, nargs="+", default=np.linspace(0.0, 1.0, 21).tolist())
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ridge-name", default="label_ridge")
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.dataset]
    cache_root = resolve_cache(args.dataset)
    dataset_root = ROOT / "outputs" / "icme" / args.dataset
    ridge_root = dataset_root / args.ridge_name
    output_name = "label_patch_fusion" if args.ridge_name == "label_ridge" else f"{args.ridge_name}_patch_fusion"
    output_root = dataset_root / output_name
    output_root.mkdir(parents=True, exist_ok=True)
    ridge_payload = torch.load(ridge_root / "adapter.pt", map_location="cpu", weights_only=True)
    validation_ridge_path = ridge_root / "validation_adapter.pt"
    if not validation_ridge_path.exists():
        raise FileNotFoundError(
            f"{validation_ridge_path} is missing; rerun run_label_ridge.py with the leakage-safe protocol"
        )
    validation_ridge_payload = torch.load(
        validation_ridge_path, map_location="cpu", weights_only=True
    )
    ridge_report = json.loads((ridge_root / "result.json").read_text(encoding="utf-8"))
    ridge_features = list(ridge_report["protocol"]["features"])
    validation_indices = np.load(ridge_root / "validation_indices.npy")
    fit_indices = np.load(ridge_root / "fit_indices.npy")
    text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float()
    bank_candidates = sorted(
        (dataset_root / "banks").glob("global_k*.pt"),
        key=lambda path: int(path.stem.split("k")[-1]),
        reverse=True,
    )
    if not bank_candidates:
        raise FileNotFoundError("run the main QAFedPASA experiment before fusion")
    full_bank = truncate_bank(
        deserialize_bank(torch.load(bank_candidates[0], map_location="cpu", weights_only=True)), args.k
    )
    full_adapter = fit_central_piaa(
        full_bank,
        text,
        purification_temperature=benchmark.purification_temperature,
    )

    # Fit the label-free PIAA branch without the held-out images as well.  This
    # makes both branches of validation prediction strictly out-of-sample.
    validation_bank_path = output_root / f"validation_global_bank_k{args.k}.pt"
    if validation_bank_path.exists():
        validation_bank = deserialize_bank(
            torch.load(validation_bank_path, map_location="cpu", weights_only=True)
        )
    else:
        assignments = np.full(benchmark.expected_train, -1, dtype=np.int64)
        assignments[fit_indices] = 0
        validation_bank = build_client_banks(
            cache_root / "train_patch.npy",
            assignments,
            text,
            clients=1,
            k=args.k,
            batch_images=args.batch_size,
            device=args.device,
        )[0]
        torch.save(serialize_bank(validation_bank), validation_bank_path)
    validation_piaa_adapter = fit_central_piaa(
        validation_bank,
        text,
        purification_temperature=benchmark.purification_temperature,
    )

    started = perf_counter()
    validation_piaa_path = output_root / "training_validation_piaa_scores.npy"
    if validation_piaa_path.exists():
        validation_piaa = np.load(validation_piaa_path)
    else:
        validation_piaa = piaa_scores(
            cache_root / "train_patch.npy",
            cache_root / "train_cls.npy",
            validation_indices,
            text,
            validation_piaa_adapter.weight,
            validation_piaa_adapter.bias,
            cls_scale=benchmark.cls_scale,
            patch_scale=benchmark.patch_scale,
            batch_images=args.batch_size,
            device=args.device,
        )
        validation_piaa_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(validation_piaa_path, validation_piaa)
    test_indices = np.arange(benchmark.expected_test, dtype=np.int64)
    existing_test = dataset_root / "scores" / "qafedpasa_representative_scores.npy"
    test_piaa = (
        np.load(existing_test)
        if existing_test.exists()
        else piaa_scores(
            cache_root / "test_patch.npy",
            cache_root / "test_cls.npy",
            test_indices,
            text,
            full_adapter.weight,
            full_adapter.bias,
            cls_scale=benchmark.cls_scale,
            patch_scale=benchmark.patch_scale,
            batch_images=args.batch_size,
            device=args.device,
        )
    )
    train_cls = load_ridge_features(cache_root, "train", ridge_features)
    test_cls = load_ridge_features(cache_root, "test", ridge_features)
    validation_ridge = ridge_scores(
        train_cls[validation_indices],
        validation_ridge_payload["weight"],
        validation_ridge_payload["bias"],
    )
    test_ridge = ridge_scores(test_cls, ridge_payload["weight"], ridge_payload["bias"])
    validation_targets = np.load(cache_root / "train_targets.npy")[validation_indices]
    test_targets = np.load(cache_root / "test_targets.npy")
    validation_piaa_z, test_piaa_z, piaa_calibration = standardize(validation_piaa, test_piaa)
    validation_ridge_z, test_ridge_z, ridge_calibration = standardize(validation_ridge, test_ridge)

    tuning: list[dict[str, float]] = []
    validation_per_class: list[np.ndarray] = []
    for ridge_weight in args.weights:
        combined = ridge_weight * validation_ridge_z + (1.0 - ridge_weight) * validation_piaa_z
        value, per_class = mean_average_precision(validation_targets, combined)
        tuning.append({"ridge_weight": ridge_weight, "validation_mAP": value})
        validation_per_class.append(per_class)
        print(f"FUSION_TUNE ridge_weight={ridge_weight:.2f} mAP={value:.4f}", flush=True)

    global_choice = max(tuning, key=lambda value: value["validation_mAP"])
    global_weight = float(global_choice["ridge_weight"])
    global_scores = global_weight * test_ridge_z + (1.0 - global_weight) * test_piaa_z
    global_map, global_per_class = mean_average_precision(test_targets, global_scores)

    validation_matrix = np.stack(validation_per_class)
    choice_indices = validation_matrix.argmax(axis=0)
    class_weights = np.asarray(args.weights, dtype=np.float32)[choice_indices]
    classwise_scores = class_weights[None, :] * test_ridge_z + (1.0 - class_weights[None, :]) * test_piaa_z
    classwise_map, classwise_per_class = mean_average_precision(test_targets, classwise_scores)
    strongest = max(FEDMPT_MAP.get(args.dataset, (float("nan"),)))
    report = {
        "dataset": args.dataset,
        "protocol": {
            "selection_data": "training-side held-out 20% split inherited from label_ridge",
            "test_labels_used_for_selection": False,
            "validation_predictions": "both ridge and PIAA fit on the disjoint 80% fit split",
            "calibration": "per-class z-score using training-side validation predictions",
            "ridge_weight_definition": "1.0 is label-aware ridge; 0.0 is QAFedPASA/PIAA",
        },
        "tuning": tuning,
        "global_weight": {
            "selection": global_choice,
            "test_mAP": global_map,
            "per_class_AP": global_per_class.tolist(),
            "margin_over_strongest_fedmpt_points": global_map - strongest if np.isfinite(strongest) else None,
            "passes_plus_2": global_map >= strongest + 2.0 if np.isfinite(strongest) else None,
        },
        "classwise_weight": {
            "weights": class_weights.tolist(),
            "test_mAP": classwise_map,
            "per_class_AP": classwise_per_class.tolist(),
            "margin_over_strongest_fedmpt_points": classwise_map - strongest if np.isfinite(strongest) else None,
            "passes_plus_2": classwise_map >= strongest + 2.0 if np.isfinite(strongest) else None,
        },
        "strongest_fedmpt_mAP": strongest if np.isfinite(strongest) else None,
        "runtime_seconds": perf_counter() - started,
    }
    np.save(output_root / "test_scores_global.npy", global_scores)
    np.save(output_root / "test_scores_classwise.npy", classwise_scores)
    np.save(output_root / "piaa_calibration.npy", piaa_calibration)
    np.save(output_root / "ridge_calibration.npy", ridge_calibration)
    temporary = output_root / ".result.json.partial"
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output_root / "result.json")
    print(
        json.dumps(
            {"global_weight": report["global_weight"], "classwise_weight": report["classwise_weight"]},
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
