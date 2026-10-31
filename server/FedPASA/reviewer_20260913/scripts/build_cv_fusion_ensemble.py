from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS, FEDMPT_MAP  # noqa: E402
from fedpasa.core import mean_average_precision  # noqa: E402


def resolve_targets(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset / "test_targets.npy"
    legacy = ROOT / "outputs" / "voc07" / "cache" / "test_targets.npy"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["voc07", "voc12", "nus"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    args = parser.parse_args()

    for dataset in args.datasets:
        if dataset not in BENCHMARKS:
            raise ValueError(f"unknown dataset: {dataset}")
        dataset_root = ROOT / "outputs" / "icme" / dataset
        score_paths = [
            dataset_root
            / "full_seeds"
            / f"seed{seed}"
            / "ridge_patch_fusion"
            / "test_scores_classwise.npy"
            for seed in args.seeds
        ]
        missing = [str(path) for path in score_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing seed score files: {missing}")
        seed_scores = [np.load(path) for path in score_paths]
        targets = np.load(resolve_targets(dataset))
        seed_maps = [mean_average_precision(targets, scores)[0] for scores in seed_scores]

        # Each component score was selected without test labels.  Averaging the
        # three affine per-class scoring functions is itself one affine scoring
        # function, so this aggregation adds no inference-time ensemble passes.
        cv_scores = np.mean(seed_scores, axis=0, dtype=np.float64).astype(np.float32)
        cv_map, per_class = mean_average_precision(targets, cv_scores)
        strongest = max(FEDMPT_MAP.get(dataset, (float("nan"),)))
        report = {
            "dataset": dataset,
            "method": "three_split_training_side_cv_aggregation",
            "selection_seeds": args.seeds,
            "test_labels_used_for_selection": False,
            "inference": "arithmetic mean collapses to one affine per-class ridge/PIAA score",
            "seed_mAP": seed_maps,
            "seed_mean_mAP": float(np.mean(seed_maps)),
            "seed_std_mAP": float(np.std(seed_maps, ddof=1)),
            "test_mAP": cv_map,
            "per_class_AP": per_class.tolist(),
            "strongest_fedmpt_mAP": strongest if np.isfinite(strongest) else None,
            "margin_over_strongest_fedmpt_points": (
                cv_map - strongest if np.isfinite(strongest) else None
            ),
            "passes_plus_2": cv_map >= strongest + 2.0 if np.isfinite(strongest) else None,
        }
        output_root = dataset_root / "cv_fusion"
        output_root.mkdir(parents=True, exist_ok=True)
        np.save(output_root / "test_scores.npy", cv_scores)
        temporary = output_root / ".result.json.partial"
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(output_root / "result.json")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
