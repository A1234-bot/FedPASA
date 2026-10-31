from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import EXTERNAL_MAPS  # noqa: E402
from fedpasa.core import mean_average_precision  # noqa: E402


T_VALUES = (10, 20, 40, 60, 80, 100)
SOURCE_URL = "https://arxiv.org/html/2605.28347"
FULL_FUSION_NAME = "label_ridge_cls_patch_mean_patch_max_text_patch_fusion"


def full_seed_records(output_root: Path, dataset: str) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for result_path in sorted(
        (output_root / dataset / "full_seeds").glob("seed*/ridge_patch_fusion/result.json")
    ):
        report = json.loads(result_path.read_text(encoding="utf-8"))
        seed = int(result_path.parents[1].name.removeprefix("seed"))
        records.append(
            {
                "seed": seed,
                "mAP": float(report["classwise_weight"]["test_mAP"]),
                "scores": result_path.parent / "test_scores_classwise.npy",
            }
        )
    return records


def full_seed_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        records = full_seed_records(output_root, dataset)
        if not records:
            continue
        values = np.asarray([record["mAP"] for record in records], dtype=np.float64)
        for record in records:
            rows.append(
                {
                    "dataset": dataset,
                    "record": f"seed_{record['seed']}",
                    "mAP": record["mAP"],
                    "std_mAP": "",
                    "seeds": record["seed"],
                }
            )
        rows.append(
            {
                "dataset": dataset,
                "record": "mean_std",
                "mAP": float(values.mean()),
                "std_mAP": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "seeds": "/".join(str(record["seed"]) for record in records),
            }
        )
    return rows


def full_cv_record(output_root: Path, dataset: str) -> dict[str, object] | None:
    result_path = output_root / dataset / "cv_fusion" / "result.json"
    score_path = output_root / dataset / "cv_fusion" / "test_scores.npy"
    if not result_path.exists() or not score_path.exists():
        return None
    report = json.loads(result_path.read_text(encoding="utf-8"))
    return {"mAP": float(report["test_mAP"]), "scores": score_path, "report": report}


def atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def bootstrap_map(
    targets: np.ndarray,
    scores: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> np.ndarray:
    """Exact non-parametric image bootstrap using multiplicities and fixed score ranks."""

    if replicates <= 0:
        return np.empty(0, dtype=np.float64)
    samples, classes = targets.shape
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, samples, size=(replicates, samples), dtype=np.int32)
    counts = np.zeros((replicates, samples), dtype=np.int32)
    rows = np.repeat(np.arange(replicates, dtype=np.int32), samples)
    np.add.at(counts, (rows, draws.ravel()), 1)
    del draws, rows
    aps = np.zeros((replicates, classes), dtype=np.float64)
    for class_id in range(classes):
        order = np.argsort(-scores[:, class_id], kind="mergesort")
        weights = counts[:, order]
        truth = targets[order, class_id].astype(np.int32)
        true_positive = np.cumsum(weights * truth[None, :], axis=1, dtype=np.int32)
        predicted_positive = np.cumsum(weights, axis=1, dtype=np.int32)
        precision = np.divide(
            true_positive,
            predicted_positive,
            out=np.zeros_like(true_positive, dtype=np.float64),
            where=predicted_positive > 0,
        )
        positive_weight = weights * truth[None, :]
        denominator = positive_weight.sum(axis=1)
        aps[:, class_id] = np.divide(
            (precision * positive_weight).sum(axis=1),
            denominator,
            out=np.zeros(replicates, dtype=np.float64),
            where=denominator > 0,
        )
        print(f"BOOTSTRAP class={class_id + 1}/{classes}", flush=True)
    return 100.0 * aps.mean(axis=1)


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def comparison_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        path = output_root / dataset / "main.json"
        if not path.exists() or dataset not in EXTERNAL_MAPS:
            continue
        main = json.loads(path.read_text(encoding="utf-8"))
        ours = float(main["summary"]["qafedpasa_mAP"])
        for method, values in EXTERNAL_MAPS[dataset].items():
            row: dict[str, object] = {"dataset": dataset, "method": method, "result_type": "published_external"}
            row.update({f"t{t}": value for t, value in zip(T_VALUES, values)})
            row["average"] = float(np.mean(values))
            rows.append(row)
        row = {"dataset": dataset, "method": "QAFedPASA", "result_type": "local_reproduction"}
        row.update({f"t{t}": ours for t in T_VALUES})
        row["average"] = ours
        rows.append(row)
        full_records = full_seed_records(output_root, dataset)
        cv_record = full_cv_record(output_root, dataset)
        if cv_record is not None or full_records:
            full_map = (
                float(cv_record["mAP"])
                if cv_record is not None
                else float(np.mean([record["mAP"] for record in full_records]))
            )
            row = {
                "dataset": dataset,
                "method": "FedPASA-Full",
                "result_type": "local_3split_cv_aggregation" if cv_record is not None else "local_3seed_mean",
            }
            row.update({f"t{t}": full_map for t in T_VALUES})
            row["average"] = full_map
            rows.append(row)
    return rows


def ablation_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        path = output_root / dataset / "ablation.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for method, value in report["evaluation"].items():
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "fixed_0.9_mAP": value["fixed_0.9"]["mAP"],
                    "patch_only_mAP": value["patch_only"]["mAP"],
                    "cls_only_mAP": value["cls_only"]["mAP"],
                    "entropy_adaptive_mAP": value["entropy_adaptive"]["mAP"],
                }
            )
    return rows


def systems_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        path = output_root / dataset / "main.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for run in report["runs"]:
            rows.append(
                {
                    "dataset": dataset,
                    "t_percent": run["t_percent"],
                    "clients": run["clients"],
                    "seed": run["seed"],
                    "communication_MiB": run["communication_MiB"],
                    "cluster_seconds": run["cluster_seconds"],
                    "adapter_fit_seconds": run["adapter_fit_seconds"],
                    "max_abs_weight_error": run["central_recovery_max_abs_weight"],
                    "max_abs_bias_error": run["central_recovery_max_abs_bias"],
                }
            )
    return rows


def annotation_invariance_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    """Document an exact consequence of the label-free training path."""

    rows: list[dict[str, object]] = []
    for dataset in datasets:
        path = output_root / dataset / "main.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        score = float(report["summary"]["qafedpasa_mAP"])
        for mask in (10, 30, 50, 70, 90):
            rows.append(
                {
                    "dataset": dataset,
                    "masked_training_annotations_percent": mask,
                    "mAP": score,
                    "invariance_basis": "training targets are not read by feature-bank construction or adapter fitting",
                    "empirical_claim": False,
                }
            )
    return rows


def dirichlet_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        path = output_root / dataset / "dirichlet.json"
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        for run in report["runs"]:
            rows.append(
                {
                    "dataset": dataset,
                    "beta": run["beta"],
                    "clients": run["clients"],
                    "seed": run["seed"],
                    "mAP": run["mAP"],
                    "communication_MiB": run["communication_MiB"],
                    "max_abs_weight_error": run["central_recovery_max_abs_weight"],
                    "max_abs_bias_error": run["central_recovery_max_abs_bias"],
                    "equivalent": run["numerically_equivalent_at_tolerance"],
                }
            )
    return rows


def partial_annotation_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    path = output_root / "paper" / "partial_annotation.json"
    if not path.exists():
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        if dataset not in report:
            continue
        for value in report[dataset]["summary"]:
            rows.append({"dataset": dataset, **value})
    return rows


def label_feature_rows(output_root: Path, datasets: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset in datasets:
        for path in sorted((output_root / dataset).glob("label_ridge*/result.json")):
            if path.parent.name.endswith("patch_fusion"):
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "dataset": dataset,
                    "features": "+".join(report["protocol"]["features"]),
                    "selected_alpha": report["selection"]["alpha"],
                    "validation_mAP": report["selection"]["validation_mAP"],
                    "test_mAP": report["test"]["mAP"],
                    "margin_over_strongest_fedmpt_points": report["test"]["margin_over_strongest_fedmpt_points"],
                    "passes_plus_2": report["test"]["passes_strongest_fedmpt_plus_2"],
                }
            )
    return rows


def build_statistics(
    output_root: Path,
    cache_root: Path,
    datasets: list[str],
    *,
    replicates: int,
    seed: int,
) -> dict[str, object]:
    result: dict[str, object] = {
        "method": "non-parametric bootstrap over test images",
        "replicates": replicates,
        "seed": seed,
        "external_baseline_uncertainty_available": False,
        "interpretation": "superiority probabilities condition on each published external mAP scalar",
        "datasets": {},
    }
    for dataset in datasets:
        full_records = full_seed_records(output_root, dataset)
        cv_record = full_cv_record(output_root, dataset)
        score_paths = [Path(cv_record["scores"])] if cv_record is not None else [
            Path(record["scores"]) for record in full_records
        ]
        method = (
            "FedPASA-Full three-split CV aggregation"
            if cv_record is not None
            else "FedPASA-Full classwise fusion, three-seed mean"
        )
        if not score_paths:
            score_paths = [output_root / dataset / "scores" / "qafedpasa_representative_scores.npy"]
            method = "QAFedPASA"
        target_path = cache_root / dataset / "test_targets.npy"
        if dataset == "voc07" and not target_path.exists():
            target_path = ROOT / "outputs" / "voc07" / "cache" / "test_targets.npy"
        if not all(path.exists() for path in score_paths) or not target_path.exists():
            continue
        targets = np.load(target_path)
        observed_per_seed: list[float] = []
        bootstrap_per_seed: list[np.ndarray] = []
        for score_path in score_paths:
            scores = np.load(score_path)
            observed, _ = mean_average_precision(targets, scores)
            observed_per_seed.append(observed)
            bootstrap_per_seed.append(
                bootstrap_map(targets, scores, replicates=replicates, seed=seed)
            )
        values = np.mean(np.stack(bootstrap_per_seed), axis=0)
        component_maps = (
            [float(record["mAP"]) for record in full_records]
            if full_records
            else observed_per_seed
        )
        dataset_report: dict[str, object] = {
            "method": method,
            "selection_seeds": [int(record["seed"]) for record in full_records] or None,
            "observed_mAP_per_seed": component_maps,
            "observed_mAP": float(np.mean(observed_per_seed)),
            "between_seed_std_mAP": (
                float(np.std(component_maps, ddof=1)) if len(component_maps) > 1 else 0.0
            ),
            "bootstrap_mean_mAP": float(values.mean()) if len(values) else None,
            "bootstrap_std_mAP": float(values.std(ddof=1)) if len(values) > 1 else None,
            "ci95_mAP": np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None,
        }
        if dataset in EXTERNAL_MAPS and len(values):
            strongest = np.max(np.asarray(list(EXTERNAL_MAPS[dataset].values())), axis=0)
            dataset_report["strongest_external_mAP"] = strongest.tolist()
            dataset_report["probability_margin_gt_0"] = [
                float(np.mean(values > baseline)) for baseline in strongest
            ]
            dataset_report["probability_margin_gt_2"] = [
                float(np.mean(values > baseline + 2.0)) for baseline in strongest
            ]
        result["datasets"][dataset] = dataset_report
    atomic_json(output_root / "paper" / "statistics.json", result)
    return result


def plot_main(output_root: Path, rows: list[dict[str, object]]) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.8), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        selected = [row for row in rows if row["dataset"] == dataset]
        for row in selected:
            method = str(row["method"])
            values = [float(row[f"t{t}"]) for t in T_VALUES]
            if method == "FedPASA-Full":
                axis.plot(T_VALUES, values, "o-", linewidth=2.7, color="#d1495b", label=method, zorder=5)
            elif method == "QAFedPASA":
                axis.plot(T_VALUES, values, "^-.", linewidth=1.8, color="#edae49", label=method, zorder=4)
            elif method == "FedMPT (CVPR 2026)":
                axis.plot(T_VALUES, values, "s--", linewidth=2.1, color="#00798c", label="FedMPT")
            else:
                axis.plot(T_VALUES, values, color="#a9a9a9", linewidth=0.8, alpha=0.55)
        axis.set_title(dataset.upper())
        axis.set_xlabel("Client count / class count, t (%)")
        axis.set_ylabel("mAP (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    figure_root = output_root / "paper" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "heterogeneity.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "heterogeneity.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(output_root: Path, rows: list[dict[str, object]]) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.3 * len(datasets), 4.0), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        selected = [row for row in rows if row["dataset"] == dataset]
        names = [str(row["method"]).replace("local_topk_no_quantile_alignment", "no quantile align") for row in selected]
        values = [float(row["fixed_0.9_mAP"]) for row in selected]
        colors = ["#d1495b" if name == "topk_512" else "#6c91bf" for name in names]
        axis.barh(names, values, color=colors)
        axis.set_title(dataset.upper())
        axis.set_xlabel("mAP (%)")
        axis.grid(axis="x", alpha=0.25)
        low = max(0.0, min(values) - 2.0)
        axis.set_xlim(low, max(values) + 0.8)
    fig.tight_layout()
    figure_root = output_root / "paper" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "ablation.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "ablation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_partial_annotation(output_root: Path, rows: list[dict[str, object]]) -> None:
    datasets = sorted({str(row["dataset"]) for row in rows})
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.0 * len(datasets), 3.8), squeeze=False)
    for axis, dataset in zip(axes[0], datasets):
        selected = sorted(
            (row for row in rows if row["dataset"] == dataset),
            key=lambda row: float(row["masked_positive_annotations_percent"]),
        )
        retained = [100.0 - float(row["masked_positive_annotations_percent"]) for row in selected]
        axis.plot(retained, [row["global_fusion_mAP_mean"] for row in selected], "o-", label="global fusion")
        axis.plot(retained, [row["classwise_fusion_mAP_mean"] for row in selected], "s--", label="classwise fusion")
        axis.invert_xaxis()
        axis.set_title(dataset.upper())
        axis.set_xlabel("Retained positive annotations (%)")
        axis.set_ylabel("mAP (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    fig.tight_layout()
    figure_root = output_root / "paper" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_root / "partial_annotation.pdf", bbox_inches="tight")
    fig.savefig(figure_root / "partial_annotation.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_latex(output_root: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        r"\begin{tabular}{llrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & $t$=10 & 20 & 40 & 60 & 80 & 100 & Avg. \\",
        r"\midrule",
    ]
    for dataset in ("voc07", "coco", "nus"):
        selected = [
            row for row in rows
            if row["dataset"] == dataset
            and row["method"] in {"FedMPT (CVPR 2026)", "QAFedPASA", "FedPASA-Full"}
        ]
        for row in selected:
            values = " & ".join(f"{float(row[f't{t}']):.2f}" for t in T_VALUES)
            lines.append(f"{dataset.upper()} & {row['method']} & {values} & {float(row['average']):.2f} " + r"\\")
        if selected:
            lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines.pop()
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path = output_root / "paper" / "main_table.tex"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["voc07", "voc12", "coco", "nus"])
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "icme")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "outputs" / "icme" / "cache")
    parser.add_argument("--bootstrap", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=2026)
    args = parser.parse_args()

    rows = comparison_rows(args.output_root, args.datasets)
    comparison_fields = ["dataset", "method", "result_type", *[f"t{t}" for t in T_VALUES], "average"]
    write_csv(args.output_root / "paper" / "main_comparison.csv", rows, comparison_fields)
    ablations = ablation_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "ablation.csv",
        ablations,
        ["dataset", "method", "fixed_0.9_mAP", "patch_only_mAP", "cls_only_mAP", "entropy_adaptive_mAP"],
    )
    systems = systems_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "systems.csv",
        systems,
        [
            "dataset", "t_percent", "clients", "seed", "communication_MiB",
            "cluster_seconds", "adapter_fit_seconds", "max_abs_weight_error", "max_abs_bias_error",
        ],
    )
    annotation_rows = annotation_invariance_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "annotation_mask_invariance.csv",
        annotation_rows,
        ["dataset", "masked_training_annotations_percent", "mAP", "invariance_basis", "empirical_claim"],
    )
    dirichlet = dirichlet_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "dirichlet.csv",
        dirichlet,
        [
            "dataset", "beta", "clients", "seed", "mAP", "communication_MiB",
            "max_abs_weight_error", "max_abs_bias_error", "equivalent",
        ],
    )
    seed_rows = full_seed_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "full_seed_summary.csv",
        seed_rows,
        ["dataset", "record", "mAP", "std_mAP", "seeds"],
    )
    partial_rows = partial_annotation_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "partial_annotation.csv",
        partial_rows,
        [
            "dataset", "masked_positive_annotations_percent", "ridge_mAP_mean",
            "ridge_mAP_std", "global_fusion_mAP_mean", "classwise_fusion_mAP_mean",
            "classwise_fusion_mAP_std",
        ],
    )
    label_rows = label_feature_rows(args.output_root, args.datasets)
    write_csv(
        args.output_root / "paper" / "label_feature_ablation.csv",
        label_rows,
        [
            "dataset", "features", "selected_alpha", "validation_mAP", "test_mAP",
            "margin_over_strongest_fedmpt_points", "passes_plus_2",
        ],
    )
    write_latex(args.output_root, rows)
    plot_main(args.output_root, rows)
    plot_ablation(args.output_root, ablations)
    plot_partial_annotation(args.output_root, partial_rows)
    manifest = {
        "external_source": SOURCE_URL,
        "external_values": "FedMPT Table 1 published mAP; not locally rerun",
        "datasets_requested": args.datasets,
        "comparison_rows": len(rows),
        "ablation_rows": len(ablations),
        "systems_rows": len(systems),
        "annotation_invariance_rows": len(annotation_rows),
        "dirichlet_rows": len(dirichlet),
        "full_seed_rows": len(seed_rows),
        "partial_annotation_rows": len(partial_rows),
        "label_feature_ablation_rows": len(label_rows),
        "reviewer_experiment_artifacts": [
            name
            for name in (
                "label_noise.json",
                "label_noise.csv",
                "client_participation.json",
                "client_participation.csv",
                "temperature_sensitivity.json",
                "temperature_sensitivity.csv",
                "communication_compression.json",
                "communication_compression.csv",
                "per_class_failure_analysis.json",
                "per_class_failure_analysis.csv",
                "qualitative_failure_examples.json",
                "official_fedmpt_runnability.json",
                "estimator_preserving_evidence.json",
                "estimator_preserving_evidence.csv",
                "estimator_preserving_summary.csv",
                "classwise_complementarity.json",
                "classwise_complementarity.csv",
                "classwise_complementarity_summary.csv",
                "communication_runtime_breakdown.json",
                "communication_runtime_breakdown.csv",
                "reviewer_checklist.json",
                "REVIEWER_CHECKLIST.md",
            )
            if (args.output_root / "paper" / name).exists()
        ],
    }
    atomic_json(args.output_root / "paper" / "artifact_manifest.json", manifest)
    if args.bootstrap:
        build_statistics(
            args.output_root,
            args.cache_root,
            args.datasets,
            replicates=args.bootstrap,
            seed=args.bootstrap_seed,
        )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
