from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import EXTERNAL_MAPS  # noqa: E402


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def optional_read(path: Path) -> dict[str, object]:
    return read(path) if path.exists() else {}


def main() -> None:
    paper_root = ROOT / "outputs" / "icme" / "paper"
    verification = read(paper_root / "verification_report.json")
    statistics = read(paper_root / "statistics.json")
    partial = read(paper_root / "partial_annotation.json")
    federated = read(paper_root / "federated_full_verification.json")
    label_noise = optional_read(paper_root / "label_noise.json")
    participation = optional_read(paper_root / "client_participation.json")
    temperature = optional_read(paper_root / "temperature_sensitivity.json")
    compression = optional_read(paper_root / "communication_compression.json")
    failures = optional_read(paper_root / "per_class_failure_analysis.json")
    official_audit = optional_read(paper_root / "official_fedmpt_runnability.json")
    estimator = optional_read(paper_root / "estimator_preserving_evidence.json")
    complementarity = optional_read(paper_root / "classwise_complementarity.json")
    communication = optional_read(paper_root / "communication_runtime_breakdown.json")
    lines = [
        "# ICME full-experiment report",
        "",
        f"Overall status: **{'PASS' if verification['overall_pass'] else 'IN PROGRESS'}**.",
        "",
        "The locked method is FedPASA-Full: quantile-aligned label-free patch statistics, "
        "one-round float64 federated ridge sufficient statistics over CLS/patch-mean/patch-max-text "
        "features, and three training-side split-selected per-class fusion rules collapsed into one "
        "affine inference score. Test labels are never used for hyperparameter or fusion selection.",
        "",
        "## Main results",
        "",
        "| Dataset | FedPASA-Full mAP | Strongest published external | Margin | 95% image-bootstrap CI | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for dataset in ("voc07", "voc12", "coco", "nus"):
        result_path = ROOT / "outputs" / "icme" / dataset / "cv_fusion" / "result.json"
        if not result_path.exists():
            lines.append(f"| {dataset.upper()} | pending | pending | pending | pending | pending |")
            continue
        result = read(result_path)
        ours = float(result["test_mAP"])
        if dataset in EXTERNAL_MAPS:
            strongest = float(np.max(np.asarray(list(EXTERNAL_MAPS[dataset].values()))))
            external = f"{strongest:.2f}"
            margin = f"{ours - strongest:+.2f}"
        else:
            external = "n/a"
            margin = "n/a"
        dataset_statistics = statistics.get("datasets", {}).get(dataset, {})
        ci = dataset_statistics.get("ci95_mAP")
        ci_text = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "pending"
        state = verification["datasets"].get(dataset, {}).get("status", "pending")
        lines.append(
            f"| {dataset.upper()} | {ours:.2f} | {external} | {margin} | {ci_text} | {state} |"
        )

    lines.extend(
        [
            "",
            "Published external values are transcribed from FedMPT Table 1 "
            "(https://arxiv.org/html/2605.28347); they are not presented as local reruns.",
            "",
            "## Federated and statistical checks",
            "",
            "| Dataset | Clients (t=60%) | Central ridge mAP | Federated ridge mAP | Max score error | Combined communication (MiB) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in federated["rows"]:
        lines.append(
            f"| {str(row['dataset']).upper()} | {row['clients']} | {row['central_ridge_mAP']:.5f} | "
            f"{row['federated_ridge_mAP']:.5f} | {row['max_abs_test_score_error']:.2e} | "
            f"{row['combined_communication_MiB']:.2f} |"
        )

    lines.extend(
        [
            "",
            "Each completed dataset has 18 KMeans heterogeneity runs (six client counts × three seeds), "
            "nine Dirichlet runs (β=0.1/0.3/0.5 × three seeds), three independent training-side "
            "selection splits, and 1,000 image-level bootstrap replicates.",
            "",
            "## Partial positive-annotation robustness",
            "",
            "Values below are three-seed global-fusion means; missing labels are selected positive "
            "training annotations only, while test labels remain complete and unused for selection.",
            "",
            "| Dataset | 10% missing | 30% missing | 50% missing | 70% missing | 90% missing |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in ("voc07", "voc12", "coco", "nus"):
        if dataset not in partial:
            lines.append(f"| {dataset.upper()} | pending | pending | pending | pending | pending |")
            continue
        values = [row["global_fusion_mAP_mean"] for row in partial[dataset]["summary"]]
        lines.append(f"| {dataset.upper()} | " + " | ".join(f"{value:.2f}" for value in values) + " |")

    if label_noise and participation and temperature and compression:
        lines.extend(
            [
                "",
                "## Reviewer-oriented robustness and sensitivity",
                "",
                "All grids and subset seeds are fixed before test evaluation. Label-noise alpha/fusion "
                "selection uses only the clean training-side validation split; client-dropout fusion and "
                "calibration are frozen before the availability shock.",
                "",
                "| Dataset | 20% pair-noise fusion mAP | 75% clients fusion mAP | Temperature range (mAP) | FP32 ridge ΔmAP |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset in ("voc07", "voc12", "coco", "nus"):
            noise_row = next(
                row for row in label_noise[dataset]["summary"]
                if row["pair_noise_percent"] == 20.0
            )
            participation_row = next(
                row for row in participation[dataset]["summary"]
                if row["requested_fraction_percent"] == 75.0
            )
            temperature_range = float(temperature[dataset]["range_mAP"])
            compression_row = next(
                row for row in compression[dataset]["rows"] if row["codec"] == "fp32"
            )
            lines.append(
                f"| {dataset.upper()} | {noise_row['global_fusion_mAP_mean']:.2f} ± "
                f"{noise_row['global_fusion_mAP_std']:.2f} | "
                f"{participation_row['frozen_fusion_mAP_mean']:.2f} ± "
                f"{participation_row['frozen_fusion_mAP_std']:.2f} | "
                f"{temperature_range:.4f} | {compression_row['mAP_delta_vs_fp64']:+.5f} |"
            )
        lines.extend(
            [
                "",
                "FP32 halves ridge-packet communication on every dataset with an absolute mAP change "
                "below 0.00002. FP16 is not lossless on COCO/NUS, and naive per-tensor INT8 fails; both "
                "negative findings are retained in `communication_compression.csv`.",
            ]
        )

    if failures:
        lines.extend(
            [
                "",
                "## Long-tail and failure analysis",
                "",
                "| Dataset | Rare-quartile AP | Common-quartile AP | Classes hurt by fusion | Lowest-AP class |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for dataset in ("voc07", "voc12", "coco", "nus"):
            value = failures[dataset]
            worst = value["five_lowest_AP"][0]
            lines.append(
                f"| {dataset.upper()} | {value['rare_quartile_full_AP_mean']:.2f} | "
                f"{value['common_quartile_full_AP_mean']:.2f} | "
                f"{value['classes_hurt_by_fusion']} | {worst['class_name']} "
                f"({worst['fedpasa_full_AP']:.2f}) |"
            )
        lines.append(
            "Per-class prevalence, QAFedPASA AP, FedPASA-Full AP, gains, largest losses, and the "
            "five weakest classes are preserved in the machine-readable failure artifacts."
        )
        lines.append(
            "`figures/qualitative_failure_examples.*` visualizes a high-score negative, "
            "low-score positive, and high-score positive for the weakest class of each dataset."
        )

    if estimator:
        lines.extend(
            [
                "",
                "## Estimator-preserving federation evidence",
                "",
                "The controlled estimator study evaluates centralized PIAA/Ridge, Naive-A local-TopK "
                "union, Naive-B local-GDA averaging, Naive-C local-Ridge averaging, and FedPASA over "
                "Dirichlet β=0.1/0.3/0.5 plus a ten-client KMeans partition.",
                "",
                "| Dataset | Partition | Naive-A mAP | Naive-B mAP | FedPASA mAP |",
                "|---|---|---:|---:|---:|",
            ]
        )
        summary = estimator["summary"]
        for dataset in ("coco", "nus"):
            for partition in (
                "dirichlet_beta_0.1",
                "dirichlet_beta_0.3",
                "dirichlet_beta_0.5",
                "kmeans",
            ):
                lookup = {
                    row["method"]: row
                    for row in summary
                    if row["dataset"] == dataset and row["partition"] == partition
                }
                lines.append(
                    f"| {dataset.upper()} | {partition} | "
                    f"{lookup['Naive-A local-TopK union']['mAP_mean']:.2f} | "
                    f"{lookup['Naive-B local-GDA average']['mAP_mean']:.2f} | "
                    f"{lookup['FedPASA quantile-aligned']['mAP_mean']:.2f} |"
                )

    if complementarity:
        lines.extend(
            [
                "",
                "## Ridge/QAFedPASA classwise complementarity",
                "",
                "| Dataset | Full > Ridge | Full ≈ Ridge | Full < Ridge | Avg positive ΔAP |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for dataset in ("voc07", "voc12", "coco", "nus"):
            row = complementarity["datasets"][dataset]["summary"]
            lines.append(
                f"| {dataset.upper()} | {row['full_gt_ridge']} | {row['full_approx_ridge']} | "
                f"{row['full_lt_ridge']} | {row['average_positive_gain_AP']:.2f} |"
            )

    if communication:
        lines.extend(
            [
                "",
                "## Communication and runtime breakdown",
                "",
                "QAFedPASA values use a dense FP32 client-to-server accounting. Its four uploads include a separate per-class `{count, sum(q), sum(q^2)}` purification packet (`3C` scalars per client) after the preliminary global GDA is returned.",
                "",
                "| Dataset | Quantile MiB | GDA moments MiB | Ridge stats MiB | Total MiB | 10-client evidence wall clock |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in communication["rows"]:
            lines.append(
                f"| {row['dataset'].upper()} | {row['quantile_MiB']:.2f} | "
                f"{row['gda_moments_MiB']:.2f} | {row['ridge_stats_MiB']:.2f} | "
                f"{row['total_MiB']:.2f} | {row['total_condition_wall_clock_seconds']:.2f}s |"
            )

    if official_audit:
        runtime = official_audit.get("repaired_one_round_diagnostic")
        lines.extend(
            [
                "",
                "## Official FedMPT same-machine audit",
                "",
                f"Audit status: **{official_audit['status']}**. The published FedMPT values remain "
                "external comparison numbers, because the released launcher required algorithmic and "
                "protocol repairs before its documented VOC configuration could execute. The repaired "
                "run is therefore reported only as a runnability/runtime diagnostic, not as an official "
                "reproduction.",
                "",
                f"Observed upstream blockers: {len(official_audit['upstream_blockers'])}. See "
                "`official_fedmpt_runnability.json` for exact exceptions, repairs, dataset cardinalities, "
                "and the diagnostic runtime/communication record.",
            ]
        )
        if runtime:
            global_metrics = runtime.get("global_metrics", {})
            lines.append(
                "After repairing the disclosed blockers, the full 5,011/4,952 VOC train/test "
                f"protocol completed one diagnostic round at {global_metrics.get('mAP', float('nan')):.2f} "
                f"mAP in {runtime['training_plus_final_evaluation_seconds']:.2f}s, with "
                f"{runtime['peak_cuda_memory_MiB']:.1f} MiB peak CUDA memory and "
                f"{runtime['estimated_bidirectional_prompt_communication_MiB']:.2f} MiB/round. "
                "The diagnostic mAP is excluded from accuracy comparisons because prediction-changing "
                "source repairs prevent an equivalence claim."
            )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `verification_report.json`: machine-readable pass/fail audit.",
            "- `main_comparison.csv` and `main_table.tex`: external comparison table.",
            "- `statistics.json`: seed variability and bootstrap confidence intervals.",
            "- `ablation.csv`, `label_feature_ablation.csv`, and `partial_annotation.csv`: component studies.",
            "- `label_noise.csv`, `client_participation.csv`, and `temperature_sensitivity.csv`: robustness and sensitivity.",
            "- `communication_compression.csv`: bandwidth/accuracy frontier, including failed low-bit settings.",
            "- `per_class_failure_analysis.csv`: long-tail, worst-class, and negative-gain evidence.",
            "- `qualitative_failure_examples.json`: auditable image IDs and scores behind the failure montage.",
            "- `official_fedmpt_runnability.json`: same-machine source audit and repaired diagnostic policy.",
            "- `estimator_preserving_evidence.*`: centralized/naive/FedPASA non-IID estimator comparison.",
            "- `classwise_complementarity.*`: Full-vs-Ridge per-class AP evidence.",
            "- `communication_runtime_breakdown.*`: exact payload decomposition and measured runtime fields.",
            "- `REVIEWER_CHECKLIST.md`: acceptance-oriented evidence map and remaining writing-only items.",
            "- `systems.csv` and `federated_full_verification.json`: runtime, communication, and numerical equivalence.",
            "- `figures/`: heterogeneity, ablation, robustness, compression, and failure-analysis plots.",
        ]
    )
    destination = paper_root / "README.md"
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
