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

from fedpasa.benchmarks import BENCHMARKS, EXTERNAL_MAPS  # noqa: E402


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["voc07", "voc12", "coco", "nus"])
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    output_root = ROOT / "outputs" / "icme"
    statistics_path = output_root / "paper" / "statistics.json"
    statistics = read(statistics_path) if statistics_path.exists() else {"replicates": 0, "datasets": {}}
    partial_path = output_root / "paper" / "partial_annotation.json"
    partial = read(partial_path) if partial_path.exists() else {}
    label_noise_path = output_root / "paper" / "label_noise.json"
    label_noise = read(label_noise_path) if label_noise_path.exists() else {}
    participation_path = output_root / "paper" / "client_participation.json"
    participation = read(participation_path) if participation_path.exists() else {}
    temperature_path = output_root / "paper" / "temperature_sensitivity.json"
    temperature = read(temperature_path) if temperature_path.exists() else {}
    compression_path = output_root / "paper" / "communication_compression.json"
    compression = read(compression_path) if compression_path.exists() else {}
    failure_path = output_root / "paper" / "per_class_failure_analysis.json"
    failures = read(failure_path) if failure_path.exists() else {}
    qualitative_path = output_root / "paper" / "qualitative_failure_examples.json"
    qualitative = read(qualitative_path) if qualitative_path.exists() else {"rows": []}
    official_path = output_root / "paper" / "official_fedmpt_runnability.json"
    official = read(official_path) if official_path.exists() else {}
    estimator_path = output_root / "paper" / "estimator_preserving_evidence.json"
    estimator = read(estimator_path) if estimator_path.exists() else {"rows": []}
    complementarity_path = output_root / "paper" / "classwise_complementarity.json"
    complementarity = read(complementarity_path) if complementarity_path.exists() else {"datasets": {}}
    communication_path = output_root / "paper" / "communication_runtime_breakdown.json"
    communication = read(communication_path) if communication_path.exists() else {"rows": []}
    reports: dict[str, object] = {}
    all_available_pass = True
    complete = True

    for dataset in args.datasets:
        dataset_root = output_root / dataset
        required = {
            "main": dataset_root / "main.json",
            "dirichlet": dataset_root / "dirichlet.json",
            "ablation": dataset_root / "ablation.json",
            "cv_fusion": dataset_root / "cv_fusion" / "result.json",
        }
        missing = [name for name, path in required.items() if not path.exists()]
        if missing:
            reports[dataset] = {"status": "pending", "missing": missing}
            complete = False
            continue

        main_report = read(required["main"])
        dirichlet_report = read(required["dirichlet"])
        cv_report = read(required["cv_fusion"])
        seed_results = sorted(dataset_root.glob("full_seeds/seed*/ridge_patch_fusion/result.json"))
        seed_reports = [read(path) for path in seed_results]
        main_runs = main_report["runs"]
        dirichlet_runs = dirichlet_report["runs"]
        main_equivalent = all(
            run["central_recovery_max_abs_weight"] <= 1e-4
            and run["central_recovery_max_abs_bias"] <= 1e-6
            for run in main_runs
        )
        dirichlet_equivalent = all(run["numerically_equivalent_at_tolerance"] for run in dirichlet_runs)
        no_test_selection = (
            not cv_report["test_labels_used_for_selection"]
            and all(not report["protocol"]["test_labels_used_for_selection"] for report in seed_reports)
        )
        threshold_checks: list[bool] = []
        margins: list[float] = []
        if dataset in EXTERNAL_MAPS:
            ours = float(cv_report["test_mAP"])
            strongest = np.max(np.asarray(list(EXTERNAL_MAPS[dataset].values())), axis=0)
            margins = (ours - strongest).tolist()
            threshold_checks = (ours >= strongest + 2.0).tolist()
        checks = {
            "main_18_runs": len(main_runs) == 18,
            "main_federated_equivalence": main_equivalent,
            "dirichlet_9_runs": len(dirichlet_runs) == 9,
            "dirichlet_equivalence": dirichlet_equivalent,
            "three_selection_seeds": len(seed_reports) == 3,
            "test_labels_unused_for_selection": no_test_selection,
            "all_external_t_values_plus_2": all(threshold_checks) if threshold_checks else None,
            "bootstrap_1000_available": (
                statistics.get("replicates") == 1000
                and dataset in statistics.get("datasets", {})
            ),
            "partial_annotation_15_configs": (
                dataset in partial and len(partial[dataset]["rows"]) == 15
            ),
            "pair_label_noise_12_configs": (
                dataset in label_noise
                and len(label_noise[dataset]["rows"]) == 12
                and all(
                    not row["test_labels_used_for_selection"]
                    for row in label_noise[dataset]["rows"]
                )
            ),
            "client_participation_12_configs": (
                dataset in participation
                and len(participation[dataset]["rows"]) == 12
                and all(
                    not row["test_labels_used_for_selection"]
                    for row in participation[dataset]["rows"]
                )
            ),
            "temperature_5_predeclared_configs": (
                dataset in temperature
                and len(temperature[dataset]["rows"]) == 5
                and sum(
                    bool(row["predeclared_reference_multiplier"])
                    for row in temperature[dataset]["rows"]
                ) == 1
            ),
            "compression_4_codecs_and_fp32_stable": (
                dataset in compression
                and len(compression[dataset]["rows"]) == 4
                and abs(
                    next(
                        row for row in compression[dataset]["rows"]
                        if row["codec"] == "fp32"
                    )["mAP_delta_vs_fp64"]
                ) < 1e-3
            ),
            "per_class_failure_analysis_available": (
                dataset in failures
                and len(failures[dataset]["five_lowest_AP"]) == 5
                and not failures[dataset]["test_labels_used_for_selection"]
            ),
            "estimator_preserving_24_rows": (
                len([row for row in estimator["rows"] if row["dataset"] == dataset]) == 24
            ),
            "fedpasa_preserves_central_estimator": all(
                row["max_abs_weight_error"] <= 1e-5
                and row["max_abs_bias_error"] <= 1e-5
                and row["max_abs_score_error"] <= 1e-6
                for row in estimator["rows"]
                if row["dataset"] == dataset
                and row["method"] == "FedPASA quantile-aligned"
            ),
            "naive_estimator_deviation_observed": any(
                row["relative_weight_l2_error"] > 1e-3
                for row in estimator["rows"]
                if row["dataset"] == dataset and row["method"].startswith("Naive-")
            ),
            "classwise_complementarity_available": (
                dataset in complementarity["datasets"]
                and len(complementarity["datasets"][dataset]["per_class"])
                == BENCHMARKS[dataset].class_count
            ),
        }
        dataset_pass = all(value for value in checks.values() if value is not None)
        all_available_pass &= dataset_pass
        reports[dataset] = {
            "status": "pass" if dataset_pass else "fail",
            "checks": checks,
            "cv_mAP": cv_report["test_mAP"],
            "margins_over_strongest_external_by_t": margins,
        }

    expected_figure_stems = {
        "ablation",
        "client_participation",
        "communication_compression",
        "heterogeneity",
        "label_noise_robustness",
        "partial_annotation",
        "qualitative_failure_examples",
        "tail_failure_analysis",
        "temperature_sensitivity",
        "estimator_preserving_non_iid",
        "classwise_complementarity",
    }
    figure_root = output_root / "paper" / "figures"
    reviewer_suite = {
        "qualitative_failure_examples_12": (
            len(qualitative["rows"]) == 12
            and not qualitative.get("protocol", {}).get(
                "test_labels_used_for_model_or_hyperparameter_selection", True
            )
        ),
        "official_fedmpt_audit_disclosed": (
            official.get("status")
            == "REPAIRED_DIAGNOSTIC_ONLY_NOT_AN_OFFICIAL_NUMERICAL_REPRODUCTION"
            and official.get("source_edits_are_disclosed") is True
            and official.get("published_external_mAP_is_not_replaced_by_diagnostic_mAP")
            is True
            and official.get("dataset_cardinality_audit", {}).get(
                "repaired_train_cache_entries"
            )
            == 5011
            and official.get("dataset_cardinality_audit", {}).get(
                "repaired_test_cache_entries"
            )
            == 4952
            and official.get("repaired_one_round_diagnostic") is not None
        ),
        "communication_breakdown_exact": (
            len(communication["rows"]) == len(args.datasets)
            and all(abs(row["breakdown_minus_reported_MiB"]) < 1e-9 for row in communication["rows"])
            and all(row.get("qafedpasa_client_uploads") == 4 for row in communication["rows"])
            and all(
                row.get("communication_scope")
                == "client-to-server uploads only, dense FP32 upper bound"
                for row in communication["rows"]
            )
            and all(row.get("purification_moments_MiB", 0.0) > 0.0 for row in communication["rows"])
        ),
        "all_submission_figures_png_and_pdf": all(
            (figure_root / f"{stem}.png").exists()
            and (figure_root / f"{stem}.png").stat().st_size > 0
            and (figure_root / f"{stem}.pdf").exists()
            and (figure_root / f"{stem}.pdf").stat().st_size > 0
            for stem in expected_figure_stems
        ),
    }
    result = {
        "complete": complete,
        "all_available_datasets_pass": all_available_pass,
        "reviewer_suite": reviewer_suite,
        "overall_pass": complete and all_available_pass and all(reviewer_suite.values()),
        "datasets": reports,
    }
    destination = output_root / "paper" / "verification_report.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    if args.require_complete and not result["overall_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
