from __future__ import annotations

import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs" / "icme" / "paper"
OFFICIAL = ROOT / "outputs" / "icme" / "official_fedmpt"


def read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def pickle_length(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return len(pickle.load(handle))


def build_official_audit() -> dict[str, object]:
    diagnostic_path = (
        OFFICIAL
        / "repaired_full_protocol_1round"
        / "fedmpt"
        / "2_0.0"
        / "34"
        / "official_runtime.json"
    )
    fallback_path = (
        OFFICIAL
        / "repaired_same_machine_1round"
        / "fedmpt"
        / "2_0.0"
        / "34"
        / "official_runtime.json"
    )
    runtime_path = diagnostic_path if diagnostic_path.exists() else fallback_path
    runtime = read(runtime_path) if runtime_path.exists() else None
    train_cluster = OFFICIAL / "clusters" / "label_train" / "VOC2007_2.pkl"
    test_cluster = OFFICIAL / "clusters" / "label_test" / "VOC2007_2.pkl"
    original_cluster = OFFICIAL / "clusters" / "label" / "VOC2007_2.pkl"
    main = read(ROOT / "outputs" / "icme" / "voc07" / "main.json")
    t10 = next(
        row for row in main["runs"] if row["t_percent"] == 10 and row["seed"] == 7
    )
    ridge_one_client = float(
        read(
            ROOT
            / "outputs"
            / "icme"
            / "voc07"
            / "full_seeds"
            / "seed7"
            / "ridge"
            / "result.json"
        )["method"]["estimated_communication_MiB"]
    )
    fedpasa_fp64 = float(t10["communication_MiB"]) + 2.0 * ridge_one_client
    fedpasa_fp32 = float(t10["communication_MiB"]) + ridge_one_client
    audit = {
        "status": "REPAIRED_DIAGNOSTIC_ONLY_NOT_AN_OFFICIAL_NUMERICAL_REPRODUCTION",
        "comparison_policy": {
            "main_table": "use the published FedMPT Table 1 values as external results",
            "local_diagnostic": "use only for source runnability, memory, runtime, and communication auditing",
            "reason": "execution required semantic model repairs whose equivalence to the authors' unpublished working setup cannot be established",
        },
        "upstream_blockers": [
            {
                "stage": "checkpoint",
                "observed": "official loader only searched the user cache or network",
                "repair": "FEDMPT_CLIP_CHECKPOINT hook with official SHA256 verification",
                "changes_predictions": False,
            },
            {
                "stage": "cluster cache",
                "observed": "feature directory was not created and first-write pickle.dump return value replaced the in-memory assignments with None",
                "repair": "create output-scoped cache directory and preserve assignments after pickle.dump",
                "changes_predictions": False,
            },
            {
                "stage": "Windows DataLoader",
                "observed": "Can't pickle local object '_transform.<locals>.<lambda>'",
                "repair": "use zero DataLoader workers on Windows",
                "changes_predictions": False,
            },
            {
                "stage": "condition prompt",
                "observed": "73-token prompt was added to CLIP's 77-token positional embedding at --cond 5",
                "repair": "one condition placeholder for the one inserted attr_ctx vector",
                "changes_predictions": True,
            },
            {
                "stage": "vanilla patch aggregation",
                "observed": "conv1d received a four-dimensional weight and the patch-count logit multiplier saturated sigmoid outputs with zero prompt gradients",
                "repair": "use [C,D,1] conv1d weights and the fixed 5x factor used by sibling multi-label models",
                "changes_predictions": True,
            },
            {
                "stage": "mixed precision",
                "observed": "Float condition tokens entered a Half CLIP transformer",
                "repair": "cast attr_ctx rows to the CLIP dtype",
                "changes_predictions": False,
            },
            {
                "stage": "VOC split",
                "observed": "trainval.txt was paired with *_train.txt labels, retaining 2,501 of 5,011 training images",
                "repair": "pair trainval.txt with *_trainval.txt annotations",
                "changes_predictions": True,
            },
            {
                "stage": "test protocol",
                "observed": "train and test shared one 2,501-entry cluster cache, truncating test iteration to 2,501 of 4,952 images",
                "repair": "use split-specific clustering caches",
                "changes_predictions": True,
            },
            {
                "stage": "CLIP clustering",
                "observed": "CPU image features were multiplied by CUDA text features",
                "repair": "move text features to CPU for cached zero-shot probability construction",
                "changes_predictions": False,
            },
            {
                "stage": "CLIP clustering tensor shape",
                "observed": "the modified CLIP encoder returned [B,D,L] patch tensors where the clustering code assumed [B,D] global embeddings",
                "repair": "select the CLS token at index 0 for the intended global-image clustering feature",
                "changes_predictions": True,
            },
        ],
        "dataset_cardinality_audit": {
            "expected_train": 5011,
            "expected_test": 4952,
            "original_shared_cache_entries": pickle_length(original_cluster),
            "repaired_train_cache_entries": pickle_length(train_cluster),
            "repaired_test_cache_entries": pickle_length(test_cluster),
        },
        "repaired_one_round_diagnostic": runtime,
        "systems_comparison_at_voc07_t10": {
            "fedpasa_full_fp64_one_shot_MiB": fedpasa_fp64,
            "fedpasa_full_fp32_ridge_one_shot_MiB": fedpasa_fp32,
            "fedpasa_qafed_adapter_fit_seconds": t10["adapter_fit_seconds"],
            "fedpasa_kmeans_seconds": t10["cluster_seconds"],
            "fedmpt_prompt_MiB_per_round": None if runtime is None else runtime[
                "estimated_bidirectional_prompt_communication_MiB"
            ],
            "fedmpt_prompt_MiB_for_50_rounds": None if runtime is None else 50.0 * runtime[
                "estimated_bidirectional_prompt_communication_MiB"
            ],
        },
        "source_edits_are_disclosed": True,
        "published_external_mAP_is_not_replaced_by_diagnostic_mAP": True,
    }
    destination = PAPER / "official_fedmpt_runnability.json"
    destination.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def build_checklist(audit: dict[str, object]) -> None:
    verification = read(PAPER / "verification_report.json")
    noise = read(PAPER / "label_noise.json")
    participation = read(PAPER / "client_participation.json")
    compression = read(PAPER / "communication_compression.json")
    failures = read(PAPER / "per_class_failure_analysis.json")
    estimator = read(PAPER / "estimator_preserving_evidence.json")
    complementarity = read(PAPER / "classwise_complementarity.json")
    communication = read(PAPER / "communication_runtime_breakdown.json")
    qualitative_path = PAPER / "qualitative_failure_examples.json"
    qualitative = read(qualitative_path) if qualitative_path.exists() else {"rows": []}
    checks = [
        ("Importance and value", "covered", "Four established multi-label benchmarks; one-shot, no-example-transmission federated adaptation."),
        ("Recent and classic baselines", "covered", "Eleven published methods from 2022–2026 plus local zero-shot CLS/CLIP, central PIAA, ridge, and component baselines."),
        ("Same-protocol external reruns", "blocked", "The released FedMPT benchmark requires prediction-changing repairs, supports no VOC12 entry, and local raw data cover only VOC07/COCO. No repaired diagnostic is promoted to a controlled accuracy baseline."),
        ("Estimator-preserving evidence", "covered", "Centralized, Naive-A/B/C, and FedPASA are evaluated over Dirichlet β=0.1/0.3/0.5 plus KMeans on four datasets; adapter, score, and bank errors are reported."),
        ("Classwise complementarity", "covered", "Full-vs-Ridge AP differences cover all 201 classes, including positive, approximately equal, and negative groups."),
        ("Ablations", "covered", "K, purification, covariance, quantile alignment, reliability, CLS/patch fusion, and seven label-aware feature combinations."),
        ("Statistical significance", "covered", "Three training-side splits/seeds and 1,000-replicate image bootstrap with confidence intervals and superiority probabilities."),
        ("Heterogeneity", "covered", "18 KMeans settings and nine Dirichlet settings per completed dataset."),
        ("Missing annotations", "covered", "Five missing-positive rates × three seeds × four datasets."),
        ("Wrong annotations", "covered", "Four prevalence-preserving pair-noise rates × three seeds × four datasets."),
        ("Client participation", "covered", "25/50/75/100% availability × three client-subset seeds × four datasets with frozen deployment fusion."),
        ("Hyperparameter sensitivity", "covered", "K=128/256/512/1024 and 0.25×–4× purification-temperature grid."),
        ("Systems and communication", "covered", "Exact Quantile/GDA/Ridge payload decomposition, client/statistic fitting time, wall clock, recovery, compression, and official-source runnability audit."),
        ("Failure and long-tail analysis", "covered", "Every class is reported with prevalence, AP, gain/loss, rare/common quartiles, and weakest classes."),
        ("Human evaluation", "not applicable", "The endpoint is objective multi-label recognition with fixed ground-truth labels; subjective human preference would not validate the claimed metric."),
        ("Test-set selection leakage", "passed", "All hyperparameters and fusion weights use training-side validation only; robustness grids are predeclared."),
        ("External result provenance", "passed", "Published values and local runs are typed separately; repaired FedMPT output is excluded from the main comparison."),
    ]
    payload = {
        "source": "https://maxwellforbes.com/posts/how-to-get-a-paper-accepted/",
        "interpretation": "acceptance-oriented evidence checklist adapted to objective federated multi-label classification",
        "overall_main_plus_2_pass": verification["overall_pass"],
        "checks": [
            {"requirement": name, "status": status, "evidence": evidence}
            for name, status, evidence in checks
        ],
        "counts": {
            "pair_noise_rows": sum(len(value["rows"]) for value in noise.values()),
            "client_participation_rows": sum(
                len(value["rows"]) for value in participation.values()
            ),
            "compression_rows": sum(len(value["rows"]) for value in compression.values()),
            "per_class_rows": sum(
                1
                for line in (PAPER / "per_class_failure_analysis.csv")
                .read_text(encoding="utf-8-sig")
                .splitlines()[1:]
                if line
            ),
            "qualitative_examples": len(qualitative["rows"]),
            "official_source_blockers": len(audit["upstream_blockers"]),
            "estimator_rows": len(estimator["rows"]),
            "complementarity_classes": sum(
                len(value["per_class"]) for value in complementarity["datasets"].values()
            ),
            "communication_breakdown_rows": len(communication["rows"]),
        },
        "remaining_experimental_items": [
            "A valid same-protocol external baseline table still requires runnable author-equivalent implementations, complete raw datasets, and three-seed training for FedMPT plus two peers.",
        ],
        "remaining_non_experimental_items": [
            "Integrate the strongest new tables/figures into the ICME page limit.",
            "Write takeaway-first captions and an explicit limitations paragraph.",
            "Keep repaired-source diagnostics in the supplement, not the main accuracy table.",
        ],
    }
    (PAPER / "reviewer_checklist.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# ICME reviewer evidence checklist",
        "",
        "Mapped from Maxwell Forbes' acceptance guide: baseline completeness, ablations, statistical "
        "evidence, high-quality figures, and proactively answering likely reviewer questions.",
        "",
        "| Requirement | Status | Evidence |",
        "|---|---|---|",
    ]
    for name, status, evidence in checks:
        lines.append(f"| {name} | {status} | {evidence} |")
    lines.extend(
        [
            "",
            "## Important interpretation boundaries",
            "",
            "- Human evaluation is deliberately omitted because the task has objective multilabel ground truth; adding subjective raters would not test the paper's claim.",
            "- Naive INT8 compression failed, and extreme 25% client availability is unstable under non-IID subsets. These are reported as limitations, not hidden.",
            "- The released FedMPT launcher required prediction-changing repairs. Its repaired local diagnostic is not labeled an official reproduction; the main table continues to use published external numbers.",
            "- The same-protocol external rerun requested by reviewers remains an explicit experimental gap; it is not reclassified as writing-only work.",
            "- Naive local-union uses a conservative numerical safeguard that clamps mean+std purification thresholds to the observed maximum when saturated ties would otherwise empty a class.",
            "",
            "## Remaining experimental work",
            "",
            "- Obtain author-equivalent runnable code and complete VOC07/VOC12/COCO/NUS raw data, then run FedMPT plus two peer methods under one client split, metric, backbone, and three seeds.",
            "",
            "## Remaining writing-only work",
            "",
            "- Select the most informative new results for the ICME page budget.",
            "- Use takeaway-first captions for robustness, compression, and failure figures.",
            "- State deployment assumptions and the 25%-availability/low-bit limits explicitly.",
        ]
    )
    (PAPER / "REVIEWER_CHECKLIST.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    PAPER.mkdir(parents=True, exist_ok=True)
    audit = build_official_audit()
    build_checklist(audit)
    print(PAPER / "REVIEWER_CHECKLIST.md")


if __name__ == "__main__":
    main()
