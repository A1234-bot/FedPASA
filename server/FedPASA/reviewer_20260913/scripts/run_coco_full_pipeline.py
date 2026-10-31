from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
OUTPUT = ROOT / "outputs" / "icme"
COCO = OUTPUT / "coco"
SEEDS = (7, 17, 27)
ALPHAS = ("0.0001", "0.0003", "0.001", "0.003", "0.01", "0.03", "0.1")
FEATURE_SETS = (
    ("cls",),
    ("patch_mean",),
    ("patch_max_text",),
    ("cls", "patch_mean"),
    ("cls", "patch_max_text"),
    ("patch_mean", "patch_max_text"),
    ("cls", "patch_mean", "patch_max_text"),
)


def run(label: str, script: str, *arguments: str, marker: Path | None = None) -> None:
    if marker is not None and marker.exists():
        print(f"PIPELINE_REUSE step={label} marker={marker}", flush=True)
        return
    command = [sys.executable, str(SCRIPTS / script), *arguments]
    print(f"PIPELINE_START step={label} command={' '.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    if marker is not None and not marker.exists():
        raise RuntimeError(f"{label} finished without expected artifact: {marker}")
    print(f"PIPELINE_OK step={label}", flush=True)


def report_name(features: tuple[str, ...]) -> str:
    return "label_ridge" if features == ("cls",) else "label_ridge_" + "_".join(features)


def main() -> None:
    cache = OUTPUT / "cache" / "coco"
    run(
        "feature_extraction",
        "extract_benchmark_features.py",
        "--dataset", "coco",
        "--batch-size", "48",
        "--num-workers", "4",
        marker=cache / "test_patch.npy",
    )
    run(
        "patch_summaries",
        "extract_patch_summaries.py",
        "--dataset", "coco",
        "--batch-size", "256",
        marker=cache / "test_patch_max_text.npy",
    )
    for phase, marker in (
        ("main", COCO / "main.json"),
        ("dirichlet", COCO / "dirichlet.json"),
        ("ablation", COCO / "ablation.json"),
    ):
        run(
            f"qafedpasa_{phase}",
            "run_icme_experiments.py",
            "--dataset", "coco",
            "--phase", phase,
            "--bank-batch-size", "128",
            "--eval-batch-size", "128",
            marker=marker,
        )

    for features in FEATURE_SETS:
        name = report_name(features)
        run(
            f"ridge_{'_'.join(features)}",
            "run_label_ridge.py",
            "--dataset", "coco",
            "--features", *features,
            "--alphas", *ALPHAS,
            marker=COCO / name / "result.json",
        )

    full_name = report_name(FEATURE_SETS[-1])
    run(
        "default_full_fusion",
        "run_label_patch_fusion.py",
        "--dataset", "coco",
        "--ridge-name", full_name,
        "--batch-size", "128",
        marker=COCO / f"{full_name}_patch_fusion" / "result.json",
    )

    for seed in SEEDS:
        ridge_name = f"full_seeds/seed{seed}/ridge"
        run(
            f"seed{seed}_ridge",
            "run_label_ridge.py",
            "--dataset", "coco",
            "--seed", str(seed),
            "--report-root", str(COCO / ridge_name),
            "--features", *FEATURE_SETS[-1],
            "--alphas", *ALPHAS,
            marker=COCO / ridge_name / "result.json",
        )
        run(
            f"seed{seed}_fusion",
            "run_label_patch_fusion.py",
            "--dataset", "coco",
            "--ridge-name", ridge_name,
            "--batch-size", "128",
            marker=COCO / f"{ridge_name}_patch_fusion" / "result.json",
        )

    run(
        "cv_aggregation",
        "build_cv_fusion_ensemble.py",
        "--datasets", "coco",
        marker=COCO / "cv_fusion" / "result.json",
    )
    run("federated_full_verification", "verify_federated_full.py", "--datasets", "coco")
    run("partial_annotation", "run_partial_annotation.py", "--datasets", "coco")
    run(
        "paper_artifacts_bootstrap",
        "build_icme_artifacts.py",
        "--bootstrap", "1000",
    )
    run("strict_audit", "audit_icme_results.py", "--require-complete")
    run("paper_report", "build_icme_report.py")
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT, check=True)

    audit = json.loads((OUTPUT / "paper" / "verification_report.json").read_text(encoding="utf-8"))
    if not audit.get("overall_pass"):
        raise RuntimeError("strict audit did not pass")
    print("COCO_FULL_PIPELINE_PASS", flush=True)


if __name__ == "__main__":
    main()
