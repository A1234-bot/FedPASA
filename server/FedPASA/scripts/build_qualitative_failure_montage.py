from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS  # noqa: E402


DATASETS = ("voc07", "voc12", "coco", "nus")
FALLBACK_ROOTS = {
    "voc07": Path(
        r"D:\project\VScodeProject\BCP\data\raw\voc2007\VOCtest_06-Nov-2007\VOCdevkit\VOC2007\JPEGImages"
    ),
}


def cache_root(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


def image_root(dataset: str, cache: Path) -> Path:
    if dataset in FALLBACK_ROOTS:
        return FALLBACK_ROOTS[dataset]
    manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
    return Path(manifest["test_root"])


def image_path(dataset: str, root: Path, identifier: str) -> Path:
    name = BENCHMARKS[dataset].image_name(identifier)
    return root / name


def select_existing(
    dataset: str,
    root: Path,
    identifiers: list[str],
    scores: np.ndarray,
    candidates: np.ndarray,
    *,
    descending: bool,
) -> tuple[int, Path]:
    order = candidates[np.argsort(scores[candidates])]
    if descending:
        order = order[::-1]
    for index in order:
        path = image_path(dataset, root, identifiers[int(index)])
        if path.exists():
            return int(index), path
    raise FileNotFoundError(f"no existing qualitative image for {dataset}")


def main() -> None:
    paper = ROOT / "outputs" / "icme" / "paper"
    failure = json.loads(
        (paper / "per_class_failure_analysis.json").read_text(encoding="utf-8")
    )
    figure, axes = plt.subplots(len(DATASETS), 3, figsize=(10.2, 11.4), squeeze=False)
    records: list[dict[str, object]] = []
    for row, dataset in enumerate(DATASETS):
        cache = cache_root(dataset)
        targets = np.load(cache / "test_targets.npy")
        scores = np.load(ROOT / "outputs" / "icme" / dataset / "cv_fusion" / "test_scores.npy")
        identifiers = (cache / "test_ids.txt").read_text(encoding="utf-8").splitlines()
        root = image_root(dataset, cache)
        worst = failure[dataset]["five_lowest_AP"][0]
        class_id = int(worst["class_id"])
        values = scores[:, class_id]
        positive = np.flatnonzero(targets[:, class_id] > 0)
        negative = np.flatnonzero(targets[:, class_id] == 0)
        selections = (
            ("High-score negative", negative, True),
            ("Low-score positive", positive, False),
            ("High-score positive", positive, True),
        )
        for column, (kind, candidates, descending) in enumerate(selections):
            index, path = select_existing(
                dataset,
                root,
                identifiers,
                values,
                candidates,
                descending=descending,
            )
            with Image.open(path) as raw:
                image = raw.convert("RGB")
                axes[row, column].imshow(image)
            axes[row, column].axis("off")
            axes[row, column].set_title(
                f"{dataset.upper()} · {worst['class_name']}\n{kind}, score={values[index]:.2f}",
                fontsize=9,
            )
            records.append(
                {
                    "dataset": dataset,
                    "class_id": class_id,
                    "class_name": worst["class_name"],
                    "example_type": kind,
                    "test_index": index,
                    "image_id": identifiers[index],
                    "image_path": str(path),
                    "score": float(values[index]),
                    "ground_truth": int(targets[index, class_id]),
                }
            )
    figure.suptitle(
        "Qualitative boundary cases for each dataset's lowest-AP class",
        fontsize=13,
        y=0.995,
    )
    figure.tight_layout()
    figures = paper / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "qualitative_failure_examples.png", dpi=220, bbox_inches="tight")
    figure.savefig(figures / "qualitative_failure_examples.pdf", bbox_inches="tight")
    plt.close(figure)
    payload = {
        "protocol": {
            "class_selection": "lowest per-class AP under the final fixed model, for descriptive failure analysis only",
            "examples": "highest-scored negative, lowest-scored positive, and highest-scored positive",
            "test_labels_used_for_model_or_hyperparameter_selection": False,
        },
        "rows": records,
    }
    (paper / "qualitative_failure_examples.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(figures / "qualitative_failure_examples.png")


if __name__ == "__main__":
    main()
