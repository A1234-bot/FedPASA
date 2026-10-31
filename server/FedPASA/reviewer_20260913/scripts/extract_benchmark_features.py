from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
PIAA_ROOT = ROOT / "third_party" / "PIAA"
for value in (SRC, PIAA_ROOT):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

import clip  # noqa: E402
import scclip  # noqa: E402
from fedpasa.benchmarks import BENCHMARKS, Benchmark  # noqa: E402


class ImageDataset(Dataset):
    def __init__(self, root: Path, identifiers: list[str], benchmark: Benchmark, transform):
        self.paths = tuple(root / benchmark.image_name(identifier) for identifier in identifiers)
        self.transform = transform
        missing = [path for path in self.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def read_annotations(path: Path, classes: int) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    targets: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        target = np.zeros(classes, dtype=np.uint8)
        if len(fields) > 1:
            target[[int(value) for value in fields[1:]]] = 1
        identifiers.append(fields[0])
        targets.append(target)
    return identifiers, np.stack(targets)


def default_roots(dataset: str) -> tuple[Path, Path]:
    bcp = Path(r"D:\project\VScodeProject\BCP\data\raw")
    if dataset == "voc07":
        base = bcp / "voc2007"
        return (
            base / "VOCtrainval_06-Nov-2007" / "VOCdevkit" / "VOC2007" / "JPEGImages",
            base / "VOCtest_06-Nov-2007" / "VOCdevkit" / "VOC2007" / "JPEGImages",
        )
    if dataset == "voc12":
        root = bcp / "voc2012" / "VOCdevkit" / "VOC2012" / "JPEGImages"
        return root, root
    if dataset == "coco":
        base = ROOT / "data" / "coco2014"
        return base / "train2014", base / "val2014"
    if dataset == "nus":
        root = bcp / "nuswide" / "kaggle_tc21" / "NUS-WIDE" / "images"
        return root, root
    raise ValueError(dataset)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def make_text_features(model, class_names: tuple[str, ...], device: str) -> torch.Tensor:
    tokens = clip.tokenize([f"a photo of a {name}" for name in class_names]).to(device)
    return F.normalize(model.encode_text(tokens).float(), dim=-1)


def extract_array(model, loader: DataLoader, path: Path, kind: str, device: str) -> dict[str, object]:
    shape = (len(loader.dataset), 512) if kind == "cls" else (len(loader.dataset), 196, 512)
    temporary = path.with_name(f".{path.name}.partial.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    started = perf_counter()
    offset = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            features = (
                model.encode_image(images).float()
                if kind == "cls"
                else model.encode_image(images, return_all=True).float()
            )
            features = F.normalize(features, dim=-1)
            stop = offset + len(images)
            output[offset:stop] = features.cpu().half().numpy()
            offset = stop
            if batch_index % 100 == 0 or offset == len(loader.dataset):
                elapsed = perf_counter() - started
                rate = offset / max(elapsed, 1e-6)
                print(
                    f"EXTRACT dataset={path.parent.name} kind={kind} "
                    f"images={offset}/{len(loader.dataset)} rate={rate:.1f}/s",
                    flush=True,
                )
    output.flush()
    del output
    temporary.replace(path)
    return {
        "path": str(path),
        "shape": list(shape),
        "dtype": "float16",
        "seconds": perf_counter() - started,
        "peak_cuda_memory_MiB": torch.cuda.max_memory_allocated() / 1024**2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--train-root", type=Path)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument(
        "--weight-path",
        type=Path,
        default=Path(r"D:\project\VScodeProject\BCP\external\FedMPT-main\ViT-B-16.pt"),
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "icme" / "cache")
    parser.add_argument("--features", nargs="+", choices=("cls", "patch"), default=("cls", "patch"))
    parser.add_argument("--splits", nargs="+", choices=("train", "test"), default=("train", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    benchmark = BENCHMARKS[args.dataset]
    default_train, default_test = default_roots(args.dataset)
    roots = {"train": args.train_root or default_train, "test": args.test_root or default_test}
    lists = {
        "train": PIAA_ROOT / "imageset" / benchmark.train_list,
        "test": PIAA_ROOT / "imageset" / benchmark.test_list,
    }
    annotations = {
        split: read_annotations(lists[split], benchmark.class_count) for split in ("train", "test")
    }
    if len(annotations["train"][0]) != benchmark.expected_train:
        raise RuntimeError("unexpected training split size")
    if len(annotations["test"][0]) != benchmark.expected_test:
        raise RuntimeError("unexpected test split size")

    output = args.output_root / args.dataset
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        np.save(output / f"{split}_targets.npy", annotations[split][1])
        (output / f"{split}_ids.txt").write_text(
            "\n".join(annotations[split][0]) + "\n", encoding="utf-8"
        )

    manifest_path = output / "manifest.json"
    report = {
        "dataset": args.dataset,
        "train_images": benchmark.expected_train,
        "test_images": benchmark.expected_test,
        "classes": benchmark.class_count,
        "train_root": str(roots["train"]),
        "test_root": str(roots["test"]),
        "weight_path": str(args.weight_path),
        "weight_sha256": sha256(args.weight_path),
        "artifacts": [],
    }
    if manifest_path.exists() and not args.force:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["artifacts"] = previous.get("artifacts", [])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "cls" in args.features:
        model, preprocess = clip.load(str(args.weight_path), device=device, jit=False)
        model.eval()
        np.save(output / "text_features.npy", make_text_features(model, benchmark.classes, device).cpu().numpy())
        for split in args.splits:
            destination = output / f"{split}_cls.npy"
            if destination.exists() and not args.force:
                print(f"REUSE {destination}", flush=True)
                continue
            loader = DataLoader(
                ImageDataset(roots[split], annotations[split][0], benchmark, preprocess),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            report["artifacts"].append(extract_array(model, loader, destination, "cls", device))
            manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        del model
        torch.cuda.empty_cache()

    if "patch" in args.features:
        model, preprocess = scclip.load(str(args.weight_path), device=device, jit=False)
        model.eval()
        for split in args.splits:
            destination = output / f"{split}_patch.npy"
            if destination.exists() and not args.force:
                print(f"REUSE {destination}", flush=True)
                continue
            loader = DataLoader(
                ImageDataset(roots[split], annotations[split][0], benchmark, preprocess),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True,
            )
            report["artifacts"].append(extract_array(model, loader, destination, "patch", device))
            manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
