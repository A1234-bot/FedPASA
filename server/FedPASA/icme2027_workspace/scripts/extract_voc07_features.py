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
PIAA_ROOT = ROOT / "third_party" / "PIAA"
if str(PIAA_ROOT) not in sys.path:
    sys.path.insert(0, str(PIAA_ROOT))

import clip  # noqa: E402
import scclip  # noqa: E402


PASCAL_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "dining table", "dog", "horse", "motorbike", "person",
    "potted plant", "sheep", "sofa", "train", "tv monitor",
)


class ImageDataset(Dataset):
    def __init__(self, image_root: Path, identifiers: list[str], transform):
        self.paths = tuple(image_root / f"{identifier}.jpg" for identifier in identifiers)
        self.transform = transform
        missing = [path for path in self.paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[0]}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert("RGB"))


def read_annotations(path: Path) -> tuple[list[str], np.ndarray]:
    identifiers: list[str] = []
    targets: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.strip().split()
        if not fields:
            continue
        target = np.zeros(len(PASCAL_CLASSES), dtype=np.uint8)
        target[[int(value) for value in fields[1:]]] = 1
        identifiers.append(fields[0])
        targets.append(target)
    return identifiers, np.stack(targets)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def text_features(model, device: str) -> torch.Tensor:
    tokens = clip.tokenize([f"a photo of a {name}" for name in PASCAL_CLASSES]).to(device)
    with torch.inference_mode():
        features = model.encode_text(tokens).float()
    return F.normalize(features, dim=-1)


def extract_array(
    model,
    loader: DataLoader,
    path: Path,
    *,
    kind: str,
    device: str,
) -> dict[str, object]:
    if kind not in {"cls", "patch"}:
        raise ValueError(kind)
    shape = (len(loader.dataset), 512) if kind == "cls" else (len(loader.dataset), 196, 512)
    temporary = path.with_name(f".{path.name}.partial.npy")
    output = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float16, shape=shape)
    started = perf_counter()
    offset = 0
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for batch_index, images in enumerate(loader):
            images = images.to(device, non_blocking=True)
            if kind == "cls":
                features = model.encode_image(images).float()
            else:
                features = model.encode_image(images, return_all=True).float()
            features = F.normalize(features, dim=-1)
            stop = offset + len(images)
            output[offset:stop] = features.cpu().half().numpy()
            offset = stop
            if batch_index % 20 == 0 or offset == len(loader.dataset):
                print(f"EXTRACT kind={kind} images={offset}/{len(loader.dataset)}", flush=True)
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
    parser.add_argument(
        "--voc-root",
        type=Path,
        default=Path(r"D:\project\VScodeProject\BCP\data\raw\voc2007"),
    )
    parser.add_argument(
        "--weight-path",
        type=Path,
        default=Path(r"D:\project\VScodeProject\BCP\external\FedMPT-main\ViT-B-16.pt"),
    )
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "voc07" / "cache")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    train_root = (
        args.voc_root / "VOCtrainval_06-Nov-2007" / "VOCdevkit" / "VOC2007"
    )
    test_root = args.voc_root / "VOCtest_06-Nov-2007" / "VOCdevkit" / "VOC2007"
    split_paths = {
        "train": PIAA_ROOT / "imageset" / "voc07train_labels.txt",
        "test": PIAA_ROOT / "imageset" / "voc07val_labels.txt",
    }
    split_roots = {"train": train_root / "JPEGImages", "test": test_root / "JPEGImages"}
    annotations = {name: read_annotations(path) for name, path in split_paths.items()}
    if len(annotations["train"][0]) != 5011 or len(annotations["test"][0]) != 4952:
        raise RuntimeError("unexpected VOC07 split size")

    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "train_targets.npy", annotations["train"][1])
    np.save(args.output / "test_targets.npy", annotations["test"][1])
    (args.output / "train_ids.txt").write_text("\n".join(annotations["train"][0]) + "\n", encoding="utf-8")
    (args.output / "test_ids.txt").write_text("\n".join(annotations["test"][0]) + "\n", encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    report: dict[str, object] = {
        "dataset": "PASCAL VOC 2007",
        "train_images": len(annotations["train"][0]),
        "test_images": len(annotations["test"][0]),
        "weight_path": str(args.weight_path),
        "weight_sha256": sha256(args.weight_path),
        "source": "PIAA official image lists and preprocessing",
        "artifacts": [],
    }

    print("Loading standard CLIP for CLS features", flush=True)
    model, preprocess = clip.load(str(args.weight_path), device=device, jit=False)
    model.eval()
    np.save(args.output / "text_features.npy", text_features(model, device).cpu().numpy())
    for split in ("train", "test"):
        destination = args.output / f"{split}_cls.npy"
        if destination.exists() and not args.force:
            print(f"REUSE {destination}", flush=True)
            continue
        dataset = ImageDataset(split_roots[split], annotations[split][0], preprocess)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        report["artifacts"].append(
            extract_array(model, loader, destination, kind="cls", device=device)
        )
    del model
    torch.cuda.empty_cache()

    print("Loading SC-CLIP for patch features", flush=True)
    model, preprocess = scclip.load(str(args.weight_path), device=device, jit=False)
    model.eval()
    for split in ("train", "test"):
        destination = args.output / f"{split}_patch.npy"
        if destination.exists() and not args.force:
            print(f"REUSE {destination}", flush=True)
            continue
        dataset = ImageDataset(split_roots[split], annotations[split][0], preprocess)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        report["artifacts"].append(
            extract_array(model, loader, destination, kind="patch", device=device)
        )
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
