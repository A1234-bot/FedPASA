from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fedpasa.benchmarks import BENCHMARKS  # noqa: E402


def resolve_cache(dataset: str) -> Path:
    generic = ROOT / "outputs" / "icme" / "cache" / dataset
    legacy = ROOT / "outputs" / "voc07" / "cache"
    return legacy if dataset == "voc07" and not generic.exists() else generic


@torch.inference_mode()
def extract(split: str, cache_root: Path, *, batch_images: int, device: str) -> None:
    patch_cache = np.load(cache_root / f"{split}_patch.npy", mmap_mode="r")
    text = torch.from_numpy(np.load(cache_root / "text_features.npy")).float().to(device)
    text = F.normalize(text, dim=-1)
    paths = {
        "patch_mean": cache_root / f"{split}_patch_mean.npy",
        "patch_max_text": cache_root / f"{split}_patch_max_text.npy",
    }
    outputs = {
        "patch_mean": np.lib.format.open_memmap(
            paths["patch_mean"].with_name(f".{paths['patch_mean'].name}.partial.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(len(patch_cache), patch_cache.shape[-1]),
        ),
        "patch_max_text": np.lib.format.open_memmap(
            paths["patch_max_text"].with_name(f".{paths['patch_max_text'].name}.partial.npy"),
            mode="w+",
            dtype=np.float16,
            shape=(len(patch_cache), text.shape[0]),
        ),
    }
    for start in range(0, len(patch_cache), batch_images):
        stop = min(start + batch_images, len(patch_cache))
        patch = torch.from_numpy(np.asarray(patch_cache[start:stop], dtype=np.float32)).to(device)
        patch = F.normalize(patch, dim=-1)
        outputs["patch_mean"][start:stop] = F.normalize(patch.mean(dim=1), dim=-1).cpu().half().numpy()
        outputs["patch_max_text"][start:stop] = (patch @ text.T).max(dim=1).values.cpu().half().numpy()
        if start % (batch_images * 50) == 0 or stop == len(patch_cache):
            print(f"PATCH_SUMMARY split={split} images={stop}/{len(patch_cache)}", flush=True)
    for output in outputs.values():
        output.flush()
    del output
    outputs.clear()
    for name in paths:
        temporary = paths[name].with_name(f".{paths[name].name}.partial.npy")
        temporary.replace(paths[name])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(BENCHMARKS), required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=("train", "test"))
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cache_root = resolve_cache(args.dataset)
    for split in args.splits:
        required = (cache_root / f"{split}_patch.npy", cache_root / "text_features.npy")
        if not all(path.exists() for path in required):
            raise FileNotFoundError(required)
        if all((cache_root / f"{split}_{name}.npy").exists() for name in ("patch_mean", "patch_max_text")):
            print(f"PATCH_SUMMARY_REUSE split={split}", flush=True)
            continue
        extract(split, cache_root, batch_images=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
