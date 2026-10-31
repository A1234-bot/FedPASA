from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COCO_ROOT = ROOT / "data" / "coco2014"
DOWNLOAD_ROOT = COCO_ROOT / "_downloads"
SPECS = {
    "train": {
        "archive": DOWNLOAD_ROOT / "train2014.zip",
        "base_size": 6_035_080_815,
        "total_size": 13_510_573_713,
        "parts": [DOWNLOAD_ROOT / f"train.part{index:02d}" for index in range(9)],
        "part_sizes": [830_610_322] * 9,
    },
    "val": {
        "archive": DOWNLOAD_ROOT / "val2014.zip",
        "base_size": 4_194_799_182,
        "total_size": 6_645_013_297,
        "parts": [DOWNLOAD_ROOT / f"val.part{index:02d}" for index in range(3)],
        "part_sizes": [816_738_039, 816_738_038, 816_738_038],
    },
}


def safe_members(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        resolved = (destination / member.filename).resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"unsafe zip member: {member.filename}")


def append_parts(name: str) -> Path:
    spec = SPECS[name]
    archive = Path(spec["archive"])
    parts = list(spec["parts"])
    expected_sizes = list(spec["part_sizes"])
    for path, expected in zip(parts, expected_sizes):
        actual = path.stat().st_size if path.exists() else -1
        if actual != expected:
            raise ValueError(f"{path.name}: expected {expected} bytes, found {actual}")

    current = archive.stat().st_size
    base = int(spec["base_size"])
    total = int(spec["total_size"])
    if current < base or current > total:
        raise ValueError(f"{archive.name}: unexpected current size {current}")
    appended = current - base
    cumulative = 0
    with archive.open("ab") as output:
        for part, expected in zip(parts, expected_sizes):
            already = min(max(appended - cumulative, 0), expected)
            if already < expected:
                with part.open("rb") as source:
                    source.seek(already)
                    shutil.copyfileobj(source, output, length=16 * 1024 * 1024)
                print(f"MERGE archive={archive.name} part={part.name}", flush=True)
            cumulative += expected
    if archive.stat().st_size != total:
        raise ValueError(f"{archive.name}: merged size mismatch")
    return archive


def verify_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in {path.name}: {bad}")
        safe_members(archive, COCO_ROOT)
        print(f"ZIP_OK archive={path.name} files={len(archive.infolist())}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--cleanup-parts", action="store_true")
    args = parser.parse_args()

    archives = [append_parts(name) for name in ("train", "val")]
    for archive in archives:
        verify_zip(archive)
    if args.cleanup_parts:
        for spec in SPECS.values():
            for part in spec["parts"]:
                part.unlink()
        for temporary in (
            DOWNLOAD_ROOT / "range_speed_test.bin",
            DOWNLOAD_ROOT / "s3_range_speed_test.bin",
        ):
            if temporary.exists():
                temporary.unlink()
        print("CLEANUP removed verified range fragments", flush=True)
    if args.extract:
        for archive_path in archives:
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(COCO_ROOT)
            print(f"EXTRACT_OK archive={archive_path.name}", flush=True)


if __name__ == "__main__":
    main()
