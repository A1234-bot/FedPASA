from __future__ import annotations

import argparse
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_ROOT = ROOT / "data" / "coco2014" / "_downloads"
TRAIN_URL = "https://images.cocodataset.org/zips/train2014.zip"
VAL_URL = "https://images.cocodataset.org/zips/val2014.zip"
SEGMENTS = [
    (TRAIN_URL, 6_035_080_815, 6_865_691_136, DOWNLOAD_ROOT / "train.part00"),
    (TRAIN_URL, 6_865_691_137, 7_696_301_458, DOWNLOAD_ROOT / "train.part01"),
    (TRAIN_URL, 7_696_301_459, 8_526_911_780, DOWNLOAD_ROOT / "train.part02"),
    (TRAIN_URL, 8_526_911_781, 9_357_522_102, DOWNLOAD_ROOT / "train.part03"),
    (TRAIN_URL, 9_357_522_103, 10_188_132_424, DOWNLOAD_ROOT / "train.part04"),
    (TRAIN_URL, 10_188_132_425, 11_018_742_746, DOWNLOAD_ROOT / "train.part05"),
    (TRAIN_URL, 11_018_742_747, 11_849_353_068, DOWNLOAD_ROOT / "train.part06"),
    (TRAIN_URL, 11_849_353_069, 12_679_963_390, DOWNLOAD_ROOT / "train.part07"),
    (TRAIN_URL, 12_679_963_391, 13_510_573_712, DOWNLOAD_ROOT / "train.part08"),
    (VAL_URL, 4_194_799_182, 5_011_537_220, DOWNLOAD_ROOT / "val.part00"),
    (VAL_URL, 5_011_537_221, 5_828_275_258, DOWNLOAD_ROOT / "val.part01"),
    (VAL_URL, 5_828_275_259, 6_645_013_296, DOWNLOAD_ROOT / "val.part02"),
]


def download(url: str, start: int, end: int, path: Path, retries: int) -> str:
    expected = end - start + 1
    context = ssl._create_unverified_context()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > expected:
        raise ValueError(f"{path.name} is larger than its range")
    consecutive_failures = 0
    while True:
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return f"RANGE_OK file={path.name} bytes={expected}"
        offset = start + current
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-{end}"})
        before = current
        try:
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(
                        f"unexpected response status={response.status} range={content_range!r}"
                    )
                with path.open("ab") as output:
                    while True:
                        block = response.read(4 * 1024 * 1024)
                        if not block:
                            break
                        output.write(block)
            consecutive_failures = 0
        except Exception as error:  # Network resets are expected on this endpoint.
            after = path.stat().st_size if path.exists() else 0
            consecutive_failures = 0 if after > before else consecutive_failures + 1
            if consecutive_failures > retries:
                raise RuntimeError(f"{path.name} exhausted retries at {after}/{expected}") from error
            print(
                f"RANGE_RESUME file={path.name} bytes={after}/{expected} error={type(error).__name__}",
                flush=True,
            )
            time.sleep(min(2.0 * max(consecutive_failures, 1), 20.0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--retries", type=int, default=100)
    args = parser.parse_args()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download, url, start, end, path, args.retries)
            for url, start, end, path in SEGMENTS
        ]
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
