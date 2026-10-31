from __future__ import annotations

import argparse
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PIAA_IMAGESETS = ROOT / "third_party" / "PIAA" / "imageset"


def identifiers(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def download_one(task: tuple[str, Path], retries: int) -> tuple[str, int, bool]:
    url, destination = task
    if destination.exists() and destination.stat().st_size > 0:
        return destination.name, destination.stat().st_size, True
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "FedPASA-ICME/1.0"})
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as handle:
                while block := response.read(1024 * 1024):
                    handle.write(block)
            size = temporary.stat().st_size
            if size <= 0:
                raise OSError("downloaded an empty image")
            os.replace(temporary, destination)
            return destination.name, size, False
        except (OSError, urllib.error.URLError, TimeoutError) as error:
            last_error = error
            if temporary.exists():
                temporary.unlink()
            time.sleep(min(2**attempt, 8))
    raise RuntimeError(f"failed after {retries} attempts: {url}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "coco2014")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    tasks: list[tuple[str, Path]] = []
    for split, list_name in (
        ("train2014", "cocotrain_labels.txt"),
        ("val2014", "cocoval_labels.txt"),
    ):
        for identifier in identifiers(PIAA_IMAGESETS / list_name):
            filename = identifier if identifier.lower().endswith(".jpg") else f"{identifier}.jpg"
            tasks.append(
                (
                    f"http://images.cocodataset.org/{split}/{filename}",
                    args.output_root / split / filename,
                )
            )
    if args.limit is not None:
        tasks = tasks[: args.limit]

    started = time.perf_counter()
    downloaded_bytes = 0
    reused = 0
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, task, args.retries): task for task in tasks}
        for completed, future in enumerate(as_completed(futures), start=1):
            try:
                _, size, was_reused = future.result()
                downloaded_bytes += size
                reused += int(was_reused)
            except Exception as error:  # keep the full queue progressing, then fail loudly
                failures.append(str(error))
            if completed % 500 == 0 or completed == len(tasks):
                elapsed = time.perf_counter() - started
                print(
                    f"COCO_DOWNLOAD files={completed}/{len(tasks)} reused={reused} "
                    f"rate={completed / max(elapsed, 1e-9):.1f}_files/s "
                    f"observed_MiB={downloaded_bytes / 1024**2:.1f} failures={len(failures)}",
                    flush=True,
                )
    if failures:
        print("\n".join(failures[:20]), file=sys.stderr)
        raise RuntimeError(f"{len(failures)} COCO images failed to download")


if __name__ == "__main__":
    main()
