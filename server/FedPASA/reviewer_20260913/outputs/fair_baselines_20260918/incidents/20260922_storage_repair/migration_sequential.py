"""Byte-identical NVMe image staging; no transforms or experiment choices change."""
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fedpasa.benchmarks import BENCHMARKS

OUT = ROOT / 'outputs/fair_baselines_20260918'
EVIDENCE = OUT / 'incidents/20260922_storage_repair'
ORIGINAL = Path('/root/autodl-tmp/FedPASA/icme2027_workspace/outputs/icme/cache')


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.partial')
    tmp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    tmp.replace(path)


def main():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (EVIDENCE / 'migration.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert shutil.disk_usage('/root/autodl-tmp').free > 80 * 2**30
    started = time.time()
    total_bytes = 0
    counts = {}
    override = OUT / 'verified_image_roots.json'
    roots = json.loads(override.read_text()) if override.exists() else {}
    with (EVIDENCE / 'files.jsonl').open('a', buffering=1) as ledger:
        for dataset, part in [('coco', 'train'), ('coco', 'test'), ('nus', 'train'), ('nus', 'test')]:
            key = dataset + '_' + part
            manifest = json.loads((ORIGINAL / dataset / 'manifest.json').read_text())
            source_root = Path(manifest[part + '_root'])
            dest_root = source_root if dataset == 'coco' else Path('/root/autodl-tmp/FedPASA/data/nuswide_nvme/images')
            dest_root.mkdir(parents=True, exist_ok=True)
            assert dest_root.resolve().is_relative_to(Path('/root/autodl-tmp/FedPASA/data'))
            ids_file = ROOT / 'outputs/icme/cache' / dataset / (part + '_ids.txt')
            ids = ids_file.read_text().splitlines()
            for index, identifier in enumerate(ids):
                name = BENCHMARKS[dataset].image_name(identifier)
                src, dst = source_root / name, dest_root / name
                assert dst.parent == dest_root
                if dataset == 'coco' and not dst.is_symlink():
                    # Already committed atomic copies are checked again in the final audit.
                    assert dst.is_file()
                    continue
                original_link = os.readlink(src) if src.is_symlink() else None
                resolved = src.resolve(strict=True)
                data = resolved.read_bytes()
                sha = hashlib.sha256(data).hexdigest()
                if dst.exists() and not dst.is_symlink() and hashlib.sha256(dst.read_bytes()).hexdigest() == sha:
                    pass
                else:
                    temp = dst.with_name(dst.name + '.nvme-partial')
                    with temp.open('wb') as handle:
                        handle.write(data)
                    assert hashlib.sha256(temp.read_bytes()).hexdigest() == sha
                    # Save original link/source before atomically switching the same path.
                    ledger.write(json.dumps({'dataset': dataset, 'part': part, 'path': str(dst), 'source': str(resolved), 'original_link': original_link, 'sha256': sha, 'bytes': len(data)}) + '\n')
                    os.replace(temp, dst)
                total_bytes += len(data)
                if index % 1000 == 0:
                    progress = {'status': 'COPYING', 'dataset': dataset, 'part': part, 'processed': index + 1, 'total': len(ids), 'elapsed_seconds': time.time() - started, 'bytes_verified': total_bytes}
                    save(EVIDENCE / 'progress.json', progress)
                    print(json.dumps(progress), flush=True)
            counts[key] = len(ids)
            if dataset == 'nus':
                roots[key] = {'root': str(dest_root), 'count': len(ids), 'ids_sha256': hashlib.sha256(ids_file.read_bytes()).hexdigest(), 'source_root': str(source_root), 'verified': True}
                save(override, roots)
        # Independent readback of every materialized file against its copied-source digest.
        latest = {}
        for line in (EVIDENCE / 'files.jsonl').read_text().splitlines():
            row = json.loads(line)
            latest[row['path']] = row
        for name, row in latest.items():
            assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == row['sha256'], name
        save(EVIDENCE / 'completion.json', {'status': 'COMPLETE', 'counts': counts, 'verified_files': len(latest), 'bytes_verified': total_bytes, 'elapsed_seconds': time.time() - started, 'scope': 'Original compressed image bytes; no image transforms, labels, order, splits or selection changed. Migration time is infrastructure overhead, not fitting time.'})
        save(EVIDENCE / 'progress.json', {'status': 'COMPLETE', 'counts': counts})


if __name__ == '__main__':
    main()
