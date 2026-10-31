"""Package the existing experiment state for Linux without frozen feature arrays."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'tmp/server_migration'
DEST.mkdir(parents=True, exist_ok=True)
files = set()
for folder in ('src', 'scripts', 'tests', 'third_party/FedMPT'):
    for path in (ROOT / folder).rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix in ('.py', '.gz', '.txt', '.md', '.yaml', '.yml'):
            files.add(path)
files.add(ROOT / 'requirements.txt')
files.add(ROOT / 'outputs/icme/paper/official_fedmpt_runnability.json')
for path in (ROOT / 'outputs/icme/supplementary').rglob('*'):
    if path.is_file() and (path.suffix in ('.json', '.csv', '.md') or 'fedmpt_repaired' in path.parts):
        files.add(path)
samples = []
for dataset in ('coco', 'nus'):
    cache = ROOT / 'outputs/icme/cache' / dataset
    manifest = json.loads((cache / 'manifest.json').read_text(encoding='utf-8'))
    files.add(cache / 'manifest.json')
    files.add(ROOT / 'outputs/icme' / dataset / 'partitions/clients10_seed7.npy')
    for split in ('train', 'test'):
        files.add(cache / f'{split}_ids.txt')
        files.add(cache / f'{split}_targets.npy')
        ids = (cache / f'{split}_ids.txt').read_text(encoding='utf-8').splitlines()
        for i in sorted({round(j * (len(ids)-1) / 63) for j in range(64)}):
            identifier = ids[i]
            name = Path(identifier).name if dataset == 'nus' else identifier
            if not name.endswith('.jpg'):
                name += '.jpg'
            source = Path(manifest[split + '_root']) / name
            samples.append(dict(dataset=dataset, split=split, name=name,
                                sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
sample_path = DEST / 'image_hash_samples.json'
sample_path.write_text(json.dumps(samples, indent=2), encoding='utf-8')
files.add(sample_path)
hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
hash_path = DEST / 'file_hashes.json'
hash_path.write_text(json.dumps(hashes, indent=2), encoding='utf-8')
files.add(hash_path)
archive = DEST / 'fedpasa_resume.tar.gz'
with tarfile.open(archive, 'w:gz', compresslevel=3) as tar:
    for path in sorted(files):
        tar.add(path, arcname=path.relative_to(ROOT).as_posix(), recursive=False)
print(json.dumps(dict(archive=str(archive), bytes=archive.stat().st_size, files=len(files), image_samples=len(samples))))
