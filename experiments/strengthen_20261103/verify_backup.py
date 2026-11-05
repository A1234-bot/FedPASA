"""Read-only SHA-256 and completeness verification of this experiment snapshot."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
total = 0
for row in manifest['files']:
    path = root / 'snapshot' / row['path']
    if not path.resolve().is_relative_to((root / 'snapshot').resolve()):
        raise ValueError('Unsafe manifest path')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(block)
    assert path.stat().st_size == row['size'], row['path']
    assert digest.hexdigest() == row['sha256'], row['path']
    total += row['size']
queue = json.loads((root / 'snapshot/queue.json').read_text(encoding='utf-8'))
assert queue['status'] == 'COMPLETE' and len(queue['jobs']) == 160
for job in queue['jobs']:
    assert job['status'] == 'COMPLETE', job['id']
    path = root / 'snapshot/outputs/strengthen_20261103' / job['id'] / 'result.json'
    assert json.loads(path.read_text(encoding='utf-8'))['status'] == 'COMPLETE', job['id']
assert total == manifest['logical_bytes']
assert len(manifest['files']) == manifest['file_count']
print(json.dumps({'status': 'PASS', 'files': len(manifest['files']), 'bytes': total, 'complete_jobs': len(queue['jobs'])}))
