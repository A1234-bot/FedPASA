"""Validate uploaded state, reuse images by ID, and rewrite only server paths."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert os.name == 'posix', 'Run this on the Linux server copy only'
    inventory = ROOT / 'tmp/server_migration'
    hashes = json.loads((inventory / 'file_hashes.json').read_text())
    for rel, expected in hashes.items():
        assert digest(ROOT / rel) == expected, f'Transfer hash mismatch: {rel}'
    weight = Path('/root/BCP/external/FedMPT-main/ViT-B-16.pt')
    coco_roots = [Path('/root/BCP/data/raw/coco2017') / p for p in ('train2017', 'val2017')]
    nus_root = Path('/root/BCP/data/raw/nuswide/kaggle_tc21/NUS-WIDE/images')
    missing = []
    counts = {}
    manifests = {}
    for dataset in ('coco', 'nus'):
        cache = ROOT / 'outputs/icme/cache' / dataset
        manifest = json.loads((cache / 'manifest.json').read_text())
        assert digest(weight) == manifest['weight_sha256'], 'Wrong CLIP weights'
        (cache / 'manifest.windows_source.json').write_text(json.dumps(manifest, indent=2))
        manifest['weight_path'] = str(weight)
        for split in ('train', 'test'):
            ids = (cache / f'{split}_ids.txt').read_text().splitlines()
            dest = ROOT / 'data/coco2014' / ('train2014' if split == 'train' else 'val2014') if dataset == 'coco' else nus_root
            if dataset == 'coco':
                dest.mkdir(parents=True, exist_ok=True)
            for identifier in ids:
                name = identifier.rsplit('/', 1)[-1] if dataset == 'nus' else identifier
                name = name if name.endswith('.jpg') else name + '.jpg'
                if dataset == 'coco':
                    numeric_name = name.rsplit('_', 1)[-1]
                    source = next((p / numeric_name for p in coco_roots if (p / numeric_name).is_file()), None)
                    if source is None:
                        missing.append(dict(dataset=dataset, split=split, name=name))
                        continue
                    link = dest / name
                    if not link.exists():
                        link.symlink_to(source)
                elif not (dest / name).is_file():
                    missing.append(dict(dataset=dataset, split=split, name=name))
            manifest[split + '_root'] = str(dest)
            counts[f'{dataset}_{split}'] = len(ids)
        manifests[dataset] = manifest
    (inventory / 'missing_images.json').write_text(json.dumps(missing, indent=2))
    assert not missing, f'Missing images: {len(missing)}; see missing_images.json'
    samples = json.loads((inventory / 'image_hash_samples.json').read_text())
    for s in samples:
        path = Path(manifests[s['dataset']][s['split'] + '_root']) / s['name']
        assert digest(path) == s['sha256'], f'Image hash mismatch: {path}'
    for dataset, manifest in manifests.items():
        (ROOT / 'outputs/icme/cache' / dataset / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    import numpy as np
    import torch
    states = {}
    for dataset in ('coco', 'nus'):
        folder = ROOT / 'outputs/icme/supplementary' / dataset / 'fedmpt_repaired'
        state = torch.load(folder / 'checkpoint.pt', map_location='cpu', weights_only=False)
        assert state['target_rounds'] == 200
        assert all(torch.isfinite(t).all() for t in state['global'].values())
        if 'aggregate' in state:
            assert all(torch.isfinite(t).all() for t in state['aggregate'].values())
        states[dataset] = {k: state[k] for k in ('completed_round', 'active_round', 'next_client', 'target_rounds') if k in state}
        parts = np.load(ROOT / 'outputs/icme' / dataset / 'partitions/clients10_seed7.npy')
        assert len(parts) == counts[dataset + '_train'] and set(parts.tolist()) == set(range(10))
        for split in ('train', 'test'):
            target = np.load(ROOT / 'outputs/icme/cache' / dataset / f'{split}_targets.npy')
            assert target.shape == (counts[f'{dataset}_{split}'], 80 if dataset == 'coco' else 81)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    result = dict(status='PASS', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                  transferred_files_verified=len(hashes), image_presence=counts,
                  image_sha256_samples_verified=len(samples), checkpoints=states,
                  coco_note='Original 2014 IDs/labels/splits retained; 2017 storage reused by numeric image ID. All required IDs exist; 128 COCO image bytes sampled against local sources.',
                  numerical_note='Resumed on RTX 5090 / Linux / PyTorch 2.11; hardware/software differ from local run.')
    (inventory / 'server_validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
