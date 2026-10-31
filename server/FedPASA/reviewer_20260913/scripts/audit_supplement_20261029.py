"""Freeze existing evidence and inputs; stale queue labels are not completion proof."""
import json
from pathlib import Path
from fair_baselines_common import ROOT, digest, save

OUT=ROOT/'outputs/supplement_20261029'


def main():
    evidence=[]
    patterns=('outputs/fair_baselines_20260918/final/*/results.json',
              'outputs/fair_baselines_20260918/*/*/selection.json',
              'outputs/fair_baselines_20260918/linear/*/*/selection.json',
              'outputs/full_protocol_20260919/systems/*/result.json',
              'outputs/icme/supplementary/*/fedmpt_repaired/result.json')
    for pattern in patterns:
        for p in ROOT.glob(pattern):
            value=json.loads(p.read_text())
            evidence.append(dict(path=str(p.relative_to(ROOT)),sha256=digest(p),status=value.get('status'),
                                 reuse='historical evidence only; not a new independent outer run'))
    inputs=[]
    for d in ('coco','nus'):
        c=ROOT/'outputs/icme/cache'/d
        for name in ('train_targets.npy','test_targets.npy','train_ids.txt','test_ids.txt','text_features.npy','manifest.json'):
            p=c/name
            assert p.exists(),p
            inputs.append(dict(dataset=d,path=str(p),sha256=digest(p),bytes=p.stat().st_size))
        for part in ('train','test'):
            for name in ('cls','patch','patch_mean','patch_max_text'):
                p=c/f'{part}_{name}.npy';assert p.exists(),p
                inputs.append(dict(dataset=d,path=str(p),resolved=str(p.resolve()),bytes=p.stat().st_size,
                                   feature_hash_scope='size and immutable existing cache path; full multi-GB hash not recomputed'))
    assert any('fair_baselines' in x['path'] for x in evidence)
    save(OUT/'evidence_audit.json',dict(status='COMPLETE',evidence=evidence,inputs=inputs,
         limitations=['FedMVP/Fed-RAM historical adapted implementations retain their original scope; no author-equivalence claim.',
                      'FedMPT native protocol remains separate from frozen-feature controls.',
                      'P2 privacy attack and optional missing-class experiments are not specified in the supplied truncated plan and are not queued.']))
    print('EVIDENCE_AUDIT_COMPLETE',len(evidence),len(inputs),flush=True)


if __name__=='__main__':main()
