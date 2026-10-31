"""Isolated fairness experiment IO; never writes historical outputs."""
from pathlib import Path
import hashlib, json, os, sys, time
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/fair_baselines_20260918'
sys.path.insert(0, str(ROOT/'src'))

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, obj):
    p=Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t=p.with_suffix(p.suffix+'.partial')
    t.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); t.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def stamp(): return time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
def cache(d): return ROOT/'outputs/icme/cache'/d
def split(d,s=17):
    b=ROOT/'outputs/icme'/d/'full_seeds'/f'seed{s}'/'ridge'
    return np.load(b/'fit_indices.npy'),np.load(b/'validation_indices.npy')
def assignment(d): return np.load(ROOT/'outputs/icme'/d/'partitions/clients10_seed7.npy')
def raw_features(d,part):
    x=np.concatenate([np.asarray(np.load(cache(d)/f'{part}_{n}.npy',mmap_mode='r'),dtype=np.float32) for n in ('cls','patch_mean','patch_max_text')],axis=1)
    return x
def features(d,part):
    x=raw_features(d,part)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def metric(y,s):
    from sklearn.metrics import average_precision_score
    assert y.shape==s.shape and np.isfinite(s).all()
    return float(np.mean([average_precision_score(y[:,c],s[:,c]) if y[:,c].sum() else 0. for c in range(y.shape[1])])*100)
def freeze_inputs(d):
    f,v=split(d); y=np.load(cache(d)/'train_targets.npy'); a=assignment(d)
    assert len(a)==len(y) and not np.intersect1d(f,v).size
    np.testing.assert_array_equal(np.sort(np.r_[f,v]),np.arange(len(y)))
    assert set(np.unique(y)).issubset({0,1})
    paths=[cache(d)/'manifest.json',cache(d)/'train_ids.txt',cache(d)/'test_ids.txt',cache(d)/'train_targets.npy',ROOT/'outputs/icme'/d/'partitions/clients10_seed7.npy']
    b=ROOT/'outputs/icme'/d/'full_seeds/seed17/ridge'
    paths += [b/'fit_indices.npy',b/'validation_indices.npy']
    result={'dataset':d,'fit':len(f),'validation':len(v),'classes':y.shape[1], 'client_sizes':np.bincount(a).tolist(),'inputs':{str(p.relative_to(ROOT)):digest(p) for p in paths},'time':stamp()}
    dest=OUT/'protocol'/f'{d}_inputs.json'
    if dest.exists(): assert read(dest)['inputs']==result['inputs'], 'Input fingerprint changed'
    else: save(dest,result)
    return result
