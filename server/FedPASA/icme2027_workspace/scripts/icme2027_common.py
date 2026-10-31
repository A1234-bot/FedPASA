"""Shared paths and atomic evidence writers for the isolated server suite."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/icme2027'
sys.path.insert(0, str(ROOT/'src'))

def stamp(): return datetime.now(timezone.utc).isoformat()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p, value):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def cache(d):
    p=ROOT/'outputs/icme/cache'/d
    return ROOT/'outputs/voc07/cache' if d=='voc07' and not p.exists() else p
def dr(d): return ROOT/'outputs/icme'/d
def features(d,split):
    return np.concatenate([np.asarray(np.load(cache(d)/f'{split}_{n}.npy',mmap_mode='r'),dtype=np.float32)
                           for n in ('cls','patch_mean','patch_max_text')],axis=1)
def predict(x,w,b):
    x=np.asarray(x,dtype=np.float32)
    return (x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12))@np.asarray(w)+np.asarray(b)

