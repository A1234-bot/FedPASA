"""Atomic, fingerprinted stage checkpoints for the 20261029 experiment only."""
import hashlib
import json
import os
import time
from pathlib import Path
import torch


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_torch(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    temp.replace(path)


class StageStore:
    def __init__(self, root, identity):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.identity=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        path=self.root/'identity.json'
        if path.exists():
            assert json.loads(path.read_text())['sha256']==self.identity,'Checkpoint protocol/input/source fingerprint changed'
        else:
            temp=path.with_suffix('.partial')
            temp.write_text(json.dumps(dict(sha256=self.identity,identity=identity),indent=2))
            temp.replace(path)
        self.resumed=0

    def path(self, name):
        return self.root/(hashlib.sha256(name.encode()).hexdigest()+'.pt')

    def contains(self,name):
        return self.path(name).exists()

    def run(self,name,fn):
        path=self.path(name)
        if path.exists():
            try:
                ck=torch.load(path,map_location='cpu',weights_only=False)
                assert ck['identity']==self.identity and ck['name']==name
            except Exception as e:
                raise RuntimeError(f'Invalid checkpoint, retained for inspection: {path}') from e
            self.resumed+=1
            print('RESUME_STAGE',name,flush=True)
            return ck['value'],ck['seconds'],True
        begin=time.perf_counter();value=fn()
        if torch.cuda.is_available():torch.cuda.synchronize()
        seconds=time.perf_counter()-begin
        atomic_torch(path,dict(identity=self.identity,name=name,value=value,seconds=seconds))
        return value,seconds,False
