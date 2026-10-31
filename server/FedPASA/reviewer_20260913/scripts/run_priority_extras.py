"""Additional serialized server jobs; no overwriting original evidence."""
import os,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_priority_full_impl as r
import run_supplementary_experiments as sup
BASE=r.ROOT/'outputs/priority_20260913'

def phase(name):
    if name.startswith('controlled_'):
        d=name.split('_')[1];r.OUT=BASE/'controlled';r.full_runtime(d)
    elif name.startswith('ties_'):
        d=name.split('_')[1];sup.OUT=BASE/'ties';rows=sup.tie_stress(d)
        r.save(BASE/'ties'/d/'result.json',dict(status='COMPLETE',rows=rows))
    elif name=='backbone':subprocess.run([sys.executable,str(r.ROOT/'scripts/run_priority_backbone.py')],check=True)
    elif name.startswith('packed_'):subprocess.run([sys.executable,str(r.ROOT/'scripts/run_priority_packed.py'),name.split('_')[1]],check=True)

def queue():
    while True:
        p=BASE/'queue.json'
        if p.exists() and r.read(p)['status']!='RUNNING':break
        time.sleep(20)
    jobs=[dict(name=n,status='PENDING') for n in ('controlled_coco','controlled_nus','packed_coco','packed_nus','backbone','ties_coco','ties_nus')]
    if (BASE/'extras_queue.json').exists():jobs=r.read(BASE/'extras_queue.json')['jobs']
    for j in jobs:
        if j['status']=='COMPLETE':continue
        j.update(status='RUNNING',start=time.time());r.save(BASE/'extras_queue.json',dict(status='RUNNING',jobs=jobs))
        with (BASE/'logs'/f"{j['name']}.log").open('a') as log:
            code=subprocess.call([sys.executable,__file__,j['name']],stdout=log,stderr=subprocess.STDOUT)
        j.update(status='COMPLETE' if code==0 else 'FAILED',returncode=code,end=time.time())
    r.save(BASE/'extras_queue.json',dict(status='COMPLETE' if all(j['status']=='COMPLETE' for j in jobs) else 'NEEDS_ATTENTION',jobs=jobs))

if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if len(sys.argv)==1:queue()
        else:phase(sys.argv[1])
