"""Predeclared outer repetitions and feature ablations, isolated from historical outputs."""
import argparse, os, sys, subprocess, time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_priority_full_impl as r
from run_icme_experiments import assignments_for

BASE=r.ROOT/'outputs/priority_20260913'
RUNS=((107,(1007,1017,1027),1107),(207,(2007,2017,2027),1207),(307,(3007,3017,3027),1307))

def outer(d,index):
    ps,seeds,cs=RUNS[index];root=BASE/'outer'/d/f'run{index+1}'
    r.OUT=root;r.SEEDS=seeds
    if (root/'full_runtime'/d/'result.json').exists(): return
    a,sizes,seconds=assignments_for(np.load(r.resolve_cache(d)/'partition_features.npy'),root,clients=10,seed=ps,n_init=10)
    # Full participation is part of the estimator's protocol. Randomize client
    # processing order without dropping training observations.
    order=np.random.default_rng(cs).permutation(10);a=order[a]
    r.assignment=lambda *args,**kwargs:a
    r.sr=lambda dataset,seed=seeds[0]:root/'splits'/f'seed{seed}'
    n=len(r.targets(d,'train'))
    for seed in seeds:
        p=r.sr(d,seed)/'ridge';p.mkdir(parents=True,exist_ok=True)
        ix=np.random.default_rng(seed).permutation(n);v=int(round(.2*n))
        np.save(p/'fit_indices.npy',ix[v:]);np.save(p/'validation_indices.npy',ix[:v])
    r.save(root/'protocol.json',dict(dataset=d,partition_seed=ps,split_seeds=seeds,client_order_seed=cs,client_order=order.tolist(),client_sizes=sizes,participation=1.,sampling_note='All clients participate; random seed changes processing order only. No subsampling is invented.',ridge_grid=r.ALPHAS,fusion_grid=np.linspace(0,1,21).tolist(),k=512,partition='KMeans on original training probability features, n_init=10',test_labels_used_for_selection=False,independent_outer_run=True,cache=str(r.resolve_cache(d)),partition_seconds=seconds))
    torch.manual_seed(cs);np.random.seed(cs)
    r.full_runtime(d)

def views(d):
    combos=(('cls',),('cls','patch_mean'),('cls','patch_max_text'),('patch_mean','patch_max_text'),('cls','patch_mean','patch_max_text'))
    for names in combos:
        out=BASE/'views'/d/'_'.join(names)
        if (out/'result.json').exists():continue
        subprocess.run([sys.executable,str(r.ROOT/'scripts/run_label_ridge.py'),'--dataset',d,'--seed','17','--features',*names,'--alphas',*[str(v) for v in r.ALPHAS],'--report-root',str(out)],check=True)

def queue():
    BASE.mkdir(parents=True,exist_ok=True);jobs=[]
    for d in ('coco','nus'):
        for i in range(3): jobs.append(dict(phase='outer',dataset=d,index=i,status='PENDING'))
        jobs.append(dict(phase='views',dataset=d,index=0,status='PENDING'))
    for j in jobs:
        j.update(status='RUNNING',start=time.time());r.save(BASE/'queue.json',dict(status='RUNNING',jobs=jobs))
        p=BASE/'logs'/f"{j['phase']}_{j['dataset']}_{j['index']}.log";p.parent.mkdir(exist_ok=True)
        with p.open('a') as log:
            code=subprocess.call([sys.executable,__file__,'--phase',j['phase'],'--dataset',j['dataset'],'--index',str(j['index'])],stdout=log,stderr=subprocess.STDOUT)
        j.update(status='COMPLETE' if code==0 else 'FAILED',returncode=code,end=time.time())
        r.save(BASE/'queue.json',dict(status='RUNNING',jobs=jobs))
    r.save(BASE/'queue.json',dict(status='COMPLETE' if all(j['status']=='COMPLETE' for j in jobs) else 'NEEDS_ATTENTION',jobs=jobs))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['queue','outer','views'],default='queue');p.add_argument('--dataset',default='coco');p.add_argument('--index',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if a.phase=='queue':queue()
        elif a.phase=='outer':outer(a.dataset,a.index)
        else:views(a.dataset)
