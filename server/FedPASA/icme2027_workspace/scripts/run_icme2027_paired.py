"""Paired image bootstrap, grouping tied thresholds exactly as sklearn AP."""
import argparse
from time import perf_counter
import numpy as np
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from icme2027_common import *

def weighted_ap(y,s,w):
    order=np.argsort(-s,kind='stable'); ys=y[order]
    ends=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    sw=np.asarray(w[:,order],dtype=np.float64)
    tp=np.cumsum(sw*ys[None,:],axis=1)[:,ends]
    total=np.cumsum(sw,axis=1)[:,ends]
    delta=np.diff(tp,axis=1,prepend=0.)
    precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
    positives=tp[:,-1]
    return np.divide((precision*delta).sum(1),positives,out=np.zeros(len(w)),where=positives>0)

def check_math():
    y=np.array([1,0,1,0,1,0]); s=np.array([.8,.8,.2,.2,.2,.1])
    weights=np.array([[1,1,1,1,1,1],[2,0,1,3,0,1],[0,2,0,1,0,0]])
    for w,a in zip(weights,weighted_ap(y,s,weights)):
        expected=average_precision_score(y,s,sample_weight=w)
        np.testing.assert_allclose(a,expected,atol=1e-14)

def run(d,B):
    out=OUT/'paired'/d; out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists() and read(out/'result.json').get('replicates')==B: return
    start=perf_counter(); y=np.load(cache(d)/'test_targets.npy'); n,c=y.shape
    rng=np.random.default_rng(20270910)
    weights=np.empty((B,n),np.uint16)
    for i in range(B):weights[i]=np.bincount(rng.integers(0,n,n),minlength=n)
    score_lists={k:[] for k in ('Full','Ridge_calibrated','Global_fusion')}
    single=[]; sources=[]
    for seed in (7,17,27):
        sr=dr(d)/'full_seeds'/f'seed{seed}'
        fr=sr/'ridge_patch_fusion'
        paths=[fr/'test_scores_classwise.npy',sr/'ridge/test_scores.npy',fr/'test_scores_global.npy']
        f,r,g=[np.load(p) for p in paths]
        cal=np.load(fr/'ridge_calibration.npy'); r=(r-cal[0])/cal[1]
        for name,v in zip(score_lists,(f,r,g)):score_lists[name].append(v)
        single.append({'training_split_seed':seed,**{k:float(100*average_precision_score(y,v,average='macro')) for k,v in zip(score_lists,(f,r,g))}})
        sources.extend({'path':str(p.relative_to(ROOT)),'sha256':digest(p)} for p in paths)
    maps=np.zeros((B,len(score_lists)),np.float64); point={}
    for mi,(name,parts) in enumerate(score_lists.items()):
        scores=np.mean(parts,axis=0,dtype=np.float64).astype(np.float32)
        point[name]=float(100*average_precision_score(y,scores,average='macro'))
        unit=np.stack([weighted_ap(y[:,j],scores[:,j],np.ones((1,n),np.uint16))[0] for j in range(c)])
        np.testing.assert_allclose(unit,average_precision_score(y,scores,average=None),atol=1e-12)
        for j in range(c):
            for b in range(0,B,20):maps[b:b+20,mi]+=weighted_ap(y[:,j],scores[:,j],weights[b:b+20])/c*100
        print(f'PAIRED {d} {name} done point={point[name]:.6f}',flush=True)
        save(out/'progress.json',{'status':'RUNNING','completed_methods':mi+1,'total_methods':3,'updated':stamp()})
    np.save(out/'bootstrap_mAP.npy',maps)
    comparisons=[]
    for j,name in enumerate(list(score_lists)[1:],1):
        delta=maps[:,0]-maps[:,j]
        comparisons.append({'comparison':f'Full minus {name}','point_difference_mAP':point['Full']-point[name],
            'percentile_95_CI_mAP':np.quantile(delta,[.025,.975]).tolist(),
            'bootstrap_nonpositive_fraction':float(np.mean(delta<=0)),
            'interpretation':'paired percentile interval conditional on these fitted models; not independent training repetitions or a permutation p-value'})
    save(out/'result.json',{'status':'COMPLETE','replicates':B,'resampling':'same image multiplicities for all methods and all classes',
        'tie_handling':'group equal scores before precision/recall updates; sklearn unit-weight agreement <=1e-12',
        'rng_seed':20270910,'methods':list(score_lists),'point_mAP':point,'comparisons':comparisons,'single_training_splits':single,
        'outer_training_repeats':False,'targets_sha256':digest(cache(d)/'test_targets.npy'),'sources':sources,
        'seconds':perf_counter()-start,'completed':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--datasets',nargs='+',default=['coco','nus','voc07','voc12']);p.add_argument('--replicates',type=int,default=1000)
    a=p.parse_args();check_math()
    with threadpool_limits(limits=4):
        for d in a.datasets:run(d,a.replicates)
