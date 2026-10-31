"""VOC outer repeats using the already accepted COCO/NUS outer implementation."""
import argparse,gc,sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_priority_20260913 as prior
import run_priority_full_impl as r

BASE=r.ROOT/'outputs/full_protocol_20260919'

def run(d,index):
    assert d in ('voc07','voc12')
    prior.BASE=BASE
    device='cuda' if torch.cuda.is_available() else 'cpu'
    original_scores=r.piaa_scores
    def scores(*args,**kwargs):kwargs['device']=device;return original_scores(*args,**kwargs)
    r.piaa_scores=scores
    def banks(dataset,a,text,clients,destination):
        destination.mkdir(parents=True,exist_ok=True);paths=[]
        for cid in range(clients):
            path=destination/f'client{cid}.pt';one=np.where(a==cid,0,-1)
            bank=r.core.build_client_banks(r.resolve_cache(dataset)/'train_patch.npy',one,text,clients=1,k=512,batch_images=32,device=device)[0]
            temp=path.with_suffix('.partial');torch.save(bank,temp);temp.replace(path);del bank;gc.collect();paths.append(path)
        return r.DiskBanks(paths)
    r.disk_banks=banks
    root=BASE/'outer'/d/f'run{index+1}'
    r.save(root/'execution.json',dict(device=device,source_outer_sha256=r.digest(r.ROOT/'scripts/run_priority_20260913.py'),source_fitting_sha256=r.digest(r.ROOT/'scripts/run_priority_full_impl.py'),reuse='identical prior outer code, only output root and execution device routing changed',ridge_grid=list(r.ALPHAS)))
    prior.outer(d,index)
    fit=root/'full_runtime'/d;seeds=prior.RUNS[index][1];labels=r.targets(d)
    rows=[]
    for method,file in [('QAFedPASA','qa_scores.npy'),('3-view Ridge',f'seed{seeds[0]}_ridge_scores.npy'),('FedPASA-Full','test_scores.npy')]:
        s=np.load(fit/file);m,ap=r.metric(labels,s);rows.append(dict(method=method,mAP=m,prediction_sha256=r.digest(fit/file)))
    ensemble=np.mean([np.load(fit/f'seed{s}_test_scores.npy') for s in seeds],axis=0);np.testing.assert_array_equal(ensemble,np.load(fit/'test_scores.npy'))
    r.save(root/'verified.json',dict(status='COMPLETE',dataset=d,run=index+1,partition_seed=prior.RUNS[index][0],inner_seeds=seeds,rows=rows,hardware=device,ensemble_identical=True))
    print('OUTER_COMPLETE',d,index+1,rows,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--index',required=True,type=int);a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(a.dataset,a.index)
