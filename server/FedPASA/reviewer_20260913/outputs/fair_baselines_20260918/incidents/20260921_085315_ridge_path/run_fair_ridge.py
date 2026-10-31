"""Train-only boundary checks, with frozen QA held-out predictions and exact fusion."""
import argparse,gc,time
import numpy as np
import torch
from fair_baselines_common import *
from fedpasa.supervised import make_ridge_packet,fit_ridge_from_packets
from run_reviewer_evidence import predict,average_adapters
import run_priority_full_impl as qa

GRID=(1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,3e-3,.01,.03,.1,.3,1.)

def tune(d):
    root=OUT/'ridge'/d
    if (root/'selection.json').exists():return
    f,v=split(d);x=raw_features(d,'train');y=np.load(cache(d)/'train_targets.npy');a=assignment(d);t=time.perf_counter()
    packets=tuple(make_ridge_packet(x[f[a[f]==i]],y[f[a[f]==i]]) for i in range(10));curves={};chosen={};scores={}
    for kind in ('pooled','local_avg'):
        rows=[];models=[];vv=[]
        for alpha in GRID:
            if kind=='pooled':m=fit_ridge_from_packets(packets,regularization=alpha)
            else:m=average_adapters([fit_ridge_from_packets((p,),regularization=alpha) for p in packets],[p.count for p in packets],'independently selected local ridge')
            s=predict(x[v],m);rows.append({'alpha':alpha,'validation_mAP':metric(y[v],s)});models.append(m);vv.append(s)
        ix=max(range(len(rows)),key=lambda i:rows[i]['validation_mAP']);chosen[kind]=rows[ix];curves[kind]=rows;scores[kind]=vv[ix]
    root.mkdir(parents=True,exist_ok=True);np.save(root/'ridge_validation_scores.npy',scores['pooled'])
    # Reuse fit banks built from the exact same split17 and same patch cache;
    # if absent, reconstruct from TRAINING patches with validation excluded.
    text=torch.from_numpy(np.load(cache(d)/'text_features.npy')).float();b=qa.BENCHMARKS[d]
    base=ROOT/'outputs/reviewer_20260913/full_runtime'/d/'seed17_banks'
    paths=[base/f'client{i}.pt' for i in range(10)]
    reused=all(p.exists() for p in paths)
    if reused:bank=qa.DiskBanks(paths)
    else:
        af=a.copy();af[v]=-1;bank=qa.disk_banks(d,af,text,10,root/'seed17_banks')
    profile=qa.Profile()
    with profile.track():model=qa.core.fit_quantile_aligned_fedpasa(bank,text,k=512,purification_temperature=b.purification_temperature)
    qv=qa.piaa_scores(cache(d)/'train_patch.npy',cache(d)/'train_cls.npy',v,text,model.weight,model.bias,cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device='cuda')
    np.save(root/'qa_validation_scores.npy',qv)
    save(root/'qa_fit_profile.json',{'messages':profile.rows,'fit_uplink_bytes':sum(r['bytes'] for r in profile.rows),'bank_reused':reused})
    rv=scores['pooled'];rm=rv.mean(0);rs=rv.std(0).clip(1e-6);qm=qv.mean(0);qs=qv.std(0).clip(1e-6)
    rz=(rv-rm)/rs;qz=(qv-qm)/qs;ws=np.linspace(0,1,21,dtype='float32')
    aps=np.array([qa.metric(y[v],w*rz+(1-w)*qz)[1] for w in ws]);cw=ws[aps.argmax(0)];gw=ws[aps.mean(1).argmax()]
    save(root/'selection.json',{'status':'LOCKED','selected':chosen,'curves':curves,'classwise_weights':cw.tolist(),'global_weight':float(gw),'ridge_mean':rm.tolist(),'ridge_std':rs.tolist(),'qa_mean':qm.tolist(),'qa_std':qs.tolist(),'selection_wall_seconds':time.perf_counter()-t,'test_labels_read':False,'split_seed':17,'fit_bank_reused':reused,'fit_bank_sha256':[digest(p) for p in paths] if reused else [],'locked_at':stamp(),'input_fingerprints':read(OUT/'protocol'/f'{d}_inputs.json')['inputs']})

def final(d):
    root=OUT/'ridge'/d
    if (root/'predictions.json').exists():return
    sel=read(root/'selection.json');x=raw_features(d,'train');y=np.load(cache(d)/'train_targets.npy');z=raw_features(d,'test');a=assignment(d);t=time.perf_counter()
    packets=tuple(make_ridge_packet(x[a==i],y[a==i]) for i in range(10));rows=[]
    for kind in ('pooled','local_avg'):
        alpha=sel['selected'][kind]['alpha']
        if kind=='pooled':m=fit_ridge_from_packets(packets,regularization=alpha)
        else:m=average_adapters([fit_ridge_from_packets((p,),regularization=alpha) for p in packets],[p.count for p in packets],'local ridge independently selected')
        s=predict(z,m);np.save(root/f'{kind}_test_scores.npy',s)
        torch.save(m,root/f'{kind}_model.pt');rows.append({'method':kind,'alpha':alpha,'prediction':str(root/f'{kind}_test_scores.npy'),'sha256':digest(root/f'{kind}_test_scores.npy')})
        if kind=='pooled':ridge=s
    # Same frozen full-QA cache as the latest controlled study, never selected by test.
    qpath=ROOT/'outputs/reviewer_20260913/full_runtime'/d/'qa_scores.npy'
    if not qpath.exists():qpath=ROOT/'outputs/priority_20260913/controlled_full'/d/'qa_scores.npy'
    q=np.load(qpath);assert q.shape==ridge.shape
    rz=(ridge-np.array(sel['ridge_mean']))/np.array(sel['ridge_std']);qz=(q-np.array(sel['qa_mean']))/np.array(sel['qa_std'])
    for name,w in [('full_single',np.array(sel['classwise_weights'])),('global_fusion',sel['global_weight'])]:
        s=(w*rz+(1-w)*qz).astype('float32');np.save(root/f'{name}_test_scores.npy',s);rows.append({'method':name,'prediction':str(root/f'{name}_test_scores.npy'),'sha256':digest(root/f'{name}_test_scores.npy')})
    save(root/'predictions.json',{'status':'COMPLETE','rows':rows,'selection_sha256':digest(root/'selection.json'),'qa_source':str(qpath),'qa_sha256':digest(qpath),'refit_and_inference_seconds':time.perf_counter()-t,'test_labels_read':False,'cost_scope':'reuses archived QA full-fit prediction; this time is not end-to-end Full cost','saved_at':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','final'],required=True);p.add_argument('--dataset',choices=['coco','nus'],required=True);args=p.parse_args()
    (tune if args.phase=='tune' else final)(args.dataset)
