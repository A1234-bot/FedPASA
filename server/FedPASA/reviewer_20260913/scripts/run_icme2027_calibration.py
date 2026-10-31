"""Client-packet calibration and candidate selection; histogram AP is approximate.

Single-host simulation of 10 logical clients. The coordinator receives moments
and dense count histograms, never per-image predictions/labels. A separate
central reference reads them ONLY to audit equivalence/approximation error.
No cryptographic aggregation or formal privacy guarantee is claimed.
"""
import argparse, dataclasses, gc
from time import perf_counter
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from icme2027_common import *
from fedpasa.core import build_client_banks, fit_quantile_aligned_fedpasa, fit_central_piaa, PatchBank
from fedpasa.supervised import make_ridge_packet, fit_ridge_from_packets
from fedpasa.benchmarks import BENCHMARKS
from run_label_patch_fusion import piaa_scores

class Wire:
    def __init__(self):self.rows=[]
    def send(self,stage,direction,client,payload):
        fields=[];copied={}
        for name,value in payload.items():
            a=np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)
            # A real binary serialization boundary; count application payload,
            # not TCP/TLS headers or an estimated dtype.
            raw=a.tobytes(order='C'); copied[name]=np.frombuffer(raw,dtype=a.dtype).reshape(a.shape).copy()
            fields.append({'name':name,'shape':list(a.shape),'dtype':str(a.dtype),'bytes':len(raw)})
        self.rows.append({'stage':stage,'direction':direction,'client':client,'fields':fields,'bytes':sum(f['bytes'] for f in fields)})
        return copied

def hist_packet(y,s,bins):
    # Fixed monotone map needs no label-dependent range fitting or clipping.
    idx=np.minimum((bins*(.5+np.arctan(s)/np.pi)).astype(np.int64),bins-1).clip(0)
    c=s.shape[1]; total=np.empty((c,bins),np.int64);positive=np.empty_like(total)
    for j in range(c):
        total[j]=np.bincount(idx[:,j],minlength=bins)
        positive[j]=np.bincount(idx[y[:,j]>0,j],minlength=bins)
    return {'total':total,'positive':positive}

def hist_ap(p):
    total=np.cumsum(p['total'][:,::-1],axis=1,dtype=np.float64)
    tp=np.cumsum(p['positive'][:,::-1],axis=1,dtype=np.float64)
    precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
    return 100*np.divide((precision*p['positive'][:,::-1]).sum(1),tp[:,-1],out=np.zeros(len(tp)),where=tp[:,-1]>0)

def federated_ap(ys,ss,bins,wire,stage):
    merged=None
    for client,(y,s) in enumerate(zip(ys,ss)):
        p=wire.send(stage,'uplink',client,hist_packet(y,s,bins))
        if merged is None:merged=p
        else:
            for k in p:merged[k]+=p[k]
    return hist_ap(merged)

def moments(parts,wire,name):
    packets=[wire.send(name,'uplink',i,{'count':np.int64(len(x)),'sum':x.astype(np.float64).sum(0),'sum_square':np.square(x.astype(np.float64)).sum(0)}) for i,x in enumerate(parts)]
    n=sum(int(p['count']) for p in packets); mean=sum(p['sum'] for p in packets)/n
    sd=np.sqrt(np.maximum(sum(p['sum_square'] for p in packets)/n-mean**2,0)).clip(1e-6)
    for i in range(len(parts)):wire.send(name,'downlink',i,{'mean':mean,'std':sd})
    return mean,sd

def qa_fit(d,assignments,tag,wire):
    out=OUT/'calibration'/d/'qa'/tag;out.mkdir(parents=True,exist_ok=True)
    b=BENCHMARKS[d]; text=torch.from_numpy(np.load(cache(d)/'text_features.npy')).float()
    start=perf_counter()
    bankfile=out/'client_banks.pt'
    reused_bank=bankfile.exists()
    if reused_bank:banks=torch.load(bankfile,map_location='cpu',weights_only=False)
    else:
        banks=build_client_banks(cache(d)/'train_patch.npy',assignments,text,clients=10,k=512,batch_images=64,device='cuda')
        torch.save(banks,bankfile)
    # Trace actual moment/purification packet shapes and values by wrapping the
    # implementation at its client->coordinator boundary, preserving float64.
    import fedpasa.core as core
    originals={n:getattr(core,n) for n in ('_moment_packet','_purification_packet','_gda_from_moment_packets','_purification_threshold_from_packets','_global_quantile_selection')}
    calls={'moment':0,'purification':0,'gda':0}
    def mp(*a,**kw):
        p=originals['_moment_packet'](*a,**kw); i=calls['moment'];calls['moment']+=1
        vals=wire.send(tag+':qa_moment_'+str(i//10),'uplink',i%10,dataclasses.asdict(p))
        return type(p)(**{k:torch.from_numpy(v) for k,v in vals.items()})
    def pp(*a,**kw):
        p=originals['_purification_packet'](*a,**kw);i=calls['purification'];calls['purification']+=1
        vals=wire.send(tag+':qa_purification','uplink',i,dataclasses.asdict(p))
        return type(p)(**{k:torch.from_numpy(v) for k,v in vals.items()})
    def gda(*a,**kw):
        result=originals['_gda_from_moment_packets'](*a,**kw); stage=calls['gda'];calls['gda']+=1
        for i in range(10):wire.send(tag+':qa_adapter_'+str(stage),'downlink',i,{'weight':result[0],'bias':result[1]})
        return result
    def threshold(*a,**kw):
        value=originals['_purification_threshold_from_packets'](*a,**kw)
        for i in range(10):wire.send(tag+':qa_threshold','downlink',i,{'threshold':value})
        return value
    def selection(local,k,**kw):
        # Core accesses only entropy in ordinary tie mode for its choice.
        for i,bank in enumerate(local):
            wire.send(tag+':qa_order_statistics','uplink',i,{f'entropy_{c}':v.entropy for c,v in bank.items()})
        choices=originals['_global_quantile_selection'](local,k,**kw)
        for i,choice in enumerate(choices):wire.send(tag+':qa_selected_local_positions','downlink',i,{str(c):ix for c,ix in choice.items()})
        return choices
    core._moment_packet=mp;core._purification_packet=pp;core._gda_from_moment_packets=gda
    core._purification_threshold_from_packets=threshold;core._global_quantile_selection=selection
    try:adapter=fit_quantile_aligned_fedpasa(banks,text,k=512,purification_temperature=b.purification_temperature)
    finally:
        for name,value in originals.items():setattr(core,name,value)
    # Central independent reference over the exact same selected patch set.
    selections=originals['_global_quantile_selection'](banks,512)
    merged={}
    for c in range(len(text)):
        pieces=[(bank[c],sel[c]) for bank,sel in zip(banks,selections) if c in sel and len(sel[c])]
        if pieces:merged[c]=PatchBank(*(torch.cat([getattr(v,n)[ix] for v,ix in pieces]) for n in ('features','probabilities','entropy')))
    central=fit_central_piaa(merged,text,purification_temperature=b.purification_temperature)
    error=float((adapter.weight-central.weight).abs().max())
    torch.save({'weight':adapter.weight,'bias':adapter.bias},out/'adapter.pt')
    save(out/'result.json',{'status':'COMPLETE','fit_seconds_including_bank_load_or_build':perf_counter()-start,'central_weight_max_abs_error':error,
        'selection':'ordinary torch entropy ties; reference uses same selected set','protocol_rounds':4,'bank_cache_reused':reused_bank})
    del banks,merged;gc.collect()
    return adapter,text

def run(d,seed,bins):
    out=OUT/'calibration'/d/f'seed{seed}_bins{bins}';out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    started=perf_counter();wire=Wire();torch.cuda.reset_peak_memory_stats();timing={}
    x=features(d,'train');y=np.load(cache(d)/'train_targets.npy')
    assignment=np.load(dr(d)/'partitions/clients10_seed7.npy');sr=dr(d)/'full_seeds'/f'seed{seed}'/'ridge'
    fit=np.load(sr/'fit_indices.npy');val=np.load(sr/'validation_indices.npy')
    assert not np.intersect1d(fit,val).size and len(fit)+len(val)==len(y)
    local_fit=[fit[assignment[fit]==i] for i in range(10)];local_val=[val[assignment[val]==i] for i in range(10)]
    report=read(sr/'result.json');alphas=[v['alpha'] for v in report['tuning']]
    timing['input_loading_seconds']=perf_counter()-started;stage_start=perf_counter();packets=[]
    for i,ix in enumerate(local_fit):
        p=make_ridge_packet(x[ix],y[ix]);vals=wire.send('ridge_fit','uplink',i,dataclasses.asdict(p))
        packets.append(type(p)(**{k:(int(v) if k=='count' else torch.from_numpy(v)) for k,v in vals.items()}))
    timing['ridge_fit_statistics_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    local_y=[y[ix] for ix in local_val]; central_y=np.concatenate(local_y)
    tuning=[];adapters=[]
    for alpha in alphas:
        a=fit_ridge_from_packets(tuple(packets),regularization=alpha);adapters.append(a)
        for i in range(10):wire.send(f'ridge_candidate_{alpha}','downlink',i,{'weight':a.weight,'bias':a.bias})
        scores=[predict(x[ix],a.weight,a.bias) for ix in local_val]
        ha=federated_ap(local_y,scores,bins,wire,f'ridge_ap_{alpha}')
        ca=100*average_precision_score(central_y,np.concatenate(scores),average=None)
        tuning.append({'alpha':alpha,'histogram_mAP':float(ha.mean()),'central_mAP':float(ca.mean()),'per_class_AP_error_max':float(np.max(np.abs(ha-ca)))})
    timing['ridge_candidate_solves_validation_histograms_reference_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    choice=int(np.argmax([z['histogram_mAP'] for z in tuning]));central_choice=int(np.argmax([z['central_mAP'] for z in tuning]))
    ra=adapters[choice];rv=[predict(x[ix],ra.weight,ra.bias) for ix in local_val]
    full_packets=[make_ridge_packet(x[assignment==i],y[assignment==i]) for i in range(10)]
    for i,p in enumerate(full_packets):wire.send('ridge_refit','uplink',i,dataclasses.asdict(p))
    rf=fit_ridge_from_packets(tuple(full_packets),regularization=alphas[choice]);del x,packets,full_packets,adapters;gc.collect()
    timing['ridge_refit_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    qa_assignment=assignment.copy();qa_assignment[val]=-1
    qa,text=qa_fit(d,qa_assignment,f'fit_seed{seed}',wire)
    timing['qa_fit_bank_statistics_and_reference_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    qv=[];b=BENCHMARKS[d]
    for ix in local_val:
        qv.append(piaa_scores(cache(d)/'train_patch.npy',cache(d)/'train_cls.npy',ix,text,qa.weight,qa.bias,
            cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device='cuda'))
    timing['qa_validation_inference_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    qc=moments(qv,wire,'qa_calibration');rc=moments(rv,wire,'ridge_calibration')
    meanerr=max(float(np.max(abs(np.concatenate(parts).astype(np.float64).mean(0)-cal[0]))) for parts,cal in [(qv,qc),(rv,rc)])
    stderr=max(float(np.max(abs(np.concatenate(parts).astype(np.float64).std(0).clip(1e-6)-cal[1]))) for parts,cal in [(qv,qc),(rv,rc)])
    qz=[(z-qc[0])/qc[1] for z in qv];rz=[(z-rc[0])/rc[1] for z in rv]
    weights=np.linspace(0,1,21);ha=[];ca=[]
    for weight in weights:
        ss=[weight*r+(1-weight)*q for q,r in zip(qz,rz)]
        ha.append(federated_ap(local_y,ss,bins,wire,f'fusion_ap_{weight:.2f}'))
        ca.append(100*average_precision_score(central_y,np.concatenate(ss),average=None))
    ha,ca=np.asarray(ha),np.asarray(ca);cw=weights[ha.argmax(0)];ccw=weights[ca.argmax(0)]
    gw=weights[ha.mean(1).argmax()];cgw=weights[ca.mean(1).argmax()]
    timing['fusion_calibration_selection_and_reference_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    qfull,_=qa_fit(d,assignment,'refit_full',wire)
    timing['qa_refit_and_reference_seconds']=perf_counter()-stage_start;stage_start=perf_counter()
    tx=features(d,'test');rt=predict(tx,rf.weight,rf.bias);del tx
    qt=piaa_scores(cache(d)/'test_patch.npy',cache(d)/'test_cls.npy',np.arange(b.expected_test),text,qfull.weight,qfull.bias,
        cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device='cuda')
    qt=(qt-qc[0])/qc[1];rt=(rt-rc[0])/rc[1]
    fscore=(cw*rt+(1-cw)*qt).astype(np.float32);cscore=(ccw*rt+(1-ccw)*qt).astype(np.float32)
    for i in range(10):wire.send('final_model','downlink',i,{'ridge_weight':rf.weight,'ridge_bias':rf.bias,'qa_weight':qfull.weight,'qa_bias':qfull.bias,
        'qa_mean':qc[0],'qa_std':qc[1],'ridge_mean':rc[0],'ridge_std':rc[1],'fusion_weights':cw})
    timing['test_inference_and_final_model_delivery_seconds']=perf_counter()-stage_start
    targets=np.load(cache(d)/'test_targets.npy')
    np.save(out/'test_scores.npy',fscore);np.save(out/'central_same_candidate_test_scores.npy',cscore)
    save(out/'messages.json',wire.rows)
    # Mark approximation outcomes honestly, without changing bins based on test.
    save(out/'result.json',{'status':'COMPLETE','algorithm':'FedPASA-Full distributed histogram-selection variant',
        'bins':bins,'fixed_score_transform':'0.5+atan(score)/pi','execution':'single-server logical-client packet simulation; not a multi-machine network benchmark',
        'privacy':'no formal privacy or cryptographic secure aggregation; dense per-client histograms may disclose sparse-bin information',
        'selection_uses_test_labels':False,'partition_seed':7,'training_split_seed':seed,'clients':10,
        'ridge_candidates':tuning,'selected_alpha':alphas[choice],'central_selected_alpha':alphas[central_choice],
        'zscore_mean_max_abs_error':meanerr,'zscore_std_max_abs_error':stderr,
        'moment_equivalence_tolerance':1e-10,'moment_equivalence_passed':bool(meanerr<=1e-10 and stderr<=1e-10),
        'exact_selection_equivalence':bool(choice==central_choice and np.array_equal(cw,ccw) and gw==cgw),
        'fusion_histogram_AP':ha.tolist(),'fusion_central_AP':ca.tolist(),'fusion_AP_max_error':float(np.max(abs(ha-ca))),
        'fusion_weights':cw.tolist(),'central_fusion_weights':ccw.tolist(),'weight_mismatch_classes':int(np.sum(cw!=ccw)),
        'global_weight':float(gw),'central_global_weight':float(cgw),
        'test_mAP':float(100*average_precision_score(targets,fscore,average='macro')),
        'central_same_ridge_candidate_test_mAP':float(100*average_precision_score(targets,cscore,average='macro')),
        'central_reference_scope':'fusion reference conditions on selected ridge candidate; ridge candidate equality reported separately',
        'test_score_max_abs_error':float(np.max(abs(fscore-cscore))),
        'application_payload_uplink_bytes':sum(r['bytes'] for r in wire.rows if r['direction']=='uplink'),
        'application_payload_downlink_bytes':sum(r['bytes'] for r in wire.rows if r['direction']=='downlink'),
        'message_count':len(wire.rows),
        'communication_phases':list(dict.fromkeys(r['stage']+':'+r['direction'] for r in wire.rows)),
        'uplink_stage_count':len(set(r['stage'] for r in wire.rows if r['direction']=='uplink')),
        'transport_scope':'numeric application payload only; framing, encryption and network headers excluded',
        'candidate_and_refit_costs_included':True,
        'timing_seconds':timing,'timing_scope':'Includes independent central reference checks and serialization; conservative validation-run wall time, not production-only latency',
        'seconds':perf_counter()-started,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/1024**2,'completed':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,default=7);p.add_argument('--bins',type=int,default=8192)
    a=p.parse_args();torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(a.dataset,a.seed,a.bins)
