"""Independent prediction metrics, conditional paired intervals, cost inventory."""
import argparse,csv,time
import numpy as np
from fair_baselines_common import *
from run_icme2027_paired import weighted_ap,check_math

def validation_payload(d,curves):
    info=read(OUT/'protocol'/f'{d}_inputs.json')
    count=sum('validation_mAP' in row for curve in curves for row in curve)
    labels=info['validation']*info['classes']
    return {'evaluation_count':count,'score_bytes_FP32':count*labels*4,'labels_bytes_uint8_once':labels,'score_label_total_bytes':labels*(4*count+1),'scope':'hypothetical distributed pooled-AP payload, labels sent once per method search; implementation computes validation centrally; excludes feature/image centralization, model distribution, protocol headers and network latency'}

def prepared_ap(y,s,weights):
    order=np.argsort(-s,kind='stable');ys=y[order];ends=np.r_[np.flatnonzero(np.diff(s[order])),len(s)-1]
    result=np.zeros(len(weights))
    for start in range(0,len(weights),16):
        w=np.asarray(weights[start:start+16,order],dtype=np.float64)
        tp=np.cumsum(w*ys[None,:],axis=1)[:,ends];total=np.cumsum(w,axis=1)[:,ends]
        delta=np.diff(tp,axis=1,prepend=0.);precision=np.divide(tp,total,out=np.zeros_like(tp),where=total>0)
        result[start:start+len(w)]=np.divide((precision*delta).sum(1),tp[:,-1],out=np.zeros(len(w)),where=tp[:,-1]>0)
    return result

def run(d):
    out=OUT/'final'/d;out.mkdir(parents=True,exist_ok=True)
    from run_fair_full_cost import run as full_cost
    full_dir=full_cost(d)
    full_measurement=read(full_dir/'fair_verification.json')
    pred={};costs={};sources=[]
    rr=OUT/'ridge'/d;rs=read(rr/'selection.json');rp=read(rr/'predictions.json')
    for label,base in [('Full single split17','full_single'),('Three-view pooled Ridge','pooled'),('Local Ridge Avg','local_avg'),('Global fusion','global_fusion')]:
        p=rr/f'{base}_test_scores.npy';pred[label]=[p]
        costs[label]={'repeats':1,'selection_seconds_shared':rs['selection_wall_seconds'],'refit_inference_seconds_shared':rp['refit_and_inference_seconds'],'scope':'shared ridge/fusion computation; QA fitting and full inference reused, not a full end-to-end measurement'}
    costs['Full single split17']['fresh_pipeline_measurement']=full_measurement
    pred['Full three-split ensemble']=[full_dir/'test_scores.npy']
    costs['Full three-split ensemble']=full_measurement
    for mode,label in [('central','Central linear BCE'),('fedavg','FedAvg linear BCE')]:
        root=OUT/'linear'/d/mode;sel=read(root/'selection.json');pred[label]=[];records=[]
        for seed in (7,17,27):
            p=root/'final'/f'seed{seed}';manifest=read(p/'prediction_manifest.json');assert digest(p/'test_scores.npy')==manifest['prediction_sha256']
            pred[label].append(p/'test_scores.npy');records.append({**read(p/'result.json'),'inference_seconds':manifest['inference_seconds']})
        costs[label]={'selection_train_seconds':sel['selection_train_seconds'],'selection_validation_seconds':sel['selection_validation_seconds'],'selection_uplink_bytes':sel['selection_uplink_bytes'],'selection_downlink_bytes':sel['selection_downlink_bytes'],'config':sel['config'],'rounds':sel['rounds'],'optimization_claim':sel['optimization_claim'],'runs':records,'scope':'cached features; no image extraction; simulated clients'}
        costs[label]['selection_validation_payload']=validation_payload(d,[r['curve'] for r in sel['candidates']])
        costs[label]['centralized_data_upload_measured']=False
    for method,label in [('fedmvp','FedMVP release multilabel adaptation'),('fedram','Fed-RAM verified-core adaptation')]:
        root=OUT/method/d;sel=read(root/'selection.json');pred[label]=[];records=[]
        for seed in (7,17,27):
            p=root/'final'/f'seed{seed}';manifest=read(p/'prediction_manifest.json');assert digest(p/'test_scores.npy')==manifest['prediction_sha256']
            pred[label].append(p/'test_scores.npy');records.append({**read(p/'result.json'),'inference_seconds':manifest['inference_seconds']})
        costs[label]={'selection_training_seconds':sel['selection_training_seconds'],'selection_validation_seconds':sel['selection_validation_seconds'],'selection_uplink_bytes':sel['selection_uplink_bytes'],'selection_downlink_bytes':sel['selection_downlink_bytes'],'lr':sel['lr'],'rounds':sel['rounds'],'selected_at_budget_end':sel['selected_at_budget_end'],'runs':records,'scope':'image encoding included; simulated clients; application tensor payload; no headers'}
        costs[label]['selection_validation_payload']=validation_payload(d,[read(p)['curve'] for p in sorted((root/'tuning').glob('*/result.json'))])
        costs[label]['timing_excludes_some_checkpoint_aggregation_overhead']=True
    # Load test labels only after validating that all final prediction files exist.
    for files in pred.values():
        for p in files:assert p.is_file()
    y=np.load(cache(d)/'test_targets.npy');n,c=y.shape;B=2000;rng=np.random.default_rng(20260918);weights=np.empty((B,n),np.uint16)
    for i in range(B):weights[i]=np.bincount(rng.integers(0,n,n),minlength=n)
    check_math();rows=[];maps={}
    for label,paths in pred.items():
        bootstrap=out/(label.replace(' ','_')+'_bootstrap.npy');metrics=[]
        provenance={'predictions':[digest(p) for p in paths],'targets':digest(cache(d)/'test_targets.npy'),'replicates':B,'seed':20260918,'metric':'per-fit mean mAP, tie grouped weighted AP v1'}
        provenance_path=bootstrap.with_suffix('.json')
        already=bootstrap.exists() and provenance_path.exists() and read(provenance_path)==provenance
        total=np.load(bootstrap) if already else np.zeros(B)
        for p in paths:
            s=np.load(p);assert s.shape==y.shape and np.isfinite(s).all();metrics.append(metric(y,s));sources.append({'method':label,'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
            if not already:
                for j in range(c):
                    np.testing.assert_allclose(weighted_ap(y[:,j],s[:,j],np.ones((1,n),np.uint16))[0]*100,metric(y[:,j:j+1],s[:,j:j+1]),atol=1e-9)
                    value=prepared_ap(y[:,j],s[:,j],weights)
                    np.testing.assert_allclose(value[:3],weighted_ap(y[:,j],s[:,j],weights[:3]),atol=1e-12)
                    total+=value*100/(c*len(paths))
        if not already:np.save(bootstrap,total);save(provenance_path,provenance)
        maps[label]=total;rows.append({'dataset':d,'method':label,'mAP_mean':float(np.mean(metrics)),'mAP_sample_std':float(np.std(metrics,ddof=1)) if len(metrics)>1 else None,'runs':len(metrics),'run_mAP':metrics,'uncertainty_scope':'fixed partition7; selection split17 fixed; training randomness only; deterministic methods single fit'})
        print('REPORT',d,label,metrics,flush=True);save(out/'progress.json',{'completed_methods':len(rows),'total_methods':len(pred),'time':stamp()})
    base=maps['Full single split17'];basepoint=rows[0]['mAP_mean'];comparisons=[]
    for row in rows[1:]:
        diff=base-maps[row['method']];comparisons.append({'comparison':'Full single split17 minus '+row['method'],'difference_mAP':basepoint-row['mAP_mean'],'paired_image_bootstrap_95_CI':np.quantile(diff,[.025,.975]).tolist(),'interpretation':'conditional on fitted models; average of per-run mAP differences, not averaged predictions; uncorrected percentile CI, not training-repeat inference'})
    save(out/'results.json',{'status':'COMPLETE','rows':rows,'comparisons':comparisons,'replicates':B,'resampling_seed':20260918,'sources':sources,'test_targets_sha256':digest(cache(d)/'test_targets.npy'),'finished':stamp()});save(out/'costs.json',costs)
    with (out/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['report'],required=True);p.add_argument('--dataset',choices=['coco','nus'],required=True);a=p.parse_args();run(a.dataset)
