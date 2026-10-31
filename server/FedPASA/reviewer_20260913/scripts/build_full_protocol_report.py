"""Recompute metrics and consolidate three supplements without editing history."""
import csv
import numpy as np
from fair_baselines_common import ROOT,read,save,digest,stamp
from run_reviewer_evidence import resolve_cache
from sklearn.metrics import average_precision_score
OUT=ROOT/'outputs/full_protocol_20260919';FINAL=OUT/'final'

def metric(d,p):
    y=np.load(resolve_cache(d)/'test_targets.npy');s=np.load(p);assert s.shape==y.shape and np.isfinite(s).all()
    return float(np.mean([average_precision_score(y[:,j],s[:,j]) for j in range(y.shape[1])])*100)

def table(name,rows):
    save(FINAL/(name+'.json'),rows)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (FINAL/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    FINAL.mkdir(parents=True,exist_ok=True);cal=[];runs=[];systems=[]
    for d in ('voc07','voc12','coco','nus'):
        p=OUT/'calibration'/d;r=read(p/'result.json');assert r['status']=='COMPLETE'
        m=metric(d,p/'distributed_test_scores.npy');assert abs(m-r['distributed_mAP'])<1e-10
        assert r['max_abs_score_error']==0 and r['absolute_delta_mAP']==0
        cal.append(dict(dataset=d,centralized_Full_mAP=r['centralized_mAP'],distributed_Full_mAP=m,absolute_delta_mAP=r['absolute_delta_mAP'],historical_Full_mAP=r['historical_mAP'],historical_delta_mAP=r['historical_delta_mAP'],historical_max_abs_score_error=r['historical_max_abs_score_error'],calibration_uplink_MiB=r['calibration_uplink_framed_bytes']/2**20,total_protocol_uplink_MiB=r['network']['uplink_framed_bytes']/2**20,total_protocol_downlink_MiB=r['network']['downlink_framed_bytes']/2**20,prediction_sha256=digest(p/'distributed_test_scores.npy'),scope='10 clients + server; cached prediction replay; scores, labels, order keys and serialization framing transmitted; fitting excluded'))
    for d in ('voc07','voc12'):
        for index in range(3):
            p=OUT/'outer'/d/f'run{index+1}';r=read(p/'verified.json');protocol=read(p/'protocol.json');assert r['status']=='COMPLETE'
            for s in protocol['split_seeds']:
                root=p/'splits'/f'seed{s}'/'ridge';fit=np.load(root/'fit_indices.npy');val=np.load(root/'validation_indices.npy');assert not np.intersect1d(fit,val).size
                np.testing.assert_array_equal(np.sort(np.r_[fit,val]),np.arange(len(np.load(resolve_cache(d)/'train_targets.npy'))))
            for method,name in [('QAFedPASA','qa_scores.npy'),('3-view Ridge',f"seed{protocol['split_seeds'][0]}_ridge_scores.npy"),('FedPASA-Full','test_scores.npy')]:
                pred=p/'full_runtime'/d/name;m=metric(d,pred);record=next(x for x in r['rows'] if x['method']==method);assert abs(m-record['mAP'])<1e-10
                runs.append(dict(dataset=d,run=index+1,method=method,mAP=m,partition_seed=protocol['partition_seed'],split_seeds=protocol['split_seeds'],prediction_sha256=digest(pred)))
    # Existing accepted COCO/NUS outer records retain their original provenance.
    previous=read(ROOT/'outputs/priority_20260913/final/outer_runs.json')
    allruns=[*runs,*[dict(r,source='accepted priority_20260913 outer runs') for r in previous]];summary=[]
    for d in ('voc07','voc12','coco','nus'):
        for method in ('QAFedPASA','3-view Ridge','FedPASA-Full'):
            values=[r['mAP'] for r in allruns if r['dataset']==d and r['method']==method];assert len(values)==3
            summary.append(dict(dataset=d,method=method,mean_mAP=float(np.mean(values)),sample_std_mAP=float(np.std(values,ddof=1)),runs=3,values=values,scope='independent client repartitions and three regenerated training-side splits per outer run; cached backbone fixed'))
    for d in ('coco','nus'):
        p=OUT/'systems'/d;r=read(p/'result.json');assert r['status']=='COMPLETE';m=metric(d,p/'test_scores.npy');assert abs(m-r['mAP'])<1e-10;n=r['network']
        systems.append(dict(dataset=d,mAP=m,serialized_uplink_MiB=n['uplink_payload_bytes']/2**20,serialized_downlink_MiB=n['downlink_payload_bytes']/2**20,framed_uplink_MiB=n['uplink_framed_bytes']/2**20,framed_downlink_MiB=n['downlink_framed_bytes']/2**20,uplink_stages=n['uplink_stages'],downlink_stages=n['downlink_stages'],fitting_seconds=r['total_fitting_wall_seconds'],serialization_seconds=r['serialization_deserialization_seconds_all_processes'],server_solve_calibration_seconds=r['server_solve_calibration_seconds'],inference_seconds=r['inference_seconds'],device=r['device'],client_processes=10,server_processes=1,scope=r['scope']))
        data_suffixes={'ridge_statistics','entropy','preliminary_moments','purification_statistics','final_moments','validation_scores_labels'}
        uploads=[v for v in n['messages'] if v['direction']=='recv' and v['stage'].split('/')[-1] in data_suffixes]
        stages=len({v['stage'] for v in uploads});assert stages==23
        systems[-1].update(algorithm_uplink_stages=stages,algorithm_uplink_framed_MiB=sum(v['framed_bytes'] for v in uploads)/2**20,selection_calibration_uplink_framed_MiB=sum(v['framed_bytes'] for v in uploads if v['stage'].endswith('/validation_scores_labels'))/2**20)
    table('calibration_equivalence',cal);table('outer_runs',allruns);table('outer_summary',summary);table('tcp_systems',systems)
    save(FINAL/'completion.json',dict(status='COMPLETE',completed=stamp(),calibration_datasets=4,new_outer_runs=6,systems_datasets=2,independent_metric_recalculation=True,baseline_queue_is_separate=True))
    print('THREE_SUPPLEMENTS_COMPLETE',flush=True)

if __name__=='__main__':main()
