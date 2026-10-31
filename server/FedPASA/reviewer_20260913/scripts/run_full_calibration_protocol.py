"""Exact calibration replay through ten isolated client processes and TCP."""
import argparse,multiprocessing as mp,os,sys,time
from pathlib import Path
import numpy as np
from full_protocol_transport import Wire,listener,connect,wire_summary
from fair_baselines_common import read,save,digest,ROOT
OUT=ROOT/'outputs/full_protocol_20260919'

def calibrate(q,r,y,weights):
    from run_label_patch_fusion import standardize
    from fedpasa.core import mean_average_precision
    qz,_,qs=standardize(q,q);rz,_,rs=standardize(r,r)
    ap=np.array([mean_average_precision(y,w*rz+(1-w)*qz)[1] for w in weights])
    cw=np.asarray(weights,dtype=np.float32)[ap.argmax(0)]
    return dict(qa=qs,ridge=rs,weights=cw)

def score(q,r,state):
    qm,qs=state['qa'];rm,rs=state['ridge'];w=state['weights']
    return w[None,:]*((r-rm)/rs)+(1-w[None,:])*((q-qm)/qs)

def client_main(port,cid,folder):
    wire=connect(port,cid);wire.send('hello',{'client':cid,'pid':os.getpid()})
    while True:
        task=wire.recv()
        if task['op']=='stop':break
        seed=task['seed'];f=np.load(Path(folder)/f'seed{seed}'/f'client{cid}.npz')
        wire.send(f'seed{seed}/validation', {k:f[k] for k in f.files})
        params=wire.recv(f'seed{seed}/calibration')
        assert np.isfinite(params['weights']).all()
        wire.send(f'seed{seed}/ack',{'ok':True})
    wire.send('client_metrics',wire.rows);wire.close()

def run(d):
    import torch
    from run_reviewer_evidence import resolve_cache,load_features
    from run_label_patch_fusion import ridge_scores
    from fedpasa.core import mean_average_precision
    from run_icme_experiments import assignments_for
    torch.set_num_threads(2)
    dest=OUT/'calibration'/d
    if (dest/'result.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);base=ROOT/'outputs/icme'/d;c=resolve_cache(d)
    y=np.load(c/'train_targets.npy');x=load_features(c,'train');z=load_features(c,'test')
    a,_,_=assignments_for(np.load(c/'partition_features.npy'),dest,clients=10,seed=7,n_init=10)
    references={};test_components={};protocol=[]
    for seed in (7,17,27):
        s=base/'full_seeds'/f'seed{seed}';f=s/'ridge_patch_fusion';ix=np.load(s/'ridge/validation_indices.npy');fit=np.load(s/'ridge/fit_indices.npy')
        assert not np.intersect1d(ix,fit).size;np.testing.assert_array_equal(np.sort(np.r_[ix,fit]),np.arange(len(y)))
        q=np.load(f/'training_validation_piaa_scores.npy');m=torch.load(s/'ridge/validation_adapter.pt',map_location='cpu',weights_only=True)
        r=ridge_scores(x[ix],m['weight'],m['bias']);report=read(f/'result.json');weights=[v['ridge_weight'] for v in report['tuning']]
        # Historical QA test predictions were explicitly reused by the original script.
        qt=np.load(base/'scores/qafedpasa_representative_scores.npy');rt=np.load(s/'ridge/test_scores.npy')
        central=calibrate(q,r,y[ix],weights);historical=np.load(f/'test_scores_classwise.npy')
        recreated=score(qt,rt,central);err=float(np.max(np.abs(recreated-historical)))
        historical_weights=np.asarray(report['classwise_weight']['weights'],dtype=np.float32)
        changed=np.flatnonzero(central['weights']!=historical_weights).tolist()
        refs=dest/f'seed{seed}';refs.mkdir(exist_ok=True)
        for cid in range(10):
            pos=np.flatnonzero(a[ix]==cid)
            np.savez(refs/f'client{cid}.npz',position=pos.astype(np.int64),qa=q[pos],ridge=r[pos],labels=y[ix[pos]])
        references[seed]=(central,recreated,weights,len(ix),y.shape[1]);test_components[seed]=(qt,rt)
        protocol.append(dict(seed=seed,validation=len(ix),historical_max_score_error=err,historical_weight_changed_classes=changed,historical_weights=historical_weights.tolist(),recomputed_weights=central['weights'].tolist(),validation_indices_sha256=digest(s/'ridge/validation_indices.npy'),historical_prediction_sha256=digest(f/'test_scores_classwise.npy')))
    del x,z,y,q,r
    save(dest/'input_protocol.json',{'splits':protocol,'scope':'client-local cached validation prediction replay; fitting excluded; server calibration receives sample-level scores and labels','seeds':[7,17,27]})
    server=listener();port=server.getsockname()[1];ctx=mp.get_context('spawn');processes=[];wires={};pidlist=[];t=time.perf_counter()
    try:
        for cid in range(10):
            proc=ctx.Process(target=client_main,args=(port,cid,str(dest)));proc.start();processes.append(proc)
        for _ in range(10):
            sock,_=server.accept();sock.settimeout(7200);wire=Wire(sock,-1);hello=wire.recv('hello');wire.client=hello['client'];wires[wire.client]=wire;pidlist.append(hello['pid'])
        predictions=[];central_scores=[];checks=[];compute_seconds=0
        for seed in (7,17,27):
            ref,historical,weights,n,classes=references[seed];q=np.empty((n,classes),np.float32);r=np.empty_like(q);y=None;seen=np.zeros(n,bool)
            for cid in range(10):
                w=wires[cid];w.send(f'seed{seed}/request',{'op':'scores','seed':seed});v=w.recv(f'seed{seed}/validation');pos=v['position']
                assert not seen[pos].any();seen[pos]=True;q[pos]=v['qa'];r[pos]=v['ridge']
                if y is None:y=np.empty((n,classes),dtype=v['labels'].dtype)
                y[pos]=v['labels']
            assert seen.all();start=time.perf_counter();state=calibrate(q,r,y,weights);compute_seconds+=time.perf_counter()-start
            for k in state:np.testing.assert_array_equal(state[k],ref[k])
            for cid in range(10):wires[cid].send(f'seed{seed}/calibration',state)
            for cid in range(10):assert wires[cid].recv(f'seed{seed}/ack')['ok']
            pred=score(*test_components[seed],state);np.testing.assert_array_equal(pred,historical)
            np.save(dest/f'seed{seed}_distributed_scores.npy',pred);predictions.append(pred);central_scores.append(historical)
            checks.append({'seed':seed,'parameters_identical':True,'score_max_abs_error':float(np.max(np.abs(pred-historical)))})
        for w in wires.values():w.send('stop',{'op':'stop'})
        client_stats=[w.recv('client_metrics') for w in wires.values()]
        for proc in processes:proc.join(30);assert proc.exitcode==0
        s=np.mean(predictions,axis=0);central=np.mean(central_scores,axis=0);np.testing.assert_array_equal(s,central)
        archived=np.load(base/'cv_fusion/test_scores.npy')
        np.save(dest/'distributed_test_scores.npy',s);labels=np.load(c/'test_targets.npy');cm=mean_average_precision(labels,central)[0];dm=mean_average_precision(labels,s)[0]
        historical_map=mean_average_precision(labels,archived)[0]
        network=wire_summary(list(wires.values()));cal=[v for v in network['messages'] if v['direction']=='recv' and v['stage'].endswith('/validation')]
        result=dict(status='COMPLETE',dataset=d,centralized_mAP=cm,distributed_mAP=dm,absolute_delta_mAP=abs(dm-cm),max_abs_score_error=float(np.max(np.abs(s-central))),checks=checks,calibration_uplink_payload_bytes=sum(v['payload_bytes'] for v in cal),calibration_uplink_framed_bytes=sum(v['framed_bytes'] for v in cal),network=network,wall_seconds=time.perf_counter()-t,server_calibration_seconds=compute_seconds,client_pids=pidlist,server_pid=os.getpid(),hardware='server CPU cached prediction replay',client_metrics=client_stats,test_labels_used_for_selection=False)
        result.update(historical_mAP=historical_map,historical_delta_mAP=abs(historical_map-dm),historical_max_abs_score_error=float(np.max(np.abs(s-archived))),reference_note='Exact transport equivalence uses recomputed central calibration; archival cross-environment floating-point drift reported separately')
        save(dest/'result.json',result);print('CALIBRATION_COMPLETE',d,cm,dm,flush=True)
    finally:
        for w in wires.values():w.close()
        server.close()
        for proc in processes:
            if proc.is_alive():proc.terminate();proc.join()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);a=p.parse_args();run(a.dataset)
