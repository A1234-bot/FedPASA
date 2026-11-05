"""Actual eleven-process Full protocol over framed TCP on one host.

Client workloads are admitted sequentially to bound memory and GPU contexts.
All ten client processes remain distinct and own their local data/QA banks.
This measures real serialization and TCP payload, not WAN or parallel speedup.
"""
import argparse,dataclasses,gc,multiprocessing as mp,os,time
from pathlib import Path
import numpy as np
from supplement_transport_20261029 import Wire,listener,connect,wire_summary
from fair_baselines_common import ROOT,read,save,digest,stamp
from run_fair_ridge import GRID
OUT=Path(os.environ["STRENGTH_JOB"])
LOCKED=os.environ.get("STRENGTH_LOCKED")=="1"
SEEDS=(17,) if os.environ["SUPPLEMENT_SPLITS"]=="1" else (17,7,27)
ALPHAS=GRID
def numpy_packet(packet):
    if packet is None:return None
    return {k:(v.numpy() if hasattr(v,'numpy') else v) for k,v in dataclasses.asdict(packet).items()}

def restore_packet(data,cls):
    import torch
    return cls(**{k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in data.items()})

def client_worker(port,cid,d,folder,device):
    import torch
    from threadpoolctl import threadpool_limits
    import fedpasa.core as core
    from fedpasa.supervised import make_ridge_packet
    from run_reviewer_evidence import resolve_cache,predict
    from run_label_patch_fusion import piaa_scores
    from fedpasa.benchmarks import BENCHMARKS
    torch.set_num_threads(1);limits=threadpool_limits(limits=1)
    path=Path(folder);cache=resolve_cache(d);b=BENCHMARKS[d];text=torch.from_numpy(np.load(cache/'text_features.npy')).float()
    membership=np.load(path/'client_data'/f'client{cid}.npz');ids=membership['ids'];n=int(membership['train_size']);classes=len(text);dim=text.shape[1]
    x=np.concatenate([np.asarray(np.load(cache/f'train_{name}.npy',mmap_mode='r')[ids],dtype=np.float32) for name in ('cls','patch_mean','patch_max_text')],axis=1)
    y=np.asarray(np.load(cache/'train_targets.npy',mmap_mode='r')[ids]);wire=connect(port,cid);wire.send('hello',dict(client=cid,pid=os.getpid()));bank=None;local=None;refined=None;preds=None;records=[]
    try:
        while True:
            req=wire.recv();op=req['op'];phase=req.get('phase','');stage=req.get('reply',op);start=time.perf_counter()
            if op=='stop':break
            if op=='begin':
                fit=np.arange(len(ids)) if phase=='full' else membership[f'fit{phase}']
                assigned=np.full(n,-1,dtype=np.int64);assigned[ids[fit]]=0
                bank=core.build_client_banks(cache/'train_patch.npy',assigned,text,clients=1,k=512,batch_images=32,device=device)[0]
                target=path/'client_banks'/phase/f'client{cid}.pt';target.parent.mkdir(parents=True,exist_ok=True);temp=target.with_suffix('.partial');torch.save(bank,temp);temp.replace(target)
                packet=make_ridge_packet(x[fit],y[fit]);reply=numpy_packet(packet)
            elif op=='entropy':reply={c:v.entropy.numpy() for c,v in bank.items()}
            elif op=='moments':
                xs=[];ps=[];ys=[]
                for c,indices in sorted(req['selection'].items()):
                    ix=torch.from_numpy(indices);xs.append(bank[c].features[ix].double());ps.append(bank[c].probabilities[ix].double());ys.append(torch.full((len(ix),),c,dtype=torch.long))
                local=(torch.cat(xs),torch.cat(ps),torch.cat(ys)) if xs else (torch.empty((0,dim),dtype=torch.float64),torch.empty((0,classes),dtype=torch.float64),torch.empty(0,dtype=torch.long))
                reply=numpy_packet(core._moment_packet(*local,classes)) if len(local[0]) else None
                bank=None;gc.collect()
            elif op=='purification':
                w=torch.from_numpy(req['weight']);bias=torch.from_numpy(req['bias']);logits=local[0]@w+bias
                refined=torch.nn.functional.softmax(logits*b.purification_temperature,dim=-1);preds=logits.argmax(-1)
                reply=numpy_packet(core._purification_packet(refined,preds,classes))
            elif op=='final_moments':
                threshold=torch.from_numpy(req['threshold']);parts=[]
                for c in preds.unique().tolist():
                    ix=torch.nonzero(preds==c,as_tuple=False).flatten();parts.append(ix[refined[ix,c]>=threshold[c]])
                keep=torch.cat(parts) if parts else torch.empty(0,dtype=torch.long)
                reply=numpy_packet(core._moment_packet(local[0][keep],refined[keep],preds[keep],classes)) if len(keep) else None
                local=None;refined=None;preds=None;gc.collect()
            elif op=='validation':
                loc=membership[f'val{phase}'];pos=membership[f'pos{phase}'];qa=req['qa']
                q=piaa_scores(cache/'train_patch.npy',cache/'train_cls.npy',ids[loc],text,torch.from_numpy(qa['weight']),torch.from_numpy(qa['bias']),cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device=device) if len(loc) else np.empty((0,classes),np.float32)
                candidates=[]
                for model in req['ridge']:
                    adapter=core.Adapter(torch.from_numpy(model['weight']),torch.from_numpy(model['bias']),{})
                    candidates.append(predict(x[loc],adapter))
                reply=dict(position=pos,qa=q,ridge=np.stack(candidates),labels=y[loc])
            elif op=='calibration':
                assert np.isfinite(req['weights']).all();reply={'ok':True}
            elif op=='deploy':
                assert all(np.isfinite(v).all() for m in req['models'] for v in m.values());reply={'ok':True}
            else:raise ValueError(op)
            if device=='cuda':torch.cuda.synchronize()
            records.append(dict(stage=stage,op=op,phase=phase,seconds=time.perf_counter()-start));wire.send(stage,reply)
        import resource
        wire.send('client_metrics',dict(work=records,wire=wire.rows,pid=os.getpid(),peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024))
    except Exception as error:
        try:wire.send('error',dict(error=repr(error),client=cid))
        except Exception:pass
        raise
    finally:wire.close();limits.restore_original_limits()

def model_payload(adapter):return {'weight':adapter.weight.numpy(),'bias':adapter.bias.numpy()}

def run(d):
    import torch
    from threadpoolctl import threadpool_limits
    import fedpasa.core as core
    from fedpasa.supervised import RidgePacket,fit_ridge_from_packets
    import run_priority_full_impl as r
    from run_label_patch_fusion import standardize,piaa_scores
    from run_reviewer_evidence import resolve_cache,load_features,predict
    from run_icme_experiments import assignments_for
    torch.set_num_threads(4);limits=threadpool_limits(limits=4);dest=OUT
    if (dest/'result.json').exists():return
    dest.mkdir(parents=True,exist_ok=True);cache=resolve_cache(d);text=torch.from_numpy(np.load(cache/'text_features.npy')).float();classes=len(text);device='cuda' if torch.cuda.is_available() else 'cpu'
    # Provision immutable client memberships before timing, like pre-existing datasets.
    assignment_path=ROOT/"outputs/icme"/d/"partitions/clients10_seed7.npy"
    a=np.load(assignment_path)
    (dest/"partitions").mkdir(exist_ok=True)
    np.save(dest/"partitions/clients10_seed7.npy",a)
    n=len(a);splits={}
    for seed in SEEDS:
        root=ROOT/'outputs/icme'/d/'full_seeds'/f'seed{seed}'/'ridge';fit=np.load(root/'fit_indices.npy');val=np.load(root/'validation_indices.npy');assert not np.intersect1d(fit,val).size
        np.testing.assert_array_equal(np.sort(np.r_[fit,val]),np.arange(n));splits[str(seed)]=(fit,val)
    provision=dest/'client_data';provision.mkdir(exist_ok=True)
    for cid in range(10):
        ids=np.flatnonzero(a==cid);fields=dict(ids=ids,train_size=np.array(n))
        for seed,(fit,val) in splits.items():
            # Validation ordering follows the pre-existing randomized split exactly.
            pos=np.flatnonzero(a[val]==cid);local=np.searchsorted(ids,val[pos]);np.testing.assert_array_equal(ids[local],val[pos])
            fields.update({f'fit{seed}':np.flatnonzero(np.isin(ids,fit)),f'val{seed}':local,f'pos{seed}':pos})
        np.savez(provision/f'client{cid}.npz',**fields)
    save(dest/'protocol.json',dict(dataset=d,seeds=SEEDS,alphas=ALPHAS,clients=10,server_processes=1,device=device,client_compute_concurrency=1,partition_seed=7,assignment_sha256=digest(dest/'partitions/clients10_seed7.npy'),scope='complete cached-feature fitting plus training-side selection and calibration via TCP; cold bank construction; shared filesystem emulates prepositioned private client stores, not an access-control security boundary; pretraining/feature extraction and dataset provisioning excluded; no WAN speed claim',source_sha256=digest(Path(__file__))))
    server=listener();ctx=mp.get_context('spawn');port=server.getsockname()[1];processes=[];wires={};server_seconds=0.;checks=[];phases=[];start=time.perf_counter()
    def rpc(cid,stage,op,**kw):
        w=wires[cid];w.send(stage+'/request',dict(op=op,reply=stage,**kw));return w.recv(stage)
    def solve(fn):
        nonlocal server_seconds
        t=time.perf_counter();value=fn();server_seconds+=time.perf_counter()-t;return value
    try:
        for cid in range(10):
            p=ctx.Process(target=client_worker,args=(port,cid,d,str(dest),device));p.start();processes.append(p)
        for _ in range(10):
            sock,_=server.accept();sock.settimeout(7200);w=Wire(sock,-1);hello=w.recv('hello');w.client=hello['client'];wires[w.client]=w
        full_packets=None;full_qa=None;selections=[];qa_models={}
        for phase in (['full'] if LOCKED else ['full',*map(str,SEEDS)]):
            began=time.perf_counter();prefix=phase+'/';packets=[];banks=[]
            for cid in range(10):
                val=rpc(cid,prefix+'ridge_statistics','begin',phase=phase);packets.append(restore_packet(val,RidgePacket))
                entropy=rpc(cid,prefix+'entropy','entropy',phase=phase)
                banks.append({c:core.PatchBank(torch.empty(0),torch.empty(0),torch.from_numpy(e)) for c,e in entropy.items()})
                print('TCP_CLIENT_BANK',d,phase,cid,flush=True)
            chosen=solve(lambda:core._global_quantile_selection(tuple(banks),512));moments=[]
            for cid in range(10):
                val=rpc(cid,prefix+'preliminary_moments','moments',phase=phase,selection={c:i.numpy() for c,i in chosen[cid].items()})
                if val is not None:moments.append(restore_packet(val,core.MomentPacket))
            w,b,_=solve(lambda:core._gda_from_moment_packets(moments,text));pur=[]
            for cid in range(10):pur.append(restore_packet(rpc(cid,prefix+'purification_statistics','purification',phase=phase,weight=w.numpy(),bias=b.numpy()),core.PurificationPacket))
            threshold=solve(lambda:core._purification_threshold_from_packets(pur,classes));final=[]
            for cid in range(10):
                val=rpc(cid,prefix+'final_moments','final_moments',phase=phase,threshold=threshold.numpy())
                if val is not None:final.append(restore_packet(val,core.MomentPacket))
            w,b,_=solve(lambda:core._gda_from_moment_packets(final,text));qa=core.Adapter(w.float(),b.float(),{});qa_models[phase]=qa
            torch.save(model_payload(qa),dest/f'qa_{phase}.pt')
            if phase=='full':full_packets=packets;full_qa=qa
            else:
                ridge=[solve(lambda lam=lam:fit_ridge_from_packets(tuple(packets),regularization=lam)) for lam in ALPHAS]
                nv=len(splits[phase][1]);qv=np.empty((nv,classes),np.float32);rv=np.empty((len(ALPHAS),nv,classes),np.float32);yv=None;seen=np.zeros(nv,bool)
                for cid in range(10):
                    val=rpc(cid,prefix+'validation_scores_labels','validation',phase=phase,qa=model_payload(qa),ridge=[model_payload(m) for m in ridge]);pos=val['position'];assert not seen[pos].any();seen[pos]=True;qv[pos]=val['qa'];rv[:,pos]=val['ridge']
                    if yv is None:yv=np.empty((nv,classes),dtype=val['labels'].dtype)
                    yv[pos]=val['labels']
                assert seen.all()
                def calibrate():
                    maps=[core.mean_average_precision(yv,v)[0] for v in rv];i=int(np.argmax(maps));qz,_,qc=standardize(qv,qv);rz,_,rc=standardize(rv[i],rv[i]);weights=np.linspace(0,1,21,dtype=np.float32);aps=np.array([core.mean_average_precision(yv,weight*rz+(1-weight)*qz)[1] for weight in weights]);cw=weights[aps.argmax(0)]
                    return i,qc,rc,cw,maps
                i,qc,rc,cw,maps=solve(calibrate);selection=dict(seed=int(phase),alpha=ALPHAS[i],qa_calibration=qc.tolist(),ridge_calibration=rc.tolist(),weights=cw.tolist(),validation_mAP=maps)
                save(dest/f'seed{phase}_selection.json',selection);selections.append((int(phase),ALPHAS[i],qc,rc,cw))
                # Store reference inputs for independent calibration audit; not sent twice.
                np.savez(dest/f'seed{phase}_received_validation.npz',qa=qv,ridge=rv[i],labels=yv)
                for cid in range(10):assert rpc(cid,prefix+'calibration_ack','calibration',phase=phase,weights=cw,qa_calibration=qc,ridge_calibration=rc)['ok']
            phases.append(dict(phase=phase,wall_seconds=time.perf_counter()-began));save(dest/'progress.json',dict(status='RUNNING',completed_phase=phase,time=stamp()));print('TCP_PHASE_COMPLETE',d,phase,flush=True)
        if LOCKED:
            for seed in SEEDS:
                sel=read(Path(os.environ["STRENGTH_SEARCH"])/f"seed{seed}_selection.json")
                qc,rc,cw=(np.asarray(sel[k],dtype=np.float32) for k in ("qa_calibration","ridge_calibration","weights"))
                selections.append((seed,sel["alpha"],qc,rc,cw))
                for cid in range(10):
                    assert rpc(cid,f"locked{seed}/calibration_ack","calibration",phase=str(seed),weights=cw,qa_calibration=qc,ridge_calibration=rc)["ok"]
        full_ridge={lam:solve(lambda lam=lam:fit_ridge_from_packets(tuple(full_packets),regularization=lam)) for lam in sorted({s[1] for s in selections})}
        deployment=[model_payload(full_qa),*[model_payload(v) for v in full_ridge.values()]]
        for cid in range(10):assert rpc(cid,'deploy_ack','deploy',models=deployment)['ok']
        fitting_seconds=time.perf_counter()-start
        for w in wires.values():w.send('stop',{'op':'stop'})
        client_stats=[w.recv('client_metrics') for w in wires.values()]
        for p in processes:p.join(60);assert p.exitcode==0
        network=wire_summary(list(wires.values()))
        data_suffixes={'ridge_statistics','entropy','preliminary_moments','purification_statistics','final_moments','validation_scores_labels'}
        data_rows=[v for v in network['messages'] if v['direction']=='recv' and v['stage'].split('/')[-1] in data_suffixes]
        calibration_rows=[v for v in data_rows if v['stage'].endswith('/validation_scores_labels')]
        network.update(algorithm_uplink_stages=len({v['stage'] for v in data_rows}),algorithm_uplink_payload_bytes=sum(v['payload_bytes'] for v in data_rows),algorithm_uplink_framed_bytes=sum(v['framed_bytes'] for v in data_rows),selection_calibration_uplink_framed_bytes=sum(v['framed_bytes'] for v in calibration_rows),stage_count_note='Algorithm uploads exclude hello, acknowledgement and telemetry; all-protocol totals include these control messages.')
        assert network['algorithm_uplink_stages']==5+6*(0 if LOCKED else len(SEEDS))
        save(dest/'transport.json',network)
        # Independent original single-process core on identical local banks.
        verification_start=time.perf_counter()
        for phase,qa in qa_models.items():
            banks=r.DiskBanks([dest/'client_banks'/phase/f'client{cid}.pt' for cid in range(10)])
            with r.Profile().track():serial=core.fit_quantile_aligned_fedpasa(banks,text,k=512,purification_temperature=r.BENCHMARKS[d].purification_temperature)
            we=float((serial.weight-qa.weight).abs().max());be=float((serial.bias-qa.bias).abs().max())
            # Client BLAS uses one thread, while the serial reference uses four.
            # Compare FP32 adapters with an explicit absolute tolerance; do not
            # claim bitwise identity or use held-out test metrics for acceptance.
            check=dict(phase=phase,qa_weight_error=we,qa_bias_error=be,absolute_tolerance=1e-6,relative_tolerance=0,bitwise_identical=bool(torch.equal(serial.weight,qa.weight) and torch.equal(serial.bias,qa.bias)))
            checks.append(check);save(dest/'reference_checks.json',checks)
            np.testing.assert_allclose(serial.weight.numpy(),qa.weight.numpy(),rtol=0,atol=1e-6)
            np.testing.assert_allclose(serial.bias.numpy(),qa.bias.numpy(),rtol=0,atol=1e-6)
            del banks;gc.collect()
        verification_seconds=time.perf_counter()-verification_start
        t=time.perf_counter();benchmark=r.BENCHMARKS[d];nt=np.load(cache/'test_cls.npy',mmap_mode='r').shape[0]
        qt=piaa_scores(cache/'test_patch.npy',cache/'test_cls.npy',np.arange(nt),text,full_qa.weight,full_qa.bias,cls_scale=benchmark.cls_scale,patch_scale=benchmark.patch_scale,batch_images=128,device=device);np.save(dest/'qa_test_scores.npy',qt);xt=load_features(cache,'test');predictions=[]
        for seed,lam,qc,rc,cw in selections:
            rt=predict(xt,full_ridge[lam]);score=cw*((rt-rc[0])/rc[1])+(1-cw)*((qt-qc[0])/qc[1]);np.save(dest/f'seed{seed}_test_scores.npy',score);predictions.append(score)
        score=np.mean(predictions,axis=0);np.save(dest/'test_scores.npy',score);inference_seconds=time.perf_counter()-t
        ytest=np.load(cache/'test_targets.npy');mAP=core.mean_average_precision(ytest,score)[0]
        controls=ROOT/'outputs/priority_20260913/controlled/full_runtime'/d/'test_scores.npy';comparison=None
        if controls.exists():
            old=np.load(controls);comparison=dict(reference=str(controls),reference_mAP=core.mean_average_precision(ytest,old)[0],max_abs_score_error=float(np.max(np.abs(score-old))))
        serialization=sum(v.get('serialize_seconds',0)+v.get('deserialize_seconds',0) for v in network['messages'])+sum(v.get('serialize_seconds',0)+v.get('deserialize_seconds',0) for s in client_stats for v in s['wire'])
        result=dict(status='COMPLETE',dataset=d,mAP=mAP,device=device,clients=10,server_pid=os.getpid(),client_pids=[s['pid'] for s in client_stats],total_fitting_wall_seconds=fitting_seconds,server_solve_calibration_seconds=server_seconds,serialization_deserialization_seconds_all_processes=serialization,client_compute_seconds_sum=sum(v['seconds'] for s in client_stats for v in s['work']),inference_seconds=inference_seconds,reference_verification_seconds_excluded=verification_seconds,reference_checks=checks,historical_comparison=comparison,network=network,phases=phases,client_stats=client_stats,test_labels_used_for_selection=False,finished=stamp(),scope='TCP loopback cached-feature end-to-end fitting, selection, calibration and deployment; client concurrency limited to one; timings exclude feature extraction, provisioning, metric audit, and independent reference verification')
        import resource
        result.update(peak_server_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024, simulated_mbps=float(os.environ.get('SUPPLEMENT_MBIT','0')), simulated_rtt_ms=float(os.environ.get('SUPPLEMENT_RTT_MS','0')), network_scope='application-paced TCP loopback; fixed per-message one-way delay and sender pacing; not measured WAN or OS traffic shaping')
        result.update(scope_mode="locked_refit" if LOCKED else "search_and_refit", test_score_sha256=digest(dest/"test_scores.npy"))
        for tmp_bank in (dest/"client_banks").glob("*/client*.pt"):
            tmp_bank.unlink()
        save(dest/'result.json',result);print('SYSTEMS_COMPLETE',d,mAP,flush=True)
    finally:
        for w in wires.values():w.close()
        server.close()
        for p in processes:
            if p.is_alive():p.terminate();p.join()
        limits.restore_original_limits()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,choices=['coco','nus']);a=p.parse_args();run(a.dataset)
