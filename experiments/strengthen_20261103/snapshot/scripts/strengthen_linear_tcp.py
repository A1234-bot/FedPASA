"""Real TCP measurements for the two linear baselines using the same Wire as Full."""
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import torch
from fair_baselines_common import cache,read,save,metric,raw_features,assignment,split
from run_fair_ridge import GRID
from run_reviewer_evidence import predict
from fedpasa.supervised import RidgePacket,make_ridge_packet,fit_ridge_from_packets
from supplement_transport_20261029 import listener,connect,Wire,wire_summary
from tcp_supplement_20261029 import numpy_packet,restore_packet,model_payload
import supplement_linear_20261029 as linear


def client(port,cid,d,method):
    torch.set_num_threads(1)
    group=np.flatnonzero(assignment(d)==cid)
    fit,val=split(d,17)
    fit=np.flatnonzero(np.isin(group,fit))
    pos=np.flatnonzero(np.isin(val,group));val=np.searchsorted(group,val[pos])
    x=np.concatenate([np.asarray(np.load(cache(d)/f"train_{k}.npy",mmap_mode="r")[group],np.float32) for k in ("cls","patch_mean","patch_max_text")],1)
    y=np.load(cache(d)/"train_targets.npy",mmap_mode="r")[group]
    wire=connect(port,cid);wire.send("hello",dict(client=cid,pid=os.getpid()))
    head=None
    if method=="fedavg":
        xn=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
        tx=torch.as_tensor(xn,device="cuda");ty=torch.as_tensor(y,dtype=torch.float32,device="cuda")
        head=torch.nn.Linear(tx.shape[1],ty.shape[1],device="cuda")
    while True:
        q=wire.recv();op=q["op"];ids=fit if q.get("phase")=="fit" else np.arange(len(x))
        if op=="stop":
            wire.send("bye",True);break
        if op=="ridge":ans=numpy_packet(make_ridge_packet(x[ids],y[ids]))
        elif op=="step":
            head.load_state_dict({k:torch.as_tensor(s,device="cuda") for k,s in q["model"].items()})
            opt=linear.optimizer(head,q["config"])
            idx=torch.as_tensor(ids,device="cuda");gen=torch.Generator(device="cuda").manual_seed(17*1000000+q["round"]*100+cid)
            for batch in idx[torch.randperm(len(ids),generator=gen,device="cuda")].split(256):
                opt.zero_grad(set_to_none=True)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(head(tx[batch]),ty[batch]);loss.backward();opt.step()
            ans=(len(ids),{k:v.detach().cpu().numpy() for k,v in head.state_dict().items()})
        elif op=="validation":
            if method=="ridge":
                scores=[predict(x[val],restore_packet(m,__import__("fedpasa.core",fromlist=["Adapter"]).Adapter)) for m in q["models"]]
            else:
                model=q["model"];scores=[xn[val]@model["weight"].T+model["bias"]]
            ans=(pos,np.array(scores),y[val])
        elif op=="deploy":ans=all(np.isfinite(t).all() for t in q["model"].values())
        else:raise ValueError(op)
        wire.send(op,ans)
    wire.close()


def run(d,method,dest,locked,source):
    torch.set_num_threads(4)
    a=assignment(d);f,v=split(d,17)
    dims=np.load(cache(d)/"train_cls.npy",mmap_mode="r").shape[1]*2+np.load(cache(d)/"train_patch_max_text.npy",mmap_mode="r").shape[1]
    classes=np.load(cache(d)/"train_targets.npy",mmap_mode="r").shape[1]
    sock=listener();port=sock.getsockname()[1];workers=[];wires={};start=time.perf_counter();points=[]
    def all_clients(op,**kwargs):
        replies=[]
        for i in range(10):
            wires[i].send(op+"/request",dict(op=op,**kwargs));replies.append(wires[i].recv(op))
        return replies
    def validation(**kwargs):
        ans=all_clients("validation",**kwargs);order=np.argsort(np.concatenate([v[0] for v in ans]))
        s=np.concatenate([v[1] for v in ans],axis=1)[:,order];ys=np.concatenate([v[2] for v in ans])[order]
        return [metric(ys,q) for q in s]
    def train(config,limit,phase,state=None,offset=0):
        state=state or {"weight":np.zeros((classes,dims),np.float32),"bias":np.zeros(classes,np.float32)}
        curve=[]
        for round_id in range(offset,limit):
            ans=all_clients("step",phase=phase,round=round_id,config=config,model=state)
            n=sum(q[0] for q in ans)
            state={k:sum(q[1][k].astype("float64")*(q[0]/n) for q in ans).astype("float32") for k in state}
            if (round_id+1)%20==0 or round_id+1==limit:
                if phase=="fit":curve.append((validation(model=state)[0],round_id+1))
                else:
                    path=dest/f"round_{round_id+1}.npz";np.savez(path,**state)
                    points.append(dict(round=round_id+1,seconds=time.perf_counter()-start,framed_bytes=sum(z["framed_bytes"] for w in wires.values() for z in w.rows),path=path.name))
        return state,curve
    try:
        for i in range(10):
            p=mp.get_context("spawn").Process(target=client,args=(port,i,d,method));p.start();workers.append(p)
        for _ in workers:
            s,_=sock.accept();s.settimeout(7200);w=Wire(s,-1);hello=w.recv("hello");w.client=hello["client"];wires[w.client]=w
        if method=="ridge":
            if locked:choice=read(source/"selection.json")
            else:
                packets=tuple(restore_packet(p,RidgePacket) for p in all_clients("ridge",phase="fit"))
                models=[fit_ridge_from_packets(packets,regularization=lam) for lam in GRID]
                maps=validation(models=[dict(**model_payload(m),metadata={}) for m in models])
                choice=dict(alpha=GRID[int(np.argmax(maps))],validation=maps)
            save(dest/"selection.json",choice)
            packets=tuple(restore_packet(p,RidgePacket) for p in all_clients("ridge",phase="full"))
            model=fit_ridge_from_packets(packets,regularization=choice["alpha"]);state=model_payload(model)
        else:
            if locked:choice=read(source/"selection.json")
            else:
                runs=[]
                def candidate(c):
                    state,curve=train(c,200,"fit");runs.append(dict(config=c,state=state,curve=curve))
                for c in linear.candidates():candidate(c)
                def rank(q):return max((m,-r) for m,r in q["curve"])
                for opt,hi,ext in [("sgd",100.,1000.),("adamw",.1,1.)]:
                    best=max([q for q in runs if q["config"]["optimizer"]==opt],key=rank)
                    if best["config"]["lr"]==hi:candidate({**best["config"],"lr":ext})
                for budget in (400,800):
                    best=max(runs,key=rank);curve=best["curve"]
                    if -rank(best)[1]!=curve[-1][1] or curve[-1][0]-curve[-2][0]<.05:break
                    state,extra=train(best["config"],budget,"fit",best["state"],curve[-1][1]);best.update(state=state,curve=curve+extra)
                best=max(runs,key=rank);choice=dict(config=best["config"],rounds=-rank(best)[1],validation=[dict(config=q["config"],curve=q["curve"]) for q in runs])
            save(dest/"selection.json",choice);state,_=train(choice["config"],choice["rounds"],"full")
        assert all(all_clients("deploy",model=state))
        seconds=time.perf_counter()-start
        for w in wires.values():w.send("stop",{"op":"stop"})
        for w in wires.values():assert w.recv("bye")
        for p in workers:p.join(60);assert p.exitcode==0
        network=wire_summary(list(wires.values()))
        xt=raw_features(d,"test");xt=xt/np.maximum(np.linalg.norm(xt,axis=1,keepdims=True),1e-12)
        t=time.perf_counter()
        scores=xt@(state["weight"].T if method=="fedavg" else state["weight"])+state["bias"]
        inference=time.perf_counter()-t
        np.save(dest/"test_scores.npy",scores);yt=np.load(cache(d)/"test_targets.npy")
        for q in points:
            model=np.load(dest/q["path"]);q["mAP"]=metric(yt,xt@model["weight"].T+model["bias"])
        save(dest/"transport.json",network)
        save(dest/"result.json",dict(status="COMPLETE",mAP=metric(yt,scores),total_fitting_wall_seconds=seconds,inference_seconds=inference,network={k:v for k,v in network.items() if k!="messages"},points=points,client_pids=[p.pid for p in workers],mode="locked_refit" if locked else "search_and_refit"))
    finally:
        sock.close()
        for w in wires.values():w.close()
        for p in workers:
            if p.is_alive():p.terminate()
            p.join()
