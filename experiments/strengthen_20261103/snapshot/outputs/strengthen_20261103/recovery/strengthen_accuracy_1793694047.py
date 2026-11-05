"""ICME 2027 follow-up experiments. All selection uses training data."""

import numpy as np
import torch
import run_priority_full_impl as r
import run_supplement_20261029 as old
import supplement_linear_20261029 as linear
from fair_baselines_common import features
from run_fair_ridge import GRID

ROOT = r.ROOT
OUT = ROOT / "outputs/strengthen_20261103"
WEIGHTS = np.linspace(0, 1, 21, dtype=np.float32)


def paired_inputs(dataset, condition, seed, dest):
    hist = ROOT / "outputs/supplement_20261029/runs" / f"{dataset}_{condition}_m10_s{seed}"
    y = r.targets(dataset, "train")
    assignment = old.partition(y, condition, seed, 10, dest)
    fit, val = old.split_indices(len(y), seed * 100 + 7)
    np.save(dest / "assignment.npy", assignment)
    np.savez(dest / "split.npz", fit=fit, validation=val)
    inventory = r.read(ROOT / "inputs.json")
    for source, saved in [("assignment.npy","assignment.npy"),(f"split{seed*100+7}.npz","split.npz")]:
        expected = inventory.get((hist/source).relative_to(ROOT).as_posix())
        if expected:
            assert r.digest(dest/saved)==expected["sha256"],source
    return assignment,fit,val


def fedavg(d, condition, seed, dest):
    a,f,v = paired_inputs(d,condition,seed,dest)
    x,y = features(d,"train"),r.targets(d,"train")
    def run(config, rounds):
        folder=dest/"search"/linear.name(config)
        result=folder/"result.json"
        if result.exists() and r.read(result)["rounds"]>=rounds:
            return r.read(result)
        return linear.fit(x[f],y[f],a[f],config,"fedavg",seed,rounds,folder,x[v],y[v])[1]
    rows=[run(config,200) for config in linear.candidates()]
    def winning(values):
        return max((q for q in values if q["status"]=="COMPLETE"),key=lambda q:(q["best_validation_mAP"],-q["best_round"]))
    for opt,high,extra in [("sgd",100.,1000.),("adamw",.1,1.)]:
        best=winning([q for q in rows if q["config"]["optimizer"]==opt])
        if best["config"]["lr"]==high:
            rows.append(run({**best["config"],"lr":extra},200))
    for budget in (400,800):
        best=winning(rows)
        curve=best["curve"]
        if best["best_round"]!=best["rounds"] or curve[-1]["validation_mAP"]-curve[-2]["validation_mAP"]<.05:break
        rows[rows.index(best)]=run(best["config"],budget)
    best=winning(rows)
    r.save(dest/"selection.json",dict(config=best["config"],rounds=best["best_round"],rows=rows))
    state,stats=linear.fit(x,y,a,best["config"],"fedavg",seed,best["best_round"],dest/"refit")
    assert stats["status"]=="COMPLETE"
    score=features(d,"test")@state["weight"].numpy().T+state["bias"].numpy()
    np.save(dest/"pred.npy",score)
    m,ap=r.metric(r.targets(d),score)
    hist=r.read(ROOT/"outputs/supplement_20261029/runs"/f"{d}_{condition}_m10_s{seed}"/"result.json")
    return dict(mAP=m,class_AP=ap.tolist(),delta_vs_full1=m-next(q["mAP"] for q in hist["rows"] if q["method"]=="Full_1"),fit=stats)


def random_features(d, seed, dest):
    a,f,v=paired_inputs(d,"iid",seed,dest)
    y=r.targets(d,"train")
    out=[]
    for view in ("cls","three_view"):
        x,z=(features(d,"train"),features(d,"test")) if view=="three_view" else tuple(r.normalized(np.load(r.resolve_cache(d)/f"{part}_cls.npy").astype("float32")) for part in ("train","test"))
        # Shared Gaussian Fourier map and uniform phase; CPU float64 normal equations.
        rng=np.random.default_rng(seed); width=1024
        projection=rng.normal(size=(x.shape[1],width)).astype("float32")
        phase=rng.uniform(0,2*np.pi,width).astype("float32")
        best=None
        for scale in (.5,1.,2.):
            def encode(xx): return (np.sqrt(2./width)*np.cos((xx@projection)*scale+phase)).astype("float32")
            def moments(ids):
                gram=np.zeros((width,width));cross=np.zeros((width,y.shape[1]))
                for cid in range(10):
                    group=ids[a[ids]==cid]
                    for batch in np.array_split(group,max(1,(len(group)+2047)//2048)):
                        phi=encode(x[batch]).astype("float64")
                        gram+=phi.T@phi;cross+=phi.T@y[batch]
                return gram,cross
            gram,cross=moments(f);eig,V=np.linalg.eigh(gram);rhs=V.T@cross
            zv=encode(x[v]);curve=[]
            for lam in GRID:
                w=V@(rhs/(eig[:,None]+max(gram.trace()/width,1e-12)*lam))
                curve.append(r.metric(y[v],zv@w.astype("float32"))[0])
            ix=int(np.argmax(curve))
            if best is None or curve[ix]>best[0]:best=(curve[ix],scale,GRID[ix])
        scale=best[1];gram,cross=moments(np.arange(len(y)))
        w=np.linalg.solve(gram+np.eye(width)*gram.trace()/width*best[2],cross)
        scores=np.concatenate([encode(part)@w.astype("float32") for part in np.array_split(z,10)])
        np.save(dest/f"{view}.npy",scores)
        out.append(dict(view=view,validation_mAP=best[0],scale=scale,alpha=best[2],width=width,locked_fit_uplink_bytes=10*(gram.nbytes+cross.nbytes)))
    for row in out:row["mAP"]=r.metric(r.targets(d),np.load(dest/(row["view"]+".npy")))[0]
    return dict(rows=out,method="Fed3R-RF multilabel adaptation with shared random Fourier features, uncentered, no intercept")


def qa_score(d,model,part,ids):
    cache=r.resolve_cache(d);b=r.BENCHMARKS[d]
    return r.piaa_scores(cache/f"{part}_patch.npy",cache/f"{part}_cls.npy",ids,torch.from_numpy(np.load(cache/"text_features.npy")).float(),model.weight,model.bias,cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device="cuda")


def variants(banks,text,temp,policy,clamp):
    selected=r.core._global_quantile_selection(tuple(banks.entropy_banks),512) if policy=="global" else None
    quotas={}
    if policy=="quota":
        for c in range(len(text)):
            sizes=np.array([len(bank[c].entropy) if c in bank else 0 for bank in banks.entropy_banks])
            allocations=np.zeros(len(banks),dtype=int)
            left=min(512,int(sizes.sum()))
            while left:
                eligible=np.flatnonzero(allocations<sizes)[:left]
                allocations[eligible]+=1
                left-=len(eligible)
            quotas[c]=allocations
    def batches():
        for cid,bank in enumerate(banks):
            if selected is not None:
                bank={c:r.PatchBank(b.features[indices],b.probabilities[indices],b.entropy[indices]) for c,indices in selected[cid].items() for b in [bank[c]]}
            if policy=="quota":
                bank={c:r.truncate_bank({c:b},int(quotas[c][cid]))[c]
                      for c,b in bank.items() if quotas[c][cid]>0}
            if bank:yield r.core._stack_banks((bank,))
    classes=len(text)
    initial=[]; raw=0
    for x,p,y in batches():
        initial.append(r.core._moment_packet(x,p,y,classes));raw+=len(x)
    w,b,_=r.core._gda_from_moment_packets(initial,text)
    ps=[]; maxima=torch.full((classes,),-torch.inf,dtype=torch.float64)
    for x,_,_ in batches():
        logits=x@w+b;pred=logits.argmax(1);ref=(logits*temp).softmax(1);ps.append(r.core._purification_packet(ref,pred,classes))
        for c in pred.unique():maxima[c]=torch.maximum(maxima[c],ref[pred==c,c].max())
    threshold=r.core._purification_threshold_from_packets(ps,classes);binding=int(((threshold>maxima)&maxima.isfinite()).sum())
    if clamp: threshold=torch.minimum(threshold,maxima)
    final=[];kept=0
    for x,_,_ in batches():
        logits=x@w+b;pred=logits.argmax(1);ref=(logits*temp).softmax(1);mask=ref[torch.arange(len(x)),pred]>=threshold[pred];kept+=int(mask.sum())
        if mask.any():final.append(r.core._moment_packet(x[mask],ref[mask],pred[mask],classes))
    w,b,_=r.core._gda_from_moment_packets(final,text)
    # Logical dense uplink for this distributed-moments implementation.
    import dataclasses
    def size(p):return sum(t.numel()*t.element_size() for t in dataclasses.asdict(p).values())
    payload=sum(size(p) for p in [*initial,*ps,*final])
    if selected is not None:payload+=sum(v.entropy.numel()*v.entropy.element_size() for bank in banks.entropy_banks for v in bank.values())
    if policy=="quota":payload+=len(banks)*classes*8  # eligible counts for quota allocation
    if clamp:payload+=len(banks)*classes*8
    return r.Adapter(w.float(),b.float(),{}),dict(raw=raw,kept=kept,clamps=binding,logical_patch_uplink_bytes=payload)


def patch_study(d,kind,seed,fraction,dest,inner_offset=7):
    c=r.resolve_cache(d); x=r.load_features(c,"train");z=r.load_features(c,"test");y=r.targets(d,"train")
    text=torch.from_numpy(np.load(c/"text_features.npy")).float();temp=r.BENCHMARKS[d].purification_temperature
    if kind=="union":a,f,v=paired_inputs(d,"iid",seed,dest); ids=np.arange(len(a));m=10
    else:
        # Twenty clients allow exactly 5, 10 and 15 to participate.
        original=old.partition(y,"beta0.1",37,20,dest)
        active=np.sort(np.random.default_rng(seed).choice(20,int(20*fraction),replace=False))
        ids=np.flatnonzero(np.isin(original,active));a=np.full(len(y),-1);a[ids]=np.searchsorted(active,original[ids]);m=len(active)
        f,v=old.split_indices(len(ids),100*seed+inner_offset);f=ids[f];v=ids[v]
        r.save(dest/"participation.json",dict(clients=active.tolist(),fit=len(f),validation=len(v)))
    modes=[("global",False),("global",True),("union",False),("union",True),("quota",False),("quota",True)] if kind=="union" else [("global",False)]
    def ridge(group,central):
        return tuple(r.make_ridge_packet(x[ix],y[ix]) for ix in ([group] if central else [group[a[group]==i] for i in range(m)]) if len(ix))
    def ridge_pair(central):
        fit_packets=ridge(f,central); full_packets=ridge(ids,central)
        val_scores=[r.predict(x[v],r.fit_ridge_from_packets(fit_packets,regularization=lam)) for lam in GRID]
        best=int(np.argmax([r.metric(y[v],s)[0] for s in val_scores]))
        return val_scores[best],r.predict(z,r.fit_ridge_from_packets(full_packets,regularization=GRID[best])),GRID[best]
    rv,rt,lam=ridge_pair(False);rvz,rtz,_=r.standardize(rv,rt)
    central_rv,central_rt,central_lam=ridge_pair(True);crvz,crtz,_=r.standardize(central_rv,central_rt)
    scores={};meta={};central_scores={}
    for phase,group,part,eval_ids in [("fit",f,"train",v),("full",ids,"test",np.arange(len(z)))]:
        assignment=np.full(len(y),-1);assignment[group]=a[group]
        central=r.core.build_client_banks(c/"train_patch.npy",np.where(assignment>=0,0,-1),text,clients=1,k=512,batch_images=32,device="cuda")[0]
        central_model=r.core.fit_central_piaa(central,text,purification_temperature=temp)
        central_scores[phase]=qa_score(d,central_model,part,eval_ids)
        banks=r.disk_banks(d,assignment,text,m,dest/("banks_"+phase))
        for mode,clamp in modes:
            key=f"{mode}_{int(clamp)}"
            model,details=variants(banks,text,temp,mode,clamp)
            scores[key,phase]=qa_score(d,model,part,eval_ids)
            details.update(central_score_error=float(np.max(np.abs(scores[key,phase]-central_scores[phase]))),central_weight_error=float((model.weight-central_model.weight).abs().max()))
            meta[key,phase]=details
        for p in banks.paths:p.unlink()
    def fuse(qv,qt,rz,tz):
        qz,qtest,_=r.standardize(qv,qt)
        aps=np.array([r.metric(y[v],w*rz+(1-w)*qz)[1] for w in WEIGHTS])
        weight=WEIGHTS[aps.argmax(0)]
        return weight*tz+(1-weight)*qtest
    reference=fuse(central_scores["fit"],central_scores["full"],crvz,crtz)
    if kind=="partial":
        np.save(dest/"central.npy",reference)
        np.save(dest/"ridge.npy",rt)
    rows=[]
    for mode,clamp in modes:
        key=f"{mode}_{int(clamp)}";s=fuse(scores[key,"fit"],scores[key,"full"],rvz,rtz)
        np.save(dest/(key+".npy"),s)
        rows.append(dict(key=key,fit=meta[key,"fit"],full=meta[key,"full"],central_full_score_error=float(np.max(np.abs(reference-s)))))
    yt=r.targets(d);ref_map=r.metric(yt,reference)[0]
    for row in rows:
        key=row["key"];row.update(mAP=r.metric(yt,np.load(dest/(key+".npy")))[0],patch_mAP=r.metric(yt,scores[key,"full"])[0])
    return dict(rows=rows,ridge_mAP=r.metric(yt,rt)[0],central_full_mAP=ref_map,alpha=lam,central_alpha=central_lam,participants=m,samples=len(ids))

def participation(d,seed,fraction,dest):
    children=[]
    for offset in (7,17,27):
        folder=dest/f"inner{offset}"
        folder.mkdir(exist_ok=True)
        if not (folder/"receipt.json").exists():
            row=patch_study(d,"partial",seed,fraction,folder,offset)
            r.save(folder/"receipt.json",row)
        children.append(r.read(folder/"receipt.json"))
    result=children[0].copy()
    for name in ("global_0","central","ridge"):
        avg=np.mean([np.load(dest/f"inner{k}"/(name+".npy")) for k in (7,17,27)],axis=0)
        np.save(dest/(name+"_three.npy"),avg)
        result[name+"_three_mAP"]=r.metric(r.targets(d),avg)[0]
    result["splits"]=[7,17,27]
    return result
