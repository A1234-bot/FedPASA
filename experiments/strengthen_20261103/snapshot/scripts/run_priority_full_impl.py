"""Predeclared reviewer experiments; no test-driven selection or old output writes.

All clients are simulated serially on one server. Timings are cache-warm wall
times, not network latency. Exact Full selection explicitly accounts for pooled
validation scores/labels; it is not claimed to be a private sufficient-statistic
selection protocol. Historical three-split scores are audited separately.
"""
from __future__ import annotations
import argparse, contextlib, dataclasses, gc, hashlib, json, os, platform, sys, time
try:
    import resource
except ImportError:
    import psutil
    from types import SimpleNamespace
    class WindowsResource:
        RUSAGE_SELF=0
        @staticmethod
        def getrusage(_):
            info=psutil.Process().memory_info()
            return SimpleNamespace(ru_maxrss=getattr(info,'peak_wset',info.rss)/1024)
    resource=WindowsResource()
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import fedpasa.core as core
from fedpasa.core import Adapter, PatchBank, mean_average_precision as metric
from fedpasa.supervised import make_ridge_packet, fit_ridge_from_packets
from fedpasa.benchmarks import BENCHMARKS
from run_reviewer_evidence import resolve_cache, load_features, predict, average_adapters, fit_tie_safe_piaa
from run_icme_experiments import deserialize_bank, truncate_bank, dirichlet_assignments
from run_label_patch_fusion import piaa_scores, standardize
from run_supplementary_experiments import evaluate_many

OUT=ROOT/'outputs/priority_20260913'
SEEDS=(7,17,27)
ALPHAS=(.0001,.0003,.001,.003,.01,.03,.1)

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');t.replace(p)
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()
def dr(d):return ROOT/'outputs/icme'/d
def sr(d,s=17):return dr(d)/'full_seeds'/f'seed{s}'
def sync():
    if torch.cuda.is_available():torch.cuda.synchronize()
def timed(fn):
    sync();t=time.perf_counter();v=fn();sync();return v,time.perf_counter()-t
def normalized(x):return x/np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)
def targets(d,split='test'):return np.load(resolve_cache(d)/f'{split}_targets.npy')
def measured(d,name,scores,**kw):
    assert np.isfinite(scores).all()
    return dict(dataset=d,method=name,mAP=metric(targets(d),scores)[0],**kw)
def packet_wire(p,packed=False,dtype=np.float64):
    total=8 # integer count
    for k in ('sum_x','sum_y','xtx','xty'):
        a=getattr(p,k).numpy().astype(dtype)
        if packed and k=='xtx':a=a[np.triu_indices(len(a))]
        total+=len(a.tobytes())
    return total

class DiskBanks:
    """Bound the resident patch-bank memory to one logical client."""
    def __init__(self,paths):
        self.paths=paths;self.entropy_banks=[]
        for p in paths:
            bank=torch.load(p,weights_only=False)
            self.entropy_banks.append({c:PatchBank(torch.empty(0),torch.empty(0),v.entropy,None,v.patch_ids) for c,v in bank.items()})
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):return torch.load(self.paths[i],weights_only=False)
    def __iter__(self):
        for p in self.paths:yield torch.load(p,weights_only=False)

def disk_banks(d,a,text,clients,destination):
    destination.mkdir(parents=True,exist_ok=True);paths=[]
    for i in range(clients):
        path=destination/f'client{i}.pt';one=np.where(a==i,0,-1)
        bank=core.build_client_banks(resolve_cache(d)/'train_patch.npy',one,text,clients=1,k=512,batch_images=32,device='cuda')[0]
        tmp=path.with_suffix('.partial');torch.save(bank,tmp);tmp.replace(path);del bank;gc.collect();paths.append(path)
        print('DISK_BANK',d,i+1,'/',clients,flush=True)
    return DiskBanks(paths)
def assignment(d,m=10,seed=7,beta=None):
    if beta is not None:
        return dirichlet_assignments(targets(d,'train'),OUT/'partitions'/d,clients=m,beta=beta,seed=seed)[0]
    return np.load(dr(d)/'partitions'/f'clients{m}_seed{seed}.npy')

def audit(d):
    out=OUT/'audit'/d;out.mkdir(parents=True,exist_ok=True)
    y=targets(d);rows=[];fs=[];rs=[];fingerprints=[]
    for seed in SEEDS:
        base=sr(d,seed);fit=np.load(base/'ridge/fit_indices.npy');val=np.load(base/'ridge/validation_indices.npy')
        assert not np.intersect1d(fit,val).size
        np.testing.assert_array_equal(np.sort(np.r_[fit,val]),np.arange(len(targets(d,'train'))))
        f=np.load(base/'ridge_patch_fusion/test_scores_classwise.npy');r=np.load(base/'ridge/test_scores.npy')
        fs.append(f);rs.append(r)
        rows.append(measured(d,f'seed {seed} only',f,setting='single-split',seed=seed,ridge_mAP=metric(y,r)[0]))
        for p in (base/'ridge/fit_indices.npy',base/'ridge/validation_indices.npy',base/'ridge_patch_fusion/result.json'):
            fingerprints.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
    f=np.mean(fs,axis=0);r=np.mean(rs,axis=0)
    np.testing.assert_allclose(f,np.load(dr(d)/'cv_fusion/test_scores.npy'),atol=3e-5,rtol=0)
    mean=float(np.mean([row['mAP'] for row in rows]));ensemble=metric(y,f)[0]
    rows.extend([dict(dataset=d,method='Mean single-split',mAP=mean,setting='arithmetic mean of three mAP values'),
                 measured(d,'3-split score average',f,setting='historical fixed score ensemble',gain_over_mean_single=ensemble-mean,ridge_mAP=metric(y,r)[0])])
    for p in [resolve_cache(d)/'manifest.json',resolve_cache(d)/'train_targets.npy',resolve_cache(d)/'test_targets.npy',dr(d)/'partitions/clients10_seed7.npy']:
        fingerprints.append({'path':str(p.relative_to(ROOT)),'sha256':digest(p)})
    save(out/'result.json',dict(status='COMPLETE',rows=rows,fingerprints=fingerprints,headline_seed=17,
         rule='Seed 17 fixed before new runs; no replacement based on test mAP. Historical scores are not new reruns.'))

def ridge_family(d):
    out=OUT/'linear'/d;out.mkdir(parents=True,exist_ok=True)
    x=load_features(resolve_cache(d),'train');z=load_features(resolve_cache(d),'test');y=targets(d,'train');a=assignment(d)
    fit=np.load(sr(d)/'ridge/fit_indices.npy');val=np.load(sr(d)/'ridge/validation_indices.npy')
    alpha=read(sr(d)/'ridge/result.json')['selection']['alpha'];rows=[]
    pp,client_time=timed(lambda:tuple(make_ridge_packet(x[a==i],y[a==i]) for i in range(10)))
    center,central_time=timed(lambda:fit_ridge_from_packets((make_ridge_packet(x,y),),regularization=alpha))
    pooled,solve=timed(lambda:fit_ridge_from_packets(pp,regularization=alpha))
    local,local_time=timed(lambda:average_adapters([fit_ridge_from_packets((p,),regularization=alpha) for p in pp],[p.count for p in pp],'sample-weighted local ridge'))
    for name,model,ct,st in [('Centralized Ridge',center,central_time,0),('Three-view pooled Ridge',pooled,client_time,solve),('Local Ridge + Avg (matched penalty)',local,client_time+local_time,0)]:
        scores=predict(z,model);np.save(out/(name.split(' ')[0]+'_scores.npy'),scores)
        rows.append(measured(d,name,scores,features='cls+patch_mean+patch_max_text',alpha=alpha,split_seed=17,partition_seed=7,clients=10,
            client_fitting_seconds=ct,server_solve_seconds=st,fit_wall_seconds=ct+st,uploads_per_client=0 if name.startswith('Centralized') else 1,
            uplink_FP64_bytes=0 if name.startswith('Centralized') else sum(packet_wire(p) for p in pp) if 'pooled' in name else 10*(model.weight.numel()+model.bias.numel())*4))
    # Stronger local baseline selects its own penalty on precisely the same split.
    fp=tuple(make_ridge_packet(x[fit[a[fit]==i]],y[fit[a[fit]==i]]) for i in range(10))
    tuning=[]
    for lam in ALPHAS:
        model=average_adapters([fit_ridge_from_packets((p,),regularization=lam) for p in fp],[p.count for p in fp],'local ridge validation')
        tuning.append((metric(y[val],predict(x[val],model))[0],lam))
    chosen=max(tuning,key=lambda q:q[0])[1]
    model=average_adapters([fit_ridge_from_packets((p,),regularization=chosen) for p in pp],[p.count for p in pp],'local ridge selected penalty')
    scores=predict(z,model);np.save(out/'local_tuned_scores.npy',scores)
    rows.append(measured(d,'Local Ridge + Avg',scores,features='cls+patch_mean+patch_max_text',alpha=chosen,validation_tuning=tuning,split_seed=17,clients=10))
    # Test actual symmetric FP64 and FP32 serialization, not just a formula.
    packing=[]
    for dtype in (np.float64,np.float32):
        decoded=[]
        for p in pp:
            g=p.xtx.numpy().astype(dtype);ix=np.triu_indices(len(g));wire=g[ix].tobytes();v=np.frombuffer(wire,dtype=dtype)
            restored=np.zeros_like(g);restored[ix]=v;restored[(ix[1],ix[0])]=v
            np.testing.assert_array_equal(restored,g)
            decoded.append(type(p)(p.count,torch.from_numpy(p.sum_x.numpy().astype(dtype)).double(),torch.from_numpy(p.sum_y.numpy().astype(dtype)).double(),torch.from_numpy(restored).double(),torch.from_numpy(p.xty.numpy().astype(dtype)).double()))
        score=predict(z,fit_ridge_from_packets(tuple(decoded),regularization=alpha))
        packing.append(dict(dtype=np.dtype(dtype).name,dense_bytes=sum(packet_wire(p,dtype=dtype) for p in pp),packed_bytes=sum(packet_wire(p,True,dtype) for p in pp),mAP=metric(targets(d),score)[0],max_abs_score_error=float(np.max(np.abs(score-predict(z,pooled))))))
    # Fed3R-style uses normalized CLS, uncentered X'X + lambda I, no bias.
    del pp,fp;gc.collect();cx=normalized(np.array(np.load(resolve_cache(d)/'train_cls.npy'),dtype=np.float32));cz=normalized(np.array(np.load(resolve_cache(d)/'test_cls.npy'),dtype=np.float32))
    g=cx[fit].astype('float64').T@cx[fit].astype('float64');b=cx[fit].astype('float64').T@y[fit]
    tuning=[]
    for lam in (1e-6,1e-5,1e-4,1e-3,.01,.1,1,10,100):
        w=np.linalg.solve(g+lam*np.eye(512),b);tuning.append((metric(y[val],cx[val]@w)[0],lam))
    lam=max(tuning,key=lambda q:q[0])[1]
    def fed3_stats():
        g=np.zeros((512,512));b=np.zeros((512,y.shape[1]))
        for i in range(10):
            xx=cx[a==i].astype('float64');g+=xx.T@xx;b+=xx.T@y[a==i]
        return g,b
    (g,b),ct=timed(fed3_stats);w,st=timed(lambda:np.linalg.solve(g+lam*np.eye(512),b));score=(cz@w).astype('float32');np.save(out/'fed3r_scores.npy',score)
    rows.append(measured(d,'Fed3R-style pooled ridge',score,features='normalized CLS; no bias or centering; multilabel adaptation',lambda_value=lam,split_seed=17,clients=10,validation_tuning=tuning,
        client_fitting_seconds=ct,server_solve_seconds=st,fit_wall_seconds=ct+st,uploads_per_client=1,uplink_FP64_bytes=10*(512*512+512*y.shape[1])*8))
    save(out/'result.json',dict(status='COMPLETE',rows=rows,packing=packing))

def fedavg_train(x,y,a,lr,path,rounds=50):
    """All-client FedAvg, one local epoch/round, reset local SGD momentum."""
    path.parent.mkdir(parents=True,exist_ok=True)
    tx=torch.from_numpy(normalized(x)).cuda();ty=torch.from_numpy(y.astype('float32')).cuda()
    head=torch.nn.Linear(x.shape[1],y.shape[1]).cuda();torch.nn.init.zeros_(head.weight);torch.nn.init.zeros_(head.bias)
    begin=0;elapsed=0.;server_seconds=0.
    if path.exists():
        ck=torch.load(path,weights_only=True);head.load_state_dict(ck['model']);begin=ck['round'];elapsed=ck['seconds'];server_seconds=ck['server_seconds']
    sizes=np.bincount(a,minlength=10);ids=[torch.as_tensor(np.flatnonzero(a==i),device='cuda') for i in range(10)]
    for r in range(begin,rounds):
        sync();t=time.perf_counter();global_state={k:v.detach().clone() for k,v in head.state_dict().items()};aggregate={k:torch.zeros_like(v) for k,v in global_state.items()}
        for i,ix in enumerate(ids):
            if not len(ix):continue
            head.load_state_dict(global_state);opt=torch.optim.SGD(head.parameters(),lr=lr,momentum=.9)
            gen=torch.Generator(device='cuda').manual_seed(170000+r*100+i);order=ix[torch.randperm(len(ix),generator=gen,device='cuda')]
            for batch in order.split(256):
                opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(head(tx[batch]),ty[batch]);loss.backward();opt.step()
            sync();ss=time.perf_counter()
            with torch.no_grad():
                for k,v in head.state_dict().items():aggregate[k].add_(v,alpha=float(sizes[i]/len(x)))
            sync();server_seconds+=time.perf_counter()-ss
        head.load_state_dict(aggregate);sync();elapsed+=time.perf_counter()-t
        tmp=path.with_suffix('.partial');torch.save({'model':{k:v.cpu() for k,v in aggregate.items()},'round':r+1,'seconds':elapsed,'server_seconds':server_seconds},tmp);tmp.replace(path)
        print('FEDAVG',path.name,'round',r+1,'seconds',round(elapsed,2),flush=True)
    state={k:v.detach().cpu().numpy() for k,v in head.state_dict().items()};del tx,ty,head;gc.collect();torch.cuda.empty_cache()
    return state,elapsed,server_seconds

def fedavg(d):
    out=OUT/'fedavg'/d;x=load_features(resolve_cache(d),'train');z=load_features(resolve_cache(d),'test');y=targets(d,'train');a=assignment(d)
    fit=np.load(sr(d)/'ridge/fit_indices.npy');val=np.load(sr(d)/'ridge/validation_indices.npy');tuning=[]
    for lr in (.01,.1,1.):
        model,t,s=fedavg_train(x[fit],y[fit],a[fit],lr,out/f'validation_lr{lr}.pt')
        score=normalized(x[val])@model['weight'].T+model['bias'];tuning.append(dict(lr=lr,validation_mAP=metric(y[val],score)[0],seconds=t))
    lr=max(tuning,key=lambda r:r['validation_mAP'])['lr'];model,t,s=fedavg_train(x,y,a,lr,out/'final.pt')
    scores=normalized(z)@model['weight'].T+model['bias'];np.save(out/'test_scores.npy',scores)
    row=measured(d,'FedAvg Linear Head',scores,features='cls+patch_mean+patch_max_text',rounds=50,uploads_per_client=50,local_epochs=1,batch_size=256,loss='BCEWithLogits',optimizer='SGD momentum 0.9; reset each local round',selected_lr=lr,clients=10,partition_seed=7,split_seed=17,
        client_fitting_seconds=t-s,server_solve_seconds=s,fit_wall_seconds=t,selection_seconds=sum(v['seconds'] for v in tuning),uplink_FP32_bytes=50*10*(model['weight'].size+model['bias'].size)*4)
    save(out/'result.json',dict(status='COMPLETE',rows=[row],tuning=tuning))

class Profile:
    """Time the actual simulated message boundaries; byte count is payload only."""
    def __init__(self):self.client=0.;self.server=0.;self.rows=[]
    @contextlib.contextmanager
    def track(self):
        names=['_moment_packet','_purification_packet','_gda_from_moment_packets','_purification_threshold_from_packets','_global_quantile_selection']
        originals={n:getattr(core,n) for n in names}
        def wrap(name,fn):
            def run(*a,**kw):
                if name=='_global_quantile_selection' and isinstance(a[0],DiskBanks):a=(tuple(a[0].entropy_banks),*a[1:])
                v,t=timed(lambda:fn(*a,**kw))
                if name in ('_moment_packet','_purification_packet'):
                    self.client+=t;fields=dataclasses.asdict(v);dense=packed=0
                    for k,tensor in fields.items():
                        z=tensor.numpy();dense+=len(z.tobytes());packed+=len((z[np.triu_indices(len(z))] if k=='second_moment' else z).tobytes())
                    self.rows.append(dict(stage=name,bytes=dense,packed_bytes=packed,FP32_bytes=dense//2,packed_FP32_bytes=packed//2))
                else:
                    self.server+=t
                    if name=='_global_quantile_selection':
                        size=sum(v.entropy.numel()*v.entropy.element_size() for b in a[0] for v in b.values())
                        self.rows.append(dict(stage=name,bytes=size,packed_bytes=size,FP32_bytes=size,packed_FP32_bytes=size))
                return v
            return run
        for n,f in originals.items():setattr(core,n,wrap(n,f))
        try:yield self
        finally:
            for n,f in originals.items():setattr(core,n,f)

def union_stream(banks,text,temp):
    """Exact pooled local-union GDA with global purification; bounded temporaries.

    The historical naive comparator clamps mean+std at observed class maximum.
    This is retained and counted explicitly, including saturated confidence ties.
    """
    def stacks(b):return core._stack_banks((b,))
    packets=[]
    for b in banks:
        x,p,y=stacks(b);packets.append(core._moment_packet(x,p,y,len(text)))
    w,bias,_=core._gda_from_moment_packets(packets,text);del packets
    ps=[];maxima=torch.full((len(text),),-float('inf'),dtype=torch.float64)
    for bank in banks:
        x,_,_=stacks(bank);logits=x@w+bias;p=F.softmax(logits*temp,dim=-1);y=logits.argmax(-1);ps.append(core._purification_packet(p,y,len(text)))
        for c in y.unique().tolist():maxima[c]=max(maxima[c],p[y==c,c].max())
    threshold=core._purification_threshold_from_packets(ps,len(text));clamps=int((threshold>maxima).sum());threshold=torch.minimum(threshold,maxima)
    packets=[];kept=0
    for bank in banks:
        x,_,_=stacks(bank);logits=x@w+bias;p=F.softmax(logits*temp,dim=-1);y=logits.argmax(-1);keep=p[torch.arange(len(p)),y]>=threshold[y]
        kept+=int(keep.sum());packets.append(core._moment_packet(x[keep],p[keep],y[keep],len(text)))
    w,b,_=core._gda_from_moment_packets(packets,text)
    return Adapter(w.float(),b.float(),dict(threshold_clamps=clamps,selected_patches=kept,method='streamed local union; global purification with disclosed clamp'))

def qa_condition(d,m=10,seed=7,beta=None):
    tag=f'm{m}_seed{seed}'+('' if beta is None else f'_beta{beta}');out=OUT/'qa'/d/tag
    if (out/'result.json').exists():return
    out.mkdir(parents=True,exist_ok=True);a=assignment(d,m,seed,beta);text=torch.from_numpy(np.load(resolve_cache(d)/'text_features.npy')).float();temp=BENCHMARKS[d].purification_temperature
    np.save(out/'assignments.npy',a)
    banks,banktime=timed(lambda:disk_banks(d,a,text,m,out/'client_banks'))
    p=Profile()
    with p.track():qa,qt=timed(lambda:core.fit_quantile_aligned_fedpasa(banks,text,k=512,purification_temperature=temp))
    naive,nt=timed(lambda:union_stream(banks,text,temp))
    adapters={'QAFedPASA':qa,'Local Top-K + Union + GDA':naive}
    if m==10 and seed==7 and beta is None:
        local,lt=timed(lambda:[fit_tie_safe_piaa(b,text,purification_temperature=temp) for b in banks])
        adapters['Naive local GDA averaging']=average_adapters(local,[v.metadata['selected_patches'] for v in local],'sample-weighted local GDA')
        del local
    raw=sum(len(v.features) for b in banks for v in b.values());bank_max=max(sum(v.features.numel()*4+v.probabilities.numel()*4+v.entropy.numel()*4 for v in b.values()) for b in banks)
    del banks;gc.collect()
    # COCO/NUS server patch caches were independently extracted. Rebuild the
    # central reference on THESE SAME patches, never reuse a Windows old bank.
    reference=OUT/'reference_banks'/d/'global_k512.pt';reference.parent.mkdir(parents=True,exist_ok=True)
    if reference.exists():g=torch.load(reference,weights_only=False)
    else:
        g=core.build_client_banks(resolve_cache(d)/'train_patch.npy',np.zeros(len(a),dtype=np.int64),text,clients=1,k=512,batch_images=64,device='cuda')[0]
        tmp=reference.with_suffix('.partial');torch.save(g,tmp);tmp.replace(reference)
    center,ct=timed(lambda:core.fit_central_piaa(g,text,purification_temperature=temp));adapters['Centralized PIAA-style GDA']=center
    scores,ev=timed(lambda:evaluate_many(d,adapters,out/'scores',batch=128))
    rows=[]
    for name,s in scores.items():
        rows.append(measured(d,name,s,clients=m,partition_seed=seed,beta=beta,raw_local_union_patches=raw,local_bank_payload_max_bytes=bank_max,
            bank_build_seconds=banktime,fit_seconds={'QAFedPASA':qt,'Local Top-K + Union + GDA':nt,'Centralized PIAA-style GDA':ct}.get(name,lt if 'lt' in locals() else 0),
            score_max_abs_vs_central=float(np.max(np.abs(s-scores['Centralized PIAA-style GDA'])))))
    # Refit additive ridge at every partition; never substitute central scores.
    x=load_features(resolve_cache(d),'train');z=load_features(resolve_cache(d),'test');y=targets(d,'train');packets=tuple(make_ridge_packet(x[a==i],y[a==i]) for i in range(m))
    ridges={};full=[]
    for s in SEEDS:
        alpha=read(sr(d,s)/'ridge/result.json')['selection']['alpha']
        if alpha not in ridges:ridges[alpha]=predict(z,fit_ridge_from_packets(packets,regularization=alpha))
        fr=sr(d,s)/'ridge_patch_fusion';qc=np.load(fr/'piaa_calibration.npy');rc=np.load(fr/'ridge_calibration.npy');w=np.array(read(fr/'result.json')['classwise_weight']['weights'])
        full.append((w*(ridges[alpha]-rc[0])/rc[1]+(1-w)*(scores['QAFedPASA']-qc[0])/qc[1]).astype('float32'))
    for name,s in [('Three-view Ridge',np.mean([ridges[read(sr(d,se)/'ridge/result.json')['selection']['alpha']] for se in SEEDS],axis=0)),('FedPASA-Full (locked 3-split rule)',np.mean(full,axis=0)),('FedPASA-Full (locked seed17)',full[1])]:
        np.save(out/'scores'/(name+'.npy'),s);rows.append(measured(d,name,s,clients=m,partition_seed=seed,beta=beta,calibration='historical training-only affine rules fixed across all partitions'))
    save(out/'result.json',dict(status='COMPLETE',rows=rows,qa_profile=dict(client_packet_seconds=p.client,server_solve_selection_seconds=p.server,qa_total_fit_seconds=qt,bank_build_seconds=banktime,messages=p.rows),
        total_shared_inference_seconds=ev,naive_metadata=naive.metadata,scope='real full-population local Top-K rebuild, not repartitioning preselected global bank'))

def sensitivity(d='coco'):
    out=OUT/'sensitivity'/d;out.mkdir(parents=True,exist_ok=True);x=load_features(resolve_cache(d),'train');z=load_features(resolve_cache(d),'test');y=targets(d,'train')
    fit=np.load(sr(d)/'ridge/fit_indices.npy');val=np.load(sr(d)/'ridge/validation_indices.npy');fp=make_ridge_packet(x[fit],y[fit]);ap=make_ridge_packet(x,y)
    qv=np.load(sr(d)/'ridge_patch_fusion/training_validation_piaa_scores.npy');qt=np.load(dr(d)/'scores/central_piaa_scores.npy');qvz,qtz,_=standardize(qv,qt);rows=[]
    for lam in (1e-4,1e-3,1e-2,1e-1):
        rv=predict(x[val],fit_ridge_from_packets((fp,),regularization=lam));rt=predict(z,fit_ridge_from_packets((ap,),regularization=lam));rvz,rtz,_=standardize(rv,rt)
        weights=np.linspace(0,1,21,dtype='float32');aps=np.array([metric(y[val],w*rvz+(1-w)*qvz)[1] for w in weights]);w=weights[aps.argmax(0)]
        rows.append(measured(d,'lambda sweep: reselect fusion on validation',w*rtz+(1-w)*qtz,parameter='lambda',value=lam,ridge_mAP=metric(targets(d),rt)[0]))
    # eta is defined here explicitly; the codebase names this ridge_weight.
    rt=np.load(sr(d)/'ridge/test_scores.npy');rc=np.load(sr(d)/'ridge_patch_fusion/ridge_calibration.npy');qc=np.load(sr(d)/'ridge_patch_fusion/piaa_calibration.npy')
    for eta in (0,.1,.3,.5,.7,1):
        rows.append(measured(d,'eta sweep: fixed global ridge mixing weight',eta*(rt-rc[0])/rc[1]+(1-eta)*(qt-qc[0])/qc[1],parameter='eta',value=eta))
    save(out/'result.json',dict(status='COMPLETE',rows=rows,definition='lambda is dimensionless trace-scaled ridge regularization; eta=ridge_weight, one global calibrated fusion weight, not per-class learned weights',selection='seed17 only; all preset grid test results retained without changing headline'))

def full_runtime(d):
    """Rerun exact train-side selection including its explicit pooled-data cost."""
    sync();start=time.perf_counter();stages={};messages=[];client_fit=0.;server_solve=0.
    out=OUT/'full_runtime'/d;out.mkdir(parents=True,exist_ok=True);a=assignment(d);y=targets(d,'train');x=load_features(resolve_cache(d),'train');z=load_features(resolve_cache(d),'test');text=torch.from_numpy(np.load(resolve_cache(d)/'text_features.npy')).float();b=BENCHMARKS[d]
    stages['input_loading']=time.perf_counter()-start
    allpack,stages['ridge_full_client_statistics']=timed(lambda:tuple(make_ridge_packet(x[a==i],y[a==i]) for i in range(10)))
    messages.append(dict(stage='ridge_full_statistics',bytes=sum(packet_wire(p) for p in allpack),packed_bytes=sum(packet_wire(p,True) for p in allpack)))
    p=Profile();banks,stages['qa_full_bank_build']=timed(lambda:disk_banks(d,a,text,10,out/'full_banks'))
    with p.track():qa,stages['qa_full_fit']=timed(lambda:core.fit_quantile_aligned_fedpasa(banks,text,k=512,purification_temperature=b.purification_temperature))
    client_fit+=stages['ridge_full_client_statistics']+stages['qa_full_bank_build']+stages['qa_full_fit']-p.server;server_solve+=p.server
    del banks;gc.collect();messages+=p.rows
    qt,stages['qa_test_inference']=timed(lambda:piaa_scores(resolve_cache(d)/'test_patch.npy',resolve_cache(d)/'test_cls.npy',np.arange(len(targets(d))),text,qa.weight,qa.bias,cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device='cuda'))
    np.save(out/'qa_scores.npy',qt)
    fs=[];seedrows=[];ridge_cache={}
    for seed in SEEDS:
        seedstart=time.perf_counter();fit=np.load(sr(d,seed)/'ridge/fit_indices.npy');val=np.load(sr(d,seed)/'ridge/validation_indices.npy')
        pp,t=timed(lambda:tuple(make_ridge_packet(x[fit[a[fit]==i]],y[fit[a[fit]==i]]) for i in range(10)));stages[f'seed{seed}_ridge_fit_statistics']=t
        client_fit+=t
        messages.append(dict(stage=f'seed{seed}_ridge_fit_statistics',bytes=sum(packet_wire(v) for v in pp),packed_bytes=sum(packet_wire(v,True) for v in pp)))
        tuning=[];rv=[]
        tt=time.perf_counter()
        for lam in ALPHAS:
            model,solve_time=timed(lambda:fit_ridge_from_packets(pp,regularization=lam));server_solve+=solve_time
            v=predict(x[val],model);rv.append(v);tuning.append(metric(y[val],v)[0])
        stages[f'seed{seed}_ridge_candidate_solve_predict_select']=time.perf_counter()-tt
        ix=int(np.argmax(tuning));lam=ALPHAS[ix]
        if lam not in ridge_cache:
            model,solve_time=timed(lambda:fit_ridge_from_packets(allpack,regularization=lam));server_solve+=solve_time;ridge_cache[lam]=predict(z,model)
        af=a.copy();af[val]=-1
        bank,t=timed(lambda:disk_banks(d,af,text,10,out/f'seed{seed}_banks'));stages[f'seed{seed}_qa_bank_build']=t
        client_fit+=t
        prof=Profile()
        with prof.track():q,t=timed(lambda:core.fit_quantile_aligned_fedpasa(bank,text,k=512,purification_temperature=b.purification_temperature))
        client_fit+=t-prof.server;server_solve+=prof.server
        stages[f'seed{seed}_qa_fit']=t;messages+=[dict(v,stage=f'seed{seed}_'+v['stage']) for v in prof.rows]
        del bank;gc.collect()
        qv,t=timed(lambda:piaa_scores(resolve_cache(d)/'train_patch.npy',resolve_cache(d)/'train_cls.npy',val,text,q.weight,q.bias,cls_scale=b.cls_scale,patch_scale=b.patch_scale,batch_images=128,device='cuda'));stages[f'seed{seed}_qa_validation_inference']=t
        tt=time.perf_counter();qvz,qtz,_=standardize(qv,qt);rvz,rtz,_=standardize(rv[ix],ridge_cache[lam]);weights=np.linspace(0,1,21,dtype='float32');aps=np.array([metric(y[val],w*rvz+(1-w)*qvz)[1] for w in weights]);cw=weights[aps.argmax(0)];score=cw*rtz+(1-cw)*qtz;fs.append(score)
        gw=weights[aps.mean(1).argmax()]
        np.save(out/f'seed{seed}_ridge_scores.npy',ridge_cache[lam])
        np.save(out/f'seed{seed}_global_scores.npy',gw*rtz+(1-gw)*qtz)
        np.save(out/f'seed{seed}_fixed_scores.npy',.5*rtz+.5*qtz)
        save(out/f'seed{seed}_selection.json',dict(split_seed=seed,alpha=lam,global_weight=float(gw),classwise_weights=cw.tolist(),validation_maps=aps.mean(1).tolist(),test_labels_used=False))
        stages[f'seed{seed}_exact_calibration_fusion_selection']=time.perf_counter()-tt
        # Coordinator receives candidate scores once and labels once, then can
        # evaluate the 21 mixtures locally. Explicit privacy boundary.
        byte_count=int(len(val)*y.shape[1]*((len(ALPHAS)+1)*4+1))
        messages.append(dict(stage=f'seed{seed}_pooled_validation_scores_labels',bytes=byte_count,packed_bytes=byte_count))
        np.save(out/f'seed{seed}_test_scores.npy',score);seedrows.append(measured(d,f'Full exact runtime rerun seed{seed}',score,seed=seed,alpha=lam,validation_tuning=tuning,seconds=time.perf_counter()-seedstart))
    score=np.mean(fs,axis=0);np.save(out/'test_scores.npy',score);sync()
    save(out/'result.json',dict(status='COMPLETE',rows=seedrows+[measured(d,'FedPASA-Full exact runtime rerun (3 splits)',score)],wall_seconds=time.perf_counter()-start,stages_seconds=stages,client_fitting_seconds=client_fit,server_solve_seconds=server_solve,
        uplink_bytes=sum(v['bytes'] for v in messages),packed_uplink_bytes=sum(v['packed_bytes'] for v in messages),messages=messages,logical_upload_stages=5+3*6,
        protocol='10 clients; three fit/validation splits; reuse full-fit QA and full ridge statistics. Four QA uploads, one ridge-stat upload, one pooled validation score/label upload per split.',
        scope='Actual FP64 moments plus FP32 entropy and validation scores, uint8 validation labels; no downlink/headers; exact AP pooling discloses training-side validation labels and predictions. Single-server cached-feature simulation, not private distributed selection.',
        peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2))

def memory(d,method):
    """Fresh subprocess measures the largest-image-count logical client alone."""
    out=OUT/'memory'/d/method;a=assignment(d);client=int(np.bincount(a).argmax());ix=np.flatnonzero(a==client);torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
    if method in ('qa','full'):
        text=torch.from_numpy(np.load(resolve_cache(d)/'text_features.npy')).float();one=np.where(a==client,0,-1)
        banks=core.build_client_banks(resolve_cache(d)/'train_patch.npy',one,text,clients=1,k=512,batch_images=64)
        qa=core.fit_quantile_aligned_fedpasa(banks,text,k=512,purification_temperature=BENCHMARKS[d].purification_temperature)
    if method in ('ridge','fed3r','fedavg','full'):
        names=('cls',) if method=='fed3r' else ('cls','patch_mean','patch_max_text')
        x=np.concatenate([np.array(np.load(resolve_cache(d)/f'train_{n}.npy',mmap_mode='r')[ix],dtype='float32') for n in names],axis=1);y=targets(d,'train')[ix]
        if method=='fedavg':fedavg_train(x,y,np.zeros(len(x),dtype=int),.1,out/'one_round.pt',rounds=1)
        else:packet=make_ridge_packet(x,y)
    sync();save(out/'result.json',dict(status='COMPLETE',dataset=d,method=method,client=client,images=len(ix),seconds=time.perf_counter()-t,peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,peak_cuda_MiB=torch.cuda.max_memory_allocated()/1024**2,
        scope='Isolated largest-image-count client process; includes Python/Torch baseline and touched mmap pages. QA one-client fit is a conservative local working-set proxy; not a measured maximum across every client or networked deployment.'))

def tests():
    rng=np.random.default_rng(17);x=rng.normal(size=(101,16)).astype('float32');y=(rng.random((101,3))>.6).astype('float32')
    pp=tuple(make_ridge_packet(x[i::5],y[i::5]) for i in range(5));a=fit_ridge_from_packets(pp,regularization=.01);b=fit_ridge_from_packets((make_ridge_packet(x,y),),regularization=.01)
    np.testing.assert_allclose(predict(x,a),predict(x,b),atol=1e-6)
    for p in pp:
        g=p.xtx.numpy();ix=np.triu_indices(len(g));wire=g[ix].tobytes();v=np.frombuffer(wire,dtype=g.dtype);r=np.zeros_like(g);r[ix]=v;r[(ix[1],ix[0])]=v;np.testing.assert_array_equal(g,r)
    text=F.normalize(torch.from_numpy(rng.normal(size=(3,16)).astype('float32')),dim=-1);banks=[]
    for i in range(3):
        bank={}
        for c in range(3):
            f=F.normalize(torch.from_numpy(rng.normal(size=(30,16)).astype('float32')),dim=-1);p=F.softmax(f@text.T,dim=-1);e=-(p*p.log()).sum(-1);bank[c]=PatchBank(f,p,e)
        banks.append(bank)
    merged={c:PatchBank(*(torch.cat([getattr(b[c],k) for b in banks]) for k in ('features','probabilities','entropy'))) for c in range(3)}
    a=union_stream(banks,text,1.);b=fit_tie_safe_piaa(merged,text,purification_temperature=1.)
    torch.testing.assert_close(a.weight,b.weight,atol=2e-5,rtol=2e-5);torch.testing.assert_close(a.bias,b.bias,atol=2e-5,rtol=2e-5)
    save(OUT/'tests.json',dict(status='COMPLETE',pooled_ridge=True,packing_roundtrip=True,streamed_union_matches_materialized=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['audit','linear','fedavg','qa','sensitivity','full','memory','tests']);p.add_argument('--dataset',choices=list(BENCHMARKS),default='coco');p.add_argument('--clients',type=int,default=10);p.add_argument('--seed',type=int,default=7);p.add_argument('--beta',type=float);p.add_argument('--method',default='ridge');args=p.parse_args()
    torch.set_num_threads(4);torch.manual_seed(17);np.random.seed(17)
    with threadpool_limits(limits=4):
        if args.phase=='tests':tests()
        elif args.phase=='audit':audit(args.dataset)
        elif args.phase=='linear':ridge_family(args.dataset)
        elif args.phase=='fedavg':fedavg(args.dataset)
        elif args.phase=='qa':qa_condition(args.dataset,args.clients,args.seed,args.beta)
        elif args.phase=='sensitivity':sensitivity(args.dataset)
        elif args.phase=='full':full_runtime(args.dataset)
        elif args.phase=='memory':memory(args.dataset,args.method)
    print('COMPLETE',vars(args),flush=True)
if __name__=='__main__':main()
