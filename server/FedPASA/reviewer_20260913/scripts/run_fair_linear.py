"""Train-side optimization diagnostics and locked full-data repeats."""
import argparse, copy, gc, itertools, math, time
import numpy as np
import torch
import torch.nn.functional as F
from fair_baselines_common import *

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False

def sync(): torch.cuda.synchronize()
def atomic_torch(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix('.partial.pt');torch.save(obj,t);t.replace(p)
def params_cpu(head):return {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
def candidates():
    return [dict(optimizer=o,lr=l,weight_decay=w) for o,ls,ws in [('sgd',[.1,1.,10.,100.],[0.,1e-4,1e-3]),('adamw',[.001,.01,.1],[0.,.01])] for l in ls for w in ws]
def name(c):return f"{c['optimizer']}_lr{c['lr']:g}_wd{c['weight_decay']:g}"
def optimizer(h,c):
    if c['optimizer']=='sgd':return torch.optim.SGD(h.parameters(),lr=c['lr'],momentum=.9,weight_decay=c['weight_decay'])
    return torch.optim.AdamW(h.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])

def fit(x,y,a,c,mode,seed,rounds,out,vx=None,vy=None):
    out.mkdir(parents=True,exist_ok=True);statepath=out/'checkpoint.pt'
    tx=torch.as_tensor(x,device='cuda');ty=torch.as_tensor(y,dtype=torch.float32,device='cuda')
    vv=torch.as_tensor(vx,device='cuda') if vx is not None else None
    h=torch.nn.Linear(x.shape[1],y.shape[1],device='cuda');torch.nn.init.zeros_(h.weight);torch.nn.init.zeros_(h.bias)
    opt=optimizer(h,c);curve=[];start_round=0;train_seconds=0.;eval_seconds=0.;best=-float('inf');best_round=None
    groups=[torch.as_tensor(np.flatnonzero(a==i),device='cuda') for i in range(10)] if mode=='fedavg' else [torch.arange(len(y),device='cuda')]
    if statepath.exists():
        ck=torch.load(statepath,map_location='cpu',weights_only=False)
        assert ck['config']==c and ck['seed']==seed and ck['mode']==mode
        h.load_state_dict(ck['state']);curve=ck['curve'];start_round=ck['round'];train_seconds=ck['train_seconds'];eval_seconds=ck['eval_seconds'];best=ck['best'];best_round=ck['best_round']
        if mode=='central': opt.load_state_dict(ck['optimizer'])
    torch.cuda.reset_peak_memory_stats();status='COMPLETE'
    try:
        for r in range(start_round,rounds):
            sync();t=time.perf_counter()
            global_state={k:v.detach().clone() for k,v in h.state_dict().items()}
            aggregate={k:torch.zeros_like(v,dtype=torch.float64) for k,v in global_state.items()} if mode=='fedavg' else None
            for i,ids in enumerate(groups):
                if not len(ids):continue
                if mode=='fedavg':h.load_state_dict(global_state);opt=optimizer(h,c)
                gen=torch.Generator(device='cuda').manual_seed(seed*1000000+r*100+i)
                order=ids[torch.randperm(len(ids),generator=gen,device='cuda')]
                for ix in order.split(256):
                    opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(h(tx[ix]),ty[ix]);loss.backward();opt.step()
                if mode=='fedavg':
                    with torch.no_grad():
                        for k,p in h.state_dict().items():aggregate[k].add_(p.double(),alpha=len(ids)/len(y))
            if mode=='fedavg':h.load_state_dict(aggregate)
            sync();train_seconds+=time.perf_counter()-t
            if (r+1)%20==0 or r+1==rounds:
                sync();t=time.perf_counter()
                with torch.no_grad():
                    logits=h(tx);train_loss=float(F.binary_cross_entropy_with_logits(logits,ty));scores=h(vv).cpu().numpy() if vv is not None else None
                if not math.isfinite(train_loss) or not all(torch.isfinite(p).all() for p in h.parameters()):raise FloatingPointError('non-finite model/loss')
                row={'round':r+1,'train_BCE':train_loss,'train_seconds':train_seconds}
                if scores is not None:
                    row['validation_mAP']=metric(vy,scores)
                    if row['validation_mAP']>best:
                        best=row['validation_mAP'];best_round=r+1;atomic_torch(out/'best.pt',params_cpu(h));np.save(out/'best_validation_scores.npy',scores)
                sync();eval_seconds+=time.perf_counter()-t;curve.append(row)
                save(out/'curve.json',curve)
                atomic_torch(statepath,dict(config=c,mode=mode,seed=seed,round=r+1,state=params_cpu(h),optimizer=opt.state_dict() if mode=='central' else None,curve=curve,train_seconds=train_seconds,eval_seconds=eval_seconds,best=best,best_round=best_round))
                print('LINEAR',out.name,r+1,row,flush=True)
    except FloatingPointError as e:
        status='NUMERICAL_FAILURE';save(out/'failure.json',{'error':str(e),'round':r+1})
    nparams=sum(p.numel() for p in h.parameters());done=curve[-1]['round'] if curve else 0
    result={'status':status,'mode':mode,'config':c,'seed':seed,'rounds':done,'best_round':best_round,'best_validation_mAP':best if math.isfinite(best) else None,'train_seconds':train_seconds,'validation_seconds':eval_seconds,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/2**20,'uplink_bytes':done*10*nparams*4 if mode=='fedavg' else 0,'downlink_bytes':done*10*nparams*4 if mode=='fedavg' else 0,'curve':curve}
    save(out/'result.json',result);state=params_cpu(h)
    del h,opt,tx,ty,vv;gc.collect();torch.cuda.empty_cache()
    return state,result

def sanity(d,x,y,f):
    out=OUT/'linear'/d/'sanity';out.mkdir(parents=True,exist_ok=True)
    ids=f[:256];tx=torch.as_tensor(x[ids],device='cuda');ty=torch.as_tensor(y[ids],device='cuda')
    h=torch.nn.Linear(x.shape[1],y.shape[1],device='cuda');torch.nn.init.zeros_(h.weight);torch.nn.init.zeros_(h.bias)
    opt=torch.optim.Adam(h.parameters(),lr=.05)
    first=float(F.binary_cross_entropy_with_logits(h(tx),ty).detach())
    for _ in range(1000):
        opt.zero_grad(set_to_none=True);loss=F.binary_cross_entropy_with_logits(h(tx),ty);loss.backward();opt.step()
    with torch.no_grad():s=h(tx).cpu().numpy();last=float(F.binary_cross_entropy_with_logits(h(tx),ty))
    r={'initial_BCE':first,'final_BCE':last,'training_subset_mAP':metric(y[ids],s),'training_samples':256,'test_access':False,'passed':last<first*.5}
    save(out/'result.json',r);assert r['passed'],r
    del tx,ty,h,opt;torch.cuda.empty_cache()

def lbfgs_reference(d,x,y,f,v):
    tx=torch.as_tensor(x[f],dtype=torch.float64,device='cuda');ty=torch.as_tensor(y[f],dtype=torch.float64,device='cuda')
    rows=[]
    for wd in (0.,1e-4,1e-3):
        out=OUT/'linear'/d/'lbfgs'/f'wd{wd:g}'
        if (out/'result.json').exists():rows.append(read(out/'result.json'));continue
        h=torch.nn.Linear(x.shape[1],y.shape[1],device='cuda',dtype=torch.float64);torch.nn.init.zeros_(h.weight);torch.nn.init.zeros_(h.bias)
        opt=torch.optim.LBFGS(h.parameters(),lr=1,max_iter=500,tolerance_grad=1e-7,tolerance_change=1e-12,history_size=30,line_search_fn='strong_wolfe')
        calls=[0];t=time.perf_counter()
        def closure():
            opt.zero_grad();loss=F.binary_cross_entropy_with_logits(h(tx),ty)+.5*wd*sum(p.square().sum() for p in h.parameters());loss.backward();calls[0]+=1;return loss
        opt.step(closure);closure();sync()
        with torch.no_grad():scores=h(torch.as_tensor(x[v],dtype=torch.float64,device='cuda')).float().cpu().numpy()
        row={'weight_decay':wd,'validation_mAP':metric(y[v],scores),'max_abs_gradient':max(float(p.grad.abs().max()) for p in h.parameters()),'iterations':int(opt.state[h.weight]['n_iter']),'function_calls':calls[0],'seconds':time.perf_counter()-t,'scope':'central full-batch convex diagnostic; both weight and bias L2; not a federated baseline'}
        save(out/'result.json',row);atomic_torch(out/'model.pt',params_cpu(h));rows.append(row)
        print('LBFGS',d,row,flush=True)
        del h,opt;gc.collect();torch.cuda.empty_cache()
    del tx,ty;torch.cuda.empty_cache();return rows

def tune(d):
    freeze_inputs(d);x=features(d,'train');y=np.load(cache(d)/'train_targets.npy').astype('float32');a=assignment(d);f,v=split(d)
    sanity(d,x,y,f);lbfgs_reference(d,x,y,f,v)
    for mode in ('central','fedavg'):
        root=OUT/'linear'/d/mode
        if (root/'selection.json').exists():continue
        rows=[]
        def one(c):
            p=root/'tuning'/name(c)
            if (p/'result.json').exists():row=read(p/'result.json')
            else:_,row=fit(x[f],y[f],a[f],c,mode,17,200,p,x[v],y[v])
            row['path']=str(p);rows.append(row)
        for c in candidates():one(c)
        for optimizer_name,high,extra in [('sgd',100.,1000.),('adamw',.1,1.)]:
            good=[q for q in rows if q['config']['optimizer']==optimizer_name and q['status']=='COMPLETE']
            best=max(good,key=lambda q:q['best_validation_mAP'])
            if best['config']['lr']==high:one({**best['config'],'lr':extra})
        good=[q for q in rows if q['status']=='COMPLETE']
        best=max(good,key=lambda q:(q['best_validation_mAP'],-q['best_round']))
        for budget in (400,800):
            curve=best['curve'];rising=len(curve)>=2 and best['best_round']==best['rounds'] and curve[-1]['validation_mAP']-curve[-2]['validation_mAP']>=.05
            if not rising:break
            _,new=fit(x[f],y[f],a[f],best['config'],mode,17,budget,Path(best['path']),x[v],y[v]);new['path']=best['path']
            rows=[new if q['path']==best['path'] else q for q in rows];best=max([q for q in rows if q['status']=='COMPLETE'],key=lambda q:(q['best_validation_mAP'],-q['best_round']))
        boundary=best['best_round']==best['rounds']
        choice={'status':'LOCKED','dataset':d,'mode':mode,'config':best['config'],'rounds':best['best_round'],'validation_mAP':best['best_validation_mAP'],'selected_at_last_evaluation':boundary,'optimization_claim':'budget-limited; inspect curve' if boundary else 'validation optimum attained before budget','selection_train_seconds':sum(q['train_seconds'] for q in rows),'selection_validation_seconds':sum(q['validation_seconds'] for q in rows),'selection_uplink_bytes':sum(q['uplink_bytes'] for q in rows),'selection_downlink_bytes':sum(q['downlink_bytes'] for q in rows),'candidates':rows,'locked_at':stamp(),'test_labels_read':False}
        save(root/'selection.json',choice);print('LOCKED',d,mode,choice['config'],choice['rounds'],flush=True)

def repeat(d):
    x=features(d,'train');y=np.load(cache(d)/'train_targets.npy').astype('float32');a=assignment(d)
    for mode in ('central','fedavg'):
        root=OUT/'linear'/d/mode;selection=read(root/'selection.json')
        for seed in (7,17,27):
            p=root/'final'/f'seed{seed}'
            if (p/'prediction_manifest.json').exists():continue
            state,result=fit(x,y,a,selection['config'],mode,seed,selection['rounds'],p)
            assert result['status']=='COMPLETE'
            z=features(d,'test');t=time.perf_counter();scores=z@state['weight'].numpy().T+state['bias'].numpy()
            np.save(p/'test_scores.npy',scores)
            save(p/'prediction_manifest.json',{'prediction_sha256':digest(p/'test_scores.npy'),'model_sha256':digest(p/'checkpoint.pt'),'inference_seconds':time.perf_counter()-t,'selection_sha256':digest(root/'selection.json'),'test_labels_read':False,'shape':list(scores.shape),'saved_at':stamp()})
            del z,scores

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['tune','repeat'],required=True);p.add_argument('--dataset',choices=['coco','nus'],required=True);args=p.parse_args()
    (tune if args.phase=='tune' else repeat)(args.dataset)
