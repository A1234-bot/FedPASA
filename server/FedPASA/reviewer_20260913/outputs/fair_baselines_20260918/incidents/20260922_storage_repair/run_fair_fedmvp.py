"""Validation-only successive halving and full-data FedMVP adaptation repeats."""
import argparse, gc, math, time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,Subset
from PIL import Image
from fair_baselines_common import *
import run_icme2027_baseline as old
from icme2027_fast_fedmvp import enable, disable_token_proxy
from run_matched_fedmpt import gradient_flags,save_state
from torchvision.transforms import Compose,Resize,CenterCrop,ToTensor,Normalize,InterpolationMode
from fedpasa.benchmarks import BENCHMARKS
from utilss.asl import AsymmetricLoss_Pos

ORIGINAL=Path('/root/autodl-tmp/FedPASA/icme2027_workspace')
old.cache=lambda d:ORIGINAL/'outputs/icme/cache'/d
old.OUT=OUT/'fedmvp_source'
old.ROOT=ROOT
torch.set_num_threads(4)
METHOD='fedmvp'
METHOD_LABEL='FedMVP FedMPT-release multilabel adaptation'
LRS=(.0003,.001,.003)
def evaluate_round(rnd):return rnd%20==0

class Images(Dataset):
    def __init__(self,d,part):
        self.b=BENCHMARKS[d];m=read(old.cache(d)/'manifest.json');self.root=Path(m[part+'_root'])
        self.ids=(cache(d)/(part+'_ids.txt')).read_text().splitlines()
        self.targets=np.load(cache(d)/(part+'_targets.npy')) if part=='train' else None
        self.transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),Normalize((.48145466,.4578275,.40821073),(.26862954,.26130258,.27577711))])
        assert self.root.exists()
    def __len__(self):return len(self.ids)
    def __getitem__(self,i):
        with Image.open(self.root/self.b.image_name(self.ids[i])) as im:x=self.transform(im.convert('RGB'))
        return x,torch.from_numpy(self.targets[i].astype('float32')) if self.targets is not None else i

def source_audit():
    commit='9bb5d9f688ff37b030c76b4f5c3b168fbc08999f'
    upstream=OUT/'sources'/('FedMPT-main-'+commit)
    files=[]
    for n in ['model/fedmvp.py','convclip/fdmvpmodel.py','utilss/asl.py']:
        p=ROOT/'third_party/FedMPT'/n;q=upstream/n
        equal=p.read_bytes().replace(b'\r\n',b'\n')==q.read_bytes().replace(b'\r\n',b'\n')
        files.append({'path':n,'local_sha256':digest(p),'pinned_sha256':digest(q),'equal':equal})
    r={'status':'PASS' if all(f['equal'] for f in files) else 'FAIL','commit':commit,'files':files,'original_author_equivalent':False,'identity':METHOD_LABEL}
    save(old.OUT/'baseline_audit/result.json',r);assert r['status']=='PASS';return r

def batch_loss(net,image,y,criterion):
    z,kg=net(image,req=True)
    return .02*(criterion(z.float(),y)+kg.float())

def make_optimizer(net,lr,rnd):
    return torch.optim.SGD(net.prompt_learner.parameters(),lr=lr*.5*(1+math.cos(math.pi*(rnd-1)/200)),momentum=.9,weight_decay=1e-5)

def new_model(d,seed,fast=True):
    torch.manual_seed(seed);np.random.seed(seed)
    net=old.model(d)
    if fast:enable(net)
    return net

@torch.inference_mode()
def evaluate(net,data):
    net.eval();scores=[]
    for x,_ in DataLoader(data,batch_size=32,shuffle=False,num_workers=4,pin_memory=True):
        with torch.autocast('cuda',dtype=torch.bfloat16):z=net(x.cuda(non_blocking=True)).float()
        if not torch.isfinite(z).all():raise FloatingPointError('nonfinite evaluation')
        scores.append(z.cpu().numpy())
    return np.concatenate(scores)

def smoke(d):
    dest=OUT/'fedmvp'/d/'smoke.json'
    if dest.exists():assert read(dest)['passed'];return
    source_audit();data=Images(d,'train');f,_=split(d)
    batch=next(iter(DataLoader(Subset(data,f[:32].tolist()),batch_size=8)))
    x,y=[v.cuda() for v in batch];net=new_model(d,17,False);criterion=AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1)
    def probe():
        net.train();net.zero_grad(set_to_none=True);torch.manual_seed(31415)
        with torch.autocast('cuda',dtype=torch.bfloat16):z,kg=net(x,req=True);loss=.02*(criterion(z.float(),y)+kg.float())
        loss.backward();g={n:p.grad.detach().clone() for n,p in net.prompt_learner.named_parameters() if p.grad is not None}
        return z.detach(),loss.detach(),g
    z,l,g=probe();enable(net);zz,ll,gg=probe()
    ze=float((z-zz).abs().max());le=float((l-ll).abs());ge=max(float((g[n]-gg[n]).abs().max()) for n in g)
    assert torch.equal(z,zz) and torch.equal(l,ll) and all(torch.equal(g[n],gg[n]) for n in g),'Static cache equivalence failed'
    opt=torch.optim.SGD(net.prompt_learner.parameters(),lr=.001,momentum=.9,weight_decay=1e-5);losses=[]
    for step in range(40):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):z,kg=net(x,req=True);loss=.02*(criterion(z.float(),y)+kg.float())
        loss.backward();finite,nonzero=gradient_flags(net.prompt_learner.parameters());assert finite and nonzero
        opt.step();losses.append(float(loss.detach()))
    r={'passed':bool(np.mean(losses[-5:])<np.mean(losses[:5])),'cache_score_error':ze,'cache_loss_error':le,'cache_gradient_error':ge,'training_losses':losses,'test_access':False,'images':8,'time':stamp()}
    save(dest,r);assert r['passed'],'Small training batch loss did not decrease'
    del net,g,gg,x,y;gc.collect();torch.cuda.empty_cache()

def train(d,seed,lr,target,phase):
    out=OUT/METHOD/d/phase/(f'lr{lr:g}' if phase=='tuning' else f'seed{seed}')
    out.mkdir(parents=True,exist_ok=True);statepath=out/'checkpoint.pt'
    if phase=='final' and (out/'prediction_manifest.json').exists():return read(out/'result.json')
    data=Images(d,'train');a=assignment(d);f,v=split(d);ids=f if phase=='tuning' else np.arange(len(data))
    valid=Subset(data,v.tolist())
    net=new_model(d,seed);criterion=AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1)
    initial={k:t.detach().cpu().clone() for k,t in net.prompt_learner.state_dict().items()}
    saved=torch.load(statepath,map_location='cpu',weights_only=False) if statepath.exists() else None
    curve=[];training_seconds=0.;validation_seconds=0.;best=-float('inf');best_round=None;done=0
    if saved:
        assert saved['lr']==lr and saved['seed']==seed and saved['phase']==phase
        initial=saved['global'];curve=saved['curve'];training_seconds=saved['training_seconds'];validation_seconds=saved['validation_seconds'];best=saved['best'];best_round=saved['best_round'];done=saved['completed_round']
    assert done<=target, 'Cannot replay an earlier halving stage from a later checkpoint'
    net.prompt_learner.load_state_dict(initial)
    aggregate=None
    parameter_bytes=sum(t.numel()*t.element_size() for t in initial.values());torch.cuda.reset_peak_memory_stats()
    for rnd in range(done+1,target+1):
        active=saved and saved.get('active_round')==rnd
        aggregate=saved['aggregate'] if active else {k:torch.zeros_like(t,dtype=torch.float64) for k,t in initial.items()}
        first=saved['next_client'] if active else 0;records=saved['client_records'] if active else []
        round_start=time.perf_counter()
        for client in range(first,10):
            torch.manual_seed(seed+rnd*1000+client);net.prompt_learner.load_state_dict(initial);net.train()
            ix=ids[a[ids]==client];assert len(ix)>0
            loader=DataLoader(Subset(data,ix.tolist()),batch_size=32,shuffle=True,num_workers=4,pin_memory=True,generator=torch.Generator().manual_seed(seed+rnd*1000+client))
            opt=make_optimizer(net,lr,rnd)
            t=time.perf_counter();loss_sum=0.;seen=0
            for step,(image,y) in enumerate(loader):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=batch_loss(net,image.cuda(non_blocking=True),y.cuda(non_blocking=True),criterion)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
                loss.backward()
                if step==0:
                    finite,nz=gradient_flags(net.prompt_learner.parameters());assert finite and nz
                opt.step();loss_sum+=float(loss.detach())*len(y);seen+=len(y)
                if step%200==0:print('FEDMVP',d,phase,seed,lr,rnd,client,step,float(loss.detach()),flush=True)
            assert seen==len(ix);torch.cuda.synchronize();seconds=time.perf_counter()-t;training_seconds+=seconds
            for k,tensor in net.prompt_learner.state_dict().items():
                if initial[k].is_floating_point():aggregate[k]+=tensor.detach().double().cpu()*(len(ix)/len(ids))
                else:aggregate[k]=torch.maximum(aggregate[k],tensor.detach().double().cpu())
            records.append({'client':client,'images':seen,'mean_batch_loss':loss_sum/seen,'seconds':seconds})
            saved={'global':initial,'completed_round':rnd-1,'active_round':rnd,'next_client':client+1,'aggregate':aggregate,'client_records':records,'lr':lr,'seed':seed,'phase':phase,'curve':curve,'training_seconds':training_seconds,'validation_seconds':validation_seconds,'best':best,'best_round':best_round}
            save_state(statepath,saved)
            save(out/'progress.json',{'status':'RUNNING','round':rnd,'completed_clients':client+1,'target_rounds':target,'time':stamp()})
        initial={k:t.to(initial[k].dtype) for k,t in aggregate.items()};net.prompt_learner.load_state_dict(initial)
        row={'round':rnd,'client_records':records,'training_seconds':training_seconds,'round_wall_seconds':time.perf_counter()-round_start}
        if phase=='tuning' and evaluate_round(rnd):
            t=time.perf_counter();scores=evaluate(net,valid);validation_seconds+=time.perf_counter()-t;row['validation_mAP']=metric(data.targets[v],scores)
            np.save(out/f'round{rnd:03d}_validation_scores.npy',scores)
            if row['validation_mAP']>best:best=row['validation_mAP'];best_round=rnd;save_state(out/'best.pt',initial)
        curve.append(row);save(out/f'round{rnd:03d}.json',row)
        saved={'global':initial,'completed_round':rnd,'lr':lr,'seed':seed,'phase':phase,'curve':curve,'training_seconds':training_seconds,'validation_seconds':validation_seconds,'best':best,'best_round':best_round}
        save_state(statepath,saved)
    r={'status':'COMPLETE','dataset':d,'phase':phase,'seed':seed,'lr':lr,'rounds':target,'best_round':best_round,'best_validation_mAP':best if math.isfinite(best) else None,'training_seconds':training_seconds,'validation_seconds':validation_seconds,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/2**20,'uplink_bytes':target*10*parameter_bytes,'downlink_bytes':target*10*parameter_bytes,'curve':[{k:v for k,v in q.items() if k!='client_records'} for q in curve],'author_equivalent':False,'method':METHOD_LABEL,'test_labels_read':False}
    save(out/'result.json',r)
    if phase=='tuning':save(out/f'result_budget{target}.json',r)
    if phase=='final' and not (out/'prediction_manifest.json').exists():
        t=time.perf_counter();scores=evaluate(net,Images(d,'test'));seconds=time.perf_counter()-t;np.save(out/'test_scores.npy',scores)
        save(out/'prediction_manifest.json',{'prediction_sha256':digest(out/'test_scores.npy'),'model_sha256':digest(statepath),'selection_sha256':digest(OUT/METHOD/d/'selection.json'),'inference_seconds':seconds,'shape':list(scores.shape),'test_labels_read':False,'saved_at':stamp()})
    del net,initial,aggregate,saved;gc.collect();torch.cuda.empty_cache();return r

def tune(d):
    freeze_inputs(d);smoke(d);dest=OUT/METHOD/d/'selection.json'
    if dest.exists():return
    alive=list(LRS);latest={};history=[]
    for budget,keep in [(20,2),(60,1),(200,1)]:
        stage=OUT/METHOD/d/f'halving_{budget}.json'
        if stage.exists():
            decision=read(stage)
            assert decision['budget']==budget
            assert [r['lr'] for r in decision['candidates']]==alive
            history.append(decision);alive=decision['retained']
            for candidate in decision['candidates']:
                lr=candidate['lr'];latest[lr]=read(OUT/METHOD/d/'tuning'/f'lr{lr:g}'/'result.json')
            continue
        for lr in alive:latest[lr]=train(d,17,lr,budget,'tuning')
        ranked=sorted(alive,key=lambda lr:(-latest[lr]['best_validation_mAP'],latest[lr]['best_round'],lr))
        history.append({'budget':budget,'candidates':[{'lr':lr,'validation_mAP':latest[lr]['best_validation_mAP'],'best_round':latest[lr]['best_round']} for lr in alive],'retained':ranked[:keep]})
        save(OUT/METHOD/d/f'halving_{budget}.json',history[-1]);alive=ranked[:keep]
    best=latest[alive[0]]
    save(dest,{'status':'LOCKED','lr':best['lr'],'rounds':best['best_round'],'validation_mAP':best['best_validation_mAP'],'history':history,'selection_training_seconds':sum(r['training_seconds'] for r in latest.values()),'selection_validation_seconds':sum(r['validation_seconds'] for r in latest.values()),'selection_uplink_bytes':sum(r['uplink_bytes'] for r in latest.values()),'selection_downlink_bytes':sum(r['downlink_bytes'] for r in latest.values()),'selected_at_budget_end':best['best_round']==200,'test_labels_read':False,'locked_at':stamp(),'search_scope':'predeclared successive halving, not exhaustive full-budget grid'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','tune','repeat'],required=True);p.add_argument('--dataset',choices=['coco','nus'],required=True);args=p.parse_args()
    if args.phase=='smoke':smoke(args.dataset)
    elif args.phase=='tune':tune(args.dataset)
    else:
        s=read(OUT/'fedmvp'/args.dataset/'selection.json')
        for seed in (7,17,27):train(args.dataset,seed,s['lr'],s['rounds'],'final')
