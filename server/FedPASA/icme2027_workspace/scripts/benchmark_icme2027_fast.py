"""Actual-data, identical-SGD speed control, outside formal checkpoints."""
import gc,json,os,signal,subprocess,sys
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from threadpoolctl import threadpool_limits
from icme2027_common import ROOT,OUT,save,stamp
import run_icme2027_baseline as base
from icme2027_fast_fedmvp import enable,disable_token_proxy,VERSION

def step(net,batch,opt):
    image,target=batch;opt.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits,kg=net(image.cuda(non_blocking=True),req=True)
        loss=.02*(base.AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1)(logits.float(),target.cuda(non_blocking=True))+kg.float())
    assert torch.isfinite(loss)
    loss.backward();finite,nz=base.gradient_flags(net.prompt_learner.parameters());assert finite and nz
    opt.step();float(loss.detach())
    return logits,loss

def suspend():
    q=json.load(open(OUT/'queue.json'));pids=[]
    rootpid=q['pid'];running=[j['pid'] for j in q['jobs'] if j['status']=='RUNNING' and 'pid' in j]
    for pid in [rootpid,*running]:
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
        assert (b'run_icme2027_queue.py' in cmd or b'run_icme2027_baseline.py' in cmd),cmd
        os.kill(pid,signal.SIGSTOP);pids.append(pid)
    # Parent stopped first, so it cannot launch new workers during enumeration.
    table=subprocess.check_output(['ps','-eo','pid=,ppid='],text=True)
    pairs=[tuple(map(int,l.split())) for l in table.splitlines() if l.strip()]
    changed=True
    while changed:
        changed=False
        for pid,ppid in pairs:
            if ppid in pids and pid not in pids:
                try:os.kill(pid,signal.SIGSTOP);pids.append(pid);changed=True
                except ProcessLookupError:pass
    save(OUT/'optimization/suspension.json',{'pids':pids,'started':stamp(),'reason':'isolated numerical/speed benchmark; automatically resumes in finally'})
    return pids

def run():
    destination=OUT/'optimization';destination.mkdir(parents=True,exist_ok=True);rows=[]
    for dataset in ('coco','nus'):
        data=base.Images(dataset,'train');part=np.load(ROOT/'outputs/icme'/dataset/'partitions/clients10_seed7.npy')
        ids=np.flatnonzero(part==3).tolist()
        checkpoint=OUT/'baselines/fedmvp_release_mlc'/dataset/'seed7/checkpoint.pt'
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['global'] if checkpoint.exists() else None
        for repeat,variants in enumerate([['original','cached'],['cached','original']]):
            ref=None;refname=None
            for variant in variants:
                disable_token_proxy();torch.manual_seed(17391);net=base.model(dataset)
                if state is not None:net.prompt_learner.load_state_dict(state)
                if variant=='cached':enable(net)
                net.train();opt=torch.optim.SGD(net.prompt_learner.parameters(),lr=.001,momentum=.9,weight_decay=1e-5)
                loader=DataLoader(Subset(data,ids),batch_size=32,num_workers=4,shuffle=True,pin_memory=True,
                    generator=torch.Generator().manual_seed(24791))
                torch.manual_seed(24791);it=iter(loader);snapshots=[]
                for i in range(3):
                    logits,loss=step(net,next(it),opt)
                    snapshots.append({'logits':logits.detach().cpu(),'loss':loss.detach().cpu(),
                        'grads':{n:p.grad.detach().cpu().clone() for n,p in net.prompt_learner.named_parameters() if p.grad is not None}})
                torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=perf_counter();wait=0.
                for i in range(100):
                    t=perf_counter();batch=next(it);wait+=perf_counter()-t
                    logits,loss=step(net,batch,opt)
                torch.cuda.synchronize();seconds=perf_counter()-start
                final={n:p.detach().cpu().clone() for n,p in net.prompt_learner.named_parameters()}
                rng=torch.cuda.get_rng_state().cpu()
                current=(snapshots,final,rng)
                exact=None;error=None
                if ref is None:ref=current;refname=variant
                else:
                    pairs=[]
                    for a,b in zip(ref[0],snapshots):
                        pairs.extend([(a['logits'],b['logits']),(a['loss'],b['loss'])])
                        pairs.extend((a['grads'][k],v) for k,v in b['grads'].items())
                    pairs.extend((ref[1][k],v) for k,v in final.items());pairs.append((ref[2],rng))
                    exact=all(torch.equal(a,b) for a,b in pairs)
                    error=max(float((a.float()-b.float()).abs().max()) for a,b in pairs)
                row={'dataset':dataset,'repeat':repeat,'variant':variant,'comparison_against':refname,'bitwise_equal':exact,'max_abs_error':error,
                    'optimizer_steps_compared':103,'timed_steps':100,'seconds_per_step':seconds/100,'loader_wait_seconds':wait,
                    'peak_cuda_MiB':torch.cuda.max_memory_allocated()/1024**2,'cpu_threads':1,'workers':4}
                rows.append(row);save(destination/'benchmark.json',{'version':VERSION,'status':'RUNNING','rows':rows});print(json.dumps(row),flush=True)
                del logits,loss,opt,it,loader,net,current;gc.collect();torch.cuda.empty_cache()
    comparisons=[r for r in rows if r['bitwise_equal'] is not None]
    passed=all(r['bitwise_equal'] for r in comparisons)
    # Conservatively use the smaller paired speedup: the first image pass can
    # include cold filesystem reads that would inflate a pooled speedup.
    speed={d:min(next(r['seconds_per_step'] for r in rows if r['dataset']==d and r['repeat']==i and r['variant']=='original')/
                 next(r['seconds_per_step'] for r in rows if r['dataset']==d and r['repeat']==i and r['variant']=='cached')
                 for i in [0,1]) for d in ['coco','nus']}
    save(destination/'benchmark.json',{'version':VERSION,'status':'PASS' if passed else 'FAIL','rows':rows,'speedup':speed,'completed':stamp(),
        'comparison':'same real image order, batch32, dropout RNG, optimizer and initial parameters; 103-step final state; both orderings'})
    if not passed:raise RuntimeError('Numerical equivalence failed; do not deploy')

if __name__=='__main__':
    torch.set_num_threads(1);pids=[]
    try:
        pids=suspend()
        with threadpool_limits(limits=1):run()
    finally:
        for pid in reversed(pids):
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        if pids:save(OUT/'optimization/resumed.json',{'pids':pids,'resumed':stamp()})
