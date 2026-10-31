"""Isolated speed and numerical checks; never writes formal training state."""
import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_matched_fedmpt as m
import torch
from threadpoolctl import threadpool_limits

DEST=m.ROOT/'tmp/server_performance'


def step(model,batch,optimizer,fused):
    image,target=batch
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        logits=model(image.cuda(non_blocking=True))
        loss=m.AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1)(logits.float(),target.cuda(non_blocking=True))
    if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
    loss.backward()
    if fused:
        finite,nonzero=m.gradient_flags(model.prompt_learner.parameters())
    else:
        finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.prompt_learner.parameters())
        nonzero=any(p.grad is not None and bool(p.grad.abs().max()>0) for p in model.prompt_learner.parameters())
    assert finite and nonzero
    optimizer.step()
    return logits,loss


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',default='coco');parser.add_argument('--steps',type=int,default=12)
    args=parser.parse_args();DEST.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data=m.Images(args.dataset,'train')
    batches=list(m.DataLoader(m.Subset(data,list(range(64))),batch_size=32,num_workers=2,pin_memory=True))
    state=torch.load(m.OUT/args.dataset/'fedmpt_repaired/checkpoint.pt',map_location='cpu',weights_only=False)['global']
    variants=[('baseline',True,False,False,4),('no_recompute',False,False,False,4),
              ('static_text',False,True,False,4),('fused_checks',False,True,True,4),
              ('one_cpu_thread',False,True,True,1),('two_cpu_threads',False,True,True,2)]
    rows=[];reference=None
    for name,ac,cache,fused,threads in variants:
        torch.set_num_threads(threads)
        with threadpool_limits(limits=threads):
            torch.manual_seed(1234)
            model=m.make_model(args.dataset,ac,cache);model.prompt_learner.load_state_dict(state);model.train()
            optimizer=torch.optim.SGD(model.prompt_learner.parameters(),lr=.001,momentum=.9,weight_decay=1e-5)
            torch.cuda.reset_peak_memory_stats()
            # Same batches, initial parameters and dropout seeds. Compare actual
            # outputs, gradients and SGD updates for three consecutive steps.
            snapshots=[]
            for i in range(3):
                torch.manual_seed(2000+i)
                logits,loss=step(model,batches[i%2],optimizer,fused)
                snapshots.append(dict(logits=logits.detach().cpu(),loss=loss.detach().cpu(),
                    gradients={k:p.grad.detach().cpu().clone() for k,p in model.prompt_learner.named_parameters() if p.grad is not None},
                    parameters={k:p.detach().cpu().clone() for k,p in model.prompt_learner.named_parameters()}))
            if reference is None:reference=snapshots
            exact=True;max_abs=0.;max_scaled=0.
            for ref,got in zip(reference,snapshots):
                pairs=[(ref['logits'],got['logits']),(ref['loss'],got['loss'])]
                pairs.extend((ref[category][k],v) for category in ('gradients','parameters') for k,v in got[category].items())
                for a,b in pairs:
                    exact &= torch.equal(a,b)
                    error=float((a.float()-b.float()).abs().max());max_abs=max(max_abs,error)
                    max_scaled=max(max_scaled,error/max(float(a.float().abs().max()),1e-8))
            del snapshots,logits,loss
            torch.cuda.synchronize();start=perf_counter()
            for i in range(args.steps):
                logits,loss=step(model,batches[i%2],optimizer,fused)
                float(loss.detach())
            torch.cuda.synchronize();elapsed=perf_counter()-start
            row=dict(variant=name,dataset=args.dataset,cpu_threads=threads,checkpointing=ac,cache_static_text=cache,fused_checks=fused,
                steps=args.steps,seconds_per_step=elapsed/args.steps,peak_allocated_MiB=torch.cuda.max_memory_allocated()/1024**2,
                three_step_bitwise_equal=exact,max_absolute_error=max_abs,max_scaled_error=max_scaled)
            rows.append(row);print(json.dumps(row),flush=True)
            del model,optimizer,logits,loss;gc.collect();torch.cuda.empty_cache()
            (DEST/f'{args.dataset}_benchmark.json').write_text(json.dumps(rows,indent=2))
    print('BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':main()
