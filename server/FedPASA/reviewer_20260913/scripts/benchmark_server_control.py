"""Matched end-to-end training control on real client data, outside formal state."""
import gc
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0,str(Path(__file__).resolve().parent))
from benchmark_server_fedmpt import step
import run_matched_fedmpt as m
import torch
from threadpoolctl import threadpool_limits


def main():
    folder=m.ROOT/'tmp/server_performance';rows=[]
    loader_results=json.loads((folder/'loader_benchmark.json').read_text())
    for dataset in ('coco','nus'):
        best=min((r for r in loader_results if r['dataset']==dataset),key=lambda r:r['seconds_per_step'])
        data=m.Images(dataset,'train')
        parts=m.np.load(m.ROOT/'outputs/icme'/dataset/'partitions/clients10_seed7.npy')
        ids=m.np.flatnonzero(parts==3).tolist()
        state=torch.load(m.OUT/dataset/'fedmpt_repaired/checkpoint.pt',map_location='cpu',weights_only=False)['global']
        reference=None
        configs=[('baseline',True,False,False,4,2),
                ('optimized',best['checkpointing'],best['cache_static_text'],best['fused_checks'],best['cpu_threads'],best['workers'])]
        if '--reverse' in sys.argv:configs.reverse()
        for name,ac,cache,fused,threads,workers in configs:
            torch.set_num_threads(threads)
            with threadpool_limits(limits=threads):
                torch.manual_seed(1234)
                model=m.make_model(dataset,ac,cache);model.prompt_learner.load_state_dict(state);model.train()
                optimizer=torch.optim.SGD(model.prompt_learner.parameters(),lr=.001,momentum=.9,weight_decay=1e-5)
                loader=m.DataLoader(m.Subset(data,ids),batch_size=32,num_workers=workers,pin_memory=True,
                    shuffle=True,generator=torch.Generator().manual_seed(10010))
                torch.manual_seed(10010)
                iterator=iter(loader)
                for i in range(2):step(model,next(iterator),optimizer,fused)
                torch.cuda.synchronize();start=perf_counter();wait=0
                for i in range(100):
                    before=perf_counter();batch=next(iterator);wait+=perf_counter()-before
                    step(model,batch,optimizer,fused)
                torch.cuda.synchronize();elapsed=perf_counter()-start
                current={k:p.detach().cpu().clone() for k,p in model.prompt_learner.named_parameters()}
                if reference is None:reference=current
                exact=all(torch.equal(current[k],reference[k]) for k in current)
                error=max(float((current[k]-reference[k]).abs().max()) for k in current)
                row=dict(dataset=dataset,variant=name,seconds_per_step=elapsed/100,loader_wait_seconds=wait,
                    steps=100,comparison_optimizer_steps=102,final_parameters_bitwise_equal=exact,max_parameter_error=error,
                    cpu_threads=threads,workers=workers,checkpointing=ac,cache_static_text=cache,fused_checks=fused)
                rows.append(row);print(json.dumps(row),flush=True)
                filename='end_to_end_control_reverse.json' if '--reverse' in sys.argv else 'end_to_end_control.json'
                (folder/filename).write_text(json.dumps(rows,indent=2))
                del iterator,loader,optimizer,model;gc.collect();torch.cuda.empty_cache()
    print('CONTROL_COMPLETE',flush=True)


if __name__=='__main__':main()
