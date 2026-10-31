"""Measure the complete image-loader/training pipeline on train images only."""
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
    rows=[];folder=m.ROOT/'tmp/server_performance'
    for dataset in ('coco','nus'):
        data=m.Images(dataset,'train')
        state=torch.load(m.OUT/dataset/'fedmpt_repaired/checkpoint.pt',map_location='cpu',weights_only=False)['global']
        variants=json.loads((folder/f'{dataset}_benchmark.json').read_text())
        compatible=[r for r in variants if r['three_step_bitwise_equal']]
        best=min(compatible,key=lambda r:r['seconds_per_step'])
        threads=best['cpu_threads'];torch.set_num_threads(threads)
        for workers in (2,4,8):
            with threadpool_limits(limits=threads):
                torch.manual_seed(1234)
                model=m.make_model(dataset,best['checkpointing'],best['cache_static_text']);model.prompt_learner.load_state_dict(state);model.train()
                optimizer=torch.optim.SGD(model.prompt_learner.parameters(),lr=.001,momentum=.9,weight_decay=1e-5)
                # Identical image order for every loader candidate; startup and
                # warmup excluded, following 32 complete steps timed.
                loader=m.DataLoader(m.Subset(data,list(range(34*32))),batch_size=32,num_workers=workers,
                    pin_memory=True,shuffle=True,generator=torch.Generator().manual_seed(44))
                iterator=iter(loader)
                for i in range(2):step(model,next(iterator),optimizer,best['fused_checks'])
                torch.cuda.synchronize();start=perf_counter();wait=0
                for i in range(32):
                    before=perf_counter();batch=next(iterator);wait+=perf_counter()-before
                    step(model,batch,optimizer,best['fused_checks'])
                torch.cuda.synchronize();elapsed=perf_counter()-start
                row=dict(dataset=dataset,variant=best['variant'],cpu_threads=threads,workers=workers,
                    steps=32,seconds_per_step=elapsed/32,loader_wait_seconds=wait,
                    checkpointing=best['checkpointing'],cache_static_text=best['cache_static_text'],fused_checks=best['fused_checks'])
                rows.append(row);print(json.dumps(row),flush=True)
                (folder/'loader_benchmark.json').write_text(json.dumps(rows,indent=2))
                del iterator,loader,optimizer,model;gc.collect();torch.cuda.empty_cache()
    print('LOADER_BENCHMARK_COMPLETE',flush=True)


if __name__=='__main__':main()
