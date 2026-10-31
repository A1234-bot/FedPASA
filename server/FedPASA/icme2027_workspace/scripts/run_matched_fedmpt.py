"""Disclosed repaired FedMPT, same images/labels/partition/checkpoint as FedPASA.

Uses images because visual prompt training cannot reuse final frozen features.
Never promotes an intermediate round to the completed baseline. Resumes at client
boundaries and writes round metrics; the final round is fixed before seeing test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import subprocess
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.checkpoint import checkpoint
from threadpoolctl import threadpool_limits
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'third_party/FedMPT'))
from fedpasa.benchmarks import BENCHMARKS
from fedpasa.core import mean_average_precision
from convclip import clip
from model.fedmpt import FedMPTCLIP
from utilss.asl import AsymmetricLoss_Pos

OUT=ROOT/'outputs/icme/supplementary'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def save_state(path,value):
    tmp=path.with_suffix('.partial.pt');torch.save(value,tmp);tmp.replace(path)


class Images(Dataset):
    def __init__(self,dataset,split):
        self.b=BENCHMARKS[dataset]
        cache=ROOT/'outputs/icme/cache'/dataset
        manifest=read(cache/'manifest.json')
        self.root=Path(manifest[split+'_root'])
        self.ids=(cache/(split+'_ids.txt')).read_text(encoding='utf-8').splitlines()
        self.targets=np.load(cache/(split+'_targets.npy'))
        assert len(self.ids)==len(self.targets)
        # Images are converted to RGB in __getitem__; remove CLIP's local lambda
        # so this numerically identical transform is picklable on Windows.
        self.transform=Compose([Resize(224,interpolation=InterpolationMode.BICUBIC),CenterCrop(224),ToTensor(),
            Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))])
    def __len__(self):return len(self.ids)
    def __getitem__(self,index):
        with Image.open(self.root/self.b.image_name(self.ids[index])) as image:
            tensor=self.transform(image.convert('RGB'))
        return tensor,torch.from_numpy(self.targets[index].astype(np.float32))


def checkpoint_transformer(transformer):
    # Same layers and arithmetic. Recompute frozen-block activations for backward.
    def forward(x,prompt_embeddings=None):
        for i,layer in enumerate(transformer.resblocks):
            prompt=prompt_embeddings if i==0 else None
            if torch.is_grad_enabled() and (x.requires_grad or (prompt is not None and prompt.requires_grad)):
                if prompt is None:
                    x=checkpoint(lambda a,block=layer:block([a,None]),x,use_reentrant=False)
                else:
                    x=checkpoint(lambda a,b,block=layer:block([a,b]),x,prompt,use_reentrant=False)
            else:x=layer([x,prompt])
        return x
    transformer.forward=forward


def make_model(dataset,activation_checkpointing=True,cache_static_text=False):
    manifest=read(ROOT/'outputs/icme/cache'/dataset/'manifest.json')
    source=Path(manifest['weight_path'])
    if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['weight_sha256']:
        raise ValueError('CLIP checkpoint digest mismatch')
    jit=torch.jit.load(str(source),map_location='cpu')
    backbone=clip.build_model_fedmvp(jit.state_dict()).float();del jit
    cfg=SimpleNamespace(TRAINER=SimpleNamespace(ML=SimpleNamespace(CLS=4,COND=4,LAT=64,TEMP=4.,AGG='vanilla',OT_REG=0.1)))
    model=FedMPTCLIP(cfg,backbone,list(BENCHMARKS[dataset].classes),device='cuda')
    for name,p in model.named_parameters():p.requires_grad_('prompt_learner' in name)
    # Keep trainable master parameters in FP32; autocast handles the forward.
    model.prompt_learner.float()
    model.cache_static_text=cache_static_text
    if activation_checkpointing:
        checkpoint_transformer(model.image_encoder.transformer)
        checkpoint_transformer(model.text_encoder.transformer)
    return model.cuda()


def gradient_flags(parameters,check_nonzero=True):
    """Keep finite-gradient checks but transfer their flags in one synchronization."""
    grads=[p.grad for p in parameters if p.grad is not None]
    if not grads:return False,False
    finite=torch.stack([torch.isfinite(g).all() for g in grads]).all()
    nonzero=torch.stack([g.abs().max()>0 for g in grads]).any() if check_nonzero else finite.new_tensor(True)
    return torch.stack((finite,nonzero)).tolist()


@torch.inference_mode()
def evaluate(model,data,batch_size,workers):
    model.eval();outputs=[]
    for i,(image,_) in enumerate(DataLoader(data,batch_size=batch_size,num_workers=workers,pin_memory=True)):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=model(image.cuda(non_blocking=True)).float()
        if not torch.isfinite(output).all():raise FloatingPointError('non-finite test logits')
        outputs.append(output.cpu().numpy())
        if i%200==0:print(f'FEDMPT_EVAL batch={i}',flush=True)
    return np.concatenate(outputs)


def run_round(dataset,args,round_index):
    workers=getattr(args,f'{dataset}_workers',None)
    workers=args.workers if workers is None else workers
    out=OUT/dataset/'fedmpt_repaired';out.mkdir(parents=True,exist_ok=True)
    state_path=out/'checkpoint.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=False) if state_path.exists() else None
    if state and state['completed_round']>=round_index:return
    if state and state['target_rounds']!=args.rounds:raise ValueError('resume must preserve the predeclared round budget')
    torch.manual_seed(args.seed+round_index)
    np.random.seed(args.seed+round_index)
    model=make_model(dataset,not getattr(args,'no_activation_checkpointing',False),getattr(args,'cache_static_text',False))
    train=Images(dataset,'train')
    assignments_path=ROOT/'outputs/icme'/dataset/'partitions'/f'clients{args.clients}_seed{args.seed}.npy'
    assignments=np.load(assignments_path)
    assert len(assignments)==len(train)
    if state:
        initial=state['global']
    else:
        initial={k:v.detach().cpu().clone() for k,v in model.prompt_learner.state_dict().items()}
    parameter_count=sum(v.numel() for v in initial.values())
    protocol=dict(dataset=dataset,method='FedMPT repaired image-input rerun',status='RUNNING',
        target_rounds=args.rounds,clients=args.clients,seed=args.seed,
        participation=1.,local_epochs=1,batch_size=args.batch_size,backbone='CLIP ViT-B/16',input_resolution=224,
        train_images=len(train),test_images=BENCHMARKS[dataset].expected_test,
        partition_sha256=hashlib.sha256(assignments_path.read_bytes()).hexdigest(),
        label_order='exact cache train/test targets and image IDs',
        parameter_count=parameter_count,upload_MiB_per_round=args.clients*parameter_count*4/1024**2,
        runtime=dict(activation_checkpointing=not getattr(args,'no_activation_checkpointing',False),
            cache_static_text=getattr(args,'cache_static_text',False),fused_gradient_check=getattr(args,'fused_gradient_check',False),
            cpu_threads=torch.get_num_threads(),workers=workers,torch_version=torch.__version__,gpu=torch.cuda.get_device_name()),
        bidirectional_MiB_per_round=2*args.clients*parameter_count*4/1024**2,
        test_labels_used_for_selection=False,final_checkpoint='fixed last round, never best test mAP',
        original_source='https://github.com/xuc865/FedMPT-main',
        repairs=read(ROOT/'outputs/icme/paper/official_fedmpt_runnability.json')['blockers_and_repairs'] if 'blockers_and_repairs' in read(ROOT/'outputs/icme/paper/official_fedmpt_runnability.json') else read(ROOT/'outputs/icme/paper/official_fedmpt_runnability.json'),
        disclosed_differences=[
            'Uses previously repaired FedMPT forward; not an official numerical reproduction.',
            'Final frozen patch cache cannot replace image encoder with visual prompts.',
            'Deterministic CLIP resize/center crop matches feature extraction; no random image augmentation.',
            'Four built-in conditions, four class context tokens, latent width 64, temperature 4.',
            'Repaired vanilla conv/softmax aggregation retained; not paper OT equivalence.',
            'Image-count-weighted FedAvg; optimizer momentum resets each client-round.',
            'SGD lr 0.001 cosine to zero, momentum 0.9, weight decay 1e-5; ASL neg=2,pos=1, summed reduction.',
            'BF16 AMP; master parameters/aggregation FP32. Activation checkpointing was used on the 8 GiB GPU; current performance options are recorded in runtime. FP16 smoke overflowed gradients.',
        ])
    save(out/'protocol.json',protocol)
    resuming=bool(state and state.get('active_round')==round_index)
    next_client=state['next_client'] if resuming else 0
    aggregate=state['aggregate'] if resuming else {k:torch.zeros_like(v,dtype=torch.float64) for k,v in initial.items()}
    accumulated=state.get('round_train_seconds',0.) if resuming else 0.
    losses=state.get('client_losses',[]) if resuming else []
    criterion=AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1)
    for client in range(next_client,args.clients):
        model.prompt_learner.load_state_dict(initial)
        model.train()
        # Frozen CLIP has no dropout; the prompt cross-attention dropout is trained.
        ids=np.flatnonzero(assignments==client)
        generator=torch.Generator().manual_seed(args.seed+round_index*1000+client)
        torch.manual_seed(args.seed+round_index*1000+client)
        loader=DataLoader(Subset(train,ids.tolist()),batch_size=args.batch_size,shuffle=True,
            generator=generator,num_workers=workers,pin_memory=True)
        lr=.001*.5*(1+math.cos(math.pi*(round_index-1)/args.rounds))
        optimizer=torch.optim.SGD(model.prompt_learner.parameters(),lr=lr,momentum=.9,weight_decay=1e-5)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        start=perf_counter();total_loss=0.;gradients_seen=False;steps=0
        for batch,(image,target) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits=model(image.cuda(non_blocking=True))
                loss=criterion(logits.float(),target.cuda(non_blocking=True))
            if not torch.isfinite(loss):raise FloatingPointError('non-finite training loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            if getattr(args,'fused_gradient_check',False):
                finite,nonzero=gradient_flags(model.prompt_learner.parameters(),not gradients_seen)
                if not finite:raise FloatingPointError('non-finite training gradient')
                gradients_seen |= nonzero
                steps+=1
            else:
                finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.prompt_learner.parameters())
                if finite:
                    gradients_seen |= any(p.grad is not None and bool(p.grad.abs().max()>0) for p in model.prompt_learner.parameters())
                    steps+=1
            scaler.step(optimizer);scaler.update();total_loss+=float(loss.detach())
            if batch%100==0:print(f'FEDMPT_TRAIN {dataset} round={round_index}/{args.rounds} client={client+1}/{args.clients} batch={batch}/{len(loader)} loss={loss.item():.4f}',flush=True)
            if args.smoke and batch>=1:break
        if not gradients_seen or steps==0:raise RuntimeError('No finite nonzero prompt updates; do not report a baseline')
        elapsed=perf_counter()-start;accumulated+=elapsed
        losses.append(dict(client=client,images=len(ids),loss_sum=total_loss,finite_optimizer_steps=steps,seconds=elapsed))
        for k,v in model.prompt_learner.state_dict().items():aggregate[k]+=v.detach().double().cpu()*(len(ids)/len(train))
        if args.smoke:
            save(out/'smoke_test.json',dict(status='PASS',dataset=dataset,finite_nonzero_gradients=True,batches=2,seconds=elapsed,
                note='smoke test only; excluded from baseline accuracy tables'))
            return
        state={'global':initial,'aggregate':aggregate,'completed_round':round_index-1,'active_round':round_index,
            'next_client':client+1,'target_rounds':args.rounds,'round_train_seconds':accumulated,'client_losses':losses}
        save_state(state_path,state)
        save(out/'progress.json',dict(status='RUNNING',dataset=dataset,completed_round=round_index-1,
            active_round=round_index,completed_clients=client+1,clients=args.clients,target_rounds=args.rounds,
            round_train_seconds=accumulated,estimated_round_seconds=accumulated/(client+1)*args.clients))
    initial={k:v.float() for k,v in aggregate.items()}
    model.prompt_learner.load_state_dict(initial)
    record=dict(round=round_index,train_seconds=accumulated,client_losses=losses)
    if round_index in (1,5,10,25,50,100,args.rounds):
        test=Images(dataset,'test');start=perf_counter()
        scores=evaluate(model,test,args.batch_size,workers)
        np.save(out/f'round{round_index:03d}_test_scores.npy',scores)
        record.update(mAP=mean_average_precision(test.targets,scores)[0],evaluation_seconds=perf_counter()-start)
    save(out/f'round{round_index:03d}.json',record)
    state={'global':initial,'completed_round':round_index,'target_rounds':args.rounds}
    save_state(state_path,state)
    status='COMPLETE' if round_index==args.rounds else 'RUNNING'
    save(out/'progress.json',dict(status=status,dataset=dataset,completed_round=round_index,target_rounds=args.rounds,
        last_round_train_seconds=accumulated,last_round_mAP=record.get('mAP')))
    if status=='COMPLETE':
        save(out/'result.json',dict(**record,protocol=protocol,status=status,official_reproduction=False))
    subprocess.run([sys.executable,str(ROOT/'scripts/build_supplementary_report.py'),'--no-plots'],check=True)
    del model;torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=['coco','nus'])
    parser.add_argument('--rounds',type=int,default=200)
    parser.add_argument('--clients',type=int,default=10)
    parser.add_argument('--seed',type=int,default=7)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--coco-workers',type=int)
    parser.add_argument('--nus-workers',type=int)
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--no-activation-checkpointing',action='store_true')
    parser.add_argument('--cache-static-text',action='store_true')
    parser.add_argument('--fused-gradient-check',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();torch.set_num_threads(args.cpu_threads)
    job_path=OUT/('fedmpt_smoke_job.json' if args.smoke else 'fedmpt_job.json')
    save(job_path,dict(status='RUNNING',datasets=args.datasets,target_rounds=args.rounds))
    try:
        with threadpool_limits(limits=args.cpu_threads):
            for r in range(1,args.rounds+1):
                for d in args.datasets:run_round(d,args,r)
                if args.smoke:break
        save(job_path,dict(status='COMPLETE',datasets=args.datasets,target_rounds=args.rounds,smoke=args.smoke))
        if not args.smoke:
            subprocess.run([sys.executable,str(ROOT/'scripts/build_supplementary_report.py')],check=True)
    except Exception as exc:
        save(job_path,dict(status='FAILED',exception=repr(exc)))
        raise


if __name__=='__main__':main()
