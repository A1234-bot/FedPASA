"""Protocol-matched FedMVP multilabel adaptation from the FedMPT release.

The model forward is imported unchanged. The release loss is 0.02*(ASL+KG),
including the consistency term. This is explicitly the benchmark adaptation,
not the original ICCV single-label/LLM-attribute implementation.
"""
import argparse, base64, gc, math, sys, urllib.request
from types import SimpleNamespace as NS
from time import perf_counter
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from threadpoolctl import threadpool_limits
from icme2027_common import *
from run_matched_fedmpt import Images,save_state,gradient_flags
from convclip import clip
from model.fedmvp import FedMVPCLIP
from utilss.asl import AsymmetricLoss_Pos
from fedpasa.benchmarks import BENCHMARKS
from fedpasa.core import mean_average_precision

def source_audit():
    out=OUT/'baseline_audit';out.mkdir(parents=True,exist_ok=True)
    result=out/'result.json'
    if result.exists():
        r=read(result)
        for row in r['files']:assert digest(ROOT/'third_party/FedMPT'/row['path'])==row['local_sha256']
        return r
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    commit_path=out/'pinned_upstream_commit.json'
    if commit_path.exists():commit=read(commit_path)['commit']
    else:
        repo=json.loads(opener.open('https://api.github.com/repos/xuc865/FedMPT-main/commits/main',timeout=30).read())
        commit=repo['sha'];save(commit_path,{'commit':commit,'pinned':stamp()})
    rows=[]
    for name in ['model/fedmvp.py','convclip/fdmvpmodel.py','utilss/asl.py','federated/client.py','config/utils.py','config/defaults.py']:
        url=f'https://api.github.com/repos/xuc865/FedMPT-main/contents/{name}?ref={commit}'
        response=json.loads(opener.open(url,timeout=30).read())
        assert response['encoding']=='base64'
        content=base64.b64decode(response['content'])
        dest=out/'upstream'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(content)
        local=(ROOT/'third_party/FedMPT'/name).read_bytes()
        equal=local.replace(b'\r\n',b'\n')==content.replace(b'\r\n',b'\n')
        rows.append({'path':name,'url':url,'local_sha256':digest(ROOT/'third_party/FedMPT'/name),'upstream_sha256':digest(dest),'equal_ignoring_line_endings':equal})
    # Do not silently promote a modified model into a faithful-release baseline.
    critical=[r for r in rows if r['path'] in ('model/fedmvp.py','convclip/fdmvpmodel.py','utilss/asl.py')]
    if not all(r['equal_ignoring_line_endings'] for r in critical):
        save(out/'source_mismatch.json',{'status':'NEEDS_REVIEW','files':rows,'commit':commit})
        raise RuntimeError('Critical upstream source mismatch; inspect before baseline training')
    r={'status':'PASS','commit':commit,'files':rows,'baseline':'FedMVP, FedMPT-release multilabel adaptation',
       'original_FedMVP_author_equivalent':False,'reference':'https://github.com/mainaksingha01/FedMVP',
       'differences':['Multilabel patch pooling and ASL from FedMPT benchmark release; original FedMVP is single-label and uses LLM attribute features',
                      'Same predeclared FedPASA image IDs, labels, checkpoint, 224 center crop and 10-client seed7 partition',
                      'BF16 autocast, FP32 trainable parameters; 200 fixed rounds; no selection by test metrics'],
       'loss':'0.02 * (AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1) + KG cosine consistency)',
       'completed':stamp()}
    save(result,r);return r

def model(d):
    m=read(cache(d)/'manifest.json');assert digest(m['weight_path'])==m['weight_sha256']
    jit=torch.jit.load(m['weight_path'],map_location='cpu')
    backbone=clip.build_model_fedmvp(jit.state_dict()).float();del jit
    cfg=NS(MODEL=NS(N_CTX=4,D_CTX=1,CTX_INIT=''))
    net=FedMVPCLIP(cfg,backbone,list(BENCHMARKS[d].classes),device='cuda')
    for n,p in net.named_parameters():p.requires_grad_('prompt_learner' in n)
    return net.cuda()

def run(d,seed,smoke=False):
    audit=source_audit();out=OUT/'baselines/fedmvp_release_mlc'/d/f'seed{seed}';out.mkdir(parents=True,exist_ok=True)
    if not smoke and (out/'result.json').exists():return
    torch.manual_seed(seed);np.random.seed(seed);net=model(d)
    train=Images(d,'train');assignments=np.load(dr(d)/'partitions/clients10_seed7.npy')
    statepath=out/'checkpoint.pt';state=torch.load(statepath,map_location='cpu',weights_only=False) if statepath.exists() and not smoke else None
    initial=state['global'] if state else {k:v.detach().cpu().clone() for k,v in net.prompt_learner.state_dict().items()}
    protocol={'dataset':d,'training_seed':seed,'partition_seed':7,'partition_sha256':digest(dr(d)/'partitions/clients10_seed7.npy'),
        'target_rounds':200,'batch_size':32,'local_epochs':1,'clients':10,'participation':1.,'lr':.001,'momentum':.9,'weight_decay':1e-5,
        'schedule':'round cosine, initial lr 0.001; fixed last checkpoint','loss':audit['loss'],'source_commit':audit['commit'],
        'method':audit['baseline'],'official_original_reproduction':False,'test_labels_used_for_selection':False,
        'precision':'BF16 autocast / FP32 parameters / FP64 FedAvg sum','data_transform':'CLIP resize+center crop, 224',
        'communication':'all prompt learner state tensors, including PromptFormer; image-weighted FedAvg',
        'differences':audit['differences'],'created':stamp()}
    if state:
        assert state['target_rounds']==200 and state['training_seed']==seed
        assert state['source_commit']==audit['commit']
    if not (out/'protocol.json').exists():save(out/'protocol.json',protocol)
    criterion=AsymmetricLoss_Pos(gamma_neg=2,gamma_pos=1)
    parameter_bytes=sum(v.numel()*v.element_size() for v in initial.values())
    start_round=state.get('active_round',state['completed_round']+1) if state else 1
    for rnd in range(start_round,201):
        active=state and state.get('active_round')==rnd
        first=state['next_client'] if active else 0
        aggregate=state['aggregate'] if active else {k:torch.zeros_like(v,dtype=torch.float64) for k,v in initial.items()}
        records=state['client_records'] if active else []
        for client in range(first,10):
            torch.manual_seed(seed+rnd*1000+client);net.prompt_learner.load_state_dict(initial);net.train()
            ids=np.flatnonzero(assignments==client)
            loader=DataLoader(Subset(train,ids.tolist()),batch_size=32,shuffle=True,num_workers=4,pin_memory=True,
                generator=torch.Generator().manual_seed(seed+rnd*1000+client))
            opt=torch.optim.SGD(net.prompt_learner.parameters(),lr=.001*.5*(1+math.cos(math.pi*(rnd-1)/200)),momentum=.9,weight_decay=1e-5)
            torch.cuda.reset_peak_memory_stats();start=perf_counter();loss_sum=0.;steps=0;seen=0;nonzero=False
            for batch,(image,target) in enumerate(loader):
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits,kg=net(image.cuda(non_blocking=True),req=True)
                    loss=.02*(criterion(logits.float(),target.cuda(non_blocking=True))+kg.float())
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();finite,nz=gradient_flags(net.prompt_learner.parameters(),not nonzero)
                if not finite:raise FloatingPointError('nonfinite gradient')
                nonzero|=nz;opt.step();steps+=1;seen+=len(image);loss_sum+=float(loss.detach())
                if batch%100==0:print(f'FEDMVP {d} seed={seed} round={rnd}/200 client={client+1}/10 batch={batch}/{len(loader)} loss={loss.item():.6f}',flush=True)
                if smoke and steps==4:break
            torch.cuda.synchronize();elapsed=perf_counter()-start
            if not nonzero:raise RuntimeError('No nonzero gradient')
            if smoke:
                changed=sum(int(not torch.equal(v.detach().cpu(),initial[k])) for k,v in net.prompt_learner.state_dict().items())
                assert changed>0
                save(OUT/'baseline_audit'/f'{d}_smoke.json',{'status':'PASS','training_only':True,'steps':steps,'images':seen,
                    'changed_parameter_tensors':changed,'finite_nonzero_gradients':True,'seconds':elapsed,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/1024**2})
                return
            assert seen==len(ids) and steps==math.ceil(len(ids)/32)
            records.append({'client':client,'images':seen,'steps':steps,'loss_sum':loss_sum,'seconds':elapsed,'peak_cuda_MiB':torch.cuda.max_memory_allocated()/1024**2})
            for k,v in net.prompt_learner.state_dict().items():aggregate[k]+=v.detach().double().cpu()*(len(ids)/len(train))
            state={'global':initial,'aggregate':aggregate,'completed_round':rnd-1,'active_round':rnd,'next_client':client+1,
                'target_rounds':200,'training_seed':seed,'source_commit':audit['commit'],'client_records':records}
            save_state(statepath,state)
            save(out/'progress.json',{'status':'RUNNING','completed_round':rnd-1,'active_round':rnd,'completed_clients':client+1,'target_rounds':200,'updated':stamp()})
        initial={k:v.float() for k,v in aggregate.items()}
        save(out/f'round{rnd:03d}.json',{'round':rnd,'train_seconds':sum(v['seconds'] for v in records),'client_records':records,
             'uplink_bytes':10*parameter_bytes,'downlink_bytes':10*parameter_bytes})
        state={'global':initial,'completed_round':rnd,'target_rounds':200,'training_seed':seed,'source_commit':audit['commit']}
        save_state(statepath,state)
        save(out/'progress.json',{'status':'TRAINED' if rnd==200 else 'RUNNING','completed_round':rnd,'target_rounds':200,'updated':stamp()})
    # Resume after training completion still performs final evaluation if missing.
    net.prompt_learner.load_state_dict(initial);net.eval();test=Images(d,'test');values=[];start=perf_counter()
    with torch.inference_mode():
        for image,_ in DataLoader(test,batch_size=32,num_workers=4,pin_memory=True):
            with torch.autocast('cuda',dtype=torch.bfloat16):v=net(image.cuda(non_blocking=True)).float()
            assert torch.isfinite(v).all();values.append(v.cpu().numpy())
    scores=np.concatenate(values);np.save(out/'round200_test_scores.npy',scores)
    metric,perclass=mean_average_precision(test.targets,scores)
    records=[read(out/f'round{i:03d}.json') for i in range(1,201)]
    save(out/'result.json',{'status':'COMPLETE','mAP':metric,'per_class_AP':perclass.tolist(),'round':200,'protocol':protocol,
        'training_seconds':sum(v['train_seconds'] for v in records),'evaluation_seconds':perf_counter()-start,
        'uplink_bytes':sum(v['uplink_bytes'] for v in records),'downlink_bytes':sum(v['downlink_bytes'] for v in records),'completed':stamp()})
    save(out/'progress.json',{'status':'COMPLETE','completed_round':200,'target_rounds':200,'updated':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['coco','nus'],default='coco');p.add_argument('--seed',type=int,default=7)
    p.add_argument('--audit-only',action='store_true');p.add_argument('--smoke',action='store_true');a=p.parse_args();torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if a.audit_only:source_audit()
        else:run(a.dataset,a.seed,a.smoke)
