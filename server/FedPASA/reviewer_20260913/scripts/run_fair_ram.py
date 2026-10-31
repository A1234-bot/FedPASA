"""Audited RAM core + disclosed sample-weighted federation, full-label adaptation."""
import argparse,ast,copy,gc,math,types
import numpy as np
import torch
from types import SimpleNamespace as NS
from fair_baselines_common import *
import run_fair_fedmvp as engine
from convclip import clip
from model.fedram import FedRAMCLIP,mmc_loss
from model.ram.base import BaseModel
from model.ram.ot_solver import Sinkhorn

SOURCE=OUT/'sources/RAM-main'
LABEL='Fed-RAM with verified RAM core; full-label federated adaptation'

class FederatedState:
    def __init__(self,net):
        self.net=net
        self.keys={n for n,p in net.named_parameters() if p.requires_grad}
        self.keys.update(n for n,b in net.named_buffers() if 'adapter' in n)
    def state_dict(self):return {n:v for n,v in self.net.state_dict().items() if n in self.keys}
    def load_state_dict(self,state):
        assert set(state)==self.keys
        with torch.no_grad():
            dest=self.net.state_dict()
            for n,v in state.items():dest[n].copy_(v)
    def parameters(self):return [p for p in self.net.parameters() if p.requires_grad]
    def named_parameters(self):return [(n,p) for n,p in self.net.named_parameters() if p.requires_grad]

class Wrapper(torch.nn.Module):
    def __init__(self,inner):super().__init__();self.inner=inner
    @property
    def prompt_learner(self):return FederatedState(self.inner)
    def forward(self,x,req=False,target=None):return self.inner(x,target=target,gzsl=False,req=req)

def new_model(d,seed,fast=True):
    torch.manual_seed(seed);np.random.seed(seed)
    m=read(engine.old.cache(d)/'manifest.json');assert digest(m['weight_path'])==m['weight_sha256']
    jit=torch.jit.load(m['weight_path'],map_location='cpu');state=jit.state_dict();del jit
    design={'depth_vision':[9,10,11],'vision_adapt':8,'depth_text':[9,10,11],'text_ctx':4,'SAA_layer':list(range(12)),'kernel_size':3}
    student=clip.build_model_ram(dict(state),(224,224),design).float().cuda();student.visual.SAA_replace()
    teacher=clip.build_model_ram(dict(state),(224,224),{'depth_vision':[-1],'depth_text':[-1],'SAA_layer':list(range(12))}).float().cuda();teacher.visual.SAA_replace()
    cfg=NS(TRAINER=NS(RAM=NS(DIST_TRAIN=False,TRANSFER_TYPE='Adapter',TEMPERATURE=.002,OT_REG=.1,OT_REGSC=.05)))
    net=FedRAMCLIP(cfg,student,list(engine.BENCHMARKS[d].classes),[],teacher,device='cuda').cuda()
    return Wrapper(net).cuda()

def batch_loss(net,image,y,criterion):return net(image,req=True,target=y)[1].float()

class Optimizers:
    def __init__(self,*opts):self.opts=opts
    def zero_grad(self,**kw):
        for o in self.opts:o.zero_grad(**kw)
    def step(self):
        for o in self.opts:o.step()

def make_optimizer(net,lr,rnd):
    factor=.5*(1+math.cos(math.pi*(rnd-1)/200))
    cp=[];ep=[];op=[]
    for n,p in net.inner.named_parameters():
        if not p.requires_grad:continue
        (ep if 'embeds' in n else cp if n.startswith('clip') else op).append(p)
    a=torch.optim.AdamW([{'params':cp,'lr':lr*.1*factor},{'params':op,'lr':lr*factor}],weight_decay=1e-4)
    b=torch.optim.SGD(ep,lr=.001*(lr/5e-5)*factor,momentum=.9,weight_decay=1e-4)
    return Optimizers(a,b)

def ast_tree(p):return ast.parse(Path(p).read_text(encoding='utf-8'))
def source_audit():
    dest=OUT/'ram_audit/source.json';rows=[]
    p=ROOT/'third_party/FedMPT/convclip/rammodel.py';q=SOURCE/'clip/model.py';a=ast_tree(p);b=ast_tree(q)
    a.body=[n for n in a.body if not (isinstance(n,ast.Import) and any(v.name=='pdb' for v in n.names))]
    for n in a.body:
        if isinstance(n,ast.FunctionDef) and n.name=='build_model_ram':n.name='build_model'
    rows.append({'component':'backbone','equal_normalized_AST':ast.dump(a)==ast.dump(b),'difference':'builder rename and unused pdb import','benchmark_sha256':digest(p),'author_sha256':digest(q)})
    for name,p,q in [('LLA',ROOT/'third_party/FedMPT/convclip/adapters.py',SOURCE/'clip/adapters.py')]:
        rows.append({'component':name,'equal_normalized_AST':ast.dump(ast_tree(p))==ast.dump(ast_tree(q)),'benchmark_sha256':digest(p),'author_sha256':digest(q)})
    p=ROOT/'third_party/FedMPT/model/ram/ot_solver.py';q=SOURCE/'model/ot_solver.py';a=ast_tree(p);b=ast_tree(q)
    fn=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='Sinkhorn')
    fn(a).args.defaults=fn(b).args.defaults
    rows.append({'component':'Sinkhorn','equal_normalized_AST':ast.dump(a)==ast.dump(b),'difference':'unused default reg; forward passes reg explicitly','benchmark_sha256':digest(p),'author_sha256':digest(q)})
    p=ROOT/'third_party/FedMPT/model/fedram.py';q=SOURCE/'loss/mmc_loss.py'
    fn=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='mmc_loss')
    rows.append({'component':'MMC','equal_normalized_AST':ast.dump(fn(ast_tree(p)))==ast.dump(fn(ast_tree(q))),'benchmark_sha256':digest(p),'author_sha256':digest(q)})
    r={'status':'PASS' if all(x['equal_normalized_AST'] for x in rows) else 'FAIL','components':rows,'author_source_url':'https://github.com/EricTan7/RAM','archive_sha256':'49bee71f0b23fd4819396afb3f925ef104fba5d48effb5c8133e6a9b613cbfd3','scope':'core source matching; full experimental author-equivalence is not claimed','adaptations':['all target classes supervised instead of original seen/unseen split','sample-weighted FedAvg across 10 clients','BN floating buffers averaged, integer counters max','200-round cosine validation-selected duration; no EMA','common center-crop image protocol'],'time':stamp()}
    save(dest,r);assert r['status']=='PASS';return r

def smoke(d):
    dest=OUT/'fedram'/d/'smoke.json'
    if dest.exists():assert read(dest)['passed'];return
    source_audit();f,_=split(d);data=engine.Images(d,'train')
    x,y=next(iter(engine.DataLoader(engine.Subset(data,f[:8].tolist()),batch_size=4)));x=x.cuda();y=y.cuda();net=new_model(d,17)
    tree=ast_tree(SOURCE/'model/model.py');node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='RAM_Model')
    namespace={'torch':torch,'nn':torch.nn,'F':torch.nn.functional,'dist':torch.distributed,'clip':clip,'BaseModel':BaseModel,'Sinkhorn':Sinkhorn}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE/'model/model.py'),'exec'),namespace)
    author_forward=namespace['RAM_Model'].forward
    initial={k:v.detach().cpu().clone() for k,v in net.inner.state_dict().items()}
    def probe(author=False,training=True):
        net.inner.load_state_dict(initial);net.train(training);net.zero_grad(set_to_none=True);torch.manual_seed(2718)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            if author:
                result=author_forward(net.inner,x,y,gzsl=False);score=result['score'];loss=result.get('loss')
            else:
                result=net(x,req=training,target=y);score,loss=result if training else (result,None)
        grads={}
        if training:
            loss.backward();grads={n:p.grad.detach().clone() for n,p in net.prompt_learner.named_parameters() if p.grad is not None}
        return score.detach().clone(),loss.detach().clone() if loss is not None else None,grads
    a,l,g=probe(True);b,ll,gg=probe(False)
    score_error=float((a-b).abs().max());loss_error=float((l-ll).abs());grad_error=max(float((g[n]-gg[n]).abs().max()) for n in g)
    assert torch.equal(a,b) and torch.equal(l,ll) and all(torch.equal(g[n],gg[n]) for n in g),'RAM training core differs from author'
    a,_,_=probe(True,False);b,_,_=probe(False,False);eval_error=float((a-b).abs().max());assert torch.equal(a,b)
    net.inner.load_state_dict(initial);net.train();opt=make_optimizer(net,5e-5,1);losses=[]
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=batch_loss(net,x,y,None)
        loss.backward();finite,nz=engine.gradient_flags(net.prompt_learner.parameters());assert finite and nz;opt.step();losses.append(float(loss.detach()))
    r={'passed':bool(np.mean(losses[-5:])<np.mean(losses[:5])),'train_score_error':score_error,'loss_error':loss_error,'gradient_error':grad_error,'eval_score_error':eval_error,'losses':losses,'trainable_parameters':sum(p.numel() for p in net.prompt_learner.parameters()),'federated_state_bytes':sum(t.numel()*t.element_size() for t in net.prompt_learner.state_dict().values()),'test_access':False,'identity':LABEL,'time':stamp()}
    save(dest,r);assert r['passed'],'RAM small training subset did not improve'
    del net,initial,g,gg,x,y;gc.collect();torch.cuda.empty_cache()

def configure():
    engine.METHOD='fedram';engine.METHOD_LABEL=LABEL;engine.LRS=(3e-5,5e-5,1e-4)
    engine.new_model=new_model;engine.batch_loss=batch_loss;engine.make_optimizer=make_optimizer;engine.smoke=smoke
    engine.evaluate_round=lambda r:r in (2,4,6,10) or r%20==0

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['audit','tune','repeat'],required=True);p.add_argument('--dataset',choices=['coco','nus'],required=True);args=p.parse_args();configure()
    if args.phase=='audit':smoke(args.dataset)
    elif args.phase=='tune':engine.tune(args.dataset)
    else:
        s=read(OUT/'fedram'/args.dataset/'selection.json')
        for seed in (7,17,27):engine.train(args.dataset,seed,s['lr'],s['rounds'],'final')
