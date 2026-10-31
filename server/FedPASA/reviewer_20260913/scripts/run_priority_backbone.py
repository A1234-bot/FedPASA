"""COCO ViT-B/32 extraction and same-representation estimator checks."""
import gc, hashlib, json, os, shutil, sys, time, urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'third_party/PIAA'))
import packaging, packaging.version, pkg_resources
pkg_resources.packaging=packaging
import clip,scclip
import extract_benchmark_features as ex
import extract_patch_summaries as sums
import run_priority_full_impl as r

OUT=ROOT/'outputs/priority_20260913/backbone';CACHE=OUT/'cache/coco'
WEIGHT=OUT/'ViT-B-32.pt'
URL='https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt'

def download():
    OUT.mkdir(parents=True,exist_ok=True)
    if not WEIGHT.exists():
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(URL,timeout=60) as src,WEIGHT.with_suffix('.partial').open('wb') as dst:shutil.copyfileobj(src,dst)
        WEIGHT.with_suffix('.partial').replace(WEIGHT)
    assert r.digest(WEIGHT)==URL.split('/')[-2]
    print('WEIGHT VERIFIED',flush=True)

def extract():
    download();CACHE.mkdir(parents=True,exist_ok=True)
    source=r.resolve_cache('coco');manifest=r.read(ROOT.parent/'icme2027_workspace/outputs/icme/cache/coco/manifest.json')
    for split in ('train','test'):
        for name in ('targets.npy','ids.txt'):shutil.copy2(source/f'{split}_{name}',CACHE/f'{split}_{name}')
    records=[]
    cls_model,preprocess=clip.load(str(WEIGHT),device='cuda',jit=False);cls_model.eval()
    patch_model,patch_preprocess=scclip.load(str(WEIGHT),device='cuda',jit=False);patch_model.eval()
    from PIL import Image
    first_id=(CACHE/'train_ids.txt').read_text().splitlines()[0]
    with Image.open(Path(manifest['train_root'])/r.BENCHMARKS['coco'].image_name(first_id)) as sample:
        sample=sample.convert('RGB')
        torch.testing.assert_close(preprocess(sample),patch_preprocess(sample),rtol=0,atol=0)
    np.save(CACHE/'text_features.npy',ex.make_text_features(cls_model,r.BENCHMARKS['coco'].classes,'cuda').cpu().numpy())
    for split in ('train','test'):
        dests={kind:CACHE/f'{split}_{kind}.npy' for kind in ('cls','patch')}
        if all(p.exists() for p in dests.values()):continue
        ids=(CACHE/f'{split}_ids.txt').read_text().splitlines();ds=ex.ImageDataset(Path(manifest[split+'_root']),ids,r.BENCHMARKS['coco'],preprocess)
        progress=CACHE/f'{split}_dual_progress.json';offset=r.read(progress)['offset'] if progress.exists() else 0
        assert offset%64==0 or offset==len(ds)
        arrays={kind:np.lib.format.open_memmap(CACHE/f'{split}_{kind}.dual_partial.npy',mode='r+' if offset else 'w+',dtype='float16',shape=(len(ds),*shape)) for kind,shape in [('cls',(512,)),('patch',(49,512))]}
        loader=DataLoader(Subset(ds,range(offset,len(ds))),batch_size=64,shuffle=False,num_workers=12,prefetch_factor=1,pin_memory=True)
        t=time.time()
        with torch.inference_mode():
            for im in loader:
                im=im.cuda(non_blocking=True)
                fs={'cls':cls_model.encode_image(im),'patch':patch_model.encode_image(im,return_all=True)}
                for kind,f in fs.items():
                    f=F.normalize(f.float(),dim=-1);arrays[kind][offset:offset+len(f)]=f.cpu().half().numpy()
                offset+=len(im)
                if offset%1280==0:
                    for arr in arrays.values():arr.flush()
                    r.save(progress,dict(offset=offset));print('DUAL_EXTRACT',split,offset,len(ds),'seconds',round(time.time()-t,2),flush=True)
        assert offset==len(ds)
        for arr in arrays.values():arr.flush()
        del arr;arrays.clear()
        for kind,dest in dests.items():(CACHE/f'{split}_{kind}.dual_partial.npy').replace(dest)
        records.append(dict(split=split,kind='cls_and_patch',seconds=time.time()-t))
    del cls_model,patch_model;gc.collect();torch.cuda.empty_cache()
    for split in ('train','test'):
        if not (CACHE/f'{split}_patch_mean.npy').exists():sums.extract(split,CACHE,batch_images=128,device='cuda')
    r.save(CACHE/'manifest.json',dict(backbone='CLIP ViT-B/32',patch_representation='PIAA SC-CLIP',patches=49,weight_sha256=r.digest(WEIGHT),weight_url=URL,same_image_ids=True,same_labels=True,extraction_batch_size=64,scclip_source_sha256=r.digest(ROOT/'third_party/PIAA/scclip/model.py'),extraction=records))

def evaluate():
    r.OUT=OUT/'evaluation';r.resolve_cache=lambda d:CACHE
    r.SEEDS=(7,17,27)
    # Reuse exactly the original B/16 partition and train-side split IDs.
    if not (r.OUT/'full_runtime/coco/result.json').exists():r.full_runtime('coco')
    p=r.OUT/'full_runtime/coco'
    text=torch.from_numpy(np.load(CACHE/'text_features.npy')).float()
    # Build central Top-K from the complete raw patch cache independently.
    central=r.core.build_client_banks(CACHE/'train_patch.npy',np.zeros(len(r.targets('coco','train')),dtype=np.int64),text,clients=1,k=512,batch_images=32,device='cuda')[0]
    qa=r.core.fit_central_piaa(central,text,purification_temperature=r.BENCHMARKS['coco'].purification_temperature)
    score=r.piaa_scores(CACHE/'test_patch.npy',CACHE/'test_cls.npy',np.arange(len(r.targets('coco'))),text,qa.weight,qa.bias,cls_scale=r.BENCHMARKS['coco'].cls_scale,patch_scale=r.BENCHMARKS['coco'].patch_scale,batch_images=128,device='cuda')
    qs=np.load(p/'qa_scores.npy');del central;gc.collect()
    x=r.load_features(CACHE,'train');z=r.load_features(CACHE,'test');y=r.targets('coco','train');lam=r.read(p/'seed7_selection.json')['alpha']
    model=r.fit_ridge_from_packets((r.make_ridge_packet(x,y),),regularization=lam);rs=r.predict(z,model);fed=np.load(p/'seed7_ridge_scores.npy')
    r.save(OUT/'matching.json',dict(status='COMPLETE',qa_central_mAP=r.metric(r.targets('coco'),score)[0],qa_federated_mAP=r.metric(r.targets('coco'),qs)[0],qa_max_abs_score_error=float(np.max(np.abs(score-qs))),ridge_central_mAP=r.metric(r.targets('coco'),rs)[0],ridge_federated_mAP=r.metric(r.targets('coco'),fed)[0],ridge_max_abs_score_error=float(np.max(np.abs(rs-fed))),central_QA_selection='global top-K independently rebuilt from complete raw training patch cache'))

if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        if '--download-only' in sys.argv:download()
        elif '--evaluate-only' in sys.argv:evaluate()
        else:extract();evaluate()
