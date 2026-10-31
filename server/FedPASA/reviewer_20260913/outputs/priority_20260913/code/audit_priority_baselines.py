"""Fresh source check plus independent evaluation of archived training outputs."""
import ast,base64,hashlib,json,sys,urllib.request
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits
import run_priority_full_impl as r
BASE=r.ROOT/'outputs/priority_20260913/baseline_audit'

def main():
    BASE.mkdir(parents=True,exist_ok=True);rows=[]
    source=r.ROOT.parent
    for d in ('coco','nus'):
        y=r.targets(d);partition=r.digest(r.dr(d)/'partitions/clients10_seed7.npy')
        candidates=[('FedMPT repaired diagnostic',source/'outputs/icme/supplementary'/d/'fedmpt_repaired')]
        candidates += [('FedMVP release multilabel adaptation',source/'icme2027_workspace/outputs/icme2027/baselines/fedmvp_release_mlc'/d/f'seed{s}') for s in (7,17,27)]
        for name,p in candidates:
            result=r.read(p/'result.json');pr=result['protocol'];scorepath=p/'round200_test_scores.npy';scores=np.load(scorepath)
            assert result['round']==200 and pr['target_rounds']==200 and pr['clients']==10
            assert pr['partition_sha256']==partition
            assert not pr['test_labels_used_for_selection']
            assert scores.shape==y.shape and np.isfinite(scores).all()
            value=r.metric(y,scores)[0];assert abs(value-result['mAP'])<1e-9
            up=result.get('uplink_bytes',pr.get('upload_MiB_per_round',0)*1024**2*200)
            rows.append(dict(dataset=d,method=name,seed=pr.get('training_seed',pr.get('seed')),mAP=value,rounds=200,uplink_MiB=up/1024**2,partition_sha256=partition,prediction_sha256=r.digest(scorepath),source_result=str(p/'result.json'),source_prediction=str(scorepath),prediction_metric_difference=value-result['mAP'],new_training=False,author_equivalent=False,scope='same images/labels/checkpoint/partition; method adaptation disclosed',repairs=pr.get('repairs',{})))
        other=source/'icme2027_workspace/outputs/icme/cache'/d
        for split in ('train','test'):
            np.testing.assert_array_equal(np.load(other/f'{split}_targets.npy'),r.targets(d,split))
            assert (other/f'{split}_ids.txt').read_text().splitlines()==(r.resolve_cache(d)/f'{split}_ids.txt').read_text().splitlines()
    r.save(BASE/'archived_predictions_rechecked.json',dict(status='COMPLETE',rows=rows))
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    commit=json.loads(opener.open('https://api.github.com/repos/xuc865/FedMPT-main/commits/main',timeout=40).read())['sha']
    url=f'https://api.github.com/repos/xuc865/FedMPT-main/contents/model/fedmpt.py?ref={commit}'
    response=json.loads(opener.open(url,timeout=40).read());raw=base64.b64decode(response['content']);(BASE/'upstream_fedmpt.py').write_bytes(raw)
    # Exercise the exact released component, with deterministic compatible input.
    tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='FedMPTCLIP')
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_vanilla_class_scores')
    module=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));scope={'F':F}
    exec(compile(module,'pinned_upstream_fedmpt.py','exec'),scope)
    error=None
    try:scope['_vanilla_class_scores'](None,torch.zeros(2,512,49),torch.zeros(80,512),4.)
    except Exception as exc:error=f'{type(exc).__name__}: {exc}'
    r.save(BASE/'current_source_audit.json',dict(status='SOURCE_BLOCKED' if error else 'NEEDS_FURTHER_AUDIT',commit=commit,url=url,sha256=hashlib.sha256(raw).hexdigest(),vanilla_component_error=error,prompt_length_for_cond4=77-4+1,CLIP_required_prompt_length=77,two_author_equivalent_strong_baselines_complete=False,reason='Released vanilla component fails conv1d rank; condition replacement also shortens prompt for COND>1. Existing repaired prediction semantics are explicitly different. FedMVP original single-label algorithm requires disclosed multilabel adaptation.'))

if __name__=='__main__':
    torch.set_num_threads(2)
    with threadpool_limits(limits=2):main()
