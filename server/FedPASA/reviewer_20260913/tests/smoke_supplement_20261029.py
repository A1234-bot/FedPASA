"""Exercise the real CUDA fitting/fusion/control pipeline on synthetic cache data."""
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_supplement_20261029 as s

torch.set_num_threads(2)
rng=np.random.default_rng(134)
with tempfile.TemporaryDirectory(prefix='fedpasa_supplement_smoke_') as temp:
    root=Path(temp);cache=root/'cache';cache.mkdir()
    text=rng.normal(size=(3,8)).astype('float32');text/=np.linalg.norm(text,axis=1,keepdims=True)
    np.save(cache/'text_features.npy',text)
    for part,n in [('train',150),('test',30)]:
        y=(rng.random((n,3))>.5).astype('uint8')
        np.save(cache/f'{part}_targets.npy',y)
        patch=rng.normal(size=(n,16,8)).astype('float32')
        np.save(cache/f'{part}_patch.npy',patch)
        for feature in ('cls','patch_mean','patch_max_text'):
            np.save(cache/f'{part}_{feature}.npy',rng.normal(size=(n,8)).astype('float32'))
    s.BASE=root/'output'
    s.r.resolve_cache=lambda d:cache
    s.run('coco','iid',37,10,True)
    output=s.BASE/'runs/coco_iid_m10_s37'
    result=json.loads((output/'result.json').read_text())
    assert result['status']=='COMPLETE' and len(result['rows'])==10
    assert not list((output/'temporary_banks').rglob('*.pt'))
    full3=np.load(output/'Full_3_scores.npy')
    np.testing.assert_array_equal(full3,np.mean([np.load(output/f'split{x}_full_scores.npy') for x in (3707,3717,3727)],axis=0))
    # Reconstruct from committed stages with the final receipt deliberately absent.
    (output/'result.json').rename(output/'original_result.json')
    original=s.r.core.build_client_banks
    def forbidden(*args,**kwargs):raise AssertionError('Committed banks/adapters must not be recomputed')
    s.r.core.build_client_banks=forbidden
    s.run('coco','iid',37,10,True)
    s.r.core.build_client_banks=original
    np.testing.assert_array_equal(full3,np.load(output/'Full_3_scores.npy'))
    assert json.loads((output/'result.json').read_text())['checkpoint_stages_reused']>30
    import supplement_linear_20261029 as linear
    model,res=linear.fit(rng.normal(size=(100,24)).astype('float32'),(rng.random((100,3))>.5).astype('float32'),np.arange(100)%10,
                        dict(optimizer='sgd',lr=.1,weight_decay=0.),'fedavg',37,2,root/'linear')
    assert res['status']=='COMPLETE' and res['rounds']==2
    # Simulate termination immediately after the durable round-3 commit.
    xx=rng.normal(size=(100,24)).astype('float32');yy=(rng.random((100,3))>.5).astype('float32');aa=np.arange(100)%10
    cfg=dict(optimizer='sgd',lr=.1,weight_decay=0.)
    expected,_=linear.fit(xx,yy,aa,cfg,'fedavg',37,6,root/'uninterrupted')
    original_save=linear.atomic_torch
    def crash(path,obj):
        original_save(path,obj)
        if path.name=='checkpoint.pt' and obj.get('round')==3:raise RuntimeError('simulated interruption')
    linear.atomic_torch=crash
    try:linear.fit(xx,yy,aa,cfg,'fedavg',37,6,root/'interrupted')
    except RuntimeError as error:assert str(error)=='simulated interruption'
    else:raise AssertionError('interruption not triggered')
    linear.atomic_torch=original_save
    assert torch.load(root/'interrupted/checkpoint.pt',weights_only=False)['round']==3
    actual,_=linear.fit(xx,yy,aa,cfg,'fedavg',37,6,root/'interrupted')
    for k in expected:torch.testing.assert_close(expected[k],actual[k],rtol=0,atol=0)
    import queue_supplement_20261029 as queue
    queue.OUT=s.BASE;queue.report()
    assert json.loads((s.BASE/'summary.json').read_text())['complete_runs']==1
print('CUDA_END_TO_END_SMOKE_PASS')
