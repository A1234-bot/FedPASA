"""Fresh QA/Ridge Full pipeline timings with test labels inaccessible."""
import argparse,gc
import numpy as np
import torch
from fair_baselines_common import *
import run_priority_full_impl as qa
from run_fair_ridge import GRID

def run(d):
    dest=OUT/'full_cost'/d/'full_runtime'/d
    done=dest/'fair_verification.json'
    if done.exists():return dest
    original_targets=qa.targets
    def train_only_targets(dataset,split='test'):
        if split=='train':return original_targets(dataset,split)
        shape=(np.load(cache(dataset)/'test_cls.npy',mmap_mode='r').shape[0],len(qa.BENCHMARKS[dataset].classes))
        return np.zeros(shape,dtype=np.uint8)
    def no_test_metric(dataset,name,scores,**kw):
        assert np.isfinite(scores).all()
        return {'dataset':dataset,'method':name,'shape':list(scores.shape),'test_metric_deferred':True,**kw}
    qa.OUT=OUT/'full_cost'/d;qa.ALPHAS=GRID;qa.SEEDS=(7,17,27);qa.targets=train_only_targets;qa.measured=no_test_metric
    torch.set_num_threads(4);torch.cuda.reset_peak_memory_stats()
    if not (dest/'result.json').exists():qa.full_runtime(d)
    r=read(dest/'result.json');s=read(dest/'seed17_selection.json');sel=read(OUT/'ridge'/d/'selection.json')
    assert s['alpha']==sel['selected']['pooled']['alpha'];np.testing.assert_array_equal(s['classwise_weights'],sel['classwise_weights'])
    reference=np.load(OUT/'ridge'/d/'full_single_test_scores.npy');fresh=np.load(dest/'seed17_test_scores.npy')
    error=float(np.max(np.abs(reference-fresh)));np.testing.assert_allclose(reference,fresh,atol=1e-4,rtol=1e-5)
    prefix=lambda stage:stage.startswith('seed')
    shared=[m for m in r['messages'] if not prefix(m['stage'])]
    single=shared+[m for m in r['messages'] if m['stage'].startswith('seed17_')]
    cost={'status':'COMPLETE','same_selection':True,'single_score_max_abs_error':error,'single_split_uplink_bytes':sum(m['bytes'] for m in single),'single_split_packed_uplink_bytes':sum(m['packed_bytes'] for m in single),'three_split_uplink_bytes':r['uplink_bytes'],'three_split_packed_uplink_bytes':r['packed_uplink_bytes'],'single_stages':11,'three_stages':23,'three_split_pipeline_seconds':r['wall_seconds'],'scope':'fresh local banks, fit, all selections, refit and test inference on cached features; actual stage uplink payload; no image feature extraction, no measured downlink or network latency','test_labels_read':False,'completed':stamp()}
    save(done,cost);return dest

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['coco','nus'],required=True);a=p.parse_args();run(a.dataset)
