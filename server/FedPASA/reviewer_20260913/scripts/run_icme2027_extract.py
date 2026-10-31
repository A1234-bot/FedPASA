"""Measure cold (no feature cache) CLIP extraction on the experiment server."""
import argparse,subprocess,sys
from time import perf_counter
import numpy as np
from icme2027_common import *

def run(d):
    out=OUT/'extraction'/d;out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():return
    m=read(cache(d)/'manifest.json');start=perf_counter()
    cmd=[sys.executable,str(ROOT/'scripts/run_icme2027_extract_worker.py'),'--dataset',d,
         '--train-root',m['train_root'],'--test-root',m['test_root'],'--weight-path',m['weight_path'],
         '--output-root',str(OUT/'server_cache'),'--batch-size','64','--num-workers','4']
    subprocess.run(cmd,check=True)
    generated=OUT/'server_cache'/d;report=read(generated/'manifest.json')
    diffs={}
    for split in ('train','test'):
        assert (generated/f'{split}_ids.txt').read_text()==(cache(d)/f'{split}_ids.txt').read_text()
        np.testing.assert_array_equal(np.load(generated/f'{split}_targets.npy'),np.load(cache(d)/f'{split}_targets.npy'))
        a=np.load(generated/f'{split}_cls.npy',mmap_mode='r');b=np.load(cache(d)/f'{split}_cls.npy',mmap_mode='r')
        diffs[split]=max(float(np.max(np.abs(a[i:i+1024].astype(np.float32)-b[i:i+1024].astype(np.float32)))) for i in range(0,len(a),1024))
        link=cache(d)/f'{split}_patch.npy'
        if not link.exists():link.symlink_to(generated/f'{split}_patch.npy')
    save(out/'result.json',{'status':'COMPLETE','dataset':d,'cold_definition':'No precomputed feature cache; OS image page cache was not forcibly evicted',
        'seconds_wall':perf_counter()-start,'artifacts':report['artifacts'],'server_vs_archived_cls_max_abs_difference':diffs,
        'image_ids_and_labels_match':True,'checkpoint_sha256':report['weight_sha256'],'completed':stamp()})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);run(p.parse_args().dataset)
