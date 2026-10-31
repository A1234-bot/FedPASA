"""Fingerprint actual data provenance and source snapshot before external training."""
from fair_baselines_common import *

def main():
    root=OUT/'protocol';records=[]
    for d in ('coco','nus'):
        freeze_inputs(d);c=cache(d);original=Path('/root/autodl-tmp/FedPASA/icme2027_workspace/outputs/icme/cache')/d
        for name in ('train_ids.txt','test_ids.txt','train_targets.npy','test_targets.npy'):
            assert digest(c/name)==digest(original/name),(d,name,'historical image/label protocol changed')
        m=read(original/'manifest.json');assert digest(m['weight_path'])==m['weight_sha256']
        files=[c/f'{part}_{n}.npy' for part in ('train','test') for n in ('cls','patch_mean','patch_max_text')]
        record={'dataset':d,'features':{str(p.relative_to(ROOT)):digest(p) for p in files},'weight_sha256':m['weight_sha256'],'image_roots':{part:m[part+'_root'] for part in ('train','test')},'matching_original_ids_and_labels':True}
        records.append(record)
    dest=root/'feature_provenance.json'
    if dest.exists():assert read(dest)['datasets']==records
    else:save(dest,{'datasets':records,'recorded':stamp()})
    codes=[p for p in (ROOT/'scripts').glob('*fair*.py')]
    save(root/'code_snapshot.json',{'files':{str(p.relative_to(ROOT)):digest(p) for p in codes},'recorded':stamp()})
    print('INPUT_PROVENANCE_PASS',flush=True)

if __name__=='__main__':main()
