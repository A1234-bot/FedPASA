"""Read completed results, independently recompute metrics, and export tables."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_priority_full_impl as r
from run_priority_20260913 import RUNS
BASE=r.ROOT/'outputs/priority_20260913';FINAL=BASE/'final'

def table(name,rows):
    r.save(FINAL/(name+'.json'),rows)
    keys=list(dict.fromkeys(k for v in rows for k in v))
    with (FINAL/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def score(p,d):
    s=np.load(p);y=r.targets(d);assert s.shape==y.shape and np.isfinite(s).all()
    return s,r.metric(y,s)[0]

def main():
    pending_backbone='--allow-pending-backbone' in sys.argv
    FINAL.mkdir(parents=True,exist_ok=True);details=[];summary=[];fusion=[];trade=[];views=[];backbones=[];verification=[]
    for q in ('queue.json','extras_queue.json'):
        v=r.read(BASE/q)
        assert all(j['status']=='COMPLETE' or (pending_backbone and j.get('name')=='backbone') for j in v['jobs']),v
    for d in ('coco','nus'):
        for i,(_,seeds,_) in enumerate(RUNS):
            root=BASE/'outer'/d/f'run{i+1}';p=root/'full_runtime'/d
            rep=r.read(p/'result.json');assert rep['status']=='COMPLETE'
            for method,name in [('QAFedPASA','qa_scores.npy'),('3-view Ridge',f'seed{seeds[0]}_ridge_scores.npy'),('FedPASA-Full','test_scores.npy')]:
                s,v=score(p/name,d);details.append(dict(dataset=d,run=i+1,method=method,mAP=v,partition_seed=RUNS[i][0],split_seeds=list(seeds),client_order_seed=RUNS[i][2]))
                if method=='FedPASA-Full':assert abs(v-rep['rows'][-1]['mAP'])<1e-9
            # Verify three independently generated holdouts in every outer run.
            for seed in seeds:
                fit=np.load(root/'splits'/f'seed{seed}'/'ridge/fit_indices.npy');val=np.load(root/'splits'/f'seed{seed}'/'ridge/validation_indices.npy')
                assert not np.intersect1d(fit,val).size
                np.testing.assert_array_equal(np.sort(np.r_[fit,val]),np.arange(len(r.targets(d,'train'))))
            parts=[np.load(p/f'seed{s}_test_scores.npy') for s in seeds]
            np.testing.assert_array_equal(np.mean(parts,axis=0),np.load(p/'test_scores.npy'))
        for method in ('QAFedPASA','3-view Ridge','FedPASA-Full'):
            vals=[v['mAP'] for v in details if v['dataset']==d and v['method']==method]
            summary.append(dict(dataset=d,method=method,n=3,mean_mAP=float(np.mean(vals)),std_mAP=float(np.std(vals,ddof=1)),ddof=1,values=vals))
        p=BASE/'controlled/full_runtime'/d;rep=r.read(p/'result.json');y=r.targets(d)
        for method,name in [('QA only','qa_scores.npy'),('Ridge only','seed7_ridge_scores.npy'),('Fixed 0.5 fusion','seed7_fixed_scores.npy'),('Training-selected global fusion','seed7_global_scores.npy'),('Classwise fusion 1 split','seed7_test_scores.npy'),('Classwise fusion 3 splits','test_scores.npy')]:
            _,value=score(p/name,d);fusion.append(dict(dataset=d,method=method,mAP=value,split='7/17/27' if '3 splits' in method else '7',test_selected=False))
        messages=rep['messages'];shared=[m for m in messages if not m['stage'].startswith('seed')]
        qa=[m for m in shared if m['stage']!='ridge_full_statistics'];ridge=[m for m in shared if m['stage']=='ridge_full_statistics']
        _,qm=score(p/'qa_scores.npy',d);_,rm=score(p/'seed7_ridge_scores.npy',d)
        trade.extend([dict(dataset=d,method='QAFedPASA',mAP=qm,uplink_MiB=sum(m['bytes'] for m in qa)/2**20,stages=4,scope='full label-free QA'),dict(dataset=d,method='Ridge locked penalty',mAP=rm,uplink_MiB=sum(m['bytes'] for m in ridge)/2**20,stages=1,scope='final fit only; penalty already fixed; selection excluded')])
        rv=len(np.load(r.sr(d,7)/'ridge/validation_indices.npy'))*y.shape[1]*(len(r.ALPHAS)*4+1)
        ridfit=next(m for m in messages if m['stage']=='seed7_ridge_fit_statistics')
        trade.append(dict(dataset=d,method='Ridge training-selected',mAP=rm,uplink_MiB=(sum(m['bytes'] for m in ridge)+ridfit['bytes']+rv)/2**20,stages=3,scope='full and fit moments plus pooled 7-candidate validation scores and labels'))
        for n in (1,2,3):
            seeds=(7,17,27)[:n];ss=np.mean([np.load(p/f'seed{s}_test_scores.npy') for s in seeds],axis=0);v=r.metric(y,ss)[0]
            ms=shared+[m for m in messages if any(m['stage'].startswith(f'seed{s}_') for s in seeds)]
            trade.append(dict(dataset=d,method=f'Full-{n}split',mAP=v,uplink_MiB=sum(m['bytes'] for m in ms)/2**20,stages=5+6*n,scope='complete exact selection; full QA and ridge statistics shared'))
            if n==3:
                assert abs(v-rep['rows'][-1]['mAP'])<1e-9
                pack=r.read(BASE/'packed'/d/'result.json');assert pack['status']=='COMPLETE' and pack['max_abs_full_score_error']==0
                trade.append(dict(dataset=d,method='Full-3split packed',mAP=pack['mAP'],uplink_MiB=pack['packed_uplink_bytes']/2**20,stages=23,scope=pack['scope']))
        for p in sorted((BASE/'views'/d).glob('*/result.json')):
            v=r.read(p);_,m=score(p.parent/'test_scores.npy',d);assert abs(m-v['test']['mAP'])<1e-9
            views.append(dict(dataset=d,features='+'.join(v['protocol']['features']),mAP=m,alpha=v['selection']['alpha'],split_seed=17))
        for method,name in [('QA','qa_scores.npy'),('Ridge','seed7_ridge_scores.npy'),('Full','test_scores.npy')]:
            _,m=score(BASE/'controlled/full_runtime'/d/name,d)
            if d=='coco':backbones.append(dict(backbone='ViT-B/16',dataset=d,method=method,mAP=m,split='7/17/27' if method=='Full' else '7'))
    if not pending_backbone:
        p=BASE/'backbone/evaluation/full_runtime/coco'
        for method,name in [('QA','qa_scores.npy'),('Ridge','seed7_ridge_scores.npy'),('Full','test_scores.npy')]:
            _,m=score(p/name,'coco');backbones.append(dict(backbone='ViT-B/32',dataset='coco',method=method,mAP=m,split='7/17/27' if method=='Full' else '7'))
    for name,rows in [('outer_runs',details),('outer_summary',summary),('fusion_ablation',fusion),('accuracy_communication',trade),('view_ablation',views),('backbone',backbones)]:table(name,rows)
    ties=[]
    for d in ('coco','nus'):
        v=r.read(BASE/'ties'/d/'result.json');assert v['status']=='COMPLETE';ties+=v['rows']
    table('ties',ties)
    baseline=r.read(BASE/'baseline_audit/archived_predictions_rechecked.json');table('baselines_reverified',baseline['rows'])
    if not pending_backbone:r.save(FINAL/'backbone_matching.json',r.read(BASE/'backbone/matching.json'))
    else:r.save(FINAL/'backbone_pending.json',dict(status='PAUSED_GPU_UNAVAILABLE',reason='User switched server to GPU-free mode because GPU is occupied',complete_train_images=82081,verified_test_checkpoint_images=r.read(BASE/'backbone/cache/coco/test_dual_progress.json')['offset'],required_test_images=40137,partial_files_are_not_final_features=True,resume_command='python scripts/run_priority_backbone.py',followup_audit='python scripts/run_priority_backbone.py --evaluate-only'))
    r.save(FINAL/'baseline_source_status.json',r.read(BASE/'baseline_audit/current_source_audit.json'))
    table('full_packed',[dict(dataset=d,**r.read(BASE/'packed'/d/'result.json')) for d in ('coco','nus')])
    # A source/runability problem must not be hidden by completed numerical jobs.
    r.save(FINAL/'completion.json',dict(numerical_jobs='PARTIAL_GPU_UNAVAILABLE' if pending_backbone else 'COMPLETE',two_author_equivalent_strong_baselines='NOT_COMPLETE_SOURCE_BLOCKED',outer_runs=6,view_fits=10,controlled_full_runs=2,additional_backbones=0 if pending_backbone else 1,tie_rows=len(ties),metric_recalculation='PASS for completed experiments',split_disjointness='PASS',ensemble_reconstruction='PASS',client_sampling='full participation; randomized processing order, no participant-subset randomness'))
    figdir=FINAL/'figures';figdir.mkdir(exist_ok=True)
    for axis,unit in [('uplink_MiB','Uplink (MiB)'),('stages','Logical upload stages')]:
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        for ax,d in zip(axes,('coco','nus')):
            for row in trade:
                if row['dataset']!=d:continue
                marker='D' if 'packed' in row['method'] else 'o'
                ax.scatter(row[axis],row['mAP'],marker=marker,s=45,label=row['method'])
            ax.set(title=d.upper(),xlabel=unit,ylabel='mAP (%)');ax.grid(alpha=.2)
        axes[1].legend(fontsize=7,loc='lower right');fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(figdir/f'accuracy_{axis}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    for p in sorted(BASE.rglob('*')):
        if not p.is_file() or 'final' in p.relative_to(BASE).parts:continue
        is_result=p.suffix=='.json'
        is_prediction=p.suffix=='.npy' and ('scores' in p.name or 'indices' in p.name or 'partitions' in p.parts)
        if is_result or is_prediction:verification.append(dict(path=str(p.relative_to(BASE)),sha256=r.digest(p),bytes=p.stat().st_size))
    for p in sorted((r.ROOT/'scripts').glob('*priority*py')):
        verification.append(dict(path=str(p.relative_to(r.ROOT)),sha256=r.digest(p),bytes=p.stat().st_size))
    inputs=[]
    for d in ('coco','nus'):
        for name in ('train_ids.txt','test_ids.txt','train_targets.npy','test_targets.npy','text_features.npy','partition_features.npy','manifest.json'):
            p=r.resolve_cache(d)/name;inputs.append(dict(dataset=d,path=str(p),sha256=r.digest(p),bytes=p.stat().st_size))
    r.save(FINAL/'input_fingerprints.json',inputs)
    r.save(FINAL/'provenance.json',verification)
    print(json.dumps(summary,indent=2));print('REPORT COMPLETE')

if __name__=='__main__':main()
