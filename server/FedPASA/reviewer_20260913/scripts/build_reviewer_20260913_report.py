"""Produce auditable long-form tables and plots from completed artifacts only."""
import csv,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/reviewer_20260913'
def read(p):return json.loads(p.read_text())
def table(name,rows):
    if not rows:return
    (OUT/(name+'.json')).write_text(json.dumps(rows,indent=2)+'\n')
    with (OUT/(name+'.csv')).open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)
def main():
    OUT.mkdir(parents=True,exist_ok=True);split=[];baseline=[];scaling=[];eff=[];pack=[];memory=[];sensitivity=[]
    for p in sorted((OUT/'audit').glob('*/result.json')):split+=read(p)['rows']
    for p in sorted((OUT/'linear').glob('*/result.json')):
        r=read(p);baseline+=r['rows'];eff+=[v for v in r['rows'] if 'fit_wall_seconds' in v];pack+=[dict(dataset=p.parent.name,**v) for v in r['packing']]
    for p in sorted((OUT/'fedavg').glob('*/result.json')):baseline+=read(p)['rows'];eff+=read(p)['rows']
    for p in sorted((OUT/'qa').glob('*/*/result.json')):
        r=read(p)
        if p.parent.name=='m10_seed7':
            baseline+=r['rows'];q=r['qa_profile'];messages=q['messages'];total=sum(v['bytes'] for v in messages);packed=sum(v['packed_bytes'] for v in messages)
            eff.append(dict(dataset=p.parent.parent.name,method='QAFedPASA',client_fitting_seconds=q['bank_build_seconds']+q['qa_total_fit_seconds']-q['server_solve_selection_seconds'],server_solve_seconds=q['server_solve_selection_seconds'],fit_wall_seconds=q['bank_build_seconds']+q['qa_total_fit_seconds'],uploads_per_client=4,uplink_bytes=total,packed_uplink_bytes=packed,average_upload_per_client_bytes=total/10,scope='cached CLIP features; full local Top-K build; actual FP64 moments/FP32 entropy'))
        else:scaling+=r['rows']
    for p in sorted((OUT/'full_runtime').glob('*/result.json')):
        r=read(p);eff.append(dict(dataset=p.parent.name,method='FedPASA-Full exact selection, 3 splits',total_wall_seconds=r['wall_seconds'],client_fitting_seconds=r.get('client_fitting_seconds'),server_solve_seconds=r.get('server_solve_seconds'),uplink_bytes=r['uplink_bytes'],packed_uplink_bytes=r['packed_uplink_bytes'],uploads_per_client=r['logical_upload_stages'],average_upload_per_client_bytes=r['uplink_bytes']/10,scope=r['scope']));baseline+=r['rows']
    for p in sorted((OUT/'memory').glob('*/*/result.json')):memory.append(read(p))
    for p in sorted((OUT/'sensitivity').glob('*/result.json')):sensitivity+=read(p)['rows']
    for name,rows in [('split_aggregation_audit',split),('controlled_baselines',baseline),('performance_scalability',scaling),('efficiency',eff),('symmetric_packing',pack),('isolated_client_memory',memory),('hyperparameter_sensitivity',sensitivity)]:table(name,rows)
    methods=['Centralized Ridge','Local Ridge + Avg','FedAvg Linear Head','Fed3R-style pooled ridge','Local Top-K + Union + GDA','Naive local GDA averaging','Centralized PIAA-style GDA','QAFedPASA','Three-view pooled Ridge','FedPASA-Full (locked seed17)','FedPASA-Full (locked 3-split rule)','FedPASA-Full exact runtime rerun (3 splits)']
    wide=[]
    for name in methods:
        values={r['dataset']:r['mAP'] for r in baseline if r['method']==name}
        wide.append(dict(method=name,**{d:values.get(d) for d in ('voc07','voc12','coco','nus')}))
    table('table_IV_controlled_comparison',wide)
    # Historical prompt-adaptation runs are separate and fully attributed.
    old=ROOT.parent/'icme2027_workspace/outputs/icme2027/systems_and_repeats.json'
    if old.exists():table('historical_prompt_adaptation_efficiency',read(old)['rows'])
    (OUT/'figures').mkdir(exist_ok=True)
    for d in ('coco','nus'):
        methods=['Local Top-K + Union + GDA','Three-view Ridge','QAFedPASA','FedPASA-Full (locked 3-split rule)']
        fig,ax=plt.subplots(figsize=(6.2,4))
        anyrows=False
        for method in methods:
            rows=sorted([r for r in scaling if r['dataset']==d and r['method']==method and r.get('beta') is None],key=lambda r:r['clients'])
            if rows:ax.plot([r['clients'] for r in rows],[r['mAP'] for r in rows],'-o',label=method);anyrows=True
        if anyrows:
            ax.set(xlabel='Number of clients',ylabel='mAP (%)',title=d.upper());ax.legend(fontsize=7);fig.tight_layout()
            for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{d}_scalability.{ext}',dpi=180)
        plt.close(fig)
    if sensitivity:
        fig,axes=plt.subplots(1,2,figsize=(9,3.6))
        for ax,param in zip(axes,['lambda','eta']):
            rows=[r for r in sensitivity if r['parameter']==param];ax.plot([r['value'] for r in rows],[r['mAP'] for r in rows],'-o');ax.set(xlabel=param,ylabel='COCO mAP (%)')
            if param=='lambda':ax.set_xscale('log')
        fig.tight_layout()
        for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'coco_sensitivity.{ext}',dpi=180)
        plt.close(fig)
    q=read(OUT/'queue.json') if (OUT/'queue.json').exists() else {'jobs':[],'status':'PREPARING'};done=sum(r['status']=='COMPLETE' for r in q['jobs'])
    lines=['# Reviewer experiments — 2026-09-13','',f"Queue: {q['status']}; {done}/{len(q['jobs'])} jobs complete.",'',
        '- Controlled baseline: 10 clients, original KMeans partition seed7, same frozen CLIP cache; training split seed17 fixed in advance. Feature representation and solver family are explicit in the table.',
        '- Split audit uses all three historical prediction arrays and independently recomputes each mAP, mean single-split mAP and score-ensemble mAP. No test-based headline change.',
        '- Scaling uses original t=10/20/40/60/80/100% partitions, seed17; full-population local Top-K rebuilt at every point. Dirichlet beta=0.1/0.3/0.5 uses 10 clients, seed7.',
        '- Full curves use locked historical calibration weights, plus actual newly fitted federated QA/Ridge; Full runtime reruns reselect from training-only validation and are labeled separately.',
        '- Exact Full runtime includes all three training splits and full refits. It explicitly pools validation score vectors and labels for exact AP; this is NOT private statistic-only calibration.',
        '- Symmetric packing is an actual round-trip check for ridge Gram matrices. Both FP64 and FP32 score errors are reported. QA packet packing sizes follow recorded tensor shapes.',
        '- Memory is measured in a fresh largest-client subprocess; process RSS and CUDA memory are distinct. Single-server timings do not measure physical network latency.',
        '- Historical prompt-adaptation timings are separately attributed, not silently mixed with these new runs.',
        '- Existing tie-boundary evidence is retained as prior P2 evidence; no new artificial tie-rate experiment was scheduled.',
        '', '## Files','', 'controlled_baselines.csv, split_aggregation_audit.csv, performance_scalability.csv, efficiency.csv, symmetric_packing.csv, isolated_client_memory.csv, hyperparameter_sensitivity.csv, figures/, queue.json, logs/.','',
        'Missing/failed jobs remain visible in queue.json; no placeholder or literature number is substituted for a missing controlled result.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
