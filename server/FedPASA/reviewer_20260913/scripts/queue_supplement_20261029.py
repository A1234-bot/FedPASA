"""Resource-gated concurrent queue; measured systems jobs remain exclusive."""
import argparse
import collections
import csv
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/supplement_20261029'


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def jobs():
    entries = []
    def add(name, script, args, **kw):
        entries.append(dict(id=name, script=script, args=args, status='PENDING', **kw))
    add('evidence_audit', 'audit_supplement_20261029.py', [])
    for seed in (37, 47, 57, 67, 77):
        for d in ('coco', 'nus'):
            args = ['--dataset', d, '--condition', 'iid', '--seed', str(seed)]
            add(f'{d}_iid_s{seed}', 'run_supplement_20261029.py', args + ['--union'])
            add(f'{d}_iid_s{seed}_fedavg', 'run_supplement_20261029.py', args + ['--fedavg'])
    for condition in ('beta1.0', 'beta0.5', 'beta0.3', 'beta0.1', 'imbalance'):
        for seed in (37, 47, 57):
            for d in ('coco', 'nus'):
                args = ['--dataset', d, '--condition', condition, '--seed', str(seed)]
                add(f'{d}_{condition}_s{seed}', 'run_supplement_20261029.py', args)
                add(f'{d}_{condition}_s{seed}_fedavg', 'run_supplement_20261029.py', args + ['--fedavg'])
    for m in (20, 40, 80):
        for d in ('coco', 'nus'):
            add(f'{d}_scale_m{m}', 'run_supplement_20261029.py',
                ['--dataset', d, '--condition', 'iid', '--seed', '37', '--clients', str(m)])
    for d in ('coco', 'nus'):
        for splits in (1, 3):
            for name, mbps, rtt in (('loopback', 0, 0), ('100Mbps_10ms', 100, 10), ('10Mbps_50ms', 10, 50)):
                add(f'{d}_tcp_full{splits}_{name}', 'tcp_supplement_20261029.py', ['--dataset', d],
                    env={'SUPPLEMENT_SPLITS': str(splits), 'SUPPLEMENT_MBIT': str(mbps),
                         'SUPPLEMENT_RTT_MS': str(rtt), 'SUPPLEMENT_NETWORK': name})
    return entries


def report():
    import numpy as np
    from scipy.stats import t
    rows = []
    groups = collections.defaultdict(list)
    for path in sorted((OUT / 'runs').glob('*/result.json')):
        value = json.loads(path.read_text())
        p = value['protocol']
        methods = {row['method']: row for row in value['rows']}
        fpath = path.parent / 'fedavg/result.json'
        if fpath.exists():
            f = json.loads(fpath.read_text())
            methods['FedAvg'] = dict(method='FedAvg', mAP=f['mAP'], class_AP=f['class_AP'])
        for name, row in methods.items():
            entry = dict(dataset=p['dataset'], condition=p['condition'], clients=p['clients'],
                         seed=p['outer_seed'], method=name, mAP=row['mAP'],
                         delta_pooled=row['mAP']-methods['Pooled_Ridge']['mAP'],
                         delta_fedavg=row['mAP']-methods['FedAvg']['mAP'] if 'FedAvg' in methods else None)
            rows.append(entry)
            groups[(p['dataset'], p['condition'], p['clients'], name)].append(entry)
        ap = np.asarray(methods['Full_1']['class_AP'])-np.asarray(methods['Pooled_Ridge']['class_AP'])
        save(path.parent / 'classwise_delta.json', dict(full1_minus_pooled=ap.tolist(),
             improved=int((ap>0).sum()), degraded=int((ap<0).sum()), equal=int((ap==0).sum())))
    if rows:
        with (OUT / 'results.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = []
    for (d, condition, clients, method), values in groups.items():
        means = np.array([v['mAP'] for v in values])
        diff = np.array([v['delta_pooled'] for v in values])
        n = len(values)
        half = float(t.ppf(.975, n-1)*diff.std(ddof=1)/np.sqrt(n)) if n>1 else None
        summary.append(dict(dataset=d, condition=condition, clients=clients, method=method,
             n=n, mean_mAP=float(means.mean()), sample_std=float(means.std(ddof=1)) if n>1 else None,
             paired_gain=float(diff.mean()), paired_gain_CI95=[float(diff.mean()-half), float(diff.mean()+half)] if n>1 else None,
             positive_partitions=int((diff>0).sum()),
             uncertainty='two-sided Student-t interval over independent outer partitions; preliminary until planned count reached'))
    save(OUT / 'summary.json', dict(rows=summary, complete_runs=len(list((OUT/'runs').glob('*/result.json'))), updated=time.time()))


def exclusive(job):
    return '_tcp_' in job['id'] or '_scale_' in job['id'] or job['id']=='evidence_audit'


def eligible(job,state,active):
    if job['status']!='PENDING':return False
    if job['id']!='evidence_audit' and state['jobs'][0]['status']!='COMPLETE':return False
    if job['id'].endswith('_fedavg'):
        parent=job['id'][:-7]
        if not any(j['id']==parent and j['status']=='COMPLETE' for j in state['jobs']):return False
    if exclusive(job):
        index=state['jobs'].index(job)
        return not active and all(j['status']=='COMPLETE' for j in state['jobs'][:index])
    return not any(exclusive(j) for j in active)


def alive(job):
    import psutil
    try:
        p=psutil.Process(job['pid'])
        return (p.status()!=psutil.STATUS_ZOMBIE and
                abs(p.create_time()-job['process_created'])<.1 and
                any('execute_supplement_job.py' in v for v in p.cmdline()))
    except (psutil.NoSuchProcess,psutil.AccessDenied,KeyError):return False


def resources():
    import psutil
    gpu_free=None;gpu_util=None
    try:
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        gpu_free,gpu_util=map(float,raw.strip().splitlines()[0].split(','))
    except Exception:pass
    return dict(available_RAM_GiB=psutil.virtual_memory().available/2**30,
                free_disk_GiB=shutil.disk_usage(OUT).free/2**30,
                free_GPU_MiB=gpu_free,GPU_util_percent=gpu_util)


def admit(job,active,resource):
    # Reserve space for resident arrays and transient client moment matrices.
    import psutil
    outstanding=0
    for j in active:
        try:
            p=psutil.Process(j['pid']);rss=sum(x.memory_info().rss for x in [p,*p.children(recursive=True)])/2**30
        except (psutil.NoSuchProcess,psutil.AccessDenied):rss=0
        outstanding+=max(0,10-rss)
    return (resource['free_disk_GiB']>=60 and
            resource['available_RAM_GiB']-outstanding>=18 and
            resource['free_GPU_MiB'] is not None and resource['free_GPU_MiB']>=4096)


def main(workers=4):
    import psutil
    OUT.mkdir(parents=True,exist_ok=True)
    for folder in ('logs','attempts','resource_peaks'):(OUT/folder).mkdir(exist_ok=True)
    with (OUT/'queue.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path=OUT/'queue.json'
        state=json.loads(path.read_text()) if path.exists() else dict(jobs=jobs(),created=time.time())
        for j in state['jobs']:
            if j['status']=='STARTING':
                matches=[]
                for p in psutil.process_iter(['pid','cmdline','create_time']):
                    cmd=p.info['cmdline'] or []
                    if j['receipt'] in cmd and any('execute_supplement_job.py' in x for x in cmd):matches.append(p)
                if matches:
                    assert len(matches)==1
                    j.update(status='RUNNING',pid=matches[0].pid,process_created=matches[0].create_time())
                elif Path(j['receipt']).exists():
                    outcome=json.loads(Path(j['receipt']).read_text())
                    j.update(status='COMPLETE' if outcome['exit_code']==0 else 'FAILED',exit_code=outcome['exit_code'])
                else:j['status']='PENDING'
        if any(j['status']=='RUNNING' and not j.get('receipt') for j in state['jobs']):
            raise RuntimeError('Legacy job still marked RUNNING: drain old scheduler first')
        state.update(pid=os.getpid(),status='RUNNING',started=time.time(),max_workers=workers,
                     execution_policy='concurrent accuracy jobs; exclusive scaling and TCP timings')
        save(path,state)
        children={}
        while True:
            changed=False
            active=[j for j in state['jobs'] if j['status']=='RUNNING']
            for j in active:
                p=children.get(j['id'])
                if p is not None:p.poll() # Reap owned wrappers before checking process state.
                receipt=Path(j['receipt'])
                if receipt.exists() and not alive(j):
                    outcome=json.loads(receipt.read_text())
                    assert outcome['attempt']==j['attempt'] and outcome['job_id']==j['id']
                    j.update(status='COMPLETE' if outcome['exit_code']==0 else 'FAILED',
                             exit_code=outcome['exit_code'],finished=outcome['finished'])
                    changed=True
                elif not receipt.exists() and not alive(j):
                    j['interruptions']=j.get('interruptions',0)+1
                    j.update(status='PENDING' if j['interruptions']<=3 else 'FAILED',
                             last_error='worker disappeared without exit receipt; resume committed checkpoints')
                    changed=True
            if changed:
                save(path,state);report()
            active=[j for j in state['jobs'] if j['status']=='RUNNING']
            failing=any(j['status']=='FAILED' for j in state['jobs'])
            paused=(OUT/'PAUSE').exists()
            resource=resources()
            state.update(updated=time.time(),resources=resource,active=[j['id'] for j in active])
            if not active and (paused or failing or all(j['status']=='COMPLETE' for j in state['jobs'])):
                state.update(status='PAUSED' if paused else 'FAILED' if failing else 'COMPLETE',finished=time.time())
                save(path,state);report();return
            # Start at most one new worker per 10-second observation to see memory growth.
            if not paused and not failing and len(active)<workers:
                candidate=next((j for j in state['jobs'] if eligible(j,state,active)),None)
                if candidate is not None and admit(candidate,active,resource):
                    script=ROOT/'scripts'/candidate['script']
                    attempt=candidate.get('attempt',0)+1
                    receipt=OUT/'attempts'/f"{candidate['id']}.{attempt}.exit.json"
                    spec=OUT/'attempts'/f"{candidate['id']}.{attempt}.spec.json"
                    command=[sys.executable,'-u',str(script),*candidate['args']]
                    save(spec,dict(job_id=candidate['id'],attempt=attempt,command=command))
                    candidate.update(status='STARTING',attempt=attempt,receipt=str(receipt))
                    save(path,state)
                    env=os.environ.copy();env.update(candidate.get('env',{}))
                    env.update(SUPPLEMENT_WORKERS=str(1 if exclusive(candidate) else workers),
                               OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                    with (OUT/'logs'/(candidate['id']+'.log')).open('a',buffering=1) as log:
                        child=subprocess.Popen([sys.executable,'-u',str(ROOT/'scripts/execute_supplement_job.py'),str(receipt),str(spec)],
                            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    candidate.update(status='RUNNING',started=time.time(),pid=child.pid,
                        process_created=psutil.Process(child.pid).create_time(),attempt=attempt,receipt=str(receipt),
                        source_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),concurrency_limit=env['SUPPLEMENT_WORKERS'])
                    children[candidate['id']]=child
                    active.append(candidate);state['active']=[j['id'] for j in active];save(path,state)
            for j in active:
                peakfile=OUT/'resource_peaks'/f"{j['id']}.json"
                peaks=json.loads(peakfile.read_text()).get('peak_sampled_RSS_bytes',{}) if peakfile.exists() else {}
                try:
                    p=psutil.Process(j['pid'])
                    for q in [p,*p.children(recursive=True)]:
                        try:peaks[str(q.pid)]=max(peaks.get(str(q.pid),0),q.memory_info().rss)
                        except psutil.NoSuchProcess:pass
                except psutil.NoSuchProcess:pass
                save(peakfile,dict(peak_sampled_RSS_bytes=peaks,sampling_seconds=10,updated=time.time()))
            save(path,state)
            time.sleep(10)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report',action='store_true')
    parser.add_argument('--workers',type=int,default=4,choices=range(1,7));args=parser.parse_args()
    if args.report:report()
    else:main(args.workers)

