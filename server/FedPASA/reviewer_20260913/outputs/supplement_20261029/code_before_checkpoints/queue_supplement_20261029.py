"""Server-local single-GPU queue with OS lock, atomic state and fail-stop."""
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


def main():
    import psutil
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'logs').mkdir(exist_ok=True)
    with (OUT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = OUT / 'queue.json'
        state = json.loads(path.read_text()) if path.exists() else dict(jobs=jobs(), created=time.time())
        state.update(pid=os.getpid(), status='RUNNING', started=time.time())
        save(path, state)
        for job in state['jobs']:
            if job['status'] == 'COMPLETE':
                continue
            if (OUT / 'PAUSE').exists():
                state.update(status='PAUSED', updated=time.time());save(path, state);return
            if shutil.disk_usage(OUT).free < 40*2**30:
                state.update(status='BLOCKED_LOW_DISK', updated=time.time());save(path, state);return
            script = ROOT / 'scripts' / job['script']
            job.update(status='RUNNING', started=time.time(), source_sha256=hashlib.sha256(script.read_bytes()).hexdigest())
            state.update(active=job['id'], updated=time.time());save(path, state)
            env = os.environ.copy();env.update(job.get('env', {}))
            peaks = {}
            with (OUT / 'logs' / (job['id']+'.log')).open('a', buffering=1) as log:
                proc = subprocess.Popen([sys.executable, '-u', str(script), *job['args']],
                                        cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                job['pid'] = proc.pid;save(path, state)
                while proc.poll() is None:
                    try:
                        parent = psutil.Process(proc.pid)
                        procs = [parent, *parent.children(recursive=True)]
                        for p in procs:
                            try: peaks[str(p.pid)] = max(peaks.get(str(p.pid), 0), p.memory_info().rss)
                            except psutil.NoSuchProcess: pass
                    except psutil.NoSuchProcess:
                        pass
                    state.update(updated=time.time(), free_disk_GiB=shutil.disk_usage(OUT).free/2**30)
                    save(path, state)
                    save(OUT/'resource_peaks'/f"{job['id']}.json", dict(peak_sampled_RSS_bytes=peaks, sampling_seconds=15, updated=time.time()))
                    time.sleep(15)
            job.update(status='COMPLETE' if proc.returncode == 0 else 'FAILED',
                       exit_code=proc.returncode, finished=time.time())
            state['updated']=time.time();save(path, state)
            report()
            if proc.returncode:
                state['status']='FAILED';save(path, state);return
        state.update(status='COMPLETE', active=None, finished=time.time());save(path, state)
        report()


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--report', action='store_true');args=parser.parse_args()
    if args.report: report()
    else: main()
