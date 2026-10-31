"""Read-only forced-command SSH entry point; ignores requested remote commands."""
import collections
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/supplement_20261029'


def main():
    state=json.loads((OUT/'queue.json').read_text()) if (OUT/'queue.json').exists() else {'status':'NOT_STARTED','jobs':[]}
    counts=collections.Counter(j['status'] for j in state['jobs'])
    active=state.get('active')
    live=False
    if state.get('pid'):
        try:
            cmd=Path(f"/proc/{state['pid']}/cmdline").read_bytes()
            live=b'queue_supplement_20261029.py' in cmd
        except FileNotFoundError:pass
    payload=dict(status=state.get('status'),queue_process_alive=live,counts=dict(counts),total=len(state['jobs']),
                 active=active,heartbeat_age_seconds=time.time()-state.get('updated',time.time()),
                 free_disk_GiB=shutil.disk_usage(OUT if OUT.exists() else ROOT).free/2**30,
                 now=time.time())
    active_ids=active if isinstance(active,list) else [active] if active else []
    payload['active_jobs']=[]
    for job_id in active_ids:
        entry={'id':job_id}
        p=OUT/'logs'/f'{job_id}.log'
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-2500));entry['log_tail']=f.read().decode('utf-8',errors='replace')
            entry['log_age_seconds']=time.time()-p.stat().st_mtime
        payload['active_jobs'].append(entry)
    payload['max_workers']=state.get('max_workers',1)
    payload['resources']=state.get('resources')
    summary=OUT/'summary.json'
    if summary.exists():payload['summary']=json.loads(summary.read_text())
    try:payload['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu,temperature.gpu','--format=csv'],timeout=15,text=True)
    except Exception as e:payload['gpu_error']=str(e)
    print(json.dumps(payload,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
