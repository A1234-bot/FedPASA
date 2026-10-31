"""New supplements followed by user-reauthorized baseline continuation."""
import sys,subprocess,time
from fair_baselines_common import ROOT,read,save,stamp,digest
OUT=ROOT/'outputs/full_protocol_20260919'

def main():
    jobs=[]
    for d in ('voc07','voc12','coco','nus'):jobs.append(dict(id='calibration_'+d,script='run_full_calibration_protocol.py',args=['--dataset',d]))
    for d in ('voc07','voc12'):
        for i in range(3):jobs.append(dict(id=f'outer_{d}_{i+1}',script='run_voc_outer_20260919.py',args=['--dataset',d,'--index',str(i)]))
    for d in ('coco','nus'):jobs.append(dict(id='systems_'+d,script='run_full_tcp_systems.py',args=['--dataset',d]))
    jobs.append(dict(id='report',script='build_full_protocol_report.py',args=[]))
    jobs.append(dict(id='resume_fair_baselines',script='fair_baselines_queue.py',args=[]))
    path=OUT/'queue.json';(OUT/'logs').mkdir(parents=True,exist_ok=True)
    state=read(path) if path.exists() else dict(status='PENDING',jobs=[dict(j,status='PENDING') for j in jobs],created=stamp())
    if not any(j['id']=='resume_fair_baselines' for j in state['jobs']):state['jobs'].append(dict(jobs[-1],status='PENDING'))
    for job in state['jobs']:
        if job['status']=='COMPLETE':continue
        script=ROOT/'scripts'/job['script']
        if not script.exists():
            state.update(status='WAITING_IMPLEMENTATION',active=job['id'],updated=stamp());save(path,state);return
        state.update(status='RUNNING',active=job['id'],updated=stamp());job.update(status='RUNNING',started=stamp(),script_sha256=digest(script));save(path,state)
        with (OUT/'logs'/(job['id']+'.log')).open('a',buffering=1) as log:
            code=subprocess.call([sys.executable,'-u',str(script),*job['args']],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='COMPLETE' if code==0 else 'FAILED',code=code,finished=stamp());save(path,state)
        if code:
            state.update(status='FAILED',updated=stamp());save(path,state);return
    state.update(status='COMPLETE',active=None,finished=stamp());save(path,state)

if __name__=='__main__':main()
