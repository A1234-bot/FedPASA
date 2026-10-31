"""Sequential, restartable experiment driver. No test-ranked method selection."""
import subprocess,sys,time,traceback
from fair_baselines_common import *

def jobs():
    entries=[]
    def add(script,phase,d):entries.append({'id':f'{Path(script).stem}_{phase}_{d}','script':script,'phase':phase,'dataset':d})
    for d in ('coco','nus'):add('run_fair_ridge.py','tune',d)
    for d in ('coco','nus'):add('run_fair_fedmvp.py','tune',d)
    for d in ('coco','nus'):add('run_fair_ram.py','audit',d)
    for d in ('coco','nus'):add('run_fair_ram.py','tune',d)
    for d in ('coco','nus'):add('run_fair_linear.py','repeat',d)
    for d in ('coco','nus'):add('run_fair_ridge.py','final',d)
    for d in ('coco','nus'):add('run_fair_fedmvp.py','repeat',d)
    for d in ('coco','nus'):add('run_fair_ram.py','repeat',d)
    for d in ('coco','nus'):add('build_fair_baselines_report.py','report',d)
    return entries

def main():
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'queue.json'
    state=read(path) if path.exists() else {'status':'WAITING_LINEAR','created':stamp(),'jobs':[dict(j,status='PENDING') for j in jobs()]}
    save(path,state)
    while not (OUT/'linear_tuning_complete.txt').exists():
        if (OUT/'linear_exit_code.txt').exists() and int((OUT/'linear_exit_code.txt').read_text())!=0:raise RuntimeError('linear diagnostics failed; inspect log')
        time.sleep(30)
    for job in state['jobs']:
        if job['status']=='COMPLETE':continue
        # Never proceed to test predictions if the preceding algorithm audit failed.
        state.update(status='RUNNING',active=job['id'],updated=stamp());job.update(status='RUNNING',started=stamp());save(path,state)
        cmd=[sys.executable,str(ROOT/'scripts'/job['script']),'--phase',job['phase'],'--dataset',job['dataset']]
        job['command']=cmd;job['script_sha256']=digest(ROOT/'scripts'/job['script']);save(path,state)
        with (OUT/'logs'/f"{job['id']}.log").open('a',buffering=1) as log:
            code=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        job.update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,finished=stamp());state['updated']=stamp();save(path,state)
        if code!=0:state['status']='FAILED';save(path,state);raise RuntimeError(f"job failed {job['id']}")
    state.update(status='COMPLETE',active=None,finished=stamp());save(path,state)
    save(OUT/'completion.json',{'status':'COMPLETE','queue_sha256':digest(path),'datasets':['coco','nus'],'finished':stamp(),'scope':'numerical completion does not itself certify author-equivalence or convergence'})

if __name__=='__main__':main()
