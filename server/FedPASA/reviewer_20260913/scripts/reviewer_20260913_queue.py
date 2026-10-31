"""Detached, restartable server queue; no dependence on the SSH client."""
import fcntl,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/reviewer_20260913'
def stamp():return datetime.now(timezone.utc).isoformat()
def save(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def jobs():
    j=[('tests',['--phase','tests'],'tests.json')]
    for d in ('voc07','voc12','coco','nus'):
        for phase in ('audit','linear'):
            j.append((f'{phase}_{d}',['--phase',phase,'--dataset',d],f'{phase}/{d}/result.json'))
    for d in ('voc07','voc12','coco','nus'):
        j.append((f'fedavg_{d}',['--phase','fedavg','--dataset',d],f'fedavg/{d}/result.json'))
        j.append((f'qa_{d}',['--phase','qa','--dataset',d],f'qa/{d}/m10_seed7/result.json'))
    for d in ('voc07','voc12','coco','nus'):
        j.append((f'full_{d}',['--phase','full','--dataset',d],f'full_runtime/{d}/result.json'))
    j.append(('sensitivity_coco',['--phase','sensitivity','--dataset','coco'],'sensitivity/coco/result.json'))
    for d,counts in [('coco',[8,16,32,48,64,80]),('nus',[8,16,32,49,65,81])]:
        for m in counts:
            j.append((f'scaling_{d}_{m}',['--phase','qa','--dataset',d,'--clients',str(m),'--seed','17'],f'qa/{d}/m{m}_seed17/result.json'))
        for beta in (.1,.3,.5):
            j.append((f'heterogeneity_{d}_{beta}',['--phase','qa','--dataset',d,'--beta',str(beta)],f'qa/{d}/m10_seed7_beta{beta}/result.json'))
    for d in ('voc07','voc12','coco','nus'):
        for method in ('ridge','fed3r','fedavg','qa','full'):
            j.append((f'memory_{d}_{method}',['--phase','memory','--dataset',d,'--method',method],f'memory/{d}/{method}/result.json'))
    return j
def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p=OUT/'queue.json';old=json.loads(p.read_text()) if p.exists() else {};previous={r['id']:r for r in old.get('jobs',[])}
    rows=[previous.get(n,dict(id=n,status='PENDING',artifact=a)) for n,_,a in jobs()]
    state=dict(status='RUNNING',pid=os.getpid(),started=old.get('started',stamp()),workspace=str(ROOT),jobs=rows)
    def checkpoint():state['updated']=stamp();save(p,state)
    checkpoint()
    for row,(name,args,artifact) in zip(rows,jobs()):
        path=OUT/artifact
        if row['status']=='COMPLETE' and path.exists():continue
        if name!='tests' and rows[0]['status']!='COMPLETE':row.update(status='BLOCKED',reason='protocol tests failed');checkpoint();continue
        command=[sys.executable,'-u',str(ROOT/'scripts/run_reviewer_20260913.py'),*args]
        row.update(status='RUNNING',started=stamp(),command=command);checkpoint();t=time.monotonic()
        with (OUT/'logs'/f'{name}.log').open('a',buffering=1) as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy());row['pid']=proc.pid;checkpoint();code=proc.wait()
        valid=False
        if code==0 and path.exists():
            try:valid=json.loads(path.read_text()).get('status')=='COMPLETE'
            except (ValueError,OSError):pass
        row.update(status='COMPLETE' if valid else 'FAILED',exit_code=code,finished=stamp(),seconds=time.monotonic()-t);checkpoint()
        print(name,row['status'],flush=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/build_reviewer_20260913_report.py')],cwd=ROOT,check=False)
    state['status']='COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'NEEDS_ATTENTION';checkpoint()
    subprocess.run([sys.executable,str(ROOT/'scripts/build_reviewer_20260913_report.py')],cwd=ROOT,check=False)
if __name__=='__main__':main()
