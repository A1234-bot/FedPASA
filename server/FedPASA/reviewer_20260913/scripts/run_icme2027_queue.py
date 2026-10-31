"""Persistent server queue with atomic state, file lock and per-job logs."""
import fcntl,os,subprocess,sys,time,traceback
from icme2027_common import *

def jobs():
    j=[('protocol_tests',['test_icme2027_protocols.py'])]
    for d in ['coco','nus','voc07','voc12']:j.append((f'paired_{d}',['run_icme2027_paired.py','--datasets',d,'--replicates','1000']))
    for d in ['coco','nus']:j.append((f'extract_{d}',['run_icme2027_extract.py','--dataset',d]))
    for d in ['coco','nus']:
        for s in [7,17,27]:j.append((f'calibration_{d}_{s}',['run_icme2027_calibration.py','--dataset',d,'--seed',str(s),'--bins','8192']))
    j.append(('baseline_source_audit',['run_icme2027_baseline.py','--audit-only']))
    for d in ['coco','nus']:j.append((f'baseline_smoke_{d}',['run_icme2027_baseline.py','--dataset',d,'--smoke']))
    # Both datasets seed7 first, then independent training repetitions.
    for s in [7,17,27]:
        for d in ['coco','nus']:j.append((f'baseline_{d}_{s}',['run_icme2027_baseline.py','--dataset',d,'--seed',str(s)]))
    j.append(('systems_and_repeats',['build_icme2027_server_report.py']))
    return j

def main():
    if os.name!='posix' or not torch_ready():raise RuntimeError('This queue is intended for the authorized Linux GPU server')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'logs').mkdir(exist_ok=True)
    lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=OUT/'queue.json';old=read(path) if path.exists() else {}
    previous={v['id']:v for v in old.get('jobs',[])}
    rows=[previous.get(name,{'id':name,'status':'PENDING'}) for name,_ in jobs()]
    state={'status':'RUNNING','pid':os.getpid(),'workspace':str(ROOT),'created':old.get('created',stamp()),'jobs':rows}
    def checkpoint():state['updated']=stamp();save(path,state)
    checkpoint()
    for row,(name,args) in zip(rows,jobs()):
        if row['status']=='COMPLETE':continue
        # Dependencies are explicit; independent evidence still progresses after
        # a failed branch. Long baseline runs never bypass source/gradient checks.
        required=['protocol_tests']
        if name.startswith('calibration_'):required+=['extract_'+name.split('_')[1]]
        if name.startswith('baseline_smoke_'):required+=['baseline_source_audit']
        if name.startswith('baseline_') and name.split('_')[-1].isdigit():required+=['baseline_source_audit','baseline_smoke_'+name.split('_')[1]]
        blockers=[r['id'] for r in rows if r['id'] in required and r['status'] not in ['COMPLETE'] and r['id']!=name]
        if blockers:row.update(status='BLOCKED',blocked_by=blockers);checkpoint();continue
        row.update(status='RUNNING',started=stamp(),command=[sys.executable,str(ROOT/'scripts'/args[0]),*args[1:]])
        checkpoint();start=time.monotonic()
        with (OUT/'logs'/f'{name}.log').open('a',buffering=1) as log:
            log.write('\nQUEUE_START '+stamp()+'\n')
            proc=subprocess.Popen(row['command'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy())
            row['pid']=proc.pid;checkpoint()
            code=proc.wait()
        row.update(status='COMPLETE' if code==0 else 'FAILED',exit_code=code,finished=stamp(),wall_seconds=time.monotonic()-start)
        checkpoint();print(f'QUEUE {name} {row["status"]}',flush=True)
    state['status']='COMPLETE' if all(r['status']=='COMPLETE' for r in rows) else 'NEEDS_ATTENTION';checkpoint()
    print('QUEUE_FINAL',state['status'],flush=True)

def torch_ready():
    import torch
    return torch.cuda.is_available()

if __name__=='__main__':main()
