"""Server queue for the November ICME jobs. Run from the isolated deployment."""
import argparse
import collections
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/strengthen_20261103"
QUEUE=ROOT/"queue.json"
MAX_ATTEMPTS=2
TERMINAL={"COMPLETE","FAILED","BLOCKED"}

def save(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".partial");tmp.write_text(json.dumps(obj,indent=2,allow_nan=False));tmp.replace(p)

def spec():
    jobs=[]
    def add(kind,d,**kw):
        key="_".join(map(str,[kind,d,*kw.values()]))
        jobs.append(dict(id=key,kind=kind,d=d,**kw,status="PENDING"))
    for cond in ("iid","beta0.1","imbalance"):
        for seed in (37,47,57):
            for d in ("coco","nus"):add("fedavg",d,condition=cond,seed=seed)
    for d in ("coco","nus"):
        for seed in (37,47,57):add("rf",d,seed=seed)
    for d in ("coco","nus"):
        for seed in (37,47,57,67,77):add("union",d,seed=seed)
        for frac in (.25,.50,.75):
            for seed in (37,47,57,67,77):add("partial",d,fraction=frac,seed=seed)
    for d in ("coco","nus"):
        for network in ("loopback","100Mbps_10ms","10Mbps_50ms"):
            for method in ("full1","ridge","fedavg","full3"):
                for mode,repeat in [("inclusive",1),("locked",1),("locked",2),("locked",3)]:
                    add("tcp",d,network=network,method=method,mode=mode,repeat=repeat)
    assert len({j["id"] for j in jobs})==len(jobs)==160
    return jobs

def worker(j):
    import torch
    import strengthen_accuracy as a
    if not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    d=j["d"];dest=OUT/j["id"];dest.mkdir(parents=True,exist_ok=True)
    if j["kind"]=="tcp":
        net=j["network"];method=j["method"];locked=j["mode"]=="locked"
        src=OUT/f"tcp_{d}_{net}_{method}_inclusive_1"
        rate,delay={"loopback":(0,0),"100Mbps_10ms":(100,10),"10Mbps_50ms":(10,50)}[net]
        os.environ.update(SUPPLEMENT_MBIT=str(rate),SUPPLEMENT_RTT_MS=str(delay),STRENGTH_JOB=str(dest),STRENGTH_SEARCH=str(src),STRENGTH_LOCKED=str(int(locked)),SUPPLEMENT_SPLITS="3" if method=="full3" else "1")
        if method.startswith("full"):
            import strengthen_full_tcp as full
            full.run(d)
        else:
            import strengthen_linear_tcp as lin
            lin.run(d,method,dest,locked,src)
        assert (dest/"result.json").is_file()
        return
    if j["kind"]=="fedavg":res=a.fedavg(d,j["condition"],j["seed"],dest)
    elif j["kind"]=="rf":res=a.random_features(d,j["seed"],dest)
    elif j["kind"]=="partial":res=a.participation(d,j["seed"],j["fraction"],dest)
    else:res=a.patch_study(d,j["kind"],j["seed"],1,dest)
    save(dest/"result.json",dict(status="COMPLETE",job=j["id"],**res))

def patch_available(dataset,inputs,checked):
    """Validate published tensors independently, once per file version."""
    from fair_baselines_common import digest
    for part in ("train","test"):
        key=f"outputs/icme/cache/{dataset}/{part}_patch.npy"
        path=ROOT/key
        if not path.is_file():return False
        stat=path.stat();version=(stat.st_size,stat.st_mtime_ns)
        if stat.st_size!=inputs[key]["size"]:return False
        if checked.get(key,(None,False))[0]!=version:
            checked[key]=(version,digest(path)==inputs[key]["sha256"])
        if not checked[key][1]:return False
    return True


def rebuild_finished():
    # The supervisor retains this lock through reconstruction and archive fallback.
    if not (ROOT/"patch.ready").is_file():return False
    with (ROOT/"rebuild.lock").open("a") as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return False
    return True


def next_ready(jobs,ready,timing_ready):
    by_id={j["id"]:j for j in jobs}
    # Run untouched work first; one failed task cannot monopolize the queue.
    for job in sorted(jobs,key=lambda j:(j.get("attempts",0),not j.get("resume_checkpoint",False))):
        if job["status"] in TERMINAL:continue
        if job["kind"]=="tcp":
            if not timing_ready:continue
            if job["mode"]=="locked":
                source=by_id[f'tcp_{job["d"]}_{job["network"]}_{job["method"]}_inclusive_1']
                if source["status"] in {"FAILED","BLOCKED"}:
                    job.update(status="BLOCKED",blocked_by=source["id"]);continue
                if source["status"]!="COMPLETE":continue
        elif job["kind"] in ("union","partial") and not ready[job["d"]]:continue
        return job
    return None


def complete_result(dest):
    try:return json.loads((dest/"result.json").read_text()).get("status")=="COMPLETE"
    except (OSError,ValueError):return False


def restore_jobs(previous):
    old={j["id"]:j for j in previous.get("jobs",[])}
    jobs=spec()
    for j in jobs:
        prior=old.get(j["id"],{})
        for key in ("attempts","failures","start","finished","exit_code"):
            if key in prior:j[key]=prior[key]
        if complete_result(OUT/j["id"]):j["status"]="COMPLETE"
        elif prior.get("status")=="FAILED" and prior.get("attempts",0)>=MAX_ATTEMPTS:
            j["status"]="FAILED"
        elif prior.get("status")=="RUNNING" and prior.get("attempts",0):
            # A host interruption is distinct from a worker reporting failure.
            j["attempts"]=max(0,j["attempts"]-1)
        if j["kind"]=="partial" and j["status"]!="COMPLETE":
            j["resume_checkpoint"]=any((OUT/j["id"]).glob("inner*/receipt.json"))
    return jobs


def record_failure(job,exitcode,dest,log_start,error=None):
    log_path=dest/"run.log"
    tail=""
    if log_path.exists():
        with log_path.open("rb") as log:
            log.seek(max(log_start,log_path.stat().st_size-16000))
            tail=log.read().decode(errors="replace")
    failure=dict(attempt=job["attempts"],time=time.time(),exit_code=exitcode,
                 error=error,log_offset=log_start,log_tail=tail)
    job.setdefault("failures",[]).append({k:v for k,v in failure.items() if k!="log_tail"})
    job.update(status="RETRY_PENDING" if job["attempts"]<MAX_ATTEMPTS else "FAILED",exit_code=exitcode)
    job.pop("worker_pid",None)
    save(dest/f'failure_attempt_{job["attempts"]}.json',failure)


def clean_worker_group(pid):
    # Each attempt owns a new process group, including all TCP/inner workers.
    try:os.killpg(pid,signal.SIGTERM)
    except ProcessLookupError:return
    time.sleep(.5)
    try:os.killpg(pid,signal.SIGKILL)
    except ProcessLookupError:pass


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/"queue.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous=json.loads(QUEUE.read_text()) if QUEUE.exists() else {}
    jobs=restore_jobs(previous)
    def status(label,current=None):
        save(QUEUE,dict(status=label,current=current,pid=os.getpid(),updated=time.time(),jobs=jobs))
    status("WAITING_FOR_VERIFIED_FEATURES")
    while not (ROOT/"small.ready").exists():time.sleep(10)
    from fair_baselines_common import digest
    inputs=json.loads((ROOT/"inputs.json").read_text())
    for path,v in inputs.items():
        if path.endswith("_patch.npy") or ("/runs/" in path and not path.endswith("result.json")):continue
        assert digest(ROOT/path)==v["sha256"],path
    env={**os.environ,"OMP_NUM_THREADS":"4","MKL_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","PYTHONPATH":str(ROOT/"src"),"PYTHONUNBUFFERED":"1","PYTHONFAULTHANDLER":"1","STRENGTH_QUEUE_LOCK_FD":str(lock.fileno())}
    for j in jobs:
        dest=OUT/j["id"];dest.mkdir(exist_ok=True)
        if complete_result(dest):
            j["status"]="COMPLETE"
    checked={}
    while any(j["status"] not in TERMINAL for j in jobs):
        ready={d:patch_available(d,inputs,checked) for d in ("nus","coco")}
        j=next_ready(jobs,ready,all(ready.values()) and rebuild_finished())
        if j is None:
            if all(x["status"] in TERMINAL for x in jobs):break
            status("WAITING_FOR_VERIFIED_PATCH");time.sleep(30);continue
        dest=OUT/j["id"]
        while (ROOT/"PAUSE").exists():status("PAUSED",j["id"]);time.sleep(30)
        j.update(status="RUNNING",start=time.time(),attempts=j.get("attempts",0)+1);status("RUNNING",j["id"])
        log_start=(dest/"run.log").stat().st_size if (dest/"run.log").exists() else 0
        with (dest/"run.log").open("a") as log:
            child=None;error=None
            try:
                child=subprocess.Popen([sys.executable,"-X","faulthandler",__file__,"--job",j["id"]],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(lock.fileno(),),start_new_session=True)
                j["worker_pid"]=child.pid;status("RUNNING",j["id"]);exitcode=child.wait()
            except OSError as exc:exitcode=127;error=repr(exc)
            finally:
                if child is not None:clean_worker_group(child.pid)
        if exitcode or not complete_result(dest):
            record_failure(j,exitcode or 1,dest,log_start,error or (None if exitcode else "Worker exited without COMPLETE result"))
            status("RUNNING");continue
        j.pop("worker_pid",None)
        j.update(status="COMPLETE",finished=time.time());status("RUNNING")
    failed=[j for j in jobs if j["status"]!="COMPLETE"]
    save(OUT/"failure_summary.json",dict(updated=time.time(),jobs=failed))
    if failed:
        status("COMPLETE_WITH_FAILURES")
        return
    status("REPORTING")
    subprocess.run([sys.executable,str(ROOT/"scripts/strengthen_report.py")],cwd=ROOT,env=env,check=True)
    status("COMPLETE")

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--job");p.add_argument("--list",action="store_true");args=p.parse_args()
    if args.list:print(json.dumps(spec(),indent=2))
    elif args.job:worker(next(j for j in spec() if j["id"]==args.job))
    else:
        try:run()
        except BlockingIOError:raise
        except Exception as e:
            if QUEUE.exists():
                q=json.loads(QUEUE.read_text());q.update(status="FAILED",error=repr(e));save(QUEUE,q)
            raise
