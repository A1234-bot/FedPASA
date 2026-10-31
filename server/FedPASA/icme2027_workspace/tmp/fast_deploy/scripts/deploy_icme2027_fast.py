"""Switch the authorized server queue to validated immutable-text caching."""
import json,os,signal,shutil,subprocess,time
from pathlib import Path
from icme2027_common import ROOT,OUT,read,save,digest,stamp
from benchmark_icme2027_fast import suspend
from icme2027_fast_fedmvp import VERSION

def main():
    report=read(OUT/'optimization/benchmark.json')
    assert report['status']=='PASS' and all(v>1.05 for v in report['speedup'].values())
    assert all(r['max_abs_error']==0 for r in report['rows'] if r['bitwise_equal'] is not None)
    candidate=ROOT/'tmp/fast_deploy/scripts/run_icme2027_baseline.py'
    compile(candidate.read_text(),str(candidate),'exec')
    q=read(OUT/'queue.json');running=[r for r in q['jobs'] if r['status']=='RUNNING']
    assert len(running)==1 and running[0]['id'].startswith('baseline_')
    name=running[0]['id'];_,dataset,seed=name.split('_')
    folder=OUT/'baselines/fedmvp_release_mlc'/dataset/f'seed{seed}'
    backup=OUT/'optimization/before_deployment';backup.mkdir(parents=True,exist_ok=False)
    pids=suspend();stopped=False
    try:
        # checkpoint.pt is atomically renamed at each completed client. A
        # partially executed next client is discarded and replayed identically.
        import torch
        state=torch.load(folder/'checkpoint.pt',map_location='cpu',weights_only=False)
        assert state['target_rounds']==200 and state['training_seed']==int(seed)
        for src in [folder/'checkpoint.pt',folder/'progress.json',OUT/'queue.json',ROOT/'scripts/run_icme2027_baseline.py']:
            shutil.copy2(src,backup/src.name)
        note={'version':VERSION,'code_sha256':digest(ROOT/'scripts/icme2027_fast_fedmvp.py'),
            'benchmark_sha256':digest(OUT/'optimization/benchmark.json'),'speedup_control':report['speedup'],
            'batch_size':32,'target_rounds':200,'dataset':dataset,'training_seed':int(seed),
            'checkpoint_completed_round':state['completed_round'],'checkpoint_active_round':state.get('active_round'),
            'checkpoint_next_client':state.get('next_client',0),'checkpoint_sha256':digest(folder/'checkpoint.pt'),
            'method':'cache immutable token embeddings and fixed zero-shot text encoder result; source forward/loss unchanged',
            'restart_scope':'Completed clients preserved; partial next client replayed from saved boundary; SGD/dropout seeds unchanged',
            'deployed':stamp()}
        target=ROOT/'scripts/run_icme2027_baseline.py';tmp=target.with_suffix('.deploy.tmp')
        shutil.copy2(candidate,tmp);tmp.replace(target)
        save(OUT/'optimization/enabled.json',note)
        # End old stopped workers and parents; SIGCONT permits pending TERM to
        # be handled. The manager is in a separate tmux session.
        for pid in reversed(pids):
            try:os.kill(pid,signal.SIGTERM);os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        stopped=True
        for _ in range(50):
            alive=[]
            for pid in pids:
                p=Path(f'/proc/{pid}/stat')
                if p.exists() and p.read_text().split(') ')[1][0]!='Z':alive.append(pid)
            if not alive:break
            time.sleep(.1)
        if alive:raise RuntimeError(f'Old workers did not exit: {alive}')
        if subprocess.run(['tmux','has-session','-t','icme2027'],capture_output=True).returncode==0:
            # Only the empty/dead original queue pane should remain.
            subprocess.run(['tmux','kill-session','-t','icme2027'],check=True)
        subprocess.run(['tmux','new-session','-d','-s','icme2027','bash '+str(ROOT/'scripts/start_icme2027_server.sh')],check=True)
        save(OUT/'optimization/deployment.json',{'status':'DEPLOYED','details':note,'old_pids':pids,'finished':stamp()})
    finally:
        if not stopped:
            for pid in reversed(pids):
                try:os.kill(pid,signal.SIGCONT)
                except ProcessLookupError:pass

if __name__=='__main__':main()
