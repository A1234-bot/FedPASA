"""Validate interrupted-job recovery before promoting its queue state."""
import json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/priority_20260913'
def main():
    path=BASE/'b32_exit.txt'
    while not path.exists():time.sleep(20)
    assert path.read_text().strip()=='0',path.read_text()
    subprocess.run([sys.executable,str(ROOT/'scripts/run_priority_backbone.py'),'--evaluate-only'],check=True)
    result=json.loads((BASE/'backbone/matching.json').read_text());assert result['status']=='COMPLETE'
    assert 'complete raw training patch cache' in result['central_QA_selection']
    queue=BASE/'extras_queue.json';q=json.loads(queue.read_text())
    for j in q['jobs']:
        if j['name']=='backbone':
            old=dict(j);j.update(status='COMPLETE',returncode=0,end=time.time(),recovery_previous_state=old,recovery_note='Container restart and preprocessing repr-check failure; resumed with dual extraction, exact input preprocessing tensor check, raw-cache central QA audit.')
    assert all(j['status']=='COMPLETE' for j in q['jobs'])
    q['status']='COMPLETE';temp=queue.with_suffix('.partial');temp.write_text(json.dumps(q,indent=2));temp.replace(queue)
    subprocess.run([sys.executable,str(ROOT/'scripts/finalize_priority_archive.py')],check=True)
if __name__=='__main__':main()
