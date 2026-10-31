"""Wait for server numerical completion, verify predictions, archive lightweight evidence."""
import os,sys,time,subprocess,tarfile,shutil
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'outputs/priority_20260913'

def main():
    while True:
        path=BASE/'extras_queue.json'
        if path.exists() and json.loads(path.read_text())['status']!='RUNNING':break
        time.sleep(20)
    subprocess.run([sys.executable,str(ROOT/'scripts/build_priority_report.py'),*sys.argv[1:]],check=True)
    code=BASE/'code';code.mkdir(exist_ok=True)
    for p in (ROOT/'scripts').glob('*priority*.py'):shutil.copy2(p,code/p.name)
    target=ROOT.parent/'priority_20260913_results.tar.gz'
    with tarfile.open(target,'w:gz') as tar:
        for p in sorted(BASE.rglob('*')):
            if not p.is_file():continue
            if p.suffix in ('.json','.csv','.log','.py') or ('final' in p.relative_to(BASE).parts and p.suffix in ('.png','.pdf')):
                tar.add(p,arcname=str(Path('outputs/priority_20260913')/p.relative_to(BASE)))
    print('ARCHIVE COMPLETE',target,target.stat().st_size,flush=True)

if __name__=='__main__':main()
