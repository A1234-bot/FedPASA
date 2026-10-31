"""Durable per-job exit receipt and lock, surviving queue-supervisor restart."""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/supplement_20261029'


def main():
    receipt=Path(sys.argv[1]).resolve()
    assert receipt.is_relative_to((OUT/'attempts').resolve())
    spec=json.loads(Path(sys.argv[2]).read_text())
    with (OUT/'attempts'/(spec['job_id']+'.lock')).open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=subprocess.call(spec['command'],cwd=ROOT)
        temp=receipt.with_suffix('.partial')
        with temp.open('w') as f:
            json.dump(dict(job_id=spec['job_id'],attempt=spec['attempt'],exit_code=result,
                           finished=time.time(),wrapper_pid=os.getpid()),f)
            f.flush();os.fsync(f.fileno())
        temp.replace(receipt)
        return result


if __name__=='__main__':sys.exit(main())
