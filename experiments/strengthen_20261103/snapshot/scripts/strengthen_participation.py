"""Small parent process; heavy numerical imports occur only after children exit."""
import json
import os
from pathlib import Path
import subprocess
import sys


def run(dataset, seed, fraction, dest):
    children = []
    for offset in (7, 17, 27):
        folder = dest / f'inner{offset}'
        folder.mkdir(exist_ok=True)
        if not (folder / 'receipt.json').exists():
            inherited = os.environ.get('STRENGTH_QUEUE_LOCK_FD')
            descriptors = () if inherited is None else (int(inherited),)
            subprocess.run([sys.executable, '-X', 'faulthandler',
                            str(Path(__file__).with_name('strengthen_accuracy.py')),
                            '--participation-inner', dataset, str(seed), str(fraction),
                            str(folder.resolve()), str(offset)],
                           check=True, pass_fds=descriptors)
        children.append(json.loads((folder / 'receipt.json').read_text()))
    import numpy as np
    import run_priority_full_impl as r
    result = children[0].copy()
    for name in ('global_0', 'central', 'ridge'):
        avg = np.mean([np.load(dest / f'inner{k}' / (name + '.npy')) for k in (7, 17, 27)], axis=0)
        np.save(dest / (name + '_three.npy'), avg)
        result[name + '_three_mAP'] = r.metric(r.targets(dataset), avg)[0]
    result['splits'] = [7, 17, 27]
    return result
