"""Protocol checks before spending the GPU budget."""
import json
import os
import socket
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_supplement_20261029 import partition, split_indices
from supplement_transport_20261029 import Wire


def test_partitions_and_splits():
    rng=np.random.default_rng(5)
    y=(rng.random((2000,8))>.65).astype('uint8')
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder)
        for condition in ('iid','imbalance','beta1.0','beta0.1'):
            a=partition(y,condition,37,10,root)
            np.testing.assert_array_equal(a,partition(y,condition,37,10,root))
            assert len(a)==len(y) and np.bincount(a).min()>=10
        assert np.ptp(np.bincount(partition(y,'iid',37,10,root)))==0
        assert np.bincount(partition(y,'imbalance',37,10,root)).max()>400
        assert not np.array_equal(partition(y,'iid',37,10,root),partition(y,'iid',47,10,root))
        f,v=split_indices(len(y),3707)
        assert len(v)==400 and not np.intersect1d(f,v).size


def test_paced_wire_roundtrip():
    a,b=socket.socketpair()
    wa,wb=Wire(a,0),Wire(b,0)
    original={k:os.environ.get(k) for k in ('SUPPLEMENT_MBIT','SUPPLEMENT_RTT_MS')}
    os.environ['SUPPLEMENT_MBIT']='10';os.environ['SUPPLEMENT_RTT_MS']='20'
    payload=np.arange(25000,dtype='float32')
    t=time.perf_counter()
    thread=threading.Thread(target=wa.send,args=('test',payload));thread.start()
    received=wb.recv('test');thread.join()
    elapsed=time.perf_counter()-t
    np.testing.assert_array_equal(payload,received)
    assert elapsed>=.08
    assert wa.rows[0]['framed_bytes']==wb.rows[0]['framed_bytes']
    a.close();b.close()
    for k,v in original.items():
        if v is None:os.environ.pop(k,None)
        else:os.environ[k]=v


def test_queue_plan():
    from queue_supplement_20261029 import jobs,eligible,exclusive
    plan=jobs()
    assert len({j['id'] for j in plan})==len(plan)
    assert len(plan)==99
    ids=[j['id'] for j in plan]
    for job in plan:
        assert (Path(__file__).resolve().parents[1]/'scripts'/job['script']).exists()
        if job['id'].endswith('_fedavg'):
            assert ids.index(job['id'][:-7])<ids.index(job['id'])
    state={'jobs':plan}
    plan[0]['status']='COMPLETE'
    assert eligible(plan[1],state,[])
    assert not eligible(plan[2],state,[])
    plan[1]['status']='COMPLETE'
    assert eligible(plan[2],state,[])
    scale=next(j for j in plan if '_scale_' in j['id'])
    assert exclusive(scale) and not eligible(scale,state,[plan[2]])


def test_stage_commit_boundary():
    from supplement_checkpoints import StageStore
    with tempfile.TemporaryDirectory() as folder:
        store=StageStore(folder,{'seed':37})
        value,_,resumed=store.run('completed',lambda:np.arange(5))
        assert not resumed
        def never():raise AssertionError('committed stage recomputed')
        other=StageStore(folder,{'seed':37})
        actual,_,resumed=other.run('completed',never)
        np.testing.assert_array_equal(value,actual);assert resumed
        # A torn temporary write is never considered a committed checkpoint.
        other.path('next').with_name(other.path('next').name+'.partial').write_bytes(b'torn')
        value,_,resumed=other.run('next',lambda:42)
        assert value==42 and not resumed
        try:StageStore(folder,{'seed':47})
        except AssertionError:pass
        else:raise AssertionError('changed protocol was not rejected')


if __name__=='__main__':
    test_partitions_and_splits();test_paced_wire_roundtrip();test_queue_plan();test_stage_commit_boundary()
    print('SUPPLEMENT_PROTOCOL_TESTS_PASS')
