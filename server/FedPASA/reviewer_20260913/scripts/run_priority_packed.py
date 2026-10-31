"""Replay Full with actual lossless byte serialization of all moment packets."""
import dataclasses,sys
import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_priority_full_impl as r
BASE=r.ROOT/'outputs/priority_20260913'

def run(d):
    counters=dict(packets=0,dense_bytes=0,packed_bytes=0,symmetric_matrices=0)
    def wire(packet):
        fields={}
        for f in dataclasses.fields(packet):
            v=getattr(packet,f.name)
            if not isinstance(v,torch.Tensor):
                payload=np.asarray(v,dtype=np.int64).tobytes();fields[f.name]=int(np.frombuffer(payload,dtype=np.int64)[0])
                assert fields[f.name]==v
                counters['dense_bytes']+=8;counters['packed_bytes']+=8
                continue
            arr=v.numpy();counters['dense_bytes']+=arr.nbytes
            if f.name in ('xtx','second_moment'):
                np.testing.assert_array_equal(arr,arr.T)
                ix=np.triu_indices(len(arr));payload=arr[ix].tobytes();vals=np.frombuffer(payload,dtype=arr.dtype)
                recovered=np.zeros_like(arr);recovered[ix]=vals;recovered[(ix[1],ix[0])]=vals;counters['symmetric_matrices']+=1
            else:
                payload=arr.tobytes();recovered=np.frombuffer(payload,dtype=arr.dtype).reshape(arr.shape).copy()
            np.testing.assert_array_equal(arr,recovered)
            fields[f.name]=torch.from_numpy(recovered);counters['packed_bytes']+=len(payload)
        counters['packets']+=1
        return type(packet)(**fields)
    moment=r.core._moment_packet;purify=r.core._purification_packet;ridge=r.make_ridge_packet
    r.core._moment_packet=lambda *a,**kw:wire(moment(*a,**kw))
    r.core._purification_packet=lambda *a,**kw:wire(purify(*a,**kw))
    r.make_ridge_packet=lambda *a,**kw:wire(ridge(*a,**kw))
    control=BASE/'controlled/full_runtime'/d
    def banks(dataset,a,text,clients,destination):
        paths=[control/destination.name/f'client{i}.pt' for i in range(clients)]
        assert all(p.exists() for p in paths)
        return r.DiskBanks(paths)
    r.disk_banks=banks;r.OUT=BASE/'packed';r.full_runtime(d)
    out=r.OUT/'full_runtime'/d
    aa=np.load(out/'test_scores.npy');bb=np.load(control/'test_scores.npy')
    err=float(np.max(np.abs(aa-bb)));np.testing.assert_array_equal(aa,bb)
    rep=r.read(out/'result.json')
    r.save(BASE/'packed'/d/'result.json',dict(status='COMPLETE',scope='Single-server Full replay, actual FP64 moment serialize/decode before every solve; reuses identical local banks, refits and reselects; entropy and validation scores uncompressed; no physical network',max_abs_full_score_error=err,mAP=rep['rows'][-1]['mAP'],packed_uplink_bytes=rep['packed_uplink_bytes'],dense_uplink_bytes=rep['uplink_bytes'],logical_stages=23,wire_checks=counters))

if __name__=='__main__':
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):run(sys.argv[1])
