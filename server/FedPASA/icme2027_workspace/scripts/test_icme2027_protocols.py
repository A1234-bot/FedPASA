"""Numerical boundary checks; run on the server before the experiment queue."""
import numpy as np
from sklearn.metrics import average_precision_score
from icme2027_common import OUT,save,stamp
from run_icme2027_paired import check_math,weighted_ap
from run_icme2027_calibration import hist_packet,hist_ap,Wire,moments

def main():
    check_math();rng=np.random.default_rng(41)
    y=rng.integers(0,2,(103,4));s=rng.normal(size=(103,4));s[:15]=s[15]
    pieces=np.array_split(np.arange(103),10);bins=128
    p=[hist_packet(y[ix],s[ix],bins) for ix in pieces]
    total={k:sum(v[k] for v in p) for k in p[0]}
    quantized=np.minimum((bins*(.5+np.arctan(s)/np.pi)).astype(np.int64),bins-1).clip(0)
    np.testing.assert_allclose(hist_ap(total),100*average_precision_score(y,quantized,average=None),atol=1e-12)
    w=Wire();mean,std=moments([s[ix] for ix in pieces],w,'test')
    np.testing.assert_allclose(mean,s.mean(0),atol=1e-14);np.testing.assert_allclose(std,s.std(0),atol=1e-14)
    assert all(row['bytes']==sum(f['bytes'] for f in row['fields']) for row in w.rows)
    for _ in range(5):
        ix=rng.integers(0,103,103);weights=np.bincount(ix,minlength=103)[None,:]
        actual=weighted_ap(y[:,0],s[:,0],weights)[0]
        np.testing.assert_allclose(actual,average_precision_score(y[ix,0],s[ix,0]),atol=1e-12)
    save(OUT/'protocol_tests.json',{'status':'PASS','checks':['tied AP with integer weights','multilabel histogram AP equals sklearn on quantized scores',
        '10-client additive mean/variance','serialized numeric byte accounting','literal paired-resample equivalence'],'completed':stamp()})
    print('PROTOCOL_TESTS_PASS',flush=True)
if __name__=='__main__':main()

