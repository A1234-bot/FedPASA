from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fedpasa.core import PatchBank, _global_quantile_selection, split_global_bank_by_client


def test_shared_key_recovers_topk_under_extreme_ties_and_repartitioning():
    rng=np.random.default_rng(7)
    # Repeated entropy including a tied cutoff; IDs deliberately permuted.
    ids=rng.permutation(120)
    entropy=(ids%4).astype(np.float32)
    expected=set(ids[np.lexsort((ids,entropy))[:23]])
    for clients in (2,7,13):
        assignment=rng.integers(clients,size=120)
        banks=[]
        for c in range(clients):
            local=ids[assignment[ids]==c]
            local=local[np.lexsort((local,(local%4)))[:23]]
            banks.append({0:PatchBank(torch.zeros(len(local),2),torch.ones(len(local),1),
                torch.tensor(local%4,dtype=torch.float32),torch.tensor(local),torch.tensor(local))})
        selected=_global_quantile_selection(tuple(banks),23,shared_tie_key=True)
        got={int(v) for b,s in zip(banks,selected) for v in b[0].patch_ids[s.get(0,torch.tensor([],dtype=torch.long))]}
        assert got==expected


def test_shared_key_fails_without_unique_identity():
    b={0:PatchBank(torch.zeros(3,2),torch.ones(3,1),torch.ones(3))}
    with pytest.raises(ValueError,match='patch_ids'):
        _global_quantile_selection((b,),2,shared_tie_key=True)
    b={0:PatchBank(torch.zeros(3,2),torch.ones(3,1),torch.ones(3),patch_ids=torch.tensor([1,1,2]))}
    with pytest.raises(ValueError,match='unique'):
        _global_quantile_selection((b,),2,shared_tie_key=True)


def test_split_preserves_global_identity():
    b={0:PatchBank(torch.zeros(3,2),torch.ones(3,1),torch.ones(3),torch.tensor([0,1,0]),torch.tensor([0,2,1]))}
    local=split_global_bank_by_client(b,np.array([0,1]),2)
    assert local[0][0].patch_ids.tolist()==[0,1]
    assert local[1][0].patch_ids.tolist()==[2]
