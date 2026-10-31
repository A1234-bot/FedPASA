"""Predeclared independent partitions; no historical-output writes or test selection."""
import argparse
import gc
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits
import run_priority_full_impl as r
from run_fair_ridge import GRID
from run_icme_experiments import dirichlet_assignments

BASE = r.ROOT / 'outputs/supplement_20261029'
SEEDS = (37, 47, 57, 67, 77)


def partition(y, condition, seed, clients, root):
    rng = np.random.default_rng(seed)
    if condition.startswith('beta'):
        a = dirichlet_assignments(y, root, clients=clients,
                                 beta=float(condition[4:]), seed=seed)[0]
    else:
        order = rng.permutation(len(y))
        if condition == 'iid':
            groups = np.array_split(order, clients)
        elif condition == 'imbalance':
            # At least 20 examples/client, fixed total; labels do not set sizes.
            weights = rng.lognormal(0, 1.5, clients)
            spare = len(y) - 20 * clients
            assert spare >= 0
            sizes = 20 + rng.multinomial(spare, weights / weights.sum())
            groups = np.split(order, np.cumsum(sizes)[:-1])
        else:
            raise ValueError(condition)
        a = np.empty(len(y), np.int64)
        for cid, ids in enumerate(groups):
            a[ids] = cid
    assert len(a) == len(y) and np.array_equal(np.unique(a), np.arange(clients))
    return a


def split_indices(n, seed):
    ix = np.random.default_rng(seed).permutation(n)
    nval = round(n * .2)
    fit, val = ix[nval:], ix[:nval]
    assert not np.intersect1d(fit, val).size
    np.testing.assert_array_equal(np.sort(np.r_[fit, val]), np.arange(n))
    return fit, val


def remove_disposable_banks(banks, root):
    # Only generated .pt files below this new job's temporary bank directory.
    allowed = (root / 'temporary_banks').resolve()
    for path in banks.paths:
        path = path.resolve()
        assert path.is_relative_to(allowed) and path.suffix == '.pt'
        path.unlink()


def run(dataset, condition, seed, clients=10, union=False):
    root = BASE / 'runs' / f'{dataset}_{condition}_m{clients}_s{seed}'
    root.mkdir(parents=True, exist_ok=True)
    if (root / 'result.json').exists():
        return
    cache = r.resolve_cache(dataset)
    y = r.targets(dataset, 'train')
    x = r.load_features(cache, 'train')
    z = r.load_features(cache, 'test')
    text = torch.from_numpy(np.load(cache / 'text_features.npy')).float()
    benchmark = r.BENCHMARKS[dataset]
    a = partition(y, condition, seed, clients, root)
    np.save(root / 'assignment.npy', a)
    inner = [seed * 100 + k for k in (7, 17, 27)]
    splits = [split_indices(len(y), s) for s in inner]
    for s, (f, v) in zip(inner, splits):
        np.savez(root / f'split{s}.npz', fit=f, validation=v)
    protocol = dict(dataset=dataset, condition=condition, outer_seed=seed,
                    inner_seeds=inner, clients=clients, participation=1,
                    client_sizes=np.bincount(a, minlength=clients).tolist(),
                    client_positive_counts=[y[a == i].sum(0).tolist() for i in range(clients)],
                    ridge_grid=list(GRID), fusion_grid=np.linspace(0, 1, 21).tolist(),
                    assignment_sha256=r.digest(root / 'assignment.npy'),
                    split_sha256=[r.digest(root / f'split{s}.npz') for s in inner],
                    selection_uses_test_labels=False,
                    split_rule='global train-only random 80/20; same split across partition conditions',
                    dirichlet_rule='existing multi-label random-positive-primary-label partition with minimum size 10; not independent per-label Dirichlet',
                    imbalance_rule='label-independent lognormal sigma=1.5 sizes, minimum 20; all examples retained',
                    scope='frozen cached features; serial simulated clients; no feature extraction time',
                    source_sha256=r.digest(__file__))
    r.save(root / 'protocol.json', protocol)
    stages = []
    messages = []
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def stage(name, fn):
        r.save(root / 'progress.json', dict(stage=name, pid=os.getpid(), updated=time.time()))
        print('STAGE', root.name, name, flush=True)
        value, seconds = r.timed(fn)
        stages.append(dict(stage=name, seconds=seconds))
        return value

    def packets(ids, phase):
        pp = stage(phase + '/ridge_statistics', lambda: tuple(
            r.make_ridge_packet(x[ids[a[ids] == cid]], y[ids[a[ids] == cid]])
            for cid in range(clients)))
        messages.extend(dict(phase=phase, stage='ridge_statistics', client=i,
                             bytes=r.packet_wire(p)) for i, p in enumerate(pp))
        return pp

    def qa(ids, phase, eval_ids, part, do_union):
        assigned = np.full(len(a), -1, dtype=np.int64)
        assigned[ids] = a[ids]
        banks = stage(phase + '/bank_build', lambda: r.disk_banks(
            dataset, assigned, text, clients, root / 'temporary_banks' / phase))
        profile = r.Profile()
        with profile.track():
            model = stage(phase + '/qa_fit', lambda: r.core.fit_quantile_aligned_fedpasa(
                banks, text, k=512, purification_temperature=benchmark.purification_temperature))
        messages.extend(dict(v, phase=phase) for v in profile.rows)
        torch.save({'weight': model.weight, 'bias': model.bias}, root / f'{phase}_qa.pt')
        def score(model):
            return r.piaa_scores(cache / f'{part}_patch.npy', cache / f'{part}_cls.npy',
                                 eval_ids, text, model.weight, model.bias,
                                 cls_scale=benchmark.cls_scale, patch_scale=benchmark.patch_scale,
                                 batch_images=128, device='cuda')
        scores = stage(phase + '/qa_inference', lambda: score(model))
        us = None
        if do_union:
            um = stage(phase + '/union_fit', lambda: r.union_stream(
                banks, text, benchmark.purification_temperature))
            r.save(root / f'{phase}_union_metadata.json', um.metadata)
            us = stage(phase + '/union_inference', lambda: score(um))
        remove_disposable_banks(banks, root)
        del banks
        gc.collect()
        return scores, us

    all_ids = np.arange(len(y))
    full_packets = packets(all_ids, 'full')
    qt, ut = qa(all_ids, 'full', np.arange(len(z)), 'test', union)
    np.save(root / 'patch_scores.npy', qt)
    predictions = {'Patch': qt}
    full_scores = []
    ridge_scores = []
    weights = np.linspace(0, 1, 21, dtype=np.float32)
    for index, (s, (fit, val)) in enumerate(zip(inner, splits)):
        phase = f'split{s}'
        pp = packets(fit, phase)
        tuning = []
        candidates = []
        for lam in GRID:
            model = stage(phase + f'/ridge_alpha{lam}', lambda: r.fit_ridge_from_packets(pp, regularization=lam))
            rv = r.predict(x[val], model)
            tuning.append(r.metric(y[val], rv)[0])
            candidates.append(rv)
        best = int(np.argmax(tuning))
        lam = GRID[best]
        model = stage(phase + '/ridge_refit', lambda: r.fit_ridge_from_packets(full_packets, regularization=lam))
        rt = r.predict(z, model)
        qv, uv = qa(fit, phase, val, 'train', union and index == 0)
        qvz, qtz, qc = r.standardize(qv, qt)
        rvz, rtz, rc = r.standardize(candidates[best], rt)
        aps = np.array([r.metric(y[val], w * rvz + (1-w) * qvz)[1] for w in weights])
        cw = weights[aps.argmax(0)]
        gw = weights[aps.mean(1).argmax()]
        fs = cw * rtz + (1-cw) * qtz
        r.save(root / f'{phase}_selection.json', dict(alpha=lam, validation_ridge_maps=tuning,
                classwise_weights=cw.tolist(), global_weight=float(gw),
                qa_calibration=np.asarray(qc).tolist(), ridge_calibration=np.asarray(rc).tolist(),
                test_labels_used=False))
        np.save(root / f'{phase}_full_scores.npy', fs)
        np.save(root / f'{phase}_ridge_scores.npy', rt)
        full_scores.append(fs)
        ridge_scores.append(rt)
        # Candidate validation payload once/client, labels retain their actual dtype.
        messages.extend(dict(phase=phase, stage='validation_scores_labels', client=cid,
                             bytes=int(np.sum(a[val] == cid) * y.shape[1] * ((len(GRID)+1)*4+y.dtype.itemsize)))
                        for cid in range(clients))
        if index == 0:
            predictions.update({'Pooled_Ridge': rt, 'Full_1': fs,
                                'Fixed_0.5': .5*rtz+.5*qtz,
                                'Global_fusion': gw*rtz+(1-gw)*qtz})
            # Exact same feature/penalty centralized oracle is independently fitted.
            central = stage('central_ridge', lambda: r.fit_ridge_from_packets(
                (r.make_ridge_packet(x, y),), regularization=lam))
            central_scores = r.predict(z, central)
            np.testing.assert_allclose(central_scores, rt, atol=2e-5, rtol=2e-5)
            predictions['Central_Ridge'] = central_scores
            if union:
                local_curves = []
                for local_lam in GRID:
                    model = stage(f'local_alpha{local_lam}', lambda: r.average_adapters(
                        [r.fit_ridge_from_packets((p,), regularization=local_lam) for p in pp if p.count],
                        [p.count for p in pp if p.count], 'sample-weighted local ridge'))
                    local_curves.append(r.metric(y[val], r.predict(x[val], model))[0])
                local_lam = GRID[int(np.argmax(local_curves))]
                local = stage('local_refit', lambda: r.average_adapters(
                    [r.fit_ridge_from_packets((p,), regularization=local_lam) for p in full_packets],
                    [p.count for p in full_packets], 'sample-weighted local ridge'))
                predictions['Local_Ridge'] = r.predict(z, local)
                uvz, utz, _ = r.standardize(uv, ut)
                uaps = np.array([r.metric(y[val], w*rvz+(1-w)*uvz)[1] for w in weights])
                uw = weights[uaps.argmax(0)]
                predictions['Union_fusion'] = uw*rtz+(1-uw)*utz
                r.save(root / 'ablation_selection.json', dict(local_alpha=local_lam,
                       local_validation_maps=local_curves, union_weights=uw.tolist(),
                       union_scope='local top-K union with existing disclosed confidence threshold clamp'))
        del pp, candidates
        gc.collect()
    predictions['Full_3'] = np.mean(full_scores, axis=0)
    predictions['Pooled_Ridge_3'] = np.mean(ridge_scores, axis=0)
    for name, scores in predictions.items():
        assert np.isfinite(scores).all()
        np.save(root / f'{name}_scores.npy', scores)
    r.save(root / 'selection_complete.json', dict(test_labels_read=False, finished=time.time()))
    # First access to held-out test labels, after every selection is locked.
    yt = np.load(cache / 'test_targets.npy')
    rows = []
    for name, scores in predictions.items():
        m, ap = r.metric(yt, scores)
        rows.append(dict(method=name, mAP=m, class_AP=ap.tolist(),
                         prediction_sha256=r.digest(root / f'{name}_scores.npy')))
    r.save(root / 'result.json', dict(status='COMPLETE', rows=rows, protocol=protocol,
           wall_seconds=time.perf_counter()-start, stages=stages, messages=messages,
           uplink_payload_bytes=sum(v['bytes'] for v in messages),
           cost_scope='recorded algorithm uplink only; no downlink, wire framing or feature extraction; independent controls included in wall time, inspect stages',
           peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
           peak_cuda_MiB=torch.cuda.max_memory_allocated()/2**20))
    print('RUN_COMPLETE', root.name, [(v['method'], v['mAP']) for v in rows], flush=True)


def fedavg(dataset, condition, seed):
    """Fresh validation selection per outer run, fixed declared budget."""
    import run_fair_linear as linear
    from fair_baselines_common import features
    root = BASE / 'runs' / f'{dataset}_{condition}_m10_s{seed}'
    out = root / 'fedavg'
    if (out / 'result.json').exists():
        return
    a = np.load(root / 'assignment.npy')
    sp = np.load(root / f'split{seed*100+7}.npz')
    fit, val = sp['fit'], sp['validation']
    x, z = features(dataset, 'train'), features(dataset, 'test')
    y = r.targets(dataset, 'train')
    tuning = []
    for lr in (.1, 1., 10.):
        config = dict(optimizer='sgd', lr=lr, weight_decay=0.)
        _, result = linear.fit(x[fit], y[fit], a[fit], config, 'fedavg', seed,
                               200, out / f'validation_lr{lr}', x[val], y[val])
        if result['status'] == 'COMPLETE':
            tuning.append(dict(config=config, result=result))
    assert tuning, 'all declared FedAvg candidates failed'
    best = max(tuning, key=lambda v: v['result']['best_validation_mAP'])
    r.save(out / 'selection.json', dict(candidates=tuning, selected=best, test_labels_read=False,
                                       budget='3 SGD rates x 200 rounds; evaluation every 20 rounds; one epoch/client/round'))
    state, result = linear.fit(x, y, a, best['config'], 'fedavg', seed,
                              best['result']['best_round'], out / 'refit')
    assert result['status'] == 'COMPLETE'
    scores = z @ state['weight'].numpy().T + state['bias'].numpy()
    np.save(out / 'test_scores.npy', scores)
    m, ap = r.metric(r.targets(dataset), scores)
    r.save(out / 'result.json', dict(status='COMPLETE', mAP=m, class_AP=ap.tolist(),
           selection_uplink_bytes=sum(v['result']['uplink_bytes'] for v in tuning),
           selection_downlink_bytes=sum(v['result']['downlink_bytes'] for v in tuning),
           refit=result, scope='outer-partition reference; new 3-candidate budget, separate from historical 18-candidate strong baseline'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['coco', 'nus'])
    p.add_argument('--condition', default='iid')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--clients', type=int, default=10)
    p.add_argument('--union', action='store_true')
    p.add_argument('--fedavg', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    with threadpool_limits(limits=4):
        if args.fedavg:
            fedavg(args.dataset, args.condition, args.seed)
        else:
            run(args.dataset, args.condition, args.seed, args.clients, args.union)
