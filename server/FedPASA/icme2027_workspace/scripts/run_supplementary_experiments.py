"""Reviewer supplements; existing artifacts are inputs, never overwritten.

All model/ensemble choices are locked from training-side validation. Test labels
are loaded only for reporting. Expensive phases checkpoint at dataset/condition.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fedpasa.benchmarks import BENCHMARKS
from fedpasa.core import (Adapter, PatchBank, _gda_from_samples,
    _global_quantile_selection, _purify_bank, fit_central_piaa,
    fit_quantile_aligned_fedpasa, mean_average_precision, split_global_bank_by_client)
from fedpasa.supervised import make_ridge_packet, fit_ridge_from_packets
from run_reviewer_evidence import (resolve_cache, load_features, predict,
    adapter_errors, score_errors, average_adapters, fit_tie_safe_piaa)
from run_icme_experiments import deserialize_bank, truncate_bank
from run_label_patch_fusion import piaa_scores

OUT = ROOT / 'outputs/icme/supplementary'
SEEDS = (7, 17, 27)
K = 512


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.partial')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def table(name, rows):
    save(OUT / (name + '.json'), rows)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        with (OUT / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def figure(fig, name):
    (OUT / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / 'figures' / (name + '.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / 'figures' / (name + '.pdf'), bbox_inches='tight')
    plt.close(fig)


def take(value, ix):
    ix = torch.as_tensor(ix, dtype=torch.long)
    return PatchBank(value.features[ix], value.probabilities[ix], value.entropy[ix],
        None if value.image_indices is None else value.image_indices[ix],
        None if value.patch_ids is None else value.patch_ids[ix])


def merge(banks):
    result = {}
    for c in sorted({c for b in banks for c in b}):
        values = [b[c] for b in banks if c in b]
        def cat(attr):
            parts = [getattr(v, attr) for v in values]
            return torch.cat(parts) if all(p is not None for p in parts) else None
        result[c] = PatchBank(*(cat(a) for a in
            ('features', 'probabilities', 'entropy', 'image_indices', 'patch_ids')))
    return result


def evaluate_many(dataset, adapters, output_dir, batch=256):
    """Direct full-test inference, sharing disk reads without copying scores."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = {n: a for n, a in adapters.items() if not (output_dir / (n+'.npy')).exists()}
    cache = resolve_cache(dataset)
    patches = np.load(cache / 'test_patch.npy', mmap_mode='r')
    cls = np.load(cache / 'test_cls.npy', mmap_mode='r')
    text = F.normalize(torch.from_numpy(np.load(cache / 'text_features.npy')).cuda().float(), dim=-1)
    b = BENCHMARKS[dataset]
    outputs = {n: np.empty((len(cls), len(text)), np.float32) for n in pending}
    weights = {n: (a.weight.cuda(), a.bias.cuda()) for n, a in pending.items()}
    with torch.inference_mode():
        for start in range(0, len(cls) if pending else 0, batch):
            stop = min(len(cls), start+batch)
            x = F.normalize(torch.from_numpy(np.array(patches[start:stop], dtype=np.float32)).cuda(), dim=-1)
            z = F.normalize(torch.from_numpy(np.array(cls[start:stop], dtype=np.float32)).cuda(), dim=-1)
            cp = torch.softmax(z @ text.T * b.cls_scale, dim=-1)
            for name, (w, bias) in weights.items():
                p = torch.softmax((x @ w + bias)/500, dim=-1).amax(1)
                outputs[name][start:stop] = (0.1*cp + 0.9*torch.softmax(p*b.patch_scale, dim=-1)).cpu().numpy()
            if start % (batch*100) == 0:
                print(f'EVALUATE {dataset} {start}/{len(cls)} adapters={len(pending)}', flush=True)
    for name, scores in outputs.items():
        np.save(output_dir / (name+'.npy'), scores)
    return {n: np.load(output_dir / (n+'.npy')) for n in adapters}


def fusion_analysis(datasets):
    classrows, fusionrows, reprows = [], [], []
    for dataset in datasets:
        dr = ROOT / 'outputs/icme' / dataset
        targets = np.load(resolve_cache(dataset) / 'test_targets.npy')
        qa = np.load(dr / 'scores/central_piaa_scores.npy')
        collections = {n: [] for n in ('QA only', 'Ridge only', 'Fixed 0.5', 'Global learned', 'Classwise learned')}
        weights = []
        effective = []
        for seed in SEEDS:
            sr = dr / 'full_seeds' / f'seed{seed}'
            fr = sr / 'ridge_patch_fusion'
            ridge = np.load(sr / 'ridge/test_scores.npy')
            qc, rc = np.load(fr/'piaa_calibration.npy'), np.load(fr/'ridge_calibration.npy')
            qz, rz = (qa-qc[0])/qc[1], (ridge-rc[0])/rc[1]
            report = read(fr / 'result.json')
            alpha = np.array(report['classwise_weight']['weights'])
            weights.append(alpha)
            effective.append((alpha / rc[1], (1-alpha) / qc[1]))
            collections['QA only'].append(qz)
            collections['Ridge only'].append(rz)
            collections['Fixed 0.5'].append(0.5*rz+0.5*qz)
            collections['Global learned'].append(np.load(fr/'test_scores_global.npy'))
            collections['Classwise learned'].append(np.load(fr/'test_scores_classwise.npy'))
        aggregate = {n: np.mean(v, axis=0, dtype=np.float64).astype(np.float32) for n,v in collections.items()}
        for name, scores in aggregate.items():
            value, _ = mean_average_precision(targets, scores)
            seed_maps = [mean_average_precision(targets, x)[0] for x in collections[name]]
            fusionrows.append(dict(dataset=dataset, variant=name, mAP=value,
                seed_mean=float(np.mean(seed_maps)), seed_std=float(np.std(seed_maps, ddof=1)),
                selection='three fixed training-side splits; affine score averaging', test_selection=False))
        np.testing.assert_allclose(aggregate['Classwise learned'], np.load(dr/'cv_fusion/test_scores.npy'), atol=2e-6)
        # Keep the original paper's raw three-split Ridge score mean so its
        # tolerance-based class counts remain comparable (not a new baseline).
        ridge = np.mean([np.load(dr/'full_seeds'/f'seed{s}'/'ridge/test_scores.npy') for s in SEEDS],axis=0)
        _, rap = mean_average_precision(targets, ridge)
        _, qap = mean_average_precision(targets, qa)
        _, fap = mean_average_precision(targets, aggregate['Classwise learned'])
        w = np.stack(weights)
        for c,name in enumerate(BENCHMARKS[dataset].classes):
            classrows.append(dict(dataset=dataset, class_id=c, class_name=name,
                alpha_seed7=float(w[0,c]), alpha_seed17=float(w[1,c]), alpha_seed27=float(w[2,c]),
                alpha_mean=float(w[:,c].mean()), alpha_min=float(w[:,c].min()), alpha_max=float(w[:,c].max()),
                QA_AP=qap[c], Ridge_AP=rap[c], Full_AP=fap[c], Full_minus_Ridge=fap[c]-rap[c],
                QA_wins=bool(qap[c]>rap[c]+1e-6), Ridge_wins=bool(rap[c]>qap[c]+1e-6),
                fusion_beats_both=bool(fap[c]>max(rap[c],qap[c])+1e-6)))
        fig, axes = plt.subplots(2,1,figsize=(max(10,len(rap)*0.15),6), sharex=True)
        order = np.argsort(fap-rap)
        x = np.arange(len(order))
        axes[0].bar(x, w.mean(0)[order], color='#356B9A')
        axes[0].vlines(x, w.min(0)[order], w.max(0)[order], color='#111111', lw=0.8)
        axes[0].set(ylabel='Ridge weight alpha (mean; min–max)', ylim=(0,1.08), title=dataset.upper()+' — three training-side selection splits')
        delta = (fap-rap)[order]
        axes[1].bar(x, delta, color=np.where(delta>=0,'#258477','#BA4A48'))
        axes[1].axhline(0,color='black',lw=0.6)
        axes[1].set(ylabel='Full − Ridge AP (points)', xticks=x,
            xticklabels=np.array(BENCHMARKS[dataset].classes)[order])
        axes[1].tick_params(axis='x', labelrotation=90, labelsize=7)
        fig.tight_layout()
        figure(fig, dataset+'_complementarity')
        for tag in ('cls','cls_patch_mean','cls_patch_max_text','cls_patch_mean_patch_max_text'):
            rr = dr / ('label_ridge' if tag=='cls' else 'label_ridge_'+tag)
            rep = read(rr/'result.json')
            saved = np.load(rr/'test_scores.npy')
            measured = mean_average_precision(targets,saved)[0]
            assert abs(measured-rep['test']['mAP']) < 1e-6
            reprows.append(dict(dataset=dataset, features='+'.join(rep['protocol']['features']),
                dimension={'cls':512,'cls_patch_mean':1024,'cls_patch_max_text':512+len(rap),'cls_patch_mean_patch_max_text':1024+len(rap)}[tag],
                alpha=rep['selection']['alpha'], validation_mAP=rep['selection']['validation_mAP'],mAP=measured,
                source=str(rr.relative_to(ROOT)), test_selection=False))
        print('FUSION_ANALYSIS '+dataset, flush=True)
    table('classwise_complementarity',classrows)
    table('fusion_strategies',fusionrows)
    table('representation_ablation',reprows)


def fed3r(dataset, assignments):
    """Fed3R-style multi-label linear ridge: no centering, no fitted bias."""
    cache, dr = resolve_cache(dataset), ROOT/'outputs/icme'/dataset
    x = np.array(np.load(cache/'train_cls.npy'),dtype=np.float32)
    x /= np.linalg.norm(x,axis=1,keepdims=True).clip(1e-12)
    y = np.load(cache/'train_targets.npy').astype(np.float64)
    fit = np.load(dr/'full_seeds/seed7/ridge/fit_indices.npy')
    val = np.load(dr/'full_seeds/seed7/ridge/validation_indices.npy')
    xf = x[fit].astype(np.float64)
    a, b = xf.T@xf, xf.T@y[fit]
    tuning = []
    for lam in (1e-6,1e-5,1e-4,1e-3,1e-2,0.1,1.,10.,100.):
        w = np.linalg.solve(a+lam*np.eye(x.shape[1]),b)
        tuning.append(dict(regularization=lam,validation_mAP=mean_average_precision(y[val],x[val]@w)[0]))
    choice = max(tuning,key=lambda z:z['validation_mAP'])
    aa, bb = np.zeros_like(a), np.zeros_like(b)
    for client in np.unique(assignments):
        mask = assignments==client
        z = x[mask].astype(np.float64)
        aa += z.T@z
        bb += z.T@y[mask]
    w = np.linalg.solve(aa+choice['regularization']*np.eye(x.shape[1]),bb)
    test = np.array(np.load(cache/'test_cls.npy'),dtype=np.float32)
    test /= np.linalg.norm(test,axis=1,keepdims=True).clip(1e-12)
    scores = (test@w).astype(np.float32)
    dd=OUT/dataset/'fed3r'
    dd.mkdir(parents=True,exist_ok=True)
    np.save(dd/'test_scores.npy',scores)
    save(dd/'result.json',dict(method='Fed3R-style multi-label CLS ridge', tuning=tuning,
        selection=choice, features='same frozen CLIP CLS', bias=False,
        target='multi-hot labels; adaptation of the single-label method',
        source='https://arxiv.org/abs/2406.01116',test_labels_used_for_selection=False))
    return scores


def mechanism(dataset):
    destination=OUT/dataset/'mechanism.json'
    if destination.exists():
        return read(destination)
    cache,dr=resolve_cache(dataset),ROOT/'outputs/icme'/dataset
    assignments=np.load(dr/'partitions/clients10_seed7.npy')
    text=torch.from_numpy(np.load(cache/'text_features.npy')).float()
    global_bank=truncate_bank(deserialize_bank(torch.load(dr/'banks/global_k1024.pt',weights_only=True)),K)
    local=split_global_bank_by_client(global_bank,assignments,clients=10)
    temperature=BENCHMARKS[dataset].purification_temperature
    central=fit_central_piaa(global_bank,text,purification_temperature=temperature)
    qa=fit_quantile_aligned_fedpasa(local,text,k=K,purification_temperature=temperature)
    local_adapters=[fit_tie_safe_piaa(b,text,purification_temperature=temperature) for b in local if b]
    avg=average_adapters(local_adapters,[a.metadata['selected_patches'] for a in local_adapters],'aligned_local_GDA_average')
    # Stronger control: even the purified population is identical to centralized.
    fx,pr,labels,keep=_purify_bank(global_bank,text,temperature=temperature)
    origins=torch.cat([b.image_indices for _,b in sorted(global_bank.items())])[keep]
    owners=assignments[origins.numpy()]
    pure_adapters,pure_counts=[],[]
    for client in range(10):
        mask=torch.from_numpy(owners==client)
        if mask.any():
            w,b,_=_gda_from_samples(fx[mask],pr[mask],labels[mask],text)
            pure_adapters.append(Adapter(w.float(),b.float(),{})); pure_counts.append(int(mask.sum()))
    pure=average_adapters(pure_adapters,pure_counts,'identical_purified_population_local_GDA_average')
    adapters={'Centralized':central,'QAFedPASA':qa,'GlobalQuota-GDA-Avg':avg,'SharedPurification-GDA-Avg':pure}
    score_dir=OUT/dataset/'mechanism_scores'
    scores=evaluate_many(dataset,adapters,score_dir)
    targets=np.load(cache/'test_targets.npy')
    rows=[]
    for name,adapter in adapters.items():
        rows.append(dict(dataset=dataset,method=name,branch='QA',partition='kmeans',clients=10,seed=7,
            selection='identical centralized Global Top-K assigned to owning clients',
            purification='shared global' if name!='GlobalQuota-GDA-Avg' else 'client local (clamp safeguard disclosed)',
            mAP=mean_average_precision(targets,scores[name])[0],
            **adapter_errors(central,adapter),**score_errors(scores['Centralized'],scores[name]),
            score_provenance='direct full-test inference',reference='Centralized QA'))
    # Reuse the audited local Top-K controls, with source retained.
    old=read(ROOT/'outputs/icme/paper/estimator_preserving_evidence.json')['rows']
    for r in old:
        if r['dataset']==dataset and r['partition']=='kmeans' and r['seed']==7 and r['method'].startswith('Naive-'):
            rows.append({**r,'source':'outputs/icme/paper/estimator_preserving_evidence.json'})
    train,test=load_features(cache,'train'),load_features(cache,'test')
    y=np.load(cache/'train_targets.npy')
    alpha=read(dr/'full_seeds/seed7/ridge/result.json')['selection']['alpha']
    central_r=fit_ridge_from_packets((make_ridge_packet(train,y),),regularization=alpha)
    packets=tuple(make_ridge_packet(train[assignments==c],y[assignments==c]) for c in range(10))
    fed_r=fit_ridge_from_packets(packets,regularization=alpha)
    locals_r=[fit_ridge_from_packets((p,),regularization=alpha) for p in packets]
    avg_r=average_adapters(locals_r,[p.count for p in packets],'Local-Ridge-Avg')
    cr_scores=predict(test,central_r)
    for name,a in [('Centralized Ridge',central_r),('Pooled Ridge',fed_r),('Local-Ridge-Avg',avg_r)]:
        ss=predict(test,a)
        rows.append(dict(dataset=dataset,method=name,branch='Ridge',partition='kmeans',clients=10,seed=7,
            selection='same complete training images',purification='none',mAP=mean_average_precision(targets,ss)[0],
            **adapter_errors(central_r,a),**score_errors(cr_scores,ss),reference='Centralized Ridge',score_provenance='direct full-test inference'))
    ss=fed3r(dataset,assignments)
    rows.append(dict(dataset=dataset,method='Fed3R-style CLS ridge',branch='CLS',partition='kmeans',clients=10,seed=7,
        mAP=mean_average_precision(targets,ss)[0],selection='same complete training images',reference='standalone baseline',score_provenance='direct full-test inference'))
    full=np.load(dr/'cv_fusion/test_scores.npy')
    rows.append(dict(dataset=dataset,method='FedPASA-Full',branch='Full',partition='kmeans',clients=10,seed=7,
        mAP=mean_average_precision(targets,full)[0],selection='same complete training images',reference='locked three-split fusion',score_provenance='existing locked full scores'))
    save(destination,rows)
    return rows


def entropy_cache(dataset):
    """Compute full-population entropy BEFORE any selection or quantization."""
    dd=OUT/dataset
    if (dd/'entropy_manifest.json').exists():
        return np.load(dd/'entropy.npy',mmap_mode='r'),np.load(dd/'prediction.npy',mmap_mode='r')
    dd.mkdir(parents=True,exist_ok=True)
    cache=resolve_cache(dataset)
    patches=np.load(cache/'train_patch.npy',mmap_mode='r')
    shape=patches.shape[:2]
    entropy=np.lib.format.open_memmap(dd/'entropy.npy',mode='w+',dtype='float32',shape=shape)
    prediction=np.lib.format.open_memmap(dd/'prediction.npy',mode='w+',dtype='int16',shape=shape)
    text=F.normalize(torch.from_numpy(np.load(cache/'text_features.npy')).float().cuda(),dim=-1)
    with torch.inference_mode():
        for start in range(0,len(patches),256):
            stop=min(start+256,len(patches))
            x=F.normalize(torch.from_numpy(np.array(patches[start:stop],dtype='float32')).cuda(),dim=-1)
            logits=x@text.T
            p=torch.softmax(logits*60,dim=-1)
            entropy[start:stop]=(-(p*p.clamp_min(1e-10).log()).sum(-1)).cpu().numpy()
            prediction[start:stop]=logits.argmax(-1).cpu().numpy()
            if start%25600==0: print(f'ENTROPY {dataset} {stop}/{len(patches)}',flush=True)
    entropy.flush();prediction.flush()
    save(dd/'entropy_manifest.json',dict(full_training_images=len(patches),patches_per_image=shape[1],
        source=str(cache/'train_patch.npy'),quantization_before_selection=True))
    return entropy,prediction


def materialize(dataset,ids_by_class,entropy):
    cache=resolve_cache(dataset)
    patches=np.load(cache/'train_patch.npy',mmap_mode='r')
    pp=patches.shape[1]
    text=F.normalize(torch.from_numpy(np.load(cache/'text_features.npy')).float().cuda(),dim=-1)
    bank={}
    with torch.inference_mode():
        for c,ids in ids_by_class.items():
            if len(ids)==0: continue
            # CPU normalization is independent of client grouping / batch sizes.
            features=F.normalize(torch.from_numpy(np.array(patches[ids//pp,ids%pp],dtype='float32')),dim=-1)
            probabilities=torch.softmax(features.cuda()@text.T*60,dim=-1).cpu()
            bank[c]=PatchBank(features,probabilities,torch.from_numpy(np.array(entropy[ids])),
                torch.from_numpy(ids//pp),torch.from_numpy(ids))
    return bank


def tie_stress(dataset):
    dd=OUT/dataset/'ties';dd.mkdir(parents=True,exist_ok=True)
    raw,pred=entropy_cache(dataset)
    e0=np.asarray(raw).reshape(-1); pred=np.asarray(pred).reshape(-1)
    assignments=np.load(ROOT/'outputs/icme'/dataset/'partitions/clients10_seed7.npy')
    owners=np.repeat(assignments,raw.shape[1])
    text=torch.from_numpy(np.load(resolve_cache(dataset)/'text_features.npy')).float()
    targets=np.load(resolve_cache(dataset)/'test_targets.npy')
    rows=[]
    for decimals in (None,4,3,2):
        name='fp32' if decimals is None else f'round_1e-{decimals}'
        path=dd/(name+'.json')
        if path.exists(): rows.extend(read(path));continue
        e=e0 if decimals is None else np.round(e0,decimals)
        centers,local_ids,shared_ids={},[{} for _ in range(10)],[{} for _ in range(10)]
        boundary_tied=0;boundary_total=0
        for c in range(len(text)):
            ids=np.flatnonzero(pred==c)
            if not len(ids):continue
            # Central reference is lexicographic (rounded entropy, global ID).
            order=np.lexsort((ids,e[ids])); chosen=ids[order[:K]];centers[c]=chosen
            cutoff=e[chosen[-1]]
            boundary_tied+=int(np.count_nonzero(e[ids]==cutoff)>1)
            boundary_total+=len(ids)
            for client in range(10):
                ci=ids[owners[ids]==client]
                if not len(ci):continue
                count=min(K,len(ci))
                # Unspecified local tie handling: numpy argpartition locally,
                # torch topk at server. Actual patch IDs, no feature fingerprints.
                local_ids[client][c]=ci[np.argpartition(e[ci],count-1)[:count]]
                shared_ids[client][c]=ci[np.lexsort((ci,e[ci]))[:count]]
        central_bank=materialize(dataset,centers,e)
        central=fit_central_piaa(central_bank,text,purification_temperature=BENCHMARKS[dataset].purification_temperature)
        adapters={'central':central};recalls={};counts={}
        center_set=set(int(i) for ids in centers.values() for i in ids)
        for policy,ids in [('local',local_ids),('shared',shared_ids)]:
            banks=tuple(materialize(dataset,b,e) for b in ids)
            selections=_global_quantile_selection(banks,K,shared_tie_key=policy=='shared')
            selected={int(i) for bank,selection in zip(banks,selections) for c,ix in selection.items() for i in bank[c].patch_ids[ix]}
            recalls[policy]=len(center_set&selected)/len(center_set)
            counts[policy]=len(selected)
            adapters[policy]=fit_quantile_aligned_fedpasa(banks,text,k=K,
                purification_temperature=BENCHMARKS[dataset].purification_temperature,shared_tie_key=policy=='shared')
            if policy=='shared':assert selected==center_set
            del banks
        scores=evaluate_many(dataset,adapters,dd/name)
        current=[]
        central_map=mean_average_precision(targets,scores['central'])[0]
        for policy in ('local','shared'):
            current.append(dict(dataset=dataset,entropy=name,tie_policy=policy,clients=10,seed=7,
                full_training_patches=len(e),classes_with_tied_cutoff=boundary_tied,
                patch_id_recall=recalls[policy],selected_patches=counts[policy],
                mAP=mean_average_precision(targets,scores[policy])[0],central_mAP=central_map,
                **adapter_errors(central,adapters[policy]),**score_errors(scores['central'],scores[policy]),
                extra_tie_key_MiB=adapters[policy].metadata['tie_key_uplink_bytes']/1024**2,
                score_provenance='direct full-test inference; freshly recomputed central for each precision'))
        save(path,current);rows.extend(current)
        print('TIE_RESULT '+json.dumps(current),flush=True)
    return rows


def scaling(dataset):
    dest=OUT/dataset/'scaling.json'
    if dest.exists() and all('Full_mAP' in r for r in read(dest)):return read(dest)
    cache,dr=resolve_cache(dataset),ROOT/'outputs/icme'/dataset
    text=torch.from_numpy(np.load(cache/'text_features.npy')).float()
    bank=truncate_bank(deserialize_bank(torch.load(dr/'banks/global_k1024.pt',weights_only=True)),K)
    central=fit_central_piaa(bank,text,purification_temperature=BENCHMARKS[dataset].purification_temperature)
    train,test=load_features(cache,'train'),load_features(cache,'test')
    y=np.load(cache/'train_targets.npy');targets=np.load(cache/'test_targets.npy')
    alpha=read(dr/'full_seeds/seed17/ridge/result.json')['selection']['alpha']
    central_packet=make_ridge_packet(train,y)
    cr=fit_ridge_from_packets((central_packet,),regularization=alpha)
    crs=predict(test,cr)
    seed_alphas={seed:read(dr/'full_seeds'/f'seed{seed}'/'ridge/result.json')['selection']['alpha'] for seed in SEEDS}
    central_by_alpha={a:predict(test,fit_ridge_from_packets((central_packet,),regularization=a)) for a in set(seed_alphas.values())}
    # Reuse every existing non-IID client-count partition; no extra seeds.
    rows=[];adapters={'central':central};ridges={}
    for clients in BENCHMARKS[dataset].client_counts():
        assignment=np.load(dr/'partitions'/f'clients{clients}_seed17.npy')
        local=split_global_bank_by_client(bank,assignment,clients=clients)
        start=perf_counter()
        qa=fit_quantile_aligned_fedpasa(local,text,k=K,purification_temperature=BENCHMARKS[dataset].purification_temperature)
        qa_seconds=perf_counter()-start
        start=perf_counter()
        packets=tuple(make_ridge_packet(train[assignment==c],y[assignment==c]) for c in range(clients))
        packet_seconds=perf_counter()-start
        start=perf_counter();ridge=fit_ridge_from_packets(packets,regularization=alpha);solve=perf_counter()-start
        rs=predict(test,ridge)
        ridges[clients]={a:predict(test,fit_ridge_from_packets(packets,regularization=a)) for a in set(seed_alphas.values())}
        adapters[str(clients)]=qa
        rows.append(dict(dataset=dataset,clients=clients,seed=17,
            ridge_mAP=mean_average_precision(targets,rs)[0],ridge_weight_error=adapter_errors(cr,ridge)['max_abs_weight_error'],
            ridge_score_error=score_errors(crs,rs)['max_abs_score_error'],
            QA_weight_error=adapter_errors(central,qa)['max_abs_weight_error'],
            ridge_server_solve_seconds=solve,ridge_client_packet_seconds=packet_seconds,
            QA_protocol_fit_seconds=qa_seconds,QA_FP32_MiB=qa.metadata['estimated_uplink_MiB'],
            ridge_FP64_MiB=ridge.metadata['estimated_communication_MiB'],
            total_FP64_ridge_MiB=qa.metadata['estimated_uplink_MiB']+ridge.metadata['estimated_communication_MiB'],
            total_FP32_ridge_MiB=qa.metadata['estimated_uplink_MiB']+ridge.metadata['estimated_communication_MiB']/2,
            selection_scope='fixed centralized Top-K split to owners; selection recovery tested separately'))
        print(f'SCALING {dataset} M={clients} ridge={rows[-1]["ridge_mAP"]:.5f}',flush=True)
    scores=evaluate_many(dataset,adapters,OUT/dataset/'scaling_scores')
    # Use all three locked affine fusion rules with one partitioned ridge solve
    # per corresponding selected alpha (normally equal on COCO/NUS).
    for row in rows:
        q=scores[str(row['clients'])]
        row['QA_mAP']=mean_average_precision(targets,q)[0]
        row['QA_score_error']=score_errors(scores['central'],q)['max_abs_score_error']
        row['central_QA_mAP']=mean_average_precision(targets,scores['central'])[0]
        row['central_ridge_mAP']=mean_average_precision(targets,crs)[0]
        full_parts,central_parts=[],[]
        for seed in SEEDS:
            fr=dr/'full_seeds'/f'seed{seed}'/'ridge_patch_fusion'
            qc,rc=np.load(fr/'piaa_calibration.npy'),np.load(fr/'ridge_calibration.npy')
            w=np.array(read(fr/'result.json')['classwise_weight']['weights'],dtype=np.float32)
            a=seed_alphas[seed]
            full_parts.append(w*(ridges[row['clients']][a]-rc[0])/rc[1]+(1-w)*(q-qc[0])/qc[1])
            central_parts.append(w*(central_by_alpha[a]-rc[0])/rc[1]+(1-w)*(scores['central']-qc[0])/qc[1])
        full=np.mean(full_parts,axis=0,dtype=np.float64).astype(np.float32)
        center_full=np.mean(central_parts,axis=0,dtype=np.float64).astype(np.float32)
        row['Full_mAP']=mean_average_precision(targets,full)[0]
        row['central_Full_mAP']=mean_average_precision(targets,center_full)[0]
        row['Full_score_error']=score_errors(center_full,full)['max_abs_score_error']
    save(dest,rows)
    return rows


def communication(datasets):
    rows=[]
    for dataset in datasets:
        b=BENCHMARKS[dataset];c=b.class_count;d=512;p=196;f=2*d+c;m=b.client_counts()[3]
        n=read(resolve_cache(dataset)/'manifest.json')['train_images']
        qa=m*(c*K+2*(d*d+2*c*d+2*c)+3*c)*4
        ridge_scalars=m*(f*f+f*c+f+c+1)
        formulas=[('Global selected patch embeddings (oracle payload lower bound)',c*K*d*4,'C*K*D*4; global selection coordination excluded'),
            ('Local selected patch embeddings',m*c*K*d*4,'M*C*K*D*4; dense upper bound, selection changes'),
            ('All CLS features + labels',n*(d+c)*4,'N*(D+C)*4'),
            ('All patch features + labels',n*(p*d+c)*4,'N*(P*D+C)*4'),
            ('Local GDA models',m*(d*c+c)*4,'M*(D*C+C)*4; not estimator preserving'),
            ('FedPASA QA FP32',qa,'M*(C*K+2*(D^2+2*C*D+2*C)+3*C)*4'),
            ('FedPASA Ridge FP32',ridge_scalars*4,'M*(F^2+F*C+F+C+1)*4'),
            ('FedPASA Ridge FP64 (actual accumulation)',ridge_scalars*8,'M*(F^2+F*C+F+C+1)*8'),
            ('FedPASA total FP32 wire',qa+ridge_scalars*4,'QA FP32 + Ridge FP32'),
            ('FedPASA total FP64 ridge reference',qa+ridge_scalars*8,'QA FP32 + Ridge FP64')]
        for name,size,formula in formulas:
            rows.append(dict(dataset=dataset,scheme=name,clients=m,MiB=size/1024**2,bytes=size,
                formula=formula,direction='client-to-server only',accounting='analytical dense payload; no transport headers/downlink'))
    table('communication_baselines',rows)


def manifest(datasets):
    rows=[]
    for d in datasets:
        cache=resolve_cache(d);dr=ROOT/'outputs/icme'/d
        for name in ('train_targets.npy','test_targets.npy','text_features.npy','manifest.json'):
            path=cache/name
            rows.append(dict(dataset=d,path=str(path),sha256=digest(path)))
        for path in [dr/'partitions/clients10_seed7.npy', dr/'full_seeds/seed7/ridge/fit_indices.npy',dr/'full_seeds/seed7/ridge/validation_indices.npy']:
            if path.exists():rows.append(dict(dataset=d,path=str(path),sha256=digest(path)))
    save(OUT/'protocol_manifest.json',dict(protocol='CLIP ViT-B/16, 224 input, unchanged image order, cache labels and mAP implementation',
        test_labels_used_for_selection=False,artifacts=rows,
        feature_cache_identity='paths, byte sizes, checkpoint SHA256 from extraction manifest; target/partition SHA256 above',
        torch=torch.__version__,numpy=np.__version__,device=torch.cuda.get_device_name(0)))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('analysis','mechanism','ties','scaling','all'),default='all')
    parser.add_argument('--datasets',nargs='+',default=list(BENCHMARKS))
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    with threadpool_limits(limits=4):
        manifest(args.datasets)
        if args.phase in ('analysis','all'):
            fusion_analysis(args.datasets);communication(args.datasets)
        if args.phase in ('mechanism','all'):
            table('estimator_mismatch',[r for d in args.datasets if d in ('coco','nus') for r in mechanism(d)])
        if args.phase in ('ties','all'):
            table('boundary_ties',[r for d in args.datasets if d in ('coco','nus') for r in tie_stress(d)])
        if args.phase in ('scaling','all'):
            table('client_scaling',[r for d in args.datasets for r in scaling(d)])


if __name__=='__main__':main()
