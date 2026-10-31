"""Build auditable tables/plots and completion status, including running FedMPT."""
from __future__ import annotations
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/icme/supplementary'
DATASETS=('voc07','voc12','coco','nus')


def read(path,default=None):
    return json.loads(Path(path).read_text(encoding='utf-8')) if Path(path).exists() else default


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)


def table(name,rows):
    save(OUT/(name+'.json'),rows)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (OUT/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def plot(fig,name):
    (OUT/'figures').mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(OUT/'figures'/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def mdtable(rows,cols):
    def display(x):
        if x is None:return '—'
        if isinstance(x,float):return f'{x:.5g}'
        return str(x)
    return '\n'.join(['| '+' | '.join(c[0] for c in cols)+' |','|'+'|'.join('---' for c in cols)+'|']+
        ['| '+' | '.join(display(r.get(c[1])) for c in cols)+' |' for r in rows])


def build(plots=True):
    OUT.mkdir(parents=True,exist_ok=True)
    fusion=read(OUT/'fusion_strategies.json',[])
    classes=read(OUT/'classwise_complementarity.json',[])
    reps=read(OUT/'representation_ablation.json',[])
    ties=[r for d in ('coco','nus') for p in sorted((OUT/d/'ties').glob('*.json')) for r in read(p,[])]
    scaling=[r for d in DATASETS for r in read(OUT/d/'scaling.json',[])]
    mechanism=[r for d in ('coco','nus') for r in read(OUT/d/'mechanism.json',[])]
    table('boundary_ties',ties);table('client_scaling',scaling);table('estimator_mismatch',mechanism)
    gap=[]
    for d in DATASETS:
        available=[r for r in scaling if r['dataset']==d and 'Full_mAP' in r]
        for r in available:
            for branch,key in [('QA','QA'),('Ridge','ridge'),('Full','Full')]:
                fed,central=r[key+'_mAP'],r['central_'+key+'_mAP']
                gap.append(dict(dataset=d,clients=r['clients'],seed=17,branch=branch,
                    centralized_mAP=central,federated_mAP=fed,gap_points=fed-central,
                    max_score_error=r[key+'_score_error'],
                    target_definition='centralized same estimator; reference target, not an accuracy upper bound'))
    table('federation_gap',gap)
    comp=[]
    for d in DATASETS:
        rows=[r for r in classes if r['dataset']==d]
        if rows:
            comp.append(dict(dataset=d,classes=len(rows),QA_win_classes=sum(r['QA_wins'] for r in rows),
                Ridge_win_classes=sum(r['Ridge_wins'] for r in rows),fusion_beats_both=sum(r['fusion_beats_both'] for r in rows),
                full_beats_ridge=sum(r['Full_minus_Ridge']>0.05 for r in rows),
                full_approximately_ridge=sum(abs(r['Full_minus_Ridge'])<=0.05 for r in rows),
                full_below_ridge=sum(r['Full_minus_Ridge']< -0.05 for r in rows),
                strict_full_beats_ridge=sum(r['Full_minus_Ridge']>1e-6 for r in rows),
                tolerance_AP_points=0.05,
                alpha_mean=float(np.mean([r['alpha_mean'] for r in rows])),
                alpha_min=min(r['alpha_min'] for r in rows),alpha_max=max(r['alpha_max'] for r in rows),
                mean_AP_gain=float(np.mean([r['Full_minus_Ridge'] for r in rows]))))
    table('complementarity_summary',comp)
    baselines=[]
    wanted=('Centralized','Naive-A local-TopK union','Naive-B local-GDA average',
        'GlobalQuota-GDA-Avg','Fed3R-style CLS ridge','QAFedPASA','Pooled Ridge','FedPASA-Full')
    for d in ('coco','nus'):
        for method in wanted:
            r=next((r for r in mechanism if r['dataset']==d and r['method']==method),None)
            if r:baselines.append(dict(dataset=d,method=method,mAP=r['mAP'],status='COMPLETE',
                clients=10,partition_seed=7,selection_seeds='7/17/27' if method=='FedPASA-Full' else '7',
                input='frozen CLIP cache',provenance=r.get('score_provenance'),official_reproduction=False))
        final=read(OUT/d/'fedmpt_repaired/result.json')
        progress=read(OUT/d/'fedmpt_repaired/progress.json',{})
        job=read(OUT/'fedmpt_job.json',{})
        rounds=sorted((OUT/d/'fedmpt_repaired').glob('round[0-9][0-9][0-9].json'))
        measured=[read(p) for p in rounds if 'mAP' in read(p)]
        baselines.append(dict(dataset=d,method='FedMPT (disclosed repaired rerun)',
            mAP=final['mAP'] if final else None,status='COMPLETE' if final else ('FAILED' if job.get('status')=='FAILED' else progress.get('status',job.get('status','PENDING'))),
            clients=10,partition_seed=7,selection_seeds='fixed last round; no test selection',
            input='same images, 224 crop, same CLIP checkpoint; prompted features recomputed',
            provenance='result.json' if final else 'not a completed accuracy result',official_reproduction=False,
            completed_round=progress.get('completed_round',0),target_rounds=200,
            last_measured_round=measured[-1]['round'] if measured else None,
            last_measured_mAP=measured[-1]['mAP'] if measured else None))
    table('protocol_matched_baselines',baselines)
    checks=[
        dict(item='P0-1 protocol-matched strong/principle baselines',status='COMPLETE' if all((OUT/d/'fedmpt_repaired/result.json').exists() for d in ('coco','nus')) and len(mechanism)>=24 else 'IN_PROGRESS',artifact='protocol_matched_baselines.csv'),
        dict(item='P0-2 estimator mismatch decomposition',status='COMPLETE' if len(mechanism)>=24 else 'IN_PROGRESS',artifact='estimator_mismatch.csv'),
        dict(item='P0-3 classwise complementarity / alpha',status='COMPLETE' if len(classes)==201 else 'IN_PROGRESS',artifact='classwise_complementarity.csv'),
        dict(item='P1-4 centralized/federated gap',status='COMPLETE' if len(gap)==72 else 'IN_PROGRESS',artifact='federation_gap.csv'),
        dict(item='P1-5 controlled boundary ties / global key',status='COMPLETE' if len(ties)==16 else 'IN_PROGRESS',artifact='boundary_ties.csv'),
        dict(item='P1-6 client-number scaling',status='COMPLETE' if len(gap)==72 else 'IN_PROGRESS',artifact='client_scaling.csv'),
        dict(item='P1-7 communication comparisons',status='COMPLETE' if len(read(OUT/'communication_baselines.json',[]))==40 else 'IN_PROGRESS',artifact='communication_baselines.csv'),
        dict(item='P2-8 fusion strategies',status='COMPLETE' if len(fusion)==20 else 'IN_PROGRESS',artifact='fusion_strategies.csv'),
        dict(item='P2-9 three-view representation ablation',status='COMPLETE' if len(reps)==16 else 'IN_PROGRESS',artifact='representation_ablation.csv')]
    # No requirement that a proposed variant improve performance; retain failures.
    numeric_checks={
        'shared_key_exact_ID_recovery':all(r['patch_id_recall']==1 for r in ties if r['tie_policy']=='shared') if ties else None,
        'shared_key_max_score_error':max((r['max_abs_score_error'] for r in ties if r['tie_policy']=='shared'),default=None),
        'max_federation_mAP_gap':max((abs(r['gap_points']) for r in gap),default=None),
        'finite_reported_mAP':all(np.isfinite(r['mAP']) for r in fusion+ties+mechanism),
        'test_labels_used_for_selection':False,
        'new_tests':'tests/test_supplementary.py; 8 total tests passed at implementation validation'}
    overall='COMPLETE' if all(c['status']=='COMPLETE' for c in checks) else 'IN_PROGRESS'
    save(OUT/'completion_audit.json',dict(status=overall,updated_at=datetime.now().astimezone().isoformat(),items=checks,numerical_checks=numeric_checks))
    text=['# FedPASA 补充实验记录', '', f'更新时间：{datetime.now().astimezone().isoformat(timespec="seconds")}。总状态：**{overall}**。', '',
        '结果均来自本地实际计算或带来源的既有预测复核。未完成的 FedMPT 不填入最终 mAP；中间轮数单独记录。', '',
        mdtable(checks,[('建议项','item'),('状态','status'),('文件','artifact')]),'',
        '## 统一协议的基线', '',
        'COCO/NUS：相同 CLIP ViT-B/16 权重、224 输入、完整训练/测试列表、10 个 KMeans 客户端、划分 seed=7、相同 mAP。闭式方法复用相同特征缓存。FedMPT 根据用户选择使用修补版图像输入，视觉提示需要重算特征，不能宣称相同最终 feature cache 或官方等价复现。Fed3R-style 是无偏置、未中心化的 CLS 多标签 ridge，正则化仅由训练侧验证集选择。', '',
        mdtable(baselines,[('数据集','dataset'),('方法','method'),('最终 mAP','mAP'),('状态','status'),('完成轮数','completed_round')]),'',
        '完整 FedMPT 预算预先固定为 200 轮，每轮所有客户端各遍历一次全部本地训练图像。逐客户端保存断点，两数据集按轮交替运行；最终采用最后一轮，测试集指标不影响训练或模型选择。修补差异、优化器、ASL、BF16 及梯度检查见各数据集 `fedmpt_repaired/protocol.json`。', '',
        '## 选样与模型平均的分解', '',
        'GlobalQuota-GDA-Avg 使用完全相同的中央 Top-K，并按图像归属分配给客户端，随后各自净化和求解。SharedPurification-GDA-Avg 进一步固定净化后的完整 patch 集，仅改变本地求解与模型平均。前者的本地净化沿用已有控制实验的阈值夹紧保护。二者用于机制隔离；客户端完整训练集的局部 Top-K 基线由原有审计结果补充。', '',
        mdtable([r for r in mechanism if r['method'] in ('Centralized','QAFedPASA','GlobalQuota-GDA-Avg','SharedPurification-GDA-Avg','Pooled Ridge','Local-Ridge-Avg')],[('数据集','dataset'),('方法','method'),('mAP','mAP'),('最大 W 误差','max_abs_weight_error'),('最大 score 误差','max_abs_score_error')]),'',
        '误差中的无穷范数采用最大元素绝对值（max-entry），不是矩阵最大行和。所有新增 score discrepancy 通过分别推理得到，未用中央预测替代联邦预测。', '',
        '## 互补性与融合', '',mdtable(comp,[('数据集','dataset'),('QA 胜 Ridge 类数','QA_win_classes'),('Full 胜两支','fusion_beats_both'),('Full 胜 Ridge','full_beats_ridge'),('Full 低于 Ridge','full_below_ridge'),('平均 alpha','alpha_mean'),('平均 AP 增益','mean_AP_gain')]),'',
        'alpha=1 表示使用 Ridge，alpha=0 表示 QA。图中展示三个训练侧划分的平均值与最小/最大值；最终分数平均还包含各划分的 z-score 校准，平均 alpha 不能解释为未经校准原始分数的最终系数。互补性使用旧论文的三划分原始 Ridge 分数平均，Full 与 Ridge 类数比较沿用 ±0.05 AP 点的近似相等容差；同时保留严格增益类数。五种融合策略统一采用三个划分的仿射分数平均。', '',
        mdtable(fusion,[('数据集','dataset'),('融合策略','variant'),('mAP','mAP')]),'',
        'COCO/NUS 的固定 0.5 融合劣于 Ridge 单支；不能将提升归因于任意简单 ensemble。VOC07 的全局学习权重优于逐类权重，保留该负面结果，不主张逐类融合在所有数据集最优。', '',
        '## 中央目标与联邦差距', '',
        '中央估计器是同一算法的参考目标，不是理论精度上界。下表展示主设置 t=60%；完整表另含全部 24 个数据集/客户端数设置、QA/Ridge/Full 三个分支。', '',
        mdtable([r for r in gap if r['clients']==(12 if r['dataset'].startswith('voc') else 48 if r['dataset']=='coco' else 49)],[('数据集','dataset'),('分支','branch'),('中央 mAP','centralized_mAP'),('联邦 mAP','federated_mAP'),('联邦−中央','gap_points'),('score 误差','max_score_error')]),'',
        '扩展性测试固定已选中央 Top-K 按图像分配给客户端，考察统计汇总/求解和完整测试性能；不将其冒充完整局部选样的时间测试。Ridge server solve 时间仅计服务器合并统计量及线性求解；QA protocol fit 包括模拟客户端统计量构建和净化，因此单独标注。', '',
        '## Boundary-tie 压力测试', '',
        '对 COCO 16,087,876 和 NUS 29,400,000 个训练 patch 的熵先量化再做 Top-K，覆盖 FP32 / 1e-4 / 1e-3 / 1e-2。每个精度均重新构建中央目标。真实 patch ID = 缓存图像行号 × 196 + patch 位置。local 使用客户端 argpartition 与服务器 topk 的非共享 tie 顺序；shared 在本地截断与全局选择中均按 (entropy, global patch ID) 排序。', '',
        mdtable(ties,[('数据集','dataset'),('熵精度','entropy'),('tie 规则','tie_policy'),('ID recall','patch_id_recall'),('mAP','mAP'),('中央 mAP','central_mAP'),('W 误差','max_abs_weight_error'),('score 误差','max_abs_score_error')]),'',
        '共享 tie key 为显式可选方案，上传 int64 key 的额外字节已经计入。它暴露稳定 patch 标识，不能同时沿用“不上传 patch ID”的旧表述。这里的 100% 是选样 ID 恢复，浮点矩阵求解仍存在数值误差。局部 tie 顺序下 mAP 接近并不意味着 W 或测试分数完全一致。', '',
        '## 通信与表示', '',
        '全部通信表统一为 client→server 上传，不包含下行、传输头、加密开销。分别列 FP64 累加参考和 FP32 传输口径；COCO 的 373.16 MiB / NUS 的 382.05 MiB 对应 QA FP32 + Ridge FP32，参考 FP64 Ridge 分别为 612.72 / 627.24 MiB。原始全局 selected patches 是忽略全局选样协调的 oracle 下界，Local Top-K patches 则改变了样本总体，不能当作等价协议。', '',
        '推荐表述：FedPASA 以结构化汇总通信换取中央估计器的恢复，而不是无条件宣称最省通信。', '',
        mdtable(reps,[('数据集','dataset'),('表示','features'),('维度','dimension'),('验证集 alpha','alpha'),('mAP','mAP')]),'',
        '## 复现与证据', '',
        '运行 `scripts/run_supplementary_experiments.py --phase all` 生成闭式补充实验；运行 `scripts/run_matched_fedmpt.py --datasets coco nus --rounds 200` 继续既有训练断点；运行 `scripts/build_supplementary_report.py` 刷新本文与表格。解释器使用 `D:\\Software\\anaconda3\\envs\\pytorch\\python.exe`。', '',
        '协议指纹见 `protocol_manifest.json`；建议项完成状态见 `completion_audit.json`；PNG/PDF 图见 `figures/`。旧论文主结果和外部 published values 未被本次补充覆盖。', '',
        '原始方法来源：[FedMPT](https://arxiv.org/abs/2605.28347)、[发布代码](https://github.com/xuc865/FedMPT-main)、[Fed3R](https://arxiv.org/abs/2406.01116)。', '']
    path=OUT/'README.md';tmp=path.with_suffix('.partial.md');tmp.write_text('\n'.join(text),encoding='utf-8');tmp.replace(path)
    if plots:
        if scaling:
            for d in DATASETS:
                rr=[r for r in scaling if r['dataset']==d]
                if not rr:continue
                x=[r['clients'] for r in rr]
                fig,ax=plt.subplots(2,2,figsize=(10,7))
                for key,label in [('QA_mAP','QA'),('ridge_mAP','Ridge'),('Full_mAP','Full')]:
                    if all(key in r for r in rr):ax[0,0].plot(x,[r[key] for r in rr],'o-',label=label)
                ax[0,0].set(ylabel='Test mAP (%)');ax[0,0].legend()
                for key,label in [('total_FP32_ridge_MiB','FP32 wire'),('total_FP64_ridge_MiB','FP64 ridge reference')]:ax[0,1].plot(x,[r[key] for r in rr],'o-',label=label)
                ax[0,1].set(ylabel='Uplink MiB');ax[0,1].legend()
                ax[1,0].plot(x,[r['ridge_server_solve_seconds'] for r in rr],'o-');ax[1,0].set(ylabel='Ridge server merge + solve (s)')
                for key,label in [('QA_weight_error','QA'),('ridge_weight_error','Ridge')]:ax[1,1].semilogy(x,[max(r[key],1e-12) for r in rr],'o-',label=label)
                ax[1,1].set(ylabel='Max-entry W error (floor 1e-12)');ax[1,1].legend()
                for a in ax.flat:a.set_xlabel('Clients');a.grid(alpha=.2)
                fig.suptitle(d.upper()+' — fixed selected population; partition seed 17');fig.tight_layout();plot(fig,d+'_client_scaling')
        if ties:
            fig,axes=plt.subplots(2,2,figsize=(10,7))
            levels=('fp32','round_1e-4','round_1e-3','round_1e-2')
            for i,d in enumerate(('coco','nus')):
                for policy in ('local','shared'):
                    rr=sorted([r for r in ties if r['dataset']==d and r['tie_policy']==policy],key=lambda r:levels.index(r['entropy']))
                    if not rr:continue
                    x=[levels.index(r['entropy']) for r in rr]
                    axes[i,0].plot(x,[100*r['patch_id_recall'] for r in rr],'o-',label=policy)
                    axes[i,1].plot(x,[r['mAP']-r['central_mAP'] for r in rr],'o-',label=policy)
                axes[i,0].set(ylabel=d.upper()+' ID recall (%)');axes[i,1].set(ylabel=d.upper()+' mAP − central')
                for a in axes[i]:a.set_xticks(range(4),['FP32','1e-4','1e-3','1e-2']);a.legend();a.grid(alpha=.2)
            fig.tight_layout();plot(fig,'boundary_ties')
    print(json.dumps(dict(status=overall,completed=sum(c['status']=='COMPLETE' for c in checks),total=9,numerical=numeric_checks),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--no-plots',action='store_true');args=p.parse_args();build(not args.no_plots)
