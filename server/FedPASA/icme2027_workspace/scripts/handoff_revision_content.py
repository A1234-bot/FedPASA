"""Shared, evidence-scoped wording and commands for the handoff revision."""

REVISION_DATE = "2026-09-10 最终结果核验与 ICME 2027 投稿缺口更新"
STATUS = "原定补充实验 9/9 已完成并归档；投稿前补证据清单见第 12 节；无训练在运行"
SERVER_NOTE = (
    "运行交接：2026-09-10 已核验服务器 /root/autodl-tmp/FedPASA。"
    "FedMPT 修补版 COCO/NUS 均完成 200 轮，任务于 2026-09-09 22:50 完成；"
    "当前无该训练进程或 tmux 会话，属于完成后的正常退出。"
    "最终预测、断点和报告已同步本机并复算指标，本机未恢复训练。"
    "审计见 supplementary/final_result_audit_20260910.json；后续实验按第 12 节另建输出目录。"
)
CALIBRATION_SCOPE = (
    "校准与通信边界：现有 run_label_patch_fusion.py 集中读取训练侧 validation 的逐样本预测与标签，"
    "计算校准参数及候选权重的 AP。GDA/Ridge 拟合已有充分统计量接口；分布式校准和选参通信协议尚未实现或验证。"
    "因此，拟合阶段不传单样本的证据不能扩展为整个 Full 流程的端到端传输保证。"
    "现有通信表仅计超参数确定后的拟合统计上传，未包含三划分校准、选参、重拟合全过程及服务器下行成本。"
)
ESTIMATOR_NOTE = (
    "四数据集 × 4 划分 × 6 方法共 96 行。表中记录误差为 0.0e+00 的 16 行 QAFedPASA 结果，"
    "均在权重和偏置误差不超过 1e-5 后复用中央预测；该零值不是独立推理的误差测量。"
    "来源：run_reviewer_evidence.py 的复用分支及 estimator_preserving_evidence.json 的 score_provenance。"
    "Naive 方法最大相对权重误差为 0.909；patch-ID recall 为 98.350%-99.961%，不能将不同 ID 自动视为等价替换。"
    "独立推理证据见第 11 节的 supplementary/estimator_mismatch.csv 与 federation_gap.csv；"
    "这些实验的条件与旧表不同，不互换数值。"
)
RECOVERY_SCOPE = (
    "统计量在同一选样总体下可加；恢复同一中央估计器要求一致的选样及边界 tie 规则，浮点计算按给定容差判断。"
    "默认协议不上传 patch ID，不能保证任意 tie 情况下选中相同 patch；可选 shared-key 协议额外上传 int64 patch ID，见第 11 节。"
)
COMMUNICATION_SCOPE = (
    "本表是超参数已确定后的拟合统计上传量：QAFedPASA 为 4 次上传，按 FP32 dense upper bound 计量；"
    "Ridge 为 1 次上传，按 FP64 参考口径计量。COCO/NUS 合计 612.72/627.24 MiB；"
    "若 Ridge 也按 FP32 传输口径计量，则为 373.16/382.05 MiB，见第 11 节 S07。"
    "这些总量不包含三划分校准、选参及全部重拟合成本，也不包含服务器下行。"
    "运行时来自 10-client KMeans evidence condition，客户端在单机顺序模拟。"
)
REPRO_INTRO = (
    "以下 PowerShell 步骤用于从现有冻结缓存复现主方法及生成既有论文证据所需的组件结果，"
    "不是本次修改时已执行的新实验。应在独立工作区副本执行；脚本可能覆盖同名结果或复用缓存，"
    "要从头复现时请使用未含旧实验产物的输出目录，保留所需特征缓存。"
    "FedMPT 仍按服务器交接处理，不在本机恢复。"
)
REPRO_PREREQUISITES = (
    "前置条件：四数据集的 train/test CLS、patch、patch_mean、patch_max_text、targets 和 text_features 缓存齐全，"
    "并与 manifest 中图像及类别顺序一致。VOC07 兼容 outputs/voc07/cache，其他数据集使用 outputs/icme/cache/<dataset>。"
    "缺失缓存时先用阶段表中的提取脚本和原始数据构建；本节不宣称自动下载或准备全部原始数据。"
)
REPRO_BLOCKS = [
    ("8.1 固定解释器与选择参数", r"""$ErrorActionPreference = 'Stop'
$PY = 'D:\Software\anaconda3\envs\pytorch\python.exe'
$datasets = @('voc07', 'voc12', 'coco', 'nus')
$seeds = @(7, 17, 27)
$alphas = @(0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
function Invoke-Py {
    & $PY @args
    if ($LASTEXITCODE -ne 0) { throw 'Python step failed' }
}"""),
    ("8.2 主分支与七种表示", r"""$featureSets = @(
    'cls', 'patch_mean', 'patch_max_text',
    'cls patch_mean', 'cls patch_max_text',
    'patch_mean patch_max_text',
    'cls patch_mean patch_max_text'
)
foreach ($dataset in $datasets) {
    Invoke-Py scripts/run_icme_experiments.py `
        --dataset $dataset --phase all --seeds 7 17 27
    foreach ($featureSet in $featureSets) {
        $features = $featureSet.Split(' ')
        Invoke-Py scripts/run_label_ridge.py `
            --dataset $dataset --seed 2026 `
            --features @features --alphas @alphas
    }
    Invoke-Py scripts/run_label_patch_fusion.py `
        --dataset $dataset `
        --ridge-name label_ridge_cls_patch_mean_patch_max_text
}"""),
    ("8.3 三种子拟合与训练侧融合", r"""foreach ($dataset in $datasets) {
    foreach ($seed in $seeds) {
        $ridgeName = "full_seeds/seed$seed/ridge"
        $ridgeRoot = "outputs/icme/$dataset/$ridgeName"
        Invoke-Py scripts/run_label_ridge.py `
            --dataset $dataset --seed $seed `
            --features cls patch_mean patch_max_text `
            --alphas @alphas --report-root $ridgeRoot
        Invoke-Py scripts/run_label_patch_fusion.py `
            --dataset $dataset --ridge-name $ridgeName
    }
}
Invoke-Py scripts/build_cv_fusion_ensemble.py --datasets @datasets"""),
    ("8.4 既有实验汇总与验收", r"""Invoke-Py scripts/run_partial_annotation.py --datasets @datasets
Invoke-Py scripts/run_reviewer_robustness.py --datasets @datasets
Invoke-Py scripts/run_reviewer_federated_stress.py --datasets @datasets
Invoke-Py scripts/run_reviewer_evidence.py --datasets @datasets
Invoke-Py scripts/verify_federated_full.py --datasets @datasets
Invoke-Py scripts/build_icme_artifacts.py --bootstrap 1000
Invoke-Py scripts/audit_icme_results.py --require-complete
Invoke-Py -m pytest -q"""),
]
REPRO_AFTER = (
    "CV 聚合前应确认四数据集各有 seed7/seed17/seed27 的 ridge/result.json、validation_adapter.pt、"
    "fit_indices.npy、validation_indices.npy，以及 ridge_patch_fusion/test_scores_classwise.npy。"
    "选中 lambda 和 split seed 应与第 3.1.2 节一致。第 8.4 节复用既有定性样例和官方可运行性审计等证据；"
    "新工作区如缺少这些产物，必须据实补齐，不能把缺失改为 PASS。"
    "其中 run_reviewer_evidence.py 保留第 4.4 节所披露的中央预测复用行为。"
)
SUPPLEMENT_COMMANDS = r"""cd /root/autodl-tmp/FedPASA
tmux ls
tail -n 20 outputs/server_logs/fedmpt_optimized_20260907.log
cat outputs/icme/supplementary/coco/fedmpt_repaired/progress.json
cat outputs/icme/supplementary/nus/fedmpt_repaired/progress.json
# 两个数据集均已 COMPLETE 200/200，无需再次启动训练。
# 报告重建只读取已保存结果，不训练：
/root/autodl-tmp/miniconda3/envs/fbu/bin/python scripts/build_supplementary_report.py --no-plots"""
