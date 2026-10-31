# FedPASA 补充实验记录

更新时间：2026-09-09T22:50:56+08:00。总状态：**COMPLETE**。

结果均来自本地实际计算或带来源的既有预测复核。未完成的 FedMPT 不填入最终 mAP；中间轮数单独记录。

| 建议项 | 状态 | 文件 |
|---|---|---|
| P0-1 protocol-matched strong/principle baselines | COMPLETE | protocol_matched_baselines.csv |
| P0-2 estimator mismatch decomposition | COMPLETE | estimator_mismatch.csv |
| P0-3 classwise complementarity / alpha | COMPLETE | classwise_complementarity.csv |
| P1-4 centralized/federated gap | COMPLETE | federation_gap.csv |
| P1-5 controlled boundary ties / global key | COMPLETE | boundary_ties.csv |
| P1-6 client-number scaling | COMPLETE | client_scaling.csv |
| P1-7 communication comparisons | COMPLETE | communication_baselines.csv |
| P2-8 fusion strategies | COMPLETE | fusion_strategies.csv |
| P2-9 three-view representation ablation | COMPLETE | representation_ablation.csv |

## 统一协议的基线

COCO/NUS：相同 CLIP ViT-B/16 权重、224 输入、完整训练/测试列表、10 个 KMeans 客户端、划分 seed=7、相同 mAP。闭式方法复用相同特征缓存。FedMPT 根据用户选择使用修补版图像输入，视觉提示需要重算特征，不能宣称相同最终 feature cache 或官方等价复现。Fed3R-style 是无偏置、未中心化的 CLS 多标签 ridge，正则化仅由训练侧验证集选择。

| 数据集 | 方法 | 最终 mAP | 状态 | 完成轮数 |
|---|---|---|---|---|
| coco | Centralized | 72.984 | COMPLETE | — |
| coco | Naive-A local-TopK union | 66.918 | COMPLETE | — |
| coco | Naive-B local-GDA average | 71.309 | COMPLETE | — |
| coco | GlobalQuota-GDA-Avg | 65.902 | COMPLETE | — |
| coco | Fed3R-style CLS ridge | 76.128 | COMPLETE | — |
| coco | QAFedPASA | 72.984 | COMPLETE | — |
| coco | Pooled Ridge | 79.83 | COMPLETE | — |
| coco | FedPASA-Full | 80.528 | COMPLETE | — |
| coco | FedMPT (disclosed repaired rerun) | 69.806 | COMPLETE | 200 |
| nus | Centralized | 50.62 | COMPLETE | — |
| nus | Naive-A local-TopK union | 47.197 | COMPLETE | — |
| nus | Naive-B local-GDA average | 50.123 | COMPLETE | — |
| nus | GlobalQuota-GDA-Avg | 46.05 | COMPLETE | — |
| nus | Fed3R-style CLS ridge | 58.407 | COMPLETE | — |
| nus | QAFedPASA | 50.62 | COMPLETE | — |
| nus | Pooled Ridge | 60.193 | COMPLETE | — |
| nus | FedPASA-Full | 60.671 | COMPLETE | — |
| nus | FedMPT (disclosed repaired rerun) | 35.824 | COMPLETE | 200 |

完整 FedMPT 预算预先固定为 200 轮，每轮所有客户端各遍历一次全部本地训练图像。逐客户端保存断点，两数据集按轮交替运行；最终采用最后一轮，测试集指标不影响训练或模型选择。修补差异、优化器、ASL、BF16 及梯度检查见各数据集 `fedmpt_repaired/protocol.json`。

## 选样与模型平均的分解

GlobalQuota-GDA-Avg 使用完全相同的中央 Top-K，并按图像归属分配给客户端，随后各自净化和求解。SharedPurification-GDA-Avg 进一步固定净化后的完整 patch 集，仅改变本地求解与模型平均。前者的本地净化沿用已有控制实验的阈值夹紧保护。二者用于机制隔离；客户端完整训练集的局部 Top-K 基线由原有审计结果补充。

| 数据集 | 方法 | mAP | 最大 W 误差 | 最大 score 误差 |
|---|---|---|---|---|
| coco | Centralized | 72.984 | 0 | 0 |
| coco | QAFedPASA | 72.984 | 7.6294e-06 | 1.4156e-07 |
| coco | GlobalQuota-GDA-Avg | 65.902 | 3932.1 | 0.89726 |
| coco | SharedPurification-GDA-Avg | 66.119 | 5068.8 | 0.89686 |
| coco | Pooled Ridge | 79.83 | 2.3842e-07 | 2.3842e-07 |
| coco | Local-Ridge-Avg | 79.069 | 7.1175 | 0.94323 |
| nus | Centralized | 50.62 | 0 | 0 |
| nus | QAFedPASA | 50.62 | 4.7684e-07 | 1.4901e-08 |
| nus | GlobalQuota-GDA-Avg | 46.05 | 2944.5 | 0.88878 |
| nus | SharedPurification-GDA-Avg | 46.05 | 2944.9 | 0.88878 |
| nus | Pooled Ridge | 60.193 | 2.9802e-08 | 1.1921e-07 |
| nus | Local-Ridge-Avg | 60.026 | 6.322 | 0.71746 |

误差中的无穷范数采用最大元素绝对值（max-entry），不是矩阵最大行和。所有新增 score discrepancy 通过分别推理得到，未用中央预测替代联邦预测。

## 互补性与融合

| 数据集 | QA 胜 Ridge 类数 | Full 胜两支 | Full 胜 Ridge | Full 低于 Ridge | 平均 alpha | 平均 AP 增益 |
|---|---|---|---|---|---|---|
| voc07 | 6 | 12 | 12 | 4 | 0.52583 | 0.18788 |
| voc12 | 3 | 16 | 15 | 3 | 0.53917 | 0.29698 |
| coco | 4 | 63 | 59 | 3 | 0.71958 | 0.69816 |
| nus | 12 | 46 | 42 | 20 | 0.75206 | 0.47767 |

alpha=1 表示使用 Ridge，alpha=0 表示 QA。图中展示三个训练侧划分的平均值与最小/最大值；最终分数平均还包含各划分的 z-score 校准，平均 alpha 不能解释为未经校准原始分数的最终系数。互补性使用旧论文的三划分原始 Ridge 分数平均，Full 与 Ridge 类数比较沿用 ±0.05 AP 点的近似相等容差；同时保留严格增益类数。五种融合策略统一采用三个划分的仿射分数平均。

| 数据集 | 融合策略 | mAP |
|---|---|---|
| voc07 | QA only | 92.529 |
| voc07 | Ridge only | 94.033 |
| voc07 | Fixed 0.5 | 94.207 |
| voc07 | Global learned | 94.315 |
| voc07 | Classwise learned | 94.22 |
| voc12 | QA only | 92.287 |
| voc12 | Ridge only | 93.685 |
| voc12 | Fixed 0.5 | 93.901 |
| voc12 | Global learned | 93.981 |
| voc12 | Classwise learned | 93.982 |
| coco | QA only | 72.984 |
| coco | Ridge only | 79.83 |
| coco | Fixed 0.5 | 79.302 |
| coco | Global learned | 80.277 |
| coco | Classwise learned | 80.528 |
| nus | QA only | 50.62 |
| nus | Ridge only | 60.193 |
| nus | Fixed 0.5 | 59.082 |
| nus | Global learned | 60.326 |
| nus | Classwise learned | 60.671 |

COCO/NUS 的固定 0.5 融合劣于 Ridge 单支；不能将提升归因于任意简单 ensemble。VOC07 的全局学习权重优于逐类权重，保留该负面结果，不主张逐类融合在所有数据集最优。

## 中央目标与联邦差距

中央估计器是同一算法的参考目标，不是理论精度上界。下表展示主设置 t=60%；完整表另含全部 24 个数据集/客户端数设置、QA/Ridge/Full 三个分支。

| 数据集 | 分支 | 中央 mAP | 联邦 mAP | 联邦−中央 | score 误差 |
|---|---|---|---|---|---|
| voc07 | QA | 92.529 | 92.529 | 0 | 1.0803e-07 |
| voc07 | Ridge | 94.033 | 94.033 | 0 | 5.9605e-08 |
| voc07 | Full | 94.22 | 94.22 | 0 | 1.9073e-06 |
| voc12 | QA | 92.287 | 92.287 | 0 | 6.7055e-08 |
| voc12 | Ridge | 93.652 | 93.652 | 0 | 0 |
| voc12 | Full | 93.982 | 93.982 | 0 | 7.1526e-07 |
| coco | QA | 72.984 | 72.984 | -2.9572e-09 | 8.9407e-08 |
| coco | Ridge | 79.83 | 79.83 | 0 | 2.3842e-07 |
| coco | Full | 80.528 | 80.528 | 0 | 1.5497e-06 |
| nus | QA | 50.62 | 50.62 | 1.7256e-09 | 1.4901e-08 |
| nus | Ridge | 60.193 | 60.193 | 1.8769e-08 | 1.1921e-07 |
| nus | Full | 60.671 | 60.671 | -5.0072e-07 | 2.718e-05 |

扩展性测试固定已选中央 Top-K 按图像分配给客户端，考察统计汇总/求解和完整测试性能；不将其冒充完整局部选样的时间测试。Ridge server solve 时间仅计服务器合并统计量及线性求解；QA protocol fit 包括模拟客户端统计量构建和净化，因此单独标注。

## Boundary-tie 压力测试

对 COCO 16,087,876 和 NUS 29,400,000 个训练 patch 的熵先量化再做 Top-K，覆盖 FP32 / 1e-4 / 1e-3 / 1e-2。每个精度均重新构建中央目标。真实 patch ID = 缓存图像行号 × 196 + patch 位置。local 使用客户端 argpartition 与服务器 topk 的非共享 tie 顺序；shared 在本地截断与全局选择中均按 (entropy, global patch ID) 排序。

| 数据集 | 熵精度 | tie 规则 | ID recall | mAP | 中央 mAP | W 误差 | score 误差 |
|---|---|---|---|---|---|---|---|
| coco | fp32 | local | 1 | 72.984 | 72.984 | 7.6294e-06 | 5.9605e-08 |
| coco | fp32 | shared | 1 | 72.984 | 72.984 | 3.8147e-06 | 1.1921e-07 |
| coco | round_1e-2 | local | 0.91106 | 73.076 | 73.077 | 60.241 | 0.17173 |
| coco | round_1e-2 | shared | 1 | 73.077 | 73.077 | 1.9073e-06 | 1.4901e-07 |
| coco | round_1e-3 | local | 0.9877 | 72.992 | 72.996 | 20.403 | 0.036174 |
| coco | round_1e-3 | shared | 1 | 72.996 | 72.996 | 7.6294e-06 | 1.9372e-07 |
| coco | round_1e-4 | local | 0.99839 | 72.989 | 72.974 | 16.507 | 0.03864 |
| coco | round_1e-4 | shared | 1 | 72.974 | 72.974 | 7.6294e-06 | 1.3411e-07 |
| nus | fp32 | local | 1 | 50.62 | 50.62 | 9.5367e-07 | 1.2107e-08 |
| nus | fp32 | shared | 1 | 50.62 | 50.62 | 9.5367e-07 | 1.2107e-08 |
| nus | round_1e-2 | local | 0.94117 | 50.57 | 50.644 | 5.191 | 0.00041837 |
| nus | round_1e-2 | shared | 1 | 50.644 | 50.644 | 9.5367e-07 | 1.4901e-08 |
| nus | round_1e-3 | local | 0.9931 | 50.614 | 50.625 | 1.2877 | 5.0959e-05 |
| nus | round_1e-3 | shared | 1 | 50.625 | 50.625 | 4.7684e-07 | 1.5832e-08 |
| nus | round_1e-4 | local | 0.99906 | 50.623 | 50.623 | 0.33502 | 1.2734e-05 |
| nus | round_1e-4 | shared | 1 | 50.623 | 50.623 | 1.9073e-06 | 1.5832e-08 |

共享 tie key 为显式可选方案，上传 int64 key 的额外字节已经计入。它暴露稳定 patch 标识，不能同时沿用“不上传 patch ID”的旧表述。这里的 100% 是选样 ID 恢复，浮点矩阵求解仍存在数值误差。局部 tie 顺序下 mAP 接近并不意味着 W 或测试分数完全一致。

## 通信与表示

全部通信表统一为 client→server 上传，不包含下行、传输头、加密开销。分别列 FP64 累加参考和 FP32 传输口径；COCO 的 373.16 MiB / NUS 的 382.05 MiB 对应 QA FP32 + Ridge FP32，参考 FP64 Ridge 分别为 612.72 / 627.24 MiB。原始全局 selected patches 是忽略全局选样协调的 oracle 下界，Local Top-K patches 则改变了样本总体，不能当作等价协议。

推荐表述：FedPASA 以结构化汇总通信换取中央估计器的恢复，而不是无条件宣称最省通信。

| 数据集 | 表示 | 维度 | 验证集 alpha | mAP |
|---|---|---|---|---|
| voc07 | cls | 512 | 0.03 | 91.807 |
| voc07 | cls+patch_mean | 1024 | 0.1 | 92.81 |
| voc07 | cls+patch_max_text | 532 | 0.001 | 93.799 |
| voc07 | cls+patch_mean+patch_max_text | 1044 | 0.03 | 94.033 |
| voc12 | cls | 512 | 0.1 | 91.402 |
| voc12 | cls+patch_mean | 1024 | 0.1 | 92.543 |
| voc12 | cls+patch_max_text | 532 | 0.01 | 93.395 |
| voc12 | cls+patch_mean+patch_max_text | 1044 | 0.03 | 93.689 |
| coco | cls | 512 | 0.0001 | 76.115 |
| coco | cls+patch_mean | 1024 | 0.0001 | 77.851 |
| coco | cls+patch_max_text | 592 | 0.0001 | 78.85 |
| coco | cls+patch_mean+patch_max_text | 1104 | 0.0001 | 79.83 |
| nus | cls | 512 | 0.0001 | 58.413 |
| nus | cls+patch_mean | 1024 | 0.0001 | 59.781 |
| nus | cls+patch_max_text | 593 | 0.0003 | 59.227 |
| nus | cls+patch_mean+patch_max_text | 1105 | 0.0001 | 60.193 |

## 复现与证据

运行 `scripts/run_supplementary_experiments.py --phase all` 生成闭式补充实验；运行 `scripts/run_matched_fedmpt.py --datasets coco nus --rounds 200` 继续既有训练断点；运行 `scripts/build_supplementary_report.py` 刷新本文与表格。解释器使用 `D:\Software\anaconda3\envs\pytorch\python.exe`。

协议指纹见 `protocol_manifest.json`；建议项完成状态见 `completion_audit.json`；PNG/PDF 图见 `figures/`。旧论文主结果和外部 published values 未被本次补充覆盖。

原始方法来源：[FedMPT](https://arxiv.org/abs/2605.28347)、[发布代码](https://github.com/xuc865/FedMPT-main)、[Fed3R](https://arxiv.org/abs/2406.01116)。
