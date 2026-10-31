# FedPASA ICME 创新与全量实验交接文档

**状态：**原定补充实验 9/9 已归档；新增服务器队列 23/23 完成，六组基线 1200/1200 轮通过验收；校准近似边界和全部结果见第 13 节

**锁定方法：**FedPASA-Full

**日期：**2026-09-13 服务器全量补实验执行交接更新

**工作区：**D:\\project\\VScodeProject\\FedPASA

**主交付：**本 Markdown 文件；机器可读证据位于 `D:\project\VScodeProject\FedPASA\outputs\icme\paper`。

运行交接：新增服务器队列 23/23 COMPLETE；六组预测复算差值为 0，1328 个取回文件 SHA256 一致。全部结果与校准非等价限制见第 13 节。

|                                                                                                                                                                                                                 |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **一句话贡献** 将 PIAA 的集中式 patch 分类器估计改写为全局分位数对齐的联邦充分统计问题，并与多视图、闭式、标签感知联邦 Ridge 通过严格训练侧逐类别融合，形成无需迭代反向传播的多标签适配框架；拟合阶段采用联邦充分统计，校准与选参的分布式协议尚待补充。 |

# 0. 交接摘要

| **数据集** | **FedPASA-Full** | **最强公开外部** | **提升** | **95% bootstrap CI** |
|------------|------------------|------------------|----------|----------------------|
| VOC07      | 94.22            | 90.13            | +4.09    | \[93.71, 94.73\]     |
| VOC12      | 93.98            | n/a              | n/a      | \[93.50, 94.44\]     |
| COCO       | 80.53            | 67.37            | +13.16   | \[80.25, 80.83\]     |
| NUS-WIDE   | 60.67            | 58.36            | +2.31    | \[60.23, 61.70\]     |

|                                                                                                                                                                      |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **目标结论** 在 FedMPT Table 1 覆盖的 VOC07、COCO、NUS-WIDE 上，与公开最高值的最小数值差为 +2.31 mAP；它不是同协议受控增益，也不作为投稿验收目标。VOC12 没有对应联邦外部表项，只报告 93.98 mAP，不写 +2 结论。 |

当前机器验收器已将主实验、无测试集选参、96 行 estimator-preserving 对照、201 类互补性分析、通信拆分、官方基线审计、12 个定性样例以及 11 组 PNG/PDF 双格式图全部纳入；本地证据套件 `overall_pass = true`。该状态不代表可信同协议强基线和 Full 分布式校准证据已经齐全。

## 阅读导航

-   第 1-2 节：从原始 PIAA 到最终锁定创新，明确哪些能写、哪些不能写。

-   第 3-6 节：所有支持创新点的主结果、消融、统计、鲁棒性、系统和失败分析。

-   第 7-9 节：完整实验登记、复现顺序、文件索引与后续写作动作。

# 1. 原始 PIAA 血统与问题定义

原始 PIAA 将多标签识别重写为 Patch-level Inference + Adaptive Aggregation。PVCL 从未标注 patch bank 估计类别均值与共享协方差，并用 GDA 闭式获得视觉分类器；PAA 再以固定 alpha=0.9 融合 patch 与 CLS 分数。原稿第 5-6 页给出 PVCL/PAA 公式，第 9 页 limitations 明确指出初始 patch 预测和 patch bank 的 noisy/biased 风险，并提出 reliability-aware calibration 作为未来方向。

<table><tbody><tr class="odd"><td>PIAA / GDA: w_c = Sigma^{-1} mu_c<br />
b_c = -0.5 * mu_c^T Sigma^{-1} mu_c<br />
PIAA / PAA: S_f,c = alpha * S_patch,c + (1-alpha) * S_cls,c, alpha=0.9</td></tr></tbody></table>

联邦场景放大了该风险：各客户端类别覆盖、patch 数量、纯度和目标尺度不同。若每个客户端独立取 K 个 patch 后再合并，会把集中式每类 K=512 的 bank 变成最多 K×clients 的偏置 bank；若直接平均分类器参数，也不能保证恢复集中式闭式解。

|                                                                                                                              |
|------------------------------------------------------------------------------------------------------------------------------|
| **切入点** 不是 PIAA + FedAvg，而是把集中式 patch distribution estimation 改造成可合并、可审计、无需传单样本的联邦充分统计。 |

# 2. 最终锁定的创新点

| **编号** | **名称**         | **核心机制**                                                   | **证据强度**                               |
|----------|------------------|----------------------------------------------------------------|--------------------------------------------|
| I1       | QAFedPASA        | 全局分位数对齐 + 四次统计上传；一致选样下恢复中央估计器（给定容差）             | 强；直接消融、central-equivalence、Non-IID |
| I2       | 多视图联邦 Ridge | CLS/patch mean/patch max-text 的标签感知闭式适配，仅传可加统计 | 强；特征消融、集中式等价、通信             |
| I3       | 逐类别双分支融合 | 训练侧 out-of-sample 校准与选权；三划分均值折叠为一次仿射评分  | 强；三种子、bootstrap、逐类增益            |
| S1       | 系统与协议性质   | 无迭代反向传播、拟合统计不传样本、Ridge FP32 近乎无损         | 支撑项；不是正式 DP 保证                   |

## 2.1 I1 - 全局分位数对齐的 QAFedPASA

客户端首先保留本地候选 patch，但只交换每类熵顺序统计量；服务器据此下发全局 top-K 配额。客户端随后上传初始 GDA moment packet；收到初始全局 GDA 后，再上传每类 {count, sum(q), sum(q^2)} 以构造全局 mean+sample-std 净化阈值；最后上传净化后的 GDA moments。共四次客户端上传，其中 sum(q^2) 不能由另两项重构。

<table><tbody><tr class="odd"><td>Upload 1: entropy order statistics -&gt; global per-class top-K quota<br />
Upload 2: {count, sum_x, mass, weighted_sum_x, second_moment} -&gt; preliminary GDA<br />
Upload 3: {count, sum(q), sum(q^2)} -&gt; global mean + sample-std threshold<br />
Upload 4: GDA moments after global purification -&gt; final GDA<br />
Default QAFed fitting uploads no image, label, patch identity or individual embedding.<br />Optional shared tie keys transmit patch IDs; Full calibration is outside this claim.</td></tr></tbody></table>

-   直接解决 local top-K bank 过大和客户端偏置，而不是依赖启发式模型平均。

-   统计量在同一选样总体下可加；恢复同一中央估计器要求一致的选样及边界 tie 规则，浮点计算按给定容差判断。默认协议不上传 patch ID，不能保证任意 tie 情况下选中相同 patch；可选 shared-key 协议额外上传 int64 patch ID，见第 11 节。

-   标签完全不进入 QAFedPASA bank 构建和适配器拟合。

### 2.1.1 投稿前实现—公式一致性核对（已闭环）

|                                                                                                                                                                                                    |
|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **结论** 显式 PurificationPacket 是对既有计算的数据边界重构，不改变 purification threshold、保留 patch、最终 GDA 权重/偏置或 mAP；该次 PurificationPacket 重构只需修正协议轮数、通信口径和证据字段，无需重跑准确率；Full 校准协议是另外的待补项。 |

实际 q_x 是客户端收到 preliminary global GDA 后，对已选 patch 计算的预测类别 refined confidence。由于 q_x 依赖该全局 GDA，它不能并入此前的 preliminary GDA moment packet；而 sum(q_x^2) 也不能由 count 与 sum(q_x) 重构。因此客户端必须在收到 preliminary GDA 后显式上传每类 {count, sum(q), sum(q^2)}。

<table><tbody><tr class="odd"><td>r_kc = |S_kc|, a_kc = sum_{x in S_kc} q_x, b_kc = sum_{x in S_kc} q_x^2<br />
r_c = sum_k r_kc, A_c = sum_k a_kc, B_c = sum_k b_kc<br />
theta_c = A_c/r_c + sqrt(max(B_c - A_c^2/r_c, 0)/(r_c-1)) # sample std<br />
Comm_up = 4M[CK + 2(D^2 + 2CD + 2C) + 3C] bytes # dense FP32 uplink</td></tr></tbody></table>

| **核对项** | **确认结果**                                                               | **对既有实验的影响**                                     |
|------------|----------------------------------------------------------------------------|----------------------------------------------------------|
| 数值路径   | 旧实现直接累加 count/sum(q)/sum(q^2)；新实现先形成客户端 packet 再逐项求和 | 逐位等价；阈值最大绝对差 0.0                             |
| 模型与指标 | 计算仍使用 float64；聚合公式、样本标准差及边界处理不变                     | 保留 patch、GDA weight/bias、AP/mAP 不变；无需重跑准确率 |
| 协议元数据 | 原文将 preliminary GDA 与其后才能计算的 q-moments 合并称为第 2 阶段        | 修正为 4 次客户端上传；仅记录和叙述变化                  |
| 通信量     | 旧公式已用 3C 计入 count/sum(q)/sum(q^2)，但没有独立命名                   | 总 MiB 不变；新增 purification_moments_MiB 分项        |

验证证据：随机多客户端旧/新阈值逐位相同（bitwise_equal=true，max_abs_diff=0.0）；核心测试 5/5 通过；投稿证据审计 overall_pass=true 且 communication_breakdown_exact=true。VOC07 的 t=60% 配置中，12 个客户端、20 类的独立净化 packet 仅为 0.0027466 MiB。

|                                                                                                                                                                                                                                                                             |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **通信口径** 当前表格统计 dense FP32 client-to-server uplink upper bound；代码中的实际统计聚合仍为 float64 联邦逻辑模拟。论文表头应写 Uplink communication。若未来真实序列化为 FP32 或统计双向通信，则必须另做数值回归并加入 quota、preliminary GDA、threshold 等下行载荷。 |

## 2.2 I2 - 多视图标签感知联邦 Ridge

为了利用训练标签但避免样本级传输，客户端对冻结图像特征构造 RidgePacket。服务器对 n、sum_x、sum_y、X^T X、X^T Y 求和，以 float64 解正规方程。特征视图为全局 CLS、patch mean 和对文本原型的 patch max 响应，三者互补。

<table><tbody><tr class="odd"><td>RidgePacket_k = {n_k, sum_x_k, sum_y_k, X_k^T X_k, X_k^T Y_k}<br />
W = (X_c^T X_c + lambda * scale * I)^{-1} X_c^T Y_c<br />
b = mean_y - mean_x * W</td></tr></tbody></table>

|                                                                                                                                                              |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **术语边界** 该分支使用训练标签，因此 FedPASA-Full 应写成 label-aware, backpropagation-free closed-form federated adaptation；不能写成完全 annotation-free。 |

## 2.3 I3 - 无测试泄漏的逐类别融合

每个 seed 把训练集划分为 80% fit 与 20% validation。Ridge 与 PIAA 两个分支都只在 fit 子集拟合，再在 validation 上产生真正 out-of-sample 预测；按类别从 0.0-1.0 的 21 个权重中选取融合系数。三个 seed 的逐类别得分函数取平均后仍是单个仿射函数，可在部署时预先合并。

校准与通信边界：现有 run_label_patch_fusion.py 集中读取训练侧 validation 的逐样本预测与标签，计算校准参数及候选权重的 AP。GDA/Ridge 拟合已有充分统计量接口；分布式校准和选参通信协议尚未实现或验证。因此，拟合阶段不传单样本的证据不能扩展为整个 Full 流程的端到端传输保证。现有通信表仅计超参数确定后的拟合统计上传，未包含三划分校准、选参、重拟合全过程及服务器下行成本。

<table><tbody><tr class="odd"><td>score_c(x) = a_c * z_ridge,c(x) + (1-a_c) * z_piaa,c(x)<br />
score_CV,c(x) = mean_seed score_seed,c(x) # still one affine class score<br />
Selection data: training-side validation only; test labels never used.</td></tr></tbody></table>

## 2.4 已验证但未锁定的早期方案

初始创新备忘提出 client-class entropy/consensus reliability weighting 与 empirical-Bayes personalized shrinkage。代码中对应 `fit_fedpasa()` / `fit_fedpasa_from_packets()`，但 `reliability_\*` 消融在四个数据集均弱于 quantile-aligned top-K=512。因此它们应作为探索性负结果或未来工作，不应继续作为最终主贡献。

| **数据集** | **QAFed top-K=512** | **reliability+shrinkage** | **差值** |
|------------|---------------------|---------------------------|----------|
| VOC07      | 92.53               | 90.20                     | -2.33    |
| VOC12      | 92.29               | 89.40                     | -2.89    |
| COCO       | 72.98               | 66.49                     | -6.49    |
| NUS-WIDE   | 50.62               | 47.25                     | -3.37    |

# 3. 实验协议与外部基线

| **项目**       | **锁定设置**                                                                                                                                |
|----------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 骨干/输入      | 冻结 CLIP ViT-B/16；PIAA/SC-CLIP patch 与 CLS 缓存                                                                                          |
| 数据集         | VOC07 20 类、VOC12 20 类、COCO 80 类、NUS-WIDE 81 类                                                                                        |
| 指标           | class-wise Average Precision 的均值 mAP (%)                                                                                                 |
| KMeans 划分    | 最终 ICME 流水线对 L2-normalized ViT-B/16 CLS 与文本特征的零样本概率向量做聚类；t={10,20,40,60,80,100}%，seeds={7,17,27}，n_init=10，Lloyd |
| Dirichlet 划分 | 原等价性实验 beta={0.1,0.3,0.5}、10 clients、3 seeds；naive 证据实验为相同 beta 加 KMeans、seed=7                                           |
| 选参           | 训练侧 80/20；Ridge alpha、分支校准、融合权重均不读 test label                                                                              |
| 外部结果       | FedMPT Table 1 的 11 个 published_external 方法；官方源码审计不能产生 author-equivalent controlled rerun，禁止把修复版诊断值写入准确率主表 |
| 主方法         | FedPASA-Full = QAFedPASA + multi-view federated Ridge + classwise CV fusion                                                                 |

|                                                                                                                                                          |
|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **比较范围** “超过最强外部基线约 2 点”只适用于 FedMPT 表中有可比数值的 VOC07、COCO、NUS-WIDE；不延伸到 VOC12，也不延伸到不同监督级别的集中式全监督模型。 |

## 3.1 四类可复现性与实验协议核对（投稿锁定版）

|                                                                                                                                                                                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **核对结论** 数据划分、Ridge、QAFed/Patch 与鲁棒性四类信息均已从最终结果文件及其生成脚本回查。现有结果不需要因为补写这些参数而重跑；其中客户端参与率实验本身已经包含 101/202/303 三个预声明子集种子。只有当论文改称“每轮重新采样客户端”或“逐标签独立 Bernoulli 噪声”时，才会与当前实现不一致并需要新实验。 |

### 3.1.1 数据划分与随机性设置

最终 ICME 多数据集协议以 `scripts/run_icme_experiments.py` 和各数据集 `main.json` 为准。先分别 L2 归一化冻结的 CLIP ViT-B/16 全局 CLS 特征与单模板文本特征，再计算 q_i=softmax(z_i T^T)，得到每幅训练图像的 C 维零样本类别概率向量。K-means 直接作用于 q_i：没有使用标签、没有 PCA，也没有额外 logit temperature。

<table><tbody><tr class="odd"><td>partition_feature_i = softmax(L2(cls_i) @ L2(text)^T) # C-dimensional<br />
KMeans(n_clusters=N, init='k-means++', n_init=10, algorithm='lloyd', random_state=seed)<br />
partition seeds = {7, 17, 27}; feature extraction DataLoader shuffle=False</td></tr></tbody></table>

| **数据集** | **类别 C** | **t=10/20/40/60/80/100% 的客户端数** | **partition seeds** | **runs** |
|------------|------------|--------------------------------------|---------------------|----------|
| VOC07      | 20         | 2/4/8/12/16/20                       | 7/17/27             | 18       |
| VOC12      | 20         | 2/4/8/12/16/20                       | 7/17/27             | 18       |
| COCO       | 80         | 8/16/32/48/64/80                     | 7/17/27             | 18       |
| NUS-WIDE   | 81         | 8/16/32/49/65/81                     | 7/17/27             | 18       |

客户端数由 max(2, round(C×t)) 得到。`init` 未在构造器中显式传入，因而采用 scikit-learn KMeans 的默认 `k-means++`；`n_init=10` 与 `algorithm='lloyd'` 是显式锁定项。旧脚本 `scripts/run_voc07_fedpasa.py` 曾直接对归一化 CLS 聚类并使用 n_init=20，但它是遗留的 VOC-only 路径，不是本文最终多数据集主结果协议，投稿文字不得混用。

### 3.1.2 Ridge baseline 的超参数与选择规则

最终 Ridge 输入为 `cls + patch_mean + patch_max_text`：CLS 与 patch mean 各 512 维，patch max-text 为 C 维；拼接完成后对整条向量做一次 L2 归一化。每个客户端只上传 {n, sum_x, sum_y, X^T X, X^T Y}，服务器在 float64 中聚合并求解带截距的中心化正规方程。这里的 lambda 是按训练特征能量缩放的系数，不等同于未经缩放的 sklearn Ridge alpha。

<table><tbody><tr class="odd"><td>scale = trace(X_centered^T X_centered) / D<br />
W = (X_centered^T X_centered + lambda * scale * I)^(-1) X_centered^T Y_centered<br />
lambda grid = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}</td></tr></tbody></table>

| **数据集** | **seed 7 选中 lambda** | **seed 17 选中 lambda** | **seed 27 选中 lambda** |
|------------|------------------------|-------------------------|-------------------------|
| VOC07      | 0.03                   | 0.03                    | 0.03                    |
| VOC12      | 0.10                   | 0.10                    | 0.03                    |
| COCO       | 0.0001                 | 0.0001                  | 0.0001                  |
| NUS-WIDE   | 0.0001                 | 0.0001                  | 0.0001                  |

对每个 seed，`np.random.default_rng(seed).permutation` 将训练集随机分为 80% fit/20% validation；lambda 按 validation mAP 最大值独立选择，随后用所选 lambda 在完整训练集上重拟合，测试标签从不参与选择。三视图先拼接后共用一个 lambda，并非每个视图单独设正则。特征消融使用 split seed=2026，并对每个视图组合单独选 lambda。需要特别注意：`run_label_ridge.py` 当前解析器默认网格仍是较旧的 {0.01,...,10}，最终结果实际使用的是上面的七点小正则网格；论文与复现命令必须显式传 `--alphas`，不能依赖脚本默认值。

FedPASA-Full 的融合选择同样只读训练侧 validation：Ridge 与 QAFedPASA 都在不含 validation 图像的 80% fit 上拟合；两分支分别按类别用 validation 预测的均值/标准差做 z-score（std 下限 1e-6），再从 0.00 到 1.00、步长 0.05 的 21 个 Ridge 权重中逐类别选择。三组 seed 的最终逐类得分取平均。

### 3.1.3 QAFedPASA / Patch 分支的具体实现参数

| **实现项**        | **投稿锁定设置**                                                                                                                                                                        |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 冻结骨干          | OpenAI-style CLIP ViT-B/16 提取全局 CLS；PIAA 的 SC-CLIP ViT-B/16 分支提取 patch token；同一 `ViT-B-16.pt`，SHA256=5806e77c...88df416f                                                |
| 输入与 patch      | 官方 CLIP preprocess，224×224；patch size=16，因此每图 14×14=196 个 patch token，每个 512 维；缓存为 float16，计算时转 float32 并 L2 归一化                                             |
| Prompt            | 每类仅一个模板：`a photo of a {class_name}`；无 prompt ensemble                                                                                                                      |
| 初始置信度/候选   | logit=x·text；confidence=softmax(60×logit)，不是 sigmoid/raw logit；按 argmax 类别分组，每类保留最低 entropy 的全局 Top-K，K=512                                                        |
| 全局估计          | QAFed 用局部候选和全局分位数对齐恢复 Top-K；同一选样总体要求一致的边界 tie 规则；GDA 使用原始 softmax 类别置信度加权原型、合并 full covariance 与 trace shrinkage                                            |
| Purification      | 初步 GDA 后 refined=softmax(T_pur×logits)；q 为预测类别 refined confidence；每类阈值为全局 mean(q)+unbiased sample std(q)，由显式 {count,sum_q,sum_q2} packet 聚合；保留 q≥threshold |
| Patch aggregation | 最终每个 patch 先计算 softmax((xW+b)/500)，再对 196 个 patch 按类别取 max，最后 softmax(patch_scale×max)；不是 mean pooling                                                            |
| 固定分支融合      | QAFedPASA 分支内部为 0.1×CLS probability + 0.9×patch probability；entropy-adaptive 融合只做过消融，未用于最终结果                                                                       |
| Full 校准         | QAFed 内部尺度为固定 PIAA 参数；与 Ridge 组成 FedPASA-Full 时，才使用训练侧 validation 的逐类 z-score 与逐类融合权重                                                                    |

| **数据集** | **purification T_pur** | **CLS scale** | **patch scale** |
|------------|-------------------------|---------------|-----------------|
| VOC07      | 100                     | 80            | 200             |
| VOC12      | 100                     | 80            | 200             |
| COCO       | 0.01                    | 80            | 400             |
| NUS-WIDE   | 500                     | 10            | 400             |

温度敏感性实验只围绕上述数据集基准值乘以 {0.25,0.5,1,2,4}，不读取测试标签选点。净化阈值、四次客户端上传与 `PurificationPacket` 的公式—实现一致性详见 2.1.1；这次显式 packet 重构保持阈值逐位一致，因此不会改变已报告的 patch、adapter、AP 或 mAP。

### 3.1.4 鲁棒性实验：label noise 与 client participation

Label noise 采用 `balanced_pair_noise`，不是每个 multi-label 位独立 Bernoulli 翻转。对每个类别 c，在整个训练集的正样本和负样本中分别无放回抽取 m_c=min(round(rho×n_c^+), n_c^-) 个，将所选正标签置 0、等量负标签置 1，因此每类正例数严格不变。噪声率为 {0,5,10,20}%，基础 seeds={7,17,27}，实际 RNG seed=seed×100000+round(rho×10000)。每个 dataset/rate/seed 只生成一张全局训练标签噪声矩阵，再按既定数据使用；不是每客户端独立重新加噪。

噪声条件下，Ridge lambda 与融合权重仍只用干净的训练侧 20% validation 标签选择；选中 lambda 后在全部带噪训练标注上重拟合。QAFedPASA 为 label-free 分支，保持固定。最终均值/样本标准差来自三个 seeds，test label 不参与调参。

<table><tbody><tr class="odd"><td>m_c = min(round(rho * positives_c), negatives_c)<br />
choose m_c positives without replacement: 1 -&gt; 0<br />
choose m_c negatives without replacement: 0 -&gt; 1 # prevalence preserved per class</td></tr></tbody></table>

Client participation 使用固定的 t=40%、partition seed=7 的 K-means 划分作为母划分：VOC07/VOC12 各 8 个客户端，COCO/NUS-WIDE 各 32 个。请求比例为 {25,50,75,100}%；每个条件在聚合/重拟合前，以 uniform sampling without replacement 一次性选择 max(1,round(fraction×N)) 个客户端。subset seeds={101,202,303}，实际 RNG seed=subset_seed+10000×N+round(100×fraction)。同一子集同时用于 QAFedPASA 与 Ridge；不存在逐轮重新采样，因为当前方法是一次性充分统计聚合/闭式拟合而非迭代训练。

| **数据集** | **母客户端数** | **25% 实际选中 ID**                                                          | **75% 实际排除 ID**                                                           | **训练图像保留率范围：25% / 75%** |
|------------|----------------|------------------------------------------------------------------------------|-------------------------------------------------------------------------------|-----------------------------------|
| VOC07      | 8              | 101:1/7; 202:4/7; 303:0/6                                                    | 101:3/4; 202:4/7; 303:2/7                                                     | 18.66-24.73 / 75.17-82.20         |
| VOC12      | 8              | 101:1/7; 202:4/7; 303:0/6                                                    | 101:3/4; 202:4/7; 303:2/7                                                     | 22.44-39.23 / 77.37-80.92         |
| COCO       | 32             | 101:2/7/13/14/22/29/30/31; 202:5/7/10/19/24/25/27/28; 303:2/4/6/7/9/11/21/26 | 101:1/3/8/10/17/25/27/30; 202:0/3/6/7/13/19/20/29; 303:2/10/11/20/22/23/27/30 | 25.20-28.08 / 74.67-76.77         |
| NUS-WIDE   | 32             | 101:2/7/13/14/22/29/30/31; 202:5/7/10/19/24/25/27/28; 303:2/4/6/7/9/11/21/26 | 101:1/3/8/10/17/25/27/30; 202:0/3/6/7/13/19/20/29; 303:2/10/11/20/22/23/27/30 | 22.71-26.17 / 70.39-78.21         |

表中 75% 为节省篇幅列出“排除 ID”；其余客户端即实际参与集合。K-means 客户端大小不均衡，所以客户端比例不等于训练图像保留比例，结果文件同时记录两者。Ridge lambda、两分支逐类校准参数和逐类融合权重均在 availability shock 之前冻结自 seed-7 训练侧 validation，不针对任何参与率条件重调；100% 条件三个子集种子得到同一完整集合，故标准差为 0。全部 25/50/75/100% 的逐条件集合与结果保存在 `outputs/icme/paper/client_participation.json`。

| **核对项**           | **是否补跑** | **原因/投稿边界**                                                             |
|----------------------|--------------|-------------------------------------------------------------------------------|
| 数据划分/K-means     | 无需         | 参数与 72 个主实验 run 已写入 main.json；只需准确补文                         |
| Ridge                | 无需         | 实际网格、split seed 与每 seed 选中 lambda 均已有结果证据                     |
| QAFed/Patch          | 无需         | 实现参数已锁定；显式 PurificationPacket 与旧阈值逐位相同                      |
| Label noise          | 无需         | 0/5/10/20% × 3 seeds 已完成；正文必须称 prevalence-preserving paired flips    |
| Client participation | 无需         | 25/50/75/100% × 3 subset seeds 已完成；正文必须称 one-shot availability shock |

## 3.2 ICME 主文 SOTA 表：单值、三数据集、六个代表方法

主文不再并列“baseline 范围”与“FedPASA 单点”。对每个已发表方法，统一取其 t=10%-100% 公开结果中的最高 mAP；这样仅避免选取外部方法较低的公开值，不能消除协议差异。主文仅保留 2024-2026 年的六个代表方法，删除没有外部可比项的 VOC12 列，并显式给出相对最强公开方法 FedMPT 的 Gain。

| **Method**         | **VOC07**    | **COCO**     | **NUS-WIDE** |
|--------------------|--------------|--------------|--------------|
| FedPGP             | 85.47        | 63.10        | 55.92        |
| FedTPG             | 88.11        | 63.91        | 53.01        |
| FedAWA             | 84.75        | 65.58        | 54.67        |
| Fed-RAM            | 88.05        | 66.06        | 56.35        |
| FedMVP             | 87.14        | 65.76        | 53.32        |
| FedMPT             | <u>90.13</u> | <u>67.37</u> | <u>58.36</u> |
| **FedPASA (Ours)** | **94.22**    | **80.53**    | **60.67**    |
| **Gain**           | **+4.09**    | **+13.16**   | **+2.31**    |

|                                                                                                                                                                                                                                                                                          |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **主表视觉规则** 按 ICME/IEEE 常见紧凑表格处理：使用单栏 table\[t\]、lccc、small、tabcolsep=4.5pt，不画竖线；最佳结果加粗，第二名（FedMPT）数值加下划线，Gain 行加粗。caption 必须说明：For methods reported under multiple settings, we conservatively use their best published result. |

## 3.3 VOC12 与组件表的安排

VOC12 不从论文删除，但不进入 Published Federated SOTA Comparison。主结果表后紧接以下英文说明；四数据集结果继续保留在 component analysis、overall results、鲁棒性和消融表中。

<table><tbody><tr class="odd"><td>FedPASA further achieves 93.98 mAP on VOC12; we report this<br />
result for completeness but exclude it from the direct SOTA comparison<br />
because the adopted published federated benchmark does not provide a<br />
corresponding VOC12 entry.</td></tr></tbody></table>

| **Variant**             | **VOC07** | **VOC12** | **COCO**  | **NUS-WIDE** |
|-------------------------|-----------|-----------|-----------|--------------|
| w/o Quantile Alignment  | 88.94     | 88.18     | 65.80     | 46.85        |
| w/o Global Purification | 92.53     | 92.29     | 67.63     | 50.61        |
| Diagonal Covariance     | 91.71     | 91.30     | 73.32     | 50.40        |
| QAFedPASA               | 92.53     | 92.29     | 72.98     | 50.62        |
| Ridge (CLS only)        | 91.81     | 91.40     | 76.12     | 58.41        |
| Ridge (Three Views)     | 94.03     | 93.69     | 79.83     | 60.19        |
| **FedPASA-Full**        | **94.22** | **93.98** | **80.53** | **60.67**    |

该组件表不是外部 SOTA 横向表，因此保留 VOC12 合理；它同时把 QAFedPASA、三视图 Ridge 与逐类别融合的贡献放在同一统一口径下。四数据集宽表可继续使用双栏 table\*\[t\]，同样不使用竖线。

## 3.4 补充材料：全部 11 个公开外部基线

| **方法**                    | **VOC07 范围** | **COCO 范围** | **NUS 范围** |
|-----------------------------|----------------|---------------|--------------|
| Fed-DualCoOp (NeurIPS 2022) | 82.02-87.67    | 51.46-61.88   | 39.40-51.02  |
| Fed-SCPNet (CVPR 2023)      | 76.48-83.37    | 46.67-60.96   | 38.49-49.56  |
| Fed-MaPLe (CVPR 2023)       | 76.08-87.87    | 47.24-64.83   | 50.98-54.78  |
| FedPGP (ICML 2024)          | 76.60-85.47    | 54.53-63.10   | 43.20-55.92  |
| Fed-TCP (CVPR 2024)         | 76.42-81.56    | 57.22-62.79   | 47.20-54.46  |
| FedTPG (ICLR 2024)          | 82.72-88.11    | 50.88-63.91   | 49.46-53.01  |
| Fed-PosCoOp (WACV 2025)     | 79.90-87.42    | 50.02-63.28   | 40.08-52.88  |
| FedAWA (CVPR 2025)          | 78.22-84.75    | 57.57-65.58   | 50.40-54.67  |
| Fed-RAM (CVPR 2025)         | 83.04-88.05    | 55.64-66.06   | 51.20-56.35  |
| FedMVP (ICCV 2025)          | 82.43-87.14    | 58.48-65.76   | 51.08-53.32  |
| FedMPT (CVPR 2026)          | 87.19-90.13    | 62.37-67.37   | 54.87-58.36  |

*本表仅作为 supplementary full comparison 的交接底稿：范围为 t=10%-100% 的最小-最大 mAP；完整逐 t 数字在 `outputs/icme/paper/main_comparison.csv`。不得再把这些范围直接复制回 ICME 主表。*

## 3.5 同协议外部 baseline 的实际状态

| **方法组**    | **已完成**                       | **阻塞**                                 | **交接口径**                       |
|---------------|----------------------------------|------------------------------------------|------------------------------------|
| FedMPT        | VOC 1 轮诊断；COCO/NUS 修补版各 200 轮完成 | 需要改变预测语义的修复                   | 不能作为 controlled accuracy rerun |
| FedMVP/FedAWA | FedMVP 适配版六组完成 | 适配差异见第 13 节；FedAWA 未训练 | FedMVP 完成 |
| FedPGP/FedTPG | 源码存在于同一 benchmark         | 共享数据/协议缺陷，未完成 3 seeds 全训练 | 未完成                             |

|                                                                                                                                                                                                                     |
|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **P0-1 边界** 9 月 7 日服务器交接已记录 COCO/NUS 图像核验；早期仅有 VOC/COCO 诊断数据的说明已过时。FedMPT benchmark 无 VOC12 入口，官方等价三基线复现仍未完成。补充 S01 的修补版 COCO/NUS 已完成 200 轮，分别为 69.806/35.824 mAP；最终结果已核验归档；Full 分布式校准协议另有证据缺口。 |

# 4. 支撑 I1 的实验：全局恢复、消融与 Non-IID

## 4.1 K 与关键组件消融

| **数据集** | **K=128** | **K=256** | **K=512** | **K=1024** |
|------------|-----------|-----------|-----------|------------|
| VOC07      | 90.13     | 91.58     | 92.53     | 91.73      |
| VOC12      | 89.17     | 90.92     | 92.29     | 91.57      |
| COCO       | 66.49     | 67.27     | 72.98     | 72.15      |
| NUS-WIDE   | 45.95     | 48.95     | 50.62     | 49.71      |

| **数据集** | **完整** | **无分位数对齐** | **对齐增益** | **无全局净化** | **对角协方差** |
|------------|----------|------------------|--------------|----------------|----------------|
| VOC07      | 92.53    | 88.94            | +3.58        | 92.53          | 91.71          |
| VOC12      | 92.29    | 88.18            | +4.11        | 92.29          | 91.30          |
| COCO       | 72.98    | 65.80            | +7.19        | 67.63          | 73.32          |
| NUS-WIDE   | 50.62    | 46.85            | +3.77        | 50.61          | 50.40          |

K=512 在四个数据集均最优。Quantile alignment 是一致且幅度最大的关键消融。全局净化在 COCO 上贡献显著，但在 VOC/NUS 上接近中性；对角协方差在 COCO 略高、其余数据集略低，不能写成所有数据集一致增益。

**图 1 QAFedPASA 关键消融（源：paper/ablation.csv）**

<img src="assets/media/image1.png" title="图 1  QAFedPASA 关键消融（源：paper/ablation.csv）" style="width:6.2in;height:0.95891in" alt="图 1 QAFedPASA 关键消融（源：paper/ablation.csv）" />

## 4.2 KMeans 客户端规模与集中式等价

| **数据集** | **runs** | **clients** | **mAP 范围**      | **最大权重误差** | **通信 MiB** |
|------------|----------|-------------|-------------------|------------------|--------------|
| VOC07      | 18       | 2-20        | 92.52867-92.52867 | 3.81e-06         | 4.39-43.92   |
| VOC12      | 18       | 2-20        | 92.28735-92.28735 | 4.77e-07         | 4.39-43.92   |
| COCO       | 18       | 8-80        | 72.98428-72.98428 | 7.63e-06         | 22.27-222.67 |
| NUS-WIDE   | 18       | 8-81        | 50.62047-50.62047 | 9.54e-07         | 22.35-226.25 |

72 个主实验配置的参数恢复检查均在登记容差内通过；其中 mAP 按 main.json 的 score_provenance 在参数检查后复用代表性完整测试结果，并非 72 次独立测试。该证据支持给定选样条件下的 central-equivalence；客户端缺失造成的退化见第 6.2 节。

**图 2 外部基线与客户端规模异质性对比**

<img src="assets/media/image2.png" title="图 2  外部基线与客户端规模异质性对比" style="width:6.2in;height:1.53309in" alt="图 2 外部基线与客户端规模异质性对比" />

## 4.3 Dirichlet 强 Non-IID

| **数据集** | **beta**    | **runs** | **mAP**  | **最大权重误差** | **等价** |
|------------|-------------|----------|----------|------------------|----------|
| VOC07      | 0.1/0.3/0.5 | 9        | 92.52867 | 3.81e-06         | True     |
| VOC12      | 0.1/0.3/0.5 | 9        | 92.28735 | 1.91e-06         | True     |
| COCO       | 0.1/0.3/0.5 | 9        | 72.98428 | 7.63e-06         | True     |
| NUS-WIDE   | 0.1/0.3/0.5 | 9        | 50.62047 | 9.54e-07         | True     |

## 4.4 Naive federation 与 estimator-preserving federation

| **数据集** | **划分** | **Naive-A** | **Naive-B** | **FedPASA** | **记录误差*** |
|------------|----------|-------------|-------------|-------------|-------------------|
| COCO       | beta=0.1 | 67.32       | 73.38       | 72.98       | 0.0e+00           |
| COCO       | beta=0.3 | 67.42       | 73.34       | 72.98       | 0.0e+00           |
| COCO       | beta=0.5 | 67.41       | 73.52       | 72.98       | 0.0e+00           |
| COCO       | KMeans   | 66.92       | 71.31       | 72.98       | 0.0e+00           |
| NUS-WIDE   | beta=0.1 | 47.13       | 50.66       | 50.62       | 0.0e+00           |
| NUS-WIDE   | beta=0.3 | 47.16       | 50.77       | 50.62       | 0.0e+00           |
| NUS-WIDE   | beta=0.5 | 47.19       | 50.82       | 50.62       | 0.0e+00           |
| NUS-WIDE   | KMeans   | 47.20       | 50.12       | 50.62       | 0.0e+00           |

四数据集 × 4 划分 × 6 方法共 96 行。表中记录误差为 0.0e+00 的 16 行 QAFedPASA 结果，均在权重和偏置误差不超过 1e-5 后复用中央预测；该零值不是独立推理的误差测量。来源：run_reviewer_evidence.py 的复用分支及 estimator_preserving_evidence.json 的 score_provenance。Naive 方法最大相对权重误差为 0.909；patch-ID recall 为 98.350%-99.961%，不能将不同 ID 自动视为等价替换。独立推理证据见第 11 节的 supplementary/estimator_mismatch.csv 与 federation_gap.csv；这些实验的条件与旧表不同，不互换数值。

Naive-A 为 local Top-K union 后 pooled solve；Naive-B 为 sample-weighted local GDA 参数平均；Naive-C 为 sample-weighted local Ridge 参数平均。为避免利用数值崩溃夸大差距，naive PIAA 的 mean+std 净化阈值在饱和 tie 时保守夹到该类观测最大值。

**图 3 Naive local-union 与 FedPASA 的 Non-IID estimator 证据**

<img src="assets/media/image3.png" title="图 3  Naive local-union 与 FedPASA 的 Non-IID estimator 证据" style="width:6.2in;height:2.58333in" alt="图 3 Naive local-union 与 FedPASA 的 Non-IID estimator 证据" />

# 5. 支撑 I2/I3 的实验：多视图闭式适配与融合

## 5.1 七种冻结特征组合

| **特征**                         | **VOC07** | **VOC12** | **COCO** | **NUS** |
|----------------------------------|-----------|-----------|----------|---------|
| cls                              | 91.81     | 91.40     | 76.12    | 58.41   |
| patch_mean                      | 90.19     | 90.23     | 71.06    | 54.23   |
| patch_max_text                 | 91.99     | 91.96     | 73.90    | 51.39   |
| cls+patch_mean                  | 92.81     | 92.54     | 77.85    | 59.78   |
| cls+patch_max_text             | 93.80     | 93.40     | 78.85    | 59.23   |
| patch_mean+patch_max_text     | 93.76     | 93.57     | 77.49    | 56.90   |
| cls+patch_mean+patch_max_text | 94.03     | 93.69     | 79.83    | 60.19   |

三视图 `cls+patch_mean+patch_max_text` 在 VOC07、VOC12、COCO 最优，并在 NUS 上接近最优；其选定 alpha 和完整验证分数保存在 `label_feature_ablation.csv`。

## 5.2 联邦 Ridge 与集中式闭式解一致

| **数据集** | **clients** | **集中式 mAP** | **联邦 mAP** | **最大分数误差** | **拟合上传 MiB*** |
|------------|-------------|----------------|--------------|------------------|------------------|
| VOC07      | 12          | 94.03261       | 94.03261     | 5.96e-08         | 128.15           |
| VOC12      | 12          | 93.65172       | 93.65172     | 0.00e+00         | 128.15           |
| COCO       | 48          | 79.82984       | 79.82984     | 2.38e-07         | 612.72           |
| NUS-WIDE   | 49          | 60.19301       | 60.19301     | 1.19e-07         | 627.24           |

本节核验 seed=17 的最终 Ridge 拟合：四数据集 mAP 绝对误差均为 0，最大测试分数误差 2.38e-7。组合通信仅计最终拟合统计上传，采用 QA FP32 + Ridge FP64；Full 校准/选参未计入。

| **数据集** | **Quantile** | **GDA moments** | **Ridge FP64** | **拟合上传 MiB*** |
|------------|--------------|-----------------|-----------------|---------------|
| VOC07      | 0.47         | 25.88           | 101.80          | 128.15        |
| VOC12      | 0.47         | 25.88           | 101.80          | 128.15        |
| COCO       | 7.50         | 126.10          | 479.12          | 612.72        |
| NUS-WIDE   | 7.75         | 129.11          | 490.37          | 627.24        |

| **数据集** | **clients** | **client bank s** | **ridge packet s** | **condition wall s** |
|------------|-------------|-------------------|--------------------|----------------------|
| VOC07      | 10          | 8.81              | 0.14               | 31.24                |
| VOC12      | 10          | 10.22             | 0.16               | 29.35                |
| COCO       | 10          | 310.65            | 1.86               | 387.08               |
| NUS-WIDE   | 10          | 230.16            | 2.81               | 311.72               |

本表是超参数已确定后的拟合统计上传量：QAFedPASA 为 4 次上传，按 FP32 dense upper bound 计量；Ridge 为 1 次上传，按 FP64 参考口径计量。COCO/NUS 合计 612.72/627.24 MiB；若 Ridge 也按 FP32 传输口径计量，则为 373.16/382.05 MiB，见第 11 节 S07。这些总量不包含三划分校准、选参及全部重拟合成本，也不包含服务器下行。运行时来自 10-client KMeans evidence condition，客户端在单机顺序模拟。

## 5.3 三训练划分与 bootstrap

| **数据集** | **3 seed mAP**    | **seed std** | **CV 聚合** | **95% bootstrap CI** |
|------------|-------------------|--------------|-------------|----------------------|
| VOC07      | 94.07/94.12/94.19 | 0.057        | 94.22       | \[93.71, 94.73\]     |
| VOC12      | 93.83/93.91/93.85 | 0.042        | 93.98       | \[93.50, 94.44\]     |
| COCO       | 80.36/80.46/80.54 | 0.089        | 80.53       | \[80.25, 80.83\]     |
| NUS-WIDE   | 60.69/60.45/60.51 | 0.122        | 60.67       | \[60.23, 61.70\]     |

每个数据集 1000 次 image-level bootstrap，seed=2026。VOC07/COCO 对所有 t 的 P(margin&gt;2)=1；NUS 在最强 t10 基线上为 0.929，其余 t 为 1.0。外部基线没有方差，因此该概率条件于公开 mAP 标量。

## 5.4 QAFedPASA 到 FedPASA-Full 的增益

| **数据集** | **QAFedPASA** | **FedPASA-Full** | **融合增益** |
|------------|---------------|------------------|--------------|
| VOC07      | 92.53         | 94.22            | +1.69        |
| VOC12      | 92.29         | 93.98            | +1.70        |
| COCO       | 72.98         | 80.53            | +7.54        |
| NUS-WIDE   | 50.62         | 60.67            | +10.05       |

## 5.5 Ridge-only 与 QAFedPASA 的逐类互补

| **数据集** | **Full&gt;Ridge** | **Full≈Ridge** | **Full&lt;Ridge** | **正增益均值** |
|------------|-------------------|----------------|-------------------|----------------|
| VOC07      | 12                | 4              | 4                 | 0.51           |
| VOC12      | 15                | 2              | 3                 | 0.44           |
| COCO       | 59                | 18             | 3                 | 0.96           |
| NUS-WIDE   | 42                | 19             | 20                | 1.06           |

| **数据集** | **Ridge** | **global fusion** | **classwise full** | **classwise-global** |
|------------|-----------|-------------------|--------------------|----------------------|
| VOC07      | 94.03     | 94.31             | 94.22              | -0.09                |
| VOC12      | 93.69     | 93.98             | 93.98              | +0.00                |
| COCO       | 79.83     | 80.28             | 80.53              | +0.25                |
| NUS-WIDE   | 60.19     | 60.33             | 60.67              | +0.34                |

COCO 有 59/80 类明确受益、平均正增益 0.96 AP；NUS-WIDE 有 42/81 类受益、平均正增益 1.06 AP。互补不是逐类单调：NUS 有 20 类下降，VOC07 的 global fusion 还比 classwise 高 0.09 mAP，均应如实保留。核心结论是 QAFedPASA 在 Ridge 失效的部分类别提供补充证据，而不是保证所有类别提高。

**图 4 COCO/NUS-WIDE 的 Full-Ridge 逐类 AP 差**

<img src="assets/media/image4.png" title="图 4  COCO/NUS-WIDE 的 Full-Ridge 逐类 AP 差" style="width:6.2in;height:3.27222in" alt="图 4 COCO/NUS-WIDE 的 Full-Ridge 逐类 AP 差" />

# 6. 鲁棒性、系统与失败分析

## 6.1 缺失正标签（60 个原始配置）

| **缺失率** | **VOC07** | **VOC12** | **COCO** | **NUS** |
|------------|-----------|-----------|----------|---------|
| 10%        | 93.97     | 93.82     | 80.24    | 60.25   |
| 30%        | 93.80     | 93.52     | 80.11    | 60.21   |
| 50%        | 93.20     | 92.72     | 79.78    | 59.99   |
| 70%        | 93.01     | 92.57     | 79.57    | 59.67   |
| 90%        | 90.99     | 92.37     | 77.92    | 56.17   |

表中为 3 seed 全局融合均值。90% 正标签缺失时仍为 90.99/92.37/77.92/56.17 mAP。QAFedPASA 分支理论上不读取训练标签，`annotation_mask_invariance.csv` 只记录该解析不变性，不把它伪装成额外经验运行。

**图 5 不完整正标签鲁棒性**

<img src="assets/media/image5.png" title="图 5  不完整正标签鲁棒性" style="width:6.2in;height:1.14757in" alt="图 5 不完整正标签鲁棒性" />

## 6.2 成对标签噪声与客户端参与率

| **pair noise** | **VOC07**  | **VOC12**  | **COCO**   | **NUS**    |
|----------------|------------|------------|------------|------------|
| 0%             | 94.30±0.01 | 93.93±0.06 | 80.28±0.02 | 60.32±0.01 |
| 5%             | 94.16±0.01 | 93.88±0.01 | 80.25±0.03 | 60.32±0.01 |
| 10%            | 94.11±0.03 | 93.74±0.08 | 80.23±0.03 | 60.30±0.04 |
| 20%            | 93.86±0.08 | 93.61±0.08 | 80.14±0.01 | 60.24±0.11 |

| **可用客户端** | **VOC07**  | **VOC12**  | **COCO**   | **NUS**    |
|----------------|------------|------------|------------|------------|
| 25%            | 88.06±1.53 | 90.38±1.99 | 76.94±0.90 | 57.19±0.67 |
| 50%            | 92.96±0.94 | 92.24±1.05 | 78.48±0.79 | 59.07±0.44 |
| 75%            | 93.91±0.06 | 93.49±0.10 | 79.95±0.30 | 59.89±0.29 |
| 100%           | 94.19±0.00 | 93.85±0.00 | 80.54±0.00 | 60.51±0.00 |

20% 标签噪声下结果仍稳定；75% 客户端可用时仅小幅下降。25% 可用时 VOC07/NUS 方差与下降明显，应作为部署下限而非稳健性亮点。所有融合/校准参数在参与率冲击前冻结。

**图 6 成对标签噪声鲁棒性**

<img src="assets/media/image6.png" title="图 6  成对标签噪声鲁棒性" style="width:6.2in;height:4.06875in" alt="图 6 成对标签噪声鲁棒性" />

**图 7 客户端参与率压力测试**

<img src="assets/media/image7.png" title="图 7  客户端参与率压力测试" style="width:6.2in;height:4.06875in" alt="图 7 客户端参与率压力测试" />

## 6.3 净化温度敏感性

| **数据集** | **倍率**             | **min** | **max** | **range** |
|------------|----------------------|---------|---------|-----------|
| VOC07      | 0.25/0.5/1.0/2.0/4.0 | 92.5287 | 92.5287 | 0.0000    |
| VOC12      | 0.25/0.5/1.0/2.0/4.0 | 92.2874 | 92.2874 | 0.0000    |
| COCO       | 0.25/0.5/1.0/2.0/4.0 | 72.9733 | 73.0132 | 0.0398    |
| NUS-WIDE   | 0.25/0.5/1.0/2.0/4.0 | 50.6205 | 50.6205 | 0.0000    |

*温度最大变化仅 0.04 mAP。投稿主文优先使用图 3 的 naive-vs-FedPASA 证据，本表和原温度图移至补充材料。*

## 6.4 通信压缩与系统边界

| **codec** | **平均压缩比** | **VOC07 Δ** | **VOC12 Δ** | **COCO Δ** | **NUS Δ** |
|-----------|----------------|-------------|-------------|------------|-----------|
| FP64      | 1.0x           | +0.000      | +0.000      | +0.000     | +0.000    |
| FP32      | 2.0x           | -0.000      | +0.000      | +0.000     | -0.000    |
| FP16      | 4.0x           | -0.027      | +0.019      | -1.376     | -1.934    |
| INT8      | 8.0x           | -42.547     | -41.775     | -75.976    | -56.140   |

FP32 将 Ridge packet 减半，绝对 mAP 变化均 &lt;0.00002。FP16 在 COCO/NUS 分别下降 1.376/1.934；朴素 per-tensor INT8 在四个数据集均失败，不能作为可用压缩方案。

**图 8 通信量-精度前沿（保留失败的低比特边界）**

<img src="assets/media/image8.png" title="图 8  通信量-精度前沿（保留失败的低比特边界）" style="width:6.2in;height:4.06875in" alt="图 8 通信量-精度前沿（保留失败的低比特边界）" />

## 6.5 长尾、逐类别与定性失败

| **数据集** | **稀有四分位 AP** | **常见四分位 AP** | **Full<QA 类数*** | **最低 AP 类**       |
|------------|-------------------|-------------------|------------------|----------------------|
| VOC07      | 96.21             | 94.58             | 2                | bottle (78.24)       |
| VOC12      | 95.39             | 94.81             | 0                | potted plant (78.96) |
| COCO       | 74.08             | 74.09             | 3                | toaster (14.97)      |
| NUS-WIDE   | 56.65             | 69.58             | 4                | town (25.01)         |

逐类 CSV 覆盖 201 个类别。COCO 的 toaster 最低（14.97 AP），NUS 的 town 最低（25.01 AP）；NUS 稀有类均值明显低于常见类。12 个定性例子分别选择每个数据集最弱类的高分负例、低分正例和高分正例。部分高分负例画面可见目标，提示公开多标签数据可能漏标。

* Full<QA 类数按 Full AP − QAFedPASA AP < 0 计数（严格小于零）；第 5.5 节比较 Full 与 Ridge，使用 ±0.05 AP 点的近似相等容差。两处参照分支和阈值不同。

**图 9 逐类 AP、训练频次与融合增益**

<img src="assets/media/image9.png" title="图 9  逐类 AP、训练频次与融合增益" style="width:6.2in;height:1.35948in" alt="图 9 逐类 AP、训练频次与融合增益" />

**图 10 四个数据集最弱类别的定性边界案例**

<img src="assets/media/image10.png" title="图 10  四个数据集最弱类别的定性边界案例" style="width:5.3in;height:5.9801in" alt="图 10 四个数据集最弱类别的定性边界案例" />

## 6.6 官方 FedMPT 同机审计

| **阻塞阶段**                 | **改变预测** | **修复**                                                                                |
|------------------------------|--------------|-----------------------------------------------------------------------------------------|
| checkpoint                   | 否           | FEDMPT_CLIP_CHECKPOINT hook with official SHA256 verification                         |
| cluster cache                | 否           | create output-scoped cache directory and preserve assignments after pickle.dump         |
| Windows DataLoader           | 否           | use zero DataLoader workers on Windows                                                  |
| condition prompt             | 是           | one condition placeholder for the one inserted attr_ctx vector                         |
| vanilla patch aggregation    | 是           | use \[C,D,1\] conv1d weights and the fixed 5x factor used by sibling multi-label models |
| mixed precision              | 否           | cast attr_ctx rows to the CLIP dtype                                                   |
| VOC split                    | 是           | pair trainval.txt with \*_trainval.txt annotations                                     |
| test protocol                | 是           | use split-specific clustering caches                                                    |
| CLIP clustering              | 否           | move text features to CPU for cached zero-shot probability construction                 |
| CLIP clustering tensor shape | 是           | select the CLS token at index 0 for the intended global-image clustering feature        |

修复后完整 VOC 5,011/4,952 协议完成 1 个诊断轮次：mAP=47.42，训练+最终评估 190.50s，峰值 CUDA 显存 3810.2 MiB，双向 prompt 通信 17.87 MiB/round。由于存在改变预测语义的修复，该 mAP 不能作为官方复现进入准确率表；主表继续使用 published external 数值。

# 7. 全部实验登记与证据路径

| **ID** | **实验**                  | **规模**                      | **主证据（均在 paper/）**                  | **状态**             |
|--------|---------------------------|-------------------------------|--------------------------------------------|----------------------|
| E01    | 外部比较                  | 主文 6 个单值；补充 11×3×6    | main_comparison.csv                       | PASS                 |
| E02    | KMeans 异质性             | 4×6×3=72                      | \*/main.json; systems.csv                  | PASS                 |
| E03    | Dirichlet Non-IID         | 4×3×3=36                      | dirichlet.csv                              | PASS                 |
| E04    | PIAA/QAFed 消融           | 4×10=40                       | ablation.csv                               | PASS                 |
| E05    | 标签特征消融              | 4×7=28                        | label_feature_ablation.csv               | PASS                 |
| E06    | 三训练划分                | 4×3=12                        | full_seed_summary.csv                    | PASS                 |
| E07    | image bootstrap           | 4×1000                        | statistics.json                            | PASS                 |
| E08    | 联邦闭式恢复              | 4 个数据集                    | federated_full_verification.json         | PASS                 |
| E09    | 缺失正标签                | 4×5×3=60                      | partial_annotation.json/csv               | PASS                 |
| E10    | 成对标签噪声              | 4×4×3=48                      | label_noise.json/csv                      | PASS                 |
| E11    | 客户端参与率              | 4×4×3=48                      | client_participation.json/csv             | PASS/25% 边界        |
| E12    | 净化温度                  | 4×5=20                        | temperature_sensitivity.json/csv          | PASS                 |
| E13    | 通信压缩                  | 4×4=16                        | communication_compression.json/csv        | FP32 PASS; INT8 FAIL |
| E14    | 逐类别失败                | 20+20+80+81=201               | per_class_failure_analysis.json/csv     | PASS                 |
| E15    | 定性失败                  | 4×3=12                        | qualitative_failure_examples.json        | PASS                 |
| E16    | FedMPT 官方代码           | 完整 VOC 1 轮                 | official_fedmpt_runnability.json         | 诊断完成             |
| E17    | 最终机器验收              | 4 数据集+投稿图               | verification_report.json                  | overall_pass=true   |
| E18    | Estimator-preserving 对照 | 4×4×6=96                      | estimator_preserving_evidence.json/csv   | PASS                 |
| E19    | 逐类互补性                | 20+20+80+81=201               | classwise_complementarity.json/csv        | PASS                 |
| E20    | 通信/运行时拆分           | 4 数据集                      | communication_runtime_breakdown.json/csv | PASS                 |
| E21    | 同协议外部三基线          | 历史扩展设想；现按第12节缩减 | official_fedmpt_runnability.json         | 见第12节最小方案       |

|                                                                                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **总量口径** `annotation_mask_invariance.csv` 是因为 QAFedPASA 不读标签而成立的解析不变性记录，不应重复计入 20 次经验实验。外部 11 个方法均为已发表数值，不是本地训练运行；主文展示其中 6 个代表方法的最佳公开值，完整 11×3×6 放补充材料。 |

# 8. 复现入口与执行顺序

以下 PowerShell 步骤用于从现有冻结缓存复现主方法及生成既有论文证据所需的组件结果，不是本次修改时已执行的新实验。应在独立工作区副本执行；脚本可能覆盖同名结果或复用缓存，要从头复现时请使用未含旧实验产物的输出目录，保留所需特征缓存。FedMPT 仍按服务器交接处理，不在本机恢复。

前置条件：四数据集的 train/test CLS、patch、patch_mean、patch_max_text、targets 和 text_features 缓存齐全，并与 manifest 中图像及类别顺序一致。VOC07 兼容 outputs/voc07/cache，其他数据集使用 outputs/icme/cache/<dataset>。缺失缓存时先用阶段表中的提取脚本和原始数据构建；本节不宣称自动下载或准备全部原始数据。

## 8.1 固定解释器与选择参数

```powershell
$ErrorActionPreference = 'Stop'
$PY = 'D:\Software\anaconda3\envs\pytorch\python.exe'
$datasets = @('voc07', 'voc12', 'coco', 'nus')
$seeds = @(7, 17, 27)
$alphas = @(0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
function Invoke-Py {
    & $PY @args
    if ($LASTEXITCODE -ne 0) { throw 'Python step failed' }
}
```

## 8.2 主分支与七种表示

```powershell
$featureSets = @(
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
}
```

## 8.3 三种子拟合与训练侧融合

```powershell
foreach ($dataset in $datasets) {
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
Invoke-Py scripts/build_cv_fusion_ensemble.py --datasets @datasets
```

## 8.4 既有实验汇总与验收

```powershell
Invoke-Py scripts/run_partial_annotation.py --datasets @datasets
Invoke-Py scripts/run_reviewer_robustness.py --datasets @datasets
Invoke-Py scripts/run_reviewer_federated_stress.py --datasets @datasets
Invoke-Py scripts/run_reviewer_evidence.py --datasets @datasets
Invoke-Py scripts/verify_federated_full.py --datasets @datasets
Invoke-Py scripts/build_icme_artifacts.py --bootstrap 1000
Invoke-Py scripts/audit_icme_results.py --require-complete
Invoke-Py -m pytest -q
```

CV 聚合前应确认四数据集各有 seed7/seed17/seed27 的 ridge/result.json、validation_adapter.pt、fit_indices.npy、validation_indices.npy，以及 ridge_patch_fusion/test_scores_classwise.npy。选中 lambda 和 split seed 应与第 3.1.2 节一致。第 8.4 节复用既有定性样例和官方可运行性审计等证据；新工作区如缺少这些产物，必须据实补齐，不能把缺失改为 PASS。其中 run_reviewer_evidence.py 保留第 4.4 节所披露的中央预测复用行为。

| **阶段**     | **脚本**                                                      | **产物**                                                      |
|--------------|---------------------------------------------------------------|---------------------------------------------------------------|
| 特征缓存     | extract_benchmark_features.py; extract_patch_summaries.py | outputs/icme/cache/&lt;dataset&gt;/                           |
| QAFed 主实验 | run_icme_experiments.py                                     | outputs/icme/&lt;dataset&gt;/{main,dirichlet,ablation}.json   |
| 多视图 Ridge | run_label_ridge.py                                          | outputs/icme/&lt;dataset&gt;/label_ridge\*/                  |
| 逐类融合     | run_label_patch_fusion.py; build_cv_fusion_ensemble.py  | outputs/icme/&lt;dataset&gt;/cv_fusion/                      |
| 审稿补充实验 | run_partial_annotation.py; run_reviewer_\*.py             | estimator/complementarity/communication 等 JSON\|CSV\|figures |
| 汇总验收     | build_icme_artifacts.py; audit_icme_results.py            | artifact_manifest.json; verification_report.json            |

# 9. 论文写作声明边界与后续动作

1.  FedPASA-Full 使用训练标签；只有 QAFedPASA 分支是 label-free。整体可写 backpropagation-free / closed-form，不写 fully training-free。

2.  拟合阶段不传 individual examples 不是差分隐私。Full 的分布式校准和选参尚未验证；没有隐私预算、成员推断或重构攻击实验，不写 formal privacy guarantee。

3.  外部 11 个方法来自 FedMPT Table 1；必须明确 published external。主文 6 个代表方法统一取各自公开设置中的最高 mAP，完整范围与逐 t 数字放补充材料。FedMPT 修复版 47.42 mAP 只放补充材料的可运行性审计。

4.  VOC12 无对应联邦外部表项：不能进入 Published Federated SOTA Comparison，也不能写“所有四个数据集均超过最强外部基线 2 点”；93.98 mAP 仅作完整性报告，并保留在四数据集组件/消融表中。

5.  25% 客户端可用、FP16/INT8 和少数负增益类别是已观测边界，limitations 中保留。

6.  部分 qualitative high-score negatives 实际含有目标，优先解释为预测错误与潜在漏标的混合边界。

7.  QAFed 的 Non-IID 不变性源于全局充分统计恢复；真实客户端掉线的退化由 participation 实验描述。

8.  Estimator-preserving 旧对照使用真实 local Top-K bank；patch-ID recall 不足 100% 不能自动解释为等价替换。旧表分数零误差来自中央预测复用；独立推理和 shared-key 条件分别见第 11 节，须保留各自来源和实验条件。

9.  同协议外部三基线仍未完成；在获得 author-equivalent 代码与完整数据前，只能写 published external comparison，不能加 \* reproduced under our protocol。

## 9.1 投稿版建议取舍

-   主文 SOTA 表：单栏 table\[t\]，仅 VOC07/COCO/NUS-WIDE，保留 FedPGP、FedTPG、FedAWA、Fed-RAM、FedMVP、FedMPT；每个 baseline 取公开范围最大值；FedMPT 下划线，FedPASA 与 Gain 加粗。

-   主文组件表：双栏 table\*\[t\]，保留 VOC07/VOC12/COCO/NUS-WIDE 和七个关键变体；这是 VOC12 93.98 的主要表格位置。另保留 central-equivalence/通信拆分表。

-   主文 Fig. 3(a) 改为 COCO/NUS 的 Naive local-union vs FedPASA under Non-IID；Fig. 3(b) 保留客户端可用率。温度图移至补充材料。

-   正文增加一小段 classwise complementarity：报告 Full&gt;Ridge/≈/&lt;Ridge 数量和平均正增益，不写所有类别单调受益。

-   补充材料保留：全部 11 外部基线的范围与逐 t 表、Dirichlet、完整 label/noise/participation、压缩负结果、逐类 CSV 和 FedMPT 审计。

-   方法段明确三层结构 I1/I2/I3；不要继续把失败的 reliability-only 分支放进贡献列表。

-   标题/摘要避免承诺 formal privacy 或完全 annotation-free；可写 no iterative backpropagation，并将 no individual-example transmission 限定于已实现的拟合统计协议。

-   最终提交前分别运行 `audit_icme_results.py --require-complete` 与 `audit_handoff_document.py`。前者核验结果元数据和既有证据，reviewer_suite 当前为四项；后者核验文档结构、关键措辞及复现命令，两类验收不能相互替代。

# 10. 来源与交接验收

| **来源**       | **位置**                                                                                       |
|----------------|------------------------------------------------------------------------------------------------|
| PIAA 原稿      | C:\\Users\\17597\\Downloads\\CLS_is_Not_Enough_ICML2026_PIAA.pdf                          |
| 初始创新备忘   | C:\\Users\\17597\\.codex\\attachments\\529b259a-b6c4-4d4e-96ba-6ca0a7640a22\\pasted-text-1.txt |
| 审稿实验原则   | https://maxwellforbes.com/posts/how-to-get-a-paper-accepted/                                   |
| 外部联邦表     | https://arxiv.org/html/2605.28347                                                              |
| 完整实验报告   | D:\\project\\VScodeProject\\FedPASA\\outputs\\icme\\paper\\README.md                           |
| 最终验收       | D:\\project\\VScodeProject\\FedPASA\\outputs\\icme\\paper\\verification_report.json           |
| 审稿清单       | D:\\project\\VScodeProject\\FedPASA\\outputs\\icme\\paper\\REVIEWER_CHECKLIST.md              |
| 本轮补实验要求 | C:\\Users\\17597\\.codex\\attachments\\0fbe7f76-b202-4327-9520-0b12203b685c\\pasted-text.txt   |

-   机器验收 overall_pass = True。

-   2026-09-09 内容审查时，现有 pytest 为 8 passed；此记录仅指当时执行的单元测试，不代表新实验训练完成。文档修改后需另行执行文档审计。

-   11 组投稿图均有非空 PNG/PDF；12 个定性样例通过无测试集选参检查。

-   96 行 estimator-preserving 对照、201 类互补性分析与 4 行通信拆分均通过机器验收。

-   文档中的关键数字均来自 outputs/icme/paper 下的机器可读 JSON/CSV。

|                                                                                                                                                                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **交接状态** P0-2 estimator-preserving、P0-3 classwise complementarity、P1-1 Non-IID 曲线与 P1-2 通信/运行时拆分已经闭环。P0-1 可信同协议强基线仍有证据缺口；此外，Full 分布式校准/选参协议仍有证据缺口；文档修订不表示这些实验或实现已经完成。 |

# 11 补充实验清单与执行交接

本节覆盖原定 9 项补充实验。2026-09-10 已核验全部完成；其中 S01 按用户确认的修补版口径验收。9/9 表示这份执行清单完成，不表示作者等价强基线、Full 全流程联邦化和全部投稿证据已齐全；剩余重点见第 12 节。

交接保留实验结论、验收条件、脚本和结果索引。未完成训练的控制台流水已清理，模型断点和已完成轮次的结构化评估仍保留；阶段性 mAP 不进入正式基线表。

## 11.1 需要补充的实验总清单

| 编号 | 实验 | 优先级 | 当前结论 |
|---|---|---|---|
| S01 | 同协议强基线与原理基线 | P0 | 修补版已完成 |
| S02 | 选样偏差与本地模型平均分解 | P0 | 已完成 |
| S03 | QA 与 Ridge 逐类互补和融合权重 | P0 | 已完成 |
| S04 | 中央目标与联邦差距 | P1 | 已完成 |
| S05 | Boundary tie 可控压力测试 | P1 | 已完成 |
| S06 | 客户端数量扩展性 | P1 | 已完成 |
| S07 | 统一口径通信基线对比 | P1 | 已完成 |
| S08 | 五种融合策略消融 | P2 | 已完成 |
| S09 | 三视图表示消融 | P2 | 已完成 |

本轮没有继续扩展 K、temperature、噪声等级、参与率、量化精度、随机种子或新增数据集。已经完成的结果可以复用，无需因 FedMPT 中断而整套重跑。

## 11.2 已完成的 FedMPT 修补版基线

S01 的闭式对照已经具备：Fed3R-style CLS ridge、Naive local PIAA/GDA averaging、中央 pooled estimator、QAFedPASA、三视图 Ridge 和锁定 FedPASA-Full。COCO 与 NUS-WIDE 的 FedMPT 修补版最终结果均已完成，见第 12.1 节。

共同协议为 CLIP ViT-B/16、224 输入、相同训练和测试图像列表、相同标签顺序、10 个 KMeans 客户端、划分 seed=7 及同一 mAP 实现。闭式方法使用相同冻结缓存；FedMPT 的视觉提示改变图像编码过程，必须重新处理图像，因此只能称数据与划分等条件匹配，不能称最终特征缓存相同。

修补版差异包括 vanilla 聚合、4 个内置条件、每客户端轮次重置优化器动量及 BF16。9 月 7 日服务器性能记录已关闭激活重计算、启用静态文本缓存，COCO/NUS workers=4/8，cpu-threads=1；执行入口为 start_server_fedmpt.sh。具体以服务器 protocol.json 的 runtime 与实际进程为准。该实现不是官方算法等价复现。

当前断点绑定的训练预算为每数据集 200 轮，每轮全客户端各遍历一遍本地训练图像。200 轮是本次预先登记的预算，也与 FedMPT 原论文 COCO/NUS 的训练轮数一致；它不是建议文本要求的最低验收轮数。预算或算法若调整，应单独登记并保留原断点；短轮数结果必须标为有限预算实验，不得直接宣称完整 200 轮基线完成。

续跑前检查实际 python 进程、checkpoint.pt 和 progress.json 的修改时间。状态文件可能残留 RUNNING，不能单独作为进程存活证据。训练曾中断且未留下异常堆栈，根因尚未确认；交接时不要把累计墙钟时间等同于持续 GPU 计算时间。

完成核验：两个 checkpoint 均为第 200 轮；result.json 与最终预测齐全，mAP 复算差值为 0。全部 400 轮、4000 个客户端轮次的记录完整，参数与预测均有限。后续新实验必须单独登记，不能依据测试集 mAP 调优或替换预先固定的最后一轮。

S01 已满足本轮验收条件：两个数据集均有 200 轮 result.json、round200_test_scores.npy、checkpoint.pt、协议和修补说明。只报告固定第 200 轮；不能称官方算法等价复现。

## 11.3 已完成实验的结果与写作要点

| 变体 | COCO mAP | NUS mAP |
|---|---|---|
| 中央 QA | 72.98 | 50.62 |
| 对齐选样后本地 GDA 平均 | 65.90 | 46.05 |
| 共享净化后本地 GDA 平均 | 66.12 | 46.05 |
| 充分统计量汇总 QA | 72.98 | 50.62 |
| Fed3R-style CLS ridge | 76.13 | 58.41 |
| Pooled Ridge | 79.83 | 60.19 |

S02 直接隔离两个问题：全局选样对齐修复 patch 总体的偏差；充分统计量汇总修复本地独立求解后平均的偏差。SharedPurification 控制进一步固定净化后的 patch 集，仍出现模型平均偏差。最大权重误差采用 max-entry 绝对误差，并分别推理中央和联邦测试分数。

| 融合策略 | COCO mAP | NUS mAP |
|---|---|---|
| QA only | 72.98 | 50.62 |
| Ridge only | 79.83 | 60.19 |
| Fixed 0.5 | 79.30 | 59.08 |
| Global learned | 80.28 | 60.33 |
| Classwise learned | 80.53 | 60.67 |

S03 和 S08：alpha=1 表示 Ridge，alpha=0 表示 QA。逐类图给出三个训练侧划分的 alpha 均值和范围，以及按增益排序的 Full−Ridge AP。COCO/NUS 固定 0.5 融合低于 Ridge，学习融合有必要；VOC07 全局权重优于逐类权重的负面结果仍保留。类别增益计数沿用 ±0.05 AP 点容差，避免与旧报告口径冲突。

S04 与 S06：四个数据集、各 6 个客户端数设置，共 72 项 QA/Ridge/Full 分支比较；最大中央与联邦 mAP 差距约 5.01×10⁻⁷ 个百分点。扩展性固定中央 Top-K 并分配给所属客户端，测的是统计构建、服务器合并求解和完整测试推理，不能冒充完整局部选样的端到端运行时间。中央目标是同一估计器的参考，不是精度上界。

S05：在完整训练 patch 总体上先量化熵再选样，覆盖 FP32、1e-4、1e-3、1e-2；COCO 与 NUS 各有 local/shared 两种 tie 规则，共 16 项。1e-2 下 local ID recall 为 91.11%/94.12%，shared 在所有精度下均为 100%。共享 key 的最大测试分数误差约 1.94×10⁻⁷。

共享 tie key 必须同时用于本地截断和服务器排序，仅在服务器增加 key 不能恢复本地已丢弃的候选。该可选方案额外上传 int64 patch ID，所以不能沿用不上传 patch identity 的旧声明。普通 tie 规则下 mAP 接近不等于参数或测试分数完全相同。

S07：统一按 client→server 上传计量。COCO/NUS 的 QA FP32 + Ridge FP32 合计为 373.16/382.05 MiB；FP64 Ridge 参考口径为 612.72/627.24 MiB。原始全局选中 patch 上传是忽略选样协调成本的 oracle 下界，局部 Top-K 上传又改变了样本总体，表中分别标注。

S09：四数据集分别核验 CLS、CLS+Patch Mean、CLS+Patch-Text、CLS+Mean+Patch-Text 四种表示，共 16 行，使用训练侧验证集选正则化系数并复核保存的测试分数。

## 11.4 证据与复现入口

新增结果根目录：D:/project/VScodeProject/FedPASA/outputs/icme/supplementary/。下列文件均相对该目录，PNG/PDF 投稿图位于 figures/。

| 实验 | 机器可读结果文件 |
|---|---|
| S01 | protocol_matched_baselines.csv |
| S02 | estimator_mismatch.csv |
| S03 | classwise_complementarity.csv 和 complementarity_summary.csv |
| S04 | federation_gap.csv |
| S05 | boundary_ties.csv |
| S06 | client_scaling.csv |
| S07 | communication_baselines.csv |
| S08 | fusion_strategies.csv |
| S09 | representation_ablation.csv |

总体状态和协议指纹：completion_audit.json、protocol_manifest.json、README.md。FedMPT 最终文件位于 coco/fedmpt_repaired/ 与 nus/fedmpt_repaired/：result.json、round200_test_scores.npy、checkpoint.pt 和 protocol.json。其余 roundNNN 文件保留训练轨迹。protocol 内嵌 RUNNING 是构建时快照，实际完成状态以最终结果、进度及断点的 200 轮交叉核验为准。

服务器最终结果已同步本机。以下命令仅供核验和重建报告，当前无需续跑；原迁移前文件另存 tmp/before_server_final_20260910/。

```bash
cd /root/autodl-tmp/FedPASA
tmux ls
tail -n 20 outputs/server_logs/fedmpt_optimized_20260907.log
cat outputs/icme/supplementary/coco/fedmpt_repaired/progress.json
cat outputs/icme/supplementary/nus/fedmpt_repaired/progress.json
# 两个数据集均已 COMPLETE 200/200，无需再次启动训练。
# 报告重建只读取已保存结果，不训练：
/root/autodl-tmp/miniconda3/envs/fbu/bin/python scripts/build_supplementary_report.py --no-plots
```


本次补充要求来源为用户提供的 pasted-text.txt 建议；方法依据为 FedMPT 论文与发布代码、Fed3R 论文。正式文稿应区分 published external、disclosed repaired rerun、Fed3R-style adaptation 和本地机制控制实验。

# 12 ICME 2027 投稿前证据审查与最小补实验计划

本节保留 2026-09-10 的历史计划。该计划已于 9 月 13 日完成服务器执行，最终结果及未满足的精确等价边界见第 13 节。下列优先级属于研究建议，不是 ICME 官方要求的实验数量。

## 12.1 完成状态和最终结果

| 数据集 | 完成轮数 | 修补版 mAP | 预测矩阵 | 指标复算差 |
| --- | --- | --- | --- | --- |
| COCO | 200/200 | 69.806112 | 40137 × 80 | 0 |
| NUS-WIDE | 200/200 | 35.823527 | 60260 × 81 | 0 |

总任务与 9 项完成审计均为 COMPLETE，完成时间为北京时间 2026-09-09 22:50:56。9 月 10 日核验时无训练进程或 tmux 会话，属于任务完成后的正常退出。480 个归档文件哈希一致；两个最终模型参数和预测均有限，400 个完整轮次和 4000 个客户端轮次记录通过检查。指标核验使用保存预测重新计算 mAP，没有重新执行图像推理。

验收口径延续用户确认的“相同数据、划分、骨干的修补版并披露差异”。原始 final result/protocol 内保留了早期诊断策略与 RUNNING 快照等历史字段；本节依据最终 result、progress、checkpoint 和数值审计判定完成，不改写原始证据。修补版可以单列报告，不能覆盖 published external 数字或标成作者等价复现。

| 评估轮数 | COCO mAP | NUS mAP |
| --- | --- | --- |
| 1 | 10.715 | 6.026 |
| 5 | 38.485 | 15.942 |
| 10 | 49.532 | 19.326 |
| 25 | 59.979 | 23.850 |
| 50 | 65.035 | 29.084 |
| 100 | 68.035 | 33.871 |
| 200 | 69.806 | 35.824 |

NUS 从第 100 轮 33.871 提升到第 200 轮 35.824，轨迹仍在改善；这不等于已经证明收敛。其最终结果较论文公开最高值 58.36 低 22.54 点，但设置和算法存在差异，不能把差距归因于某一修补或只归因于训练时长。200 轮与论文 COCO/NUS 预算相同；现在依据测试曲线追加轮数或选取最佳测试轮均不合适。

## 12.2 P0 Full 的分布式校准与选参核验

当前缺口在 run_label_patch_fusion.py：训练侧 validation 的逐样本标签和预测在中心集中计算。GDA/Ridge 充分统计量可加、没有测试集选参，都不能单独证明 Full 全流程不上传单样本。若论文保留 Full 端到端联邦化主张，这项必须先闭环；若采用公开校准集或集中式校准，应把它明确写入问题设定并限定通信声明。

最小实验：COCO 和 NUS，固定 10 个客户端，先用已有 seed=7 验证协议，再覆盖原有 7/17/27 三个训练侧划分。客户端统计 count/sum/sum-square 可以合并 z-score；全局 AP 涉及跨客户端排序，不能用“本地 AP 平均”替代。需要实现明确的排序/安全汇总协议，或将分箱 AP 标为近似变体；不得将拟议方案写成已实现。

验收：独立的集中式与分布式路径比较校准均值/方差、全部候选 lambda 和融合权重的验证 AP、最终所选参数及测试分数；给出精确或近似容差。记录客户端到服务器和服务器到客户端的实际字节、交互次数、dtype、全部候选选择与重拟合成本，以及每种消息包含哪些数据。现有“共享既定校准参数后的 Full 等价”不能替代这项验收。

## 12.3 P0 强基线的有效性与公平比较

修补版 FedMPT 的 vanilla 聚合、条件提示等与论文的最优传输和条件流程不等价；现有 seed=7 结果只说明该登记实现跑完。先检查发布代码、条件文件、聚合方式、标签映射、损失归一化和训练侧诊断；能取得作者可运行配置时再建立作者等价基线。修补版 NUS 的低值应保留，不能据此声称对原方法取得 24 点以上的受控提升。

最小方案：COCO/NUS 两个数据集，保留已完成的 Fed3R-style 与机制对照，新增至少一个可信的强提示学习基线；优先解决 FedMPT 的算法一致性，无法解决时评估 FedMVP 或 FedPGP 官方实现的可运行性与多标签适配。算力允许再加第二个强基线。Fed3R-style 是 CLS 多标签 Ridge 适配，仍需明确与原 Fed3R 的任务差异。无需为了扩表强行移植不支持 VOC12 的方法。

公平口径：逐方法登记同一 CLIP 权重、图像列表、标签顺序、客户端划分、分辨率、mAP、训练侧选参预算和最后轮数。提示方法重算图像特征是算法需要，不能要求复用最终冻结缓存。单种子主比较使用已有 Full seed7 的预测；三划分 Full 聚合单列，并报告其全部选择成本，避免与单次训练基线混为相同计算预算。

## 12.4 P1 配对统计和独立重复

先做无需 GPU 的复分析：从已有预测对 Full 与相同口径的 Ridge、全局融合做配对 image bootstrap。每次对所有方法使用同一组图像重采样，报告 mAP 差值及 95% 区间；至少 COCO/NUS，建议顺带覆盖已有四数据集。现有 statistics.json 是相对公开单值的条件概率，不能代替同协议方法之间的配对显著性证据。

三个训练侧划分 seed7/17/27 已存在，不应作为“缺少三种子”重新跑一遍；但它们是最终 Full 聚合器的三个组成部分，并非三个独立的完整 Full 实验。先重用这些单划分预测报告逐种子差值。基线有效性解决后，再给迭代强基线补 17/27 两个训练种子，并同步相同客户端划分；若主张完整聚合方法的跨运行稳定性，另登记独立外层重复。已验证参数恢复的确定性闭式算法无需重复相同计算。

统计实现检查：历史 bootstrap_map 对同分预测逐行计算 precision，而正式 sklearn AP 合并同分阈值。单位权重诊断中 COCO/NUS 的 mAP 差约 −2.06×10⁻⁷/−2.63×10⁻⁶ 点，影响很小，不能据此否定主结果；但生成新的配对区间前应修正 tie 处理并用小样本对照 sklearn 的重采样 AP。旧区间保留为历史记录，本次没有生成新的区间。证据：bootstrap_metric_audit_20260910.json。

## 12.5 系统结果与无需扩展的实验

P1 系统核验：在同一 RTX 5090 上将冻结 CLIP 特征提取、拟合、校准/选择、最终推理分别计时，同时报告缓存已存在与从原始图像开始的总时间、峰值显存和双向通信。复用已保存的 FedMPT 训练记录，不必再跑 200 轮。训练跨本机与服务器、旧慢配置与新配置，不能用累计 53.07 小时直接当作统一硬件基准；仅用最后一轮时间也不代表全程总成本。

不再扩展：K 和 temperature 网格、更多噪声率/缺失率/参与率、更多低比特压缩、更多客户端数或新数据集。已有选样与模型平均分解、201 类互补、16 个边界 tie 设置、72 个中央/联邦比较、融合及表示消融足以支持机制分析。若论文增加跨骨干泛化或正式隐私主张，再补对应证据；当前没有必要为这两项扩展研究范围。

| 优先顺序 | 最小交付 | 计算需求 |
| --- | --- | --- |
| 1 | Full 分布式校准协议及等价/通信核验 | 缓存与 CPU 为主 |
| 2 | 可信强基线的配置核验及 COCO/NUS 对照 | 先短诊断，再决定长训练 |
| 3 | Full 对 Ridge 的配对区间与单划分差值 | 已有预测，无需 GPU |
| 4 | 有效基线独立种子与统一硬件成本表 | 仅必要新训练；复用日志 |

## 12.6 ICME 2027 会议信息和提交边界

截至 2026-09-10，IEEE Circuits and Systems Society 官方活动页列出 ICME 2027 于 2027 年 7 月 13–17 日在中国厦门举行，页面注明更多细节待公布。本次未核验到正式论文截止日、页数、匿名和补充材料规定；不能把其他同名 ICME 会议或往届规则当作 2027 要求。正式投稿前以当届 CFP 和作者指南为准。

论文叙事建议：主表优先展示可解释的同协议本地比较，published external 表明确为背景参照；保留校准边界、shared tie key 上传标识、固定 0.5 融合下降、VOC07 全局权重优于逐类权重等限制。现有“比公开最高值多约 2 点”不是同协议受控增益，也不应当作投稿验收标准。

核验来源：IEEE CASS，2027 IEEE International Conference on Multimedia & Expo，https://ieee-cas.org/event/conference/2027-ieee-international-conference-multimedia-expo-icme 。

方法来源：FedMPT 原论文第 5 节实现设置与表 1，https://arxiv.org/html/2605.28347v1 ；发布代码 https://github.com/xuc865/FedMPT-main 。Fed3R 原论文 https://arxiv.org/abs/2406.01116 。访问日期均为 2026-09-10。

## 12.7 归档和接手入口

正式结果位于 outputs/icme/supplementary/。归档包为 tmp/fedpasa_final_20260910.tar.gz，传输清单为 tmp/server_final_manifest_20260910.json，旧文件备份位于 tmp/before_server_final_20260910/。以上路径相对本项目。

9 月 10 日计划形成后，新增实验已在独立目录执行完成并归档，见第 13 节。原归档仍可用 scripts/audit_final_fedmpt.py 复核；文档验收不能替代实验验收。

# 13 服务器补实验全量执行记录

本节更新第 12 节的待补计划，覆盖 2026 年 9 月 10 日至 13 日执行的全部新增实验、加速验证与故障恢复。原 S01–S09 的 9/9 完成状态保持有效；新增队列是另一批实验，不能与原清单混为一个完成率。第 12 节中的“待补”“未启动”等文字保留为 9 月 10 日计划快照，最新执行状态以本节为准。

最终验收：6 组长训练均完成固定 200 轮，末组 NUS-WIDE seed27 于北京时间 2026-09-13 02:31:33 完成。系统汇总曾因 CUDA 不可用退出；08:56 在 GPU 恢复后仅补跑汇总，23 个队列步骤全部 COMPLETE。六组最终预测复算 mAP 差值均为 0；1200 个完整轮次和 12000 个客户端轮次记录齐全，最终参数、预测与记录损失均有限。验收没有重跑图像推理。

## 13.1 实验清单与验收范围

| 编号 | 执行内容 | 数量 | 最后核验 |
| --- | --- | --- | --- |
| N01 | 四数据集配对 image bootstrap | 4 组 | 全部 COMPLETE |
| N02 | COCO 和 NUS 原始图像特征提取 | 2 组 | 全部 COMPLETE |
| N03 | 两数据集分布式校准 split7/17/27 | 6 组 | 全部 COMPLETE |
| N04 | 两数据集提示基线 seed7/17/27 | 6 组 | 全部 COMPLETE |
| N05 | 协议测试 源码核对 两个训练短测 汇总 | 5 步 | 全部 COMPLETE |
| N06 | 固定文本缓存数值一致性与吞吐对照 | 4 对照 | 全部逐位一致 |

N01–N04 共 18 个实验执行单元，加上 N05 共 23 个队列步骤。N06 是独立加速验证，不占这 23 步。步骤完成只表示脚本成功退出；校准误差是否可接受、强基线的适配差异和最终重复统计仍须按研究口径验收。原 9 项补充实验及早期机制、敏感性实验已列于前文，不重复覆盖其原始数值。

## 13.2 六组 FedMVP 多标签适配基线

| 数据集 | 种子 | 轮数 | 完成时间 北京时间 |
| --- | --- | --- | --- |
| COCO | 7 | 200/200 | 09-10 18:54:04 |
| NUS-WIDE | 7 | 200/200 | 09-11 08:45:44 |
| COCO | 17 | 200/200 | 09-11 16:19:51 |
| NUS-WIDE | 17 | 200/200 | 09-12 05:47:17 |
| COCO | 27 | 200/200 | 09-12 13:17:43 |
| NUS-WIDE | 27 | 200/200 | 09-13 02:31:33 |

| 数据集 | seed7 | seed17 | seed27 | 均值 ± 样本标准差 |
| --- | --- | --- | --- | --- |
| COCO | 71.962000 | 71.540546 | 71.478538 | 71.660361 ± 0.263060 |
| NUS | 49.758893 | 50.291595 | 50.708947 | 50.253145 ± 0.476193 |

每组均用保存预测复算，指标差值为 0；COCO 预测形状 40137×80，NUS 为 60260×81。标准差使用 ddof=1，反映固定 partition7 下训练随机性的差异。原 FedMPT 修补版的 69.806112/35.823527 是不同实现，继续单列，不能覆盖或混合求均值。

方法名称固定为“FedMPT 发布仓库中的 FedMVP 多标签适配版”。锁定上游提交 9bb5d9f688ff37b030c76b4f5c3b168fbc08999f。model/fedmvp.py、convclip/fdmvpmodel.py、utilss/asl.py 关键源码与发布版一致；config/utils.py 的 Windows DataLoader 兼容改动不影响本次 Linux 执行。原 ICCV FedMVP 使用单标签任务与 LLM 属性，本次多标签 patch pooling 和 ASL 适配不能称为原算法严格复现。

公平协议：相同图像 ID、标签顺序、ViT-B/16 CLIP 权重、224 resize 加中心裁剪；10 客户端 partition seed7 固定。训练 seed 为 7/17/27，各固定 200 轮、每轮全客户端参与、每客户端全量数据 1 epoch、batch32。SGD lr0.001、momentum0.9、weight decay1e-5、200 轮余弦调度；每客户端重置优化器。损失为 0.02 × [ASL(gamma_neg=2,gamma_pos=1) + KG 余弦一致性]。BF16 autocast、FP32 参数、FP64 加权 FedAvg；按图像数聚合。

仅固定第 200 轮在测试集评估，不用测试 mAP 选模型或延长训练。每完成一个客户端原子保存断点，随机种子按训练种子、轮次和客户端确定；恢复时只重放尚未保存的客户端。两数据集短测均完成 4 批 128 张图像，梯度有限且非零，9 组可训练参数张量更新。

## 13.3 四数据集配对统计

每个数据集 1000 次共同图像重采样；Full、标准化 Ridge 与全局融合使用同一图像抽样权重，且均采用相同三划分聚合口径。AP 实现合并同分阈值，并通过单位权重结果与 sklearn 的 1e-12 容差检查。无正样本类别按既定规则为 0。该区间描述给定已训练模型的测试样本不确定性，不是三个独立完整 Full 训练的稳定性，也不是置换检验 p 值。

| 数据集 | Full mAP | Ridge mAP | 全局融合 mAP |
| --- | --- | --- | --- |
| COCO | 80.528002 | 79.829841 | 80.277318 |
| NUS-WIDE | 60.670673 | 60.193007 | 60.325784 |
| VOC07 | 94.220487 | 94.032605 | 94.314767 |
| VOC12 | 93.982353 | 93.684665 | 93.980679 |

| 数据集 | Full 减去 | 差值 mAP 点 | 配对 95% 区间 |
| --- | --- | --- | --- |
| COCO | Ridge | +0.698161 | [0.583293, 0.809784] |
| COCO | 全局融合 | +0.250685 | [0.191336, 0.298265] |
| NUS | Ridge | +0.477666 | [0.293028, 0.663977] |
| NUS | 全局融合 | +0.344889 | [0.195819, 0.498125] |
| VOC07 | Ridge | +0.187882 | [0.028349, 0.334181] |
| VOC07 | 全局融合 | -0.094280 | [-0.158840, -0.038968] |
| VOC12 | Ridge | +0.297688 | [0.185310, 0.414052] |
| VOC12 | 全局融合 | +0.001675 | [-0.044579, 0.052095] |

四数据集的 Full−Ridge 区间均高于 0。COCO/NUS 的 Full−全局融合区间也高于 0；VOC07 区间低于 0，说明此口径下 Full 较差；VOC12 区间跨 0，不能主张相对全局融合有明确提升。区间为未做多重比较校正的配对百分位区间；bootstrap 非正比例不是置换检验 p 值。

## 13.4 分布式校准与通信核验

COCO/NUS 各运行训练侧 split7/17/27，固定 10 个逻辑客户端 partition7。客户端通过显式数值序列化 Wire 传输 Ridge 充分统计、重拟合统计、QA 四阶段统计、z-score 的 count/sum/sum-square，以及所有候选验证 AP 直方图。六组均完成且均值/方差在 1e-10 容差内；六组 exact_selection_equivalence 均为 false。执行完成并没有证明精确选参等价。

| 数据集 种子 | 变体 mAP | 条件参考 mAP | 权重不同类数 | 最大 AP 误差 |
| --- | --- | --- | --- | --- |
| COCO 7 | 80.576012 | 80.575543 | 8 | 1.125497 |
| COCO 17 | 80.542286 | 80.546308 | 10 | 0.781187 |
| COCO 27 | 80.416361 | 80.411075 | 11 | 0.232216 |
| NUS 7 | 60.518013 | 60.508302 | 30 | 3.056763 |
| NUS 17 | 60.646306 | 60.633267 | 22 | 4.166667 |
| NUS 27 | 60.489504 | 60.489220 | 23 | 2.708333 |

表中 AP 误差是全部融合候选与类别上的最大验证 AP 误差，单位为百分点，不是最终平均 mAP 的误差。COCO split17 的 Ridge alpha 在直方图与精确参考中分别为 0.0003 和 0.0001，其余五组 alpha 一致。各组测试分数最大绝对差依次为 COCO 1.220001、2.033229、1.313112，NUS 1.641702、5.588432、3.860802（种子顺序 7/17/27）。即使平均 mAP 接近，也不能称预测或选择精确等价。

固定 8192 bins，分数单调映射为 0.5 + atan(score)/pi；全局 AP 为合并直方图后的近似值，不能使用本地 AP 平均代替。此实现应称“直方图校准变体”，不能自动认定为原集中式 Full 的精确等价替换。记录各候选与精确集中式 AP 的误差、所选参数一致性、最终分数差、校准矩误差；均值/方差容差为 1e-10。中央融合参考以该轮所选 Ridge 候选为条件，验收时需连同候选选择差异解释。

每组记录 580 条消息、40 个上行阶段，包含所有候选与重拟合。COCO 每划分上行约 3.258 GB、下行 43.788 MB；NUS 上行约 3.296 GB、下行 44.361 MB（十进制单位）。直方图通信开销不能从全流程声明中省略。仅计数值应用负载，不含封装、加密与网络头。该实验是单服务器上的客户端消息仿真，未测物理网络 RTT，未实现密码学安全聚合或正式隐私保证。直方图和统计量本身仍可能暴露信息。QA 合成消息短测为 80 条消息，仅证明序列化链路工作，不能代替真实实验结果。

## 13.5 特征提取 系统计时与重复

同一 RTX 5090 完成 COCO/NUS 冻结 CLIP CLS 与 patch 特征提取，输出使用 memmap；缓存 manifest 保留图像和标签对应关系，提取路径修复旧 pkg_resources.packaging 导入兼容问题，不修改特征计算。冷启动指无预计算特征，未强制清空操作系统页缓存。

| 阶段 | COCO 秒 | NUS 秒 |
| --- | --- | --- |
| 特征提取队列墙钟 | 467.052794 | 213.961800 |
| 校准 split7 队列墙钟 | 82.358019 | 189.370640 |
| 校准 split17 队列墙钟 | 47.836193 | 102.915344 |
| 校准 split27 队列墙钟 | 47.694876 | 102.698743 |
| 优化后稳定训练每轮 | 约 133 | 约 238 |

以上队列墙钟包含启动等开销，只用于执行交接。正式系统表应使用 extraction/result.json 和 calibration/result.json 的阶段字段，分开报告无特征缓存总时间与有缓存拟合、校准、选择、推理时间、峰值显存及双向数值负载。校准运行还包括独立中央参考审计，属于保守审计成本，不能称生产延迟。

| 数据集 | 特征提取 秒 | 有缓存 秒 | 无特征缓存总秒 | 上行 下行 GB |
| --- | --- | --- | --- | --- |
| COCO | 466.997 | 173.419 | 640.416 | 9.774485 / 0.131364 |
| NUS | 213.862 | 390.597 | 604.459 | 9.888283 / 0.133084 |

该表为三个划分合计，包含全部候选、重拟合、推理和中央审计计算。校准阶段峰值显存为 COCO 166.518 MiB、NUS 166.619 MiB，不包含特征提取的峰值，不能标为端到端峰值。Full 的三个训练侧划分是聚合器组成部分，不能冒充三个独立完整 Full 重复。系统汇总已于 9 月 13 日 08:56:04 完成，missing 为空；旧 FAILED 日志保留为故障证据。

| 数据集 种子 | 训练累计 秒 | 最终评估 秒 | 单向负载 GB |
| --- | --- | --- | --- |
| COCO 7 | 29995.073 | 524.948 | 9.083392 |
| COCO 17 | 26625.001 | 582.523 | 9.083392 |
| COCO 27 | 26567.631 | 412.100 | 9.083392 |
| NUS 7 | 49239.527 | 621.088 | 9.083392 |
| NUS 17 | 47591.178 | 743.493 | 9.083392 |
| NUS 27 | 47536.764 | 48.894 | 9.083392 |

提示基线每次训练上行和下行各 9.083392 GB；表中只列一个方向，双向为两倍。训练累计为客户端计时之和，评估墙钟受图像读取与 GPU 故障影响，NUS seed27 的缓存评估明显更短；不能从这些不同缓存状态的总时间直接宣称单一公平加速倍数。

## 13.6 加速对照与故障恢复

运行时优化 immutable_text_cache_v1 缓存类别分词、冻结 token embedding 和固定 zero-shot 文本编码。学习中的提示、两次图像编码、损失、FedAvg、batch32 和所有 200 轮预算保持不变；缓存不进入 state_dict，不增加通信参数。辅助代码 SHA256 为 d852c027022142bddc24dedecba3dcb6dd5d783b1c1908093b90e83d0082f44dc。

COCO 和 NUS 各做 original→cached、cached→original 两个顺序，共 4 个配对对照；每配置 103 步。前 3 步 logits/loss/梯度、103 步后提示参数、最终 CUDA RNG 均逐位一致，最大绝对差 0。稳定批耗时由约 61ms 降至约 51ms，保守配对速度比 COCO 1.2067、NUS 1.2085。整轮 103 实测 133.009 秒，相对此前 159.72 秒减少 16.72%。

初次 NUS 冷文件读取使原 benchmark.json 汇总出现 3.84 倍，不用于加速结论；解释以 timing_review.json 为准。COCO seed7 第 102 轮混合新旧实现并包含暂停，不计入速度比较。seed7 的累计训练时间包含优化切换，COCO seed17 测试还遇到 GPU 故障，不能把受影响墙钟当作稳定硬件基准；正式比较需单列这些异常。

9 月 10 日 15:07 加速切换保留 101 个完整轮和第 102 轮已完成的 9 个客户端。部署脚本曾因 tmux 前缀匹配提前退出，已改为精确匹配并恢复，无已保存进度丢失。9 月 11 日 16:18 GPU 查询报 NVML Unknown Error，PyTorch 无可用 CUDA；第三组训练已保存 200 轮，测试随后于 16:19:51 完成。后续 3 组及汇总在故障期间启动失败，队列退出。16:21 GPU 实际计算测试通过后，备份 queue 并重建 tmux；NUS seed17 从未完成任务继续，日志批次推进、GPU 利用率 94%。

## 13.7 产物路径与最终接手步骤

服务器独立工作目录为 /root/autodl-tmp/FedPASA/icme2027_workspace，新产物均在其中 outputs/icme2027/。服务器 RTX 5090 显存约 32GB，主机内存 16GB；“32GB”不是主机内存。原 outputs/icme/supplementary/ 归档保留。本轮已取回 1328 个文件，包含全部结果 JSON、消息记录、日志、六组最终预测与断点、加速证据；逐文件 SHA256 一致。约 63GB 特征缓存和约 6.4GB 校准中间数组保留服务器，未纳入本次轻量归档。

| 相对 outputs/icme2027 的位置 | 内容 |
| --- | --- |
| queue.json 和 logs/ | 23 步状态与原始日志 |
| paired/<dataset>/ | 共同抽样区间与来源哈希 |
| extraction/<dataset>/ | 提取性能及缓存核对 |
| calibration/<dataset>/seed<seed>_bins8192/ | 候选误差 校准选择 双向负载 |
| baselines/fedmvp_release_mlc/<dataset>/seed<seed>/ | result.json 最终预测 checkpoint 与 200 个轮次记录 |
| optimization/ | 数值对照 加速审查 启用版本和切换备份 |
| systems_and_repeats.json | 统一成本与三种子统计 |

本地归档包为 tmp/icme2027_final_20260913.tar.gz，解包结果位于 outputs/icme2027/；清单为 final_archive_manifest_20260913.json，训练与预测验收为 final_audit_20260913.json。六组均已通过 200 轮断点、完整轮次和客户端记录、有限值及保存预测复算检查。三种子统计已由最终结果重新生成。六组校准的消息数、双向负载和误差值来自原始 result/messages 文件；不将执行 COMPLETE 改写为精确等价 PASS。

当前无需再启动训练。若未来核验发现遗漏且 GPU 可用，先确认没有已有队列，再在服务器工作目录执行 tmux new-session -d -s icme2027 "bash scripts/start_icme2027_server.sh"。入口持有文件锁并跳过 COMPLETE 步骤；不得删除失败日志或覆盖完成结果。SSH 断开和本机关机不影响 tmux，服务器重启会终止进程。最终完成以后 tmux 正常退出，不能据此推断任务失败。

投稿写作应明确直方图校准已改变选参，采用近似变体表述，或把原 Full 集中式校准写入设定；若坚持精确全流程联邦等价主张，还需另实现并验证精确选择协议。FedMVP 适配版、FedMPT 修补版和 published external 应分列；Full 单划分和三划分聚合的计算预算应区分；保留 VOC07 反例与 VOC12 区间跨 0 的结果。作者等价强基线、完整 Full 外层独立重复与正式隐私保证不能由本轮执行完成自动推出。无需为当前任务扩大 K、temperature 或新数据集网格。

