# 公平强基线执行交接

用户已授权本任务按顺序在指定服务器运行全部补实验，并在完成后同步结果、更新正式交接文档。不要因对话切换丢失这项任务，也不要要求重复确认。SSH 凭据见用户原消息，不写入源码、日志或文档。

## 当前部署

- 本地 E:/FedPASA。
- 服务器 SSH root@10.69.37.195，端口10117。
- 服务器工作区 /root/autodl-tmp/FedPASA/reviewer_20260913。
- Python /root/autodl-tmp/miniconda3/envs/fbu/bin/python。
- 本批输出 outputs/fair_baselines_20260918。
- 第一阶段 tmux fair_baselines_20260918：顺序执行 COCO/NUS 线性验证调参，完成标记 linear_tuning_complete.txt。
- 后续 tmux fair_baselines_queue_20260918：等待第一阶段完成，再运行 fair_baselines_queue.py。
- 线程环境 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4。
- 本任务每小时自动续查已建立，automation id 为 fedpasa；完成文档交付后将其暂停。

## 2026-09-18 当前状态

线性调参、Ridge 扩展网格已完成。FedMVP COCO 正在第一阶段，静态缓存与原实现的分数/损失/梯度差异均为0。RAM 数值核验仍待队列执行。尚未产生本批最终测试结论。

中央线性最佳验证 mAP：COCO 81.4693（60轮），NUS 63.8160（40轮）；FedAvg COCO 81.0482（400轮，仍须注明预算边界）。这是验证集诊断，不是测试集结果。两数据集 pooled Ridge 选择3e-5，故 Full 需要重新校准。

恢复修复：run_fair_fedmvp.py 现从已保存 halving 决策继续，防止较后 checkpoint 回跑旧阶段；最终轮 checkpoint 恢复时先载入实际模型再推理。AST 提取的调度逻辑模拟中断恢复测试已通过。当前已运行的 COCO 进程使用启动时旧代码；它正常运行无需中断，未来启动/恢复会采用修复版。源包 stage3 与各 queue job script_sha256 保留版本证据。

## 已固定规则

主协议 docs/fair_baselines_20260918_protocol.md；服务器冻结副本 protocol/frozen_protocol.md，SHA256 0150f124ecdebeaaab1e84208ec2e11f80224c031cb187e5df2822bff857566b。

RAM 附录 docs/fair_baselines_20260918_ram_appendix.md；源码预审通过，不代表训练侧数值核验已经完成。FedMVP 为发布仓库适配版，RAM 为原核心可验证的全标签联邦适配，均不应写成原始作者实验严格复现。

固定 partition7、selection split17，最终种子7/17/27。三种子仅描述固定分区和固定选参下的训练随机性。Full 新单划分与旧三划分分开。所有测试标签在最后保存预测后才由汇总读取；输入审计可读测试标签字节校验而不用于选参。

## 继续操作

1. 查看 queue.json、logs/queue.log、线性阶段日志；只读摘要，不打印数据目录全清单。
2. GPU 任务顺序执行，避免并发计时污染。确认 tmux/进程状态后才恢复，不能启动重复训练。
3. 如失败，保留日志、冻结协议和已成功结果；修复工程错误后续跑。若必须改变算法或预算，先写清基于训练侧证据的附录，不根据测试结果追加预算。
4. 队列已将 Ridge 边界检查、FedMVP 调参、RAM 数值审计及调参、最终三种子、配对报告串联。若 RAM 审计不通过，查看原因，不能静默放过；仅在确有算法一致性问题时按协议审查下一候选 Fed-DualCoOp。
5. FedMVP/RAM checkpoint 可在客户端边界恢复；selection.json 是冻结选择。报告脚本不得用测试数值再挑配置。
6. 数值全部完成后，同步 final、protocol、smoke、source audit、selection、round records、queue/completion 与脚本至本地；全量预测可打包取回进行独立 sklearn mAP 复算。不要将服务器大缓存覆盖本地原缓存。
7. 更新 outputs/icme/handoff/FedPASA_ICME_experiment_handoff_zh.md 和 .docx 第16节，保留原文件备份；新结果必须说明方法身份、优化边界、缓存与系统计时范围、配对区间含义。旧文献大幅增益不升级为受控主张。
8. 使用 documents 技能的 bundled Python/Node；作者操作前按技能标记，Word 修改后 render_docx.py 渲染检查。当地无LibreOffice时按既有Word原生PDF导出fallback排查。最后交付本地正式文档链接。

## 待补验收事项

报告入口会先运行 run_fair_full_cost.py，重新测量 Full 在缓存特征上的完整银行构建、拟合、选择与推理；它用占位数组代替测试标签，不做测试选参，并核对新 seed17 选择/分数。必须区分图像提取成本。验证选择涉及集中读取验证分数与标签，报告补充模拟分布式 pooled AP 的分数/标签负载估计，并明确实际实现为集中验证；不能将估计描述为真实网络测量。图像方法训练耗时未完整覆盖聚合和 checkpoint 开销。不得声称完整端到端网络时延或公平加速倍数。

报告bootstrap固定2000次、seed20260918。代码已用含并列分数的随机小例核对AP实现。区间条件于已训练模型，不是训练重复的显著性推断；各训练种子mAP先算后平均，不平均预测。
