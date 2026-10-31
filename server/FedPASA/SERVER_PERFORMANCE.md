# FedMPT 服务器性能研究与部署记录

时间：2026-09-07。设备：RTX 5090 32 GB；10 核 CPU 配额，无 CPU 限流记录；PyTorch 2.11.0+cu128。

## 已部署配置

关闭图像和文本 Transformer 的激活重计算，缓存固定分词和冻结词嵌入，合并有限梯度检查的设备同步，计算线程设为 1，COCO/NUS 数据读取进程分别设为 4/8。可训练文本提示、文本编码和视觉提示仍每批更新；没有缓存会随参数变化的最终特征。

保留原 200 轮、batch size 32、10 客户端、seed=7、图像顺序/标签/分区、SGD 学习率日程及 BF16。性能设置写入每数据集 protocol.json 的 runtime 字段。

## 实测

| 测量 | COCO | NUS |
|---|---:|---:|
| 原版纯计算，秒/批 | 0.1382 | 0.1395 |
| 优化版纯计算，秒/批 | 0.0998 | 0.1005 |
| 反向执行顺序、热缓存原版完整流程，秒/批 | 0.1413 | 0.1415 |
| 反向执行顺序、热缓存优化版完整流程，秒/批 | 0.0987 | 0.0990 |
| 热缓存速度倍率 | 1.43× | 1.43× |
| 优化版张量显存峰值，MiB | 7524 | 7610 |

第一次完整流程对照中，原版为 0.511/0.447 秒每批，优化版约 0.10 秒每批；但反向对照确认原版也会从缓存预热受益，因此不能把首次 4～5 倍差异全部归因于代码优化。额外合并同步和减少计算线程的独立收益很小；主要可确认的计算收益来自关闭重计算和静态文本缓存。读取进程数的真实收益需要结合缓存状态评估。

正式续跑观察：NUS 第 10 轮第 4 客户端的 17572 张图像、550 批训练耗时 **130.04 秒**；上一轮同客户端为 **303.68 秒**，观察速度约 **2.34×**。该客户端已保存断点，所有更新有限且断点张量有限。它是不同轮次下的运行观察，不是严格控制缓存状态的因果对照。

新配置完整 COCO/NUS 轮次尚未测完，剩余总时间需要据此重新估算；不要继续无条件沿用原来的 9～10 天，也不要直接按短测 4～5 倍推算。

## 正确性和实验隔离

- 6 种纯计算配置 × 2 数据集：同初始状态和随机种子，连续 3 步的输出、损失、梯度、更新后参数逐位一致。
- 正向及反向完整流程对照：每数据集各 102 次优化器更新，最终参数逐位一致；训练数据参与测速，没有使用测试 mAP 决定性能配置。
- 正式断点在全部对照期间 SHA256 不变；训练前断点副本保留于服务器 `tmp/server_performance/checkpoints_before/`。
- 修改的 Python 文件编译检查通过；本机 `tests/test_supplementary.py` 的 3 项测试通过。
- 一致性检查覆盖这些样本和更新步数，不等于对整个 200 轮轨迹逐步验证。

## 文件与运行入口

服务器目录 `/root/autodl-tmp/FedPASA`，tmux 会话 `fedpasa_fedmpt`。2026-09-07 19:55 恢复，启动时主进程 PID 5704；PID 仅供定位，持续存活以实际进程和日志更新判断。

```bash
cd /root/autodl-tmp/FedPASA
tail -n 20 outputs/server_logs/fedmpt_optimized_20260907.log
cat outputs/icme/supplementary/nus/fedmpt_repaired/progress.json
cat outputs/icme/supplementary/coco/fedmpt_repaired/progress.json
tmux ls
```

启动脚本：`scripts/start_server_fedmpt.sh`。当前已经运行，不要重复启动。

原始证据：`tmp/server_performance/{coco,nus}_benchmark.json`、`loader_benchmark.json`、`end_to_end_control.json`、`end_to_end_control_reverse.json`、`production_client_validation.json`；汇总为 `outputs/icme/supplementary/server_performance.json`。
