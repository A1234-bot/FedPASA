# 补充实验服务器运行交接

## 2026-09-07 19:55 性能优化已部署

服务器在保存 NUS 第 10 轮第 3 个客户端后暂停，完成独立性能测试，再从第 4 个客户端恢复。当前主进程启动 PID 为 5704，tmux 会话仍为 `fedpasa_fedmpt`。后续请查看新日志 `outputs/server_logs/fedmpt_optimized_20260907.log`。

已关闭激活重计算、缓存冻结的文本分词/词嵌入、合并梯度检查同步，计算线程设为 1，COCO/NUS 读取进程分别为 4/8。200 轮、batch=32、10 客户端、划分和优化器保持原设置。两套数据在正反执行顺序的 102 步对照中参数逐位一致；正式模型断点在对照期间哈希保持不变。

热缓存对照的可靠计算提速约 1.43 倍。首次对照受到图像缓存预热影响，不能宣称代码本身带来 4～5 倍收益。正式 NUS 第 4 客户端（17572 张，550 批）已经完成并保存：从上一轮同客户端 303.68 秒降至 130.04 秒，观察到约 2.34 倍速度；不同轮次及 I/O 状态会影响该比值，尚不代表整套实验最终加速比。

完整分析见同目录 `SERVER_PERFORMANCE.md`；原始对照 JSON 见本机和服务器的 `tmp/server_performance/`。本机继续保持暂停，最新训练进度以服务器为准。

## 当前状态：服务器训练已启动

2026-09-07 17:37（北京时间）已按用户指示启动，运行目录 `/root/autodl-tmp/FedPASA`，tmux 会话 `fedpasa_fedmpt`。本机继续暂停；后续进度以服务器文件为准。

- GPU：RTX 5090，32607 MiB 显存，驱动 580.159.03，CUDA/BF16 可用。
- 环境：Python 3.10.20、PyTorch 2.11.0+cu128、torchvision 0.26.0+cu128。
- 已上传修补代码、断点、标签、划分与现有补充实验 JSON/CSV。214 个文件的 SHA256 校验通过。
- 已核验 COCO 82081/40137 与 NUS 150000/60260 张训练/测试图像全部存在；抽样 256 张图像与本机 SHA256 一致。
- COCO2017 存储中的图像按数字 ID 建立到 train2014/val2014 的软链接，严格保留原始 COCO2014 图像列表、标签和划分。没有采用 COCO2017 的训练/测试划分。图像字节仅做抽样核验，未宣称全量哈希核验。
- 两个模型断点及未完成客户端聚合均可加载、数值有限。COCO 完成 8 轮；NUS 从第 8 轮第 7 个客户端继续。
- COCO/NUS 分别通过独立的 2 批训练检查，损失及非零梯度有限。检查输出在 `tmp/server_smoke/`，未覆盖正式断点。
- 继续沿用 200 轮、10 客户端、seed=7、batch_size=32、workers=2。RTX 5090 / Linux / 新版 PyTorch 与本机环境不同，数值差异需在实验说明中披露。
- 启动时主进程 PID 为 473，仅供历史定位，不作为持续存活证明。

连接后查看：

```bash
cd /root/autodl-tmp/FedPASA
tmux ls
tail -n 20 outputs/server_logs/fedmpt_resume_20260907.log
cat outputs/icme/supplementary/fedmpt_job.json
cat outputs/icme/supplementary/nus/fedmpt_repaired/progress.json
cat outputs/icme/supplementary/coco/fedmpt_repaired/progress.json
```

启动脚本为 `scripts/start_server_fedmpt.sh`；当前已经在运行，不要重复启动。核验记录位于服务器 `tmp/server_migration/server_validation.json`，训练前原始迁移包保留在 `/root/autodl-tmp/fedpasa_resume_20260907.tar.gz`。阶段性结果仍不能当作最终基线。

## 以下为启动前的暂停与准备记录

更新：2026-09-07。用户要求本机补充实验暂停，改在服务器运行。本机没有存活的实验 Python 进程；任务与两数据集进度文件已标为 PAUSED。不得自动在本机恢复。

## 保留进度

- 已完成 9 项补充实验中的 8 项；仅 S01 的 FedMPT 修补版 COCO/NUS 最终基线待完成。
- COCO：完成 8/200 轮。
- NUS：完成 7/200 轮；第 8 轮前 6 个客户端已保存，续跑从下一个客户端开始。
- 两个 checkpoint.pt、已完成轮次 JSON 和预测结果仍在 outputs/icme/supplementary/{coco,nus}/fedmpt_repaired/。
- 已删除 5 个未完成或失败的控制台日志，保留成功的 BF16 smoke 日志。

## 服务器检查结果

- SSH：`ssh -p 10117 root@10.69.37.195`，已成功登录。密码不写入文档或脚本。
- 实例有 GPU，但当前为无卡开机模式；当前 CUDA 不可见属于该模式的预期状态。
- Ubuntu 22.04.5；无卡模式约 1 核 CPU、4 GB 内存。
- `/root/autodl-tmp` 可用约 856 GB，适合放迁移项目及数据。
- 已有解释器 `/root/autodl-tmp/miniconda3/envs/fbu/bin/python`：Python 3.10.20、PyTorch 2.11.0+cu128。
- 系统 PATH 中没有默认 python；不要直接依赖裸 python 命令。
- 已有 tmux、rsync。
- `/root/BCP/external/FedMPT-main/ViT-B-16.pt` 的 SHA256 为 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`，与本次实验相同。
- 初步发现 `/root/BCP/data/raw/nuswide/kaggle_tc21/NUS-WIDE` 与 coco2017 目录。尚未逐文件核验 NUS 完整性，尚未确认 COCO2014；COCO2017 不能直接代替本次 COCO2014。

## 迁移待办

1. 将项目代码、修补后的 third_party/FedMPT、两数据集断点与已完成证据上传到独立项目目录；本次仅完成连接和环境检查，尚未上传。
2. 携带缓存目录中的 manifest.json、train/test_ids.txt、train/test_targets.npy，以及 `clients10_seed7.npy`。FedMPT 使用原图，无需为其训练复制巨大的冻结 patch 特征缓存。
3. 按相同图像 ID 核验 NUS 和 COCO2014 数据；仅补齐缺失数据，核验模型及分区哈希。
4. 将服务器副本 manifest.json 的 train_root、test_root、weight_path 改为实际 Linux 路径，保留本机原件和图像顺序。核验训练/报告依赖及第三方导入。
5. 正常 GPU 模式开机后检查 GPU 型号、显存、BF16 支持，再做独立 smoke 测试；不能覆盖正式断点。
6. 使用 tmux 等持久会话恢复训练，维持 200 轮、10 客户端、seed=7、batch_size=32 和原有算法设置。更换硬件的数值差异应披露；不要依据测试 mAP 更改预算或挑选模型。

待数据和环境核验完成后的续跑命令（当前未执行）：

```bash
/root/autodl-tmp/miniconda3/envs/fbu/bin/python scripts/run_matched_fedmpt.py --datasets coco nus --rounds 200 --clients 10 --seed 7 --batch-size 32 --workers 2
```

完整的 9 项实验清单、现有证据、修补版差异和最终验收条件见同目录 `FedPASA_ICME_experiment_handoff_zh.docx` 第 11 节及同名 Markdown。本文件补充其后的暂停与服务器安排。
