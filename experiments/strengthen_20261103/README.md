# 2026 年 11 月补充实验结果备份

本次备份包含 2026 年 11 月 3 日启动、11 月 5 日北京时间 20:17:15 完成的 **160 项实验**。队列及各项原始 `result.json` 均为 `COMPLETE`。

## 查看结果

- [精度汇总：均值、标准差与 95% 置信区间](snapshot/outputs/strengthen_20261103/report/accuracy.csv)
- [精度、通信量与运行时间的操作点](snapshot/outputs/strengthen_20261103/report/operating_points.csv)
- [相同预算及目标精度的成本比较](snapshot/outputs/strengthen_20261103/report/matched_budgets_targets.csv)
- [12 张结果图 PDF](snapshot/outputs/strengthen_20261103/report)
- [逐项原始结果、选参记录及日志](snapshot/outputs/strengthen_20261103)
- [最终队列](snapshot/queue.json)、[实验计划](snapshot/plan.json)、[运行代码](snapshot/scripts)、[环境版本](environment.json)

| 实验类别 | 数量 |
| --- | ---: |
| 强 FedAvg 对照 | 18 |
| 随机特征对照 | 6 |
| Patch 融合方式 | 10 |
| 部分客户端参与 | 30 |
| TCP 通信及资源测量 | 96 |
| 合计 | 160 |

## 下载和校验

[下载压缩备份及 SHA-256 信息](https://github.com/A1234-bot/FedPASA/releases/tag/experiment-results-20261105)。压缩包大小为 18,790,256 字节，包含 1,946 个原始文件及清单、环境版本。原始文件合计 187,717,747 字节；源文件内容保持不变。

压缩包 SHA-256：

```text
a1638ddfbdf3ee1d2265f062297d8d9b9d1a72a5900f9f5fa5849486a01e1308
```

仓库克隆完成后，执行以下只读校验：

```bash
python experiments/strengthen_20261103/verify_backup.py
```

预期输出 `PASS`、`files: 1946`、`complete_jobs: 160`。清单见 [manifest.json](manifest.json)，压缩包校验信息见 [archive.json](archive.json)。压缩包解压后包含同样的 `snapshot/` 和 `manifest.json`，也可将此校验脚本放到解压目录执行。

## 备份范围

包含最终结果、各类别指标、选参记录、逐轮信息、通信记录、日志、CSV/PDF 汇总、服务器上的脚本和源码、输入文件哈希及运行配置。另包含报告引用的 6 份既有 IID 基线结果。

这是实验效果与运行证据备份。约 15.39 GB 的中间数组、预测或统计缓存、模型权重及检查点（`.npy`、`.npz`、`.pt`）未包含在本次备份中；原始数据集和完整运行环境也不在此快照内。排除数量和字节数见清单。本次未删除服务器上的任何实验文件。

2026 年 10 月 31 日的历史全量备份仍见 [原有 Release](https://github.com/A1234-bot/FedPASA/releases/tag/server-backup-20261031)。
