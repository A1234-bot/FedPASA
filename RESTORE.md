# 恢复全量服务器备份

当前上传进行中；须待首页和完整性记录标记全量备份通过后再按此说明恢复。

1. 从 `server-backup-20261031` Release 下载所有 `fedpasa-server-20261031.tar.gz.part-*`、`snapshot.json` 和 `files.jsonl.gz`，放在同一目录。Git 克隆仅包含文本子集，不能代替下载分卷。
2. 使用 Python 3 运行只读校验：`python verify_backup.py /path/to/downloaded-assets`。工具核对每卷 SHA-256、清单 SHA-256、所有普通文件内容以及目录和链接记录。必须得到 `PASS`。
3. 在 Linux 中把分卷按名称顺序连接后解包。归档中含 `FedPASA/` 目录和三个同级历史归档。原服务器符号链接使用 `/root/autodl-tmp/FedPASA` 绝对路径；恢复到相同路径可保持链接有效。

```bash
# 在准备恢复的机器上执行。请选择空的恢复目录，避免覆盖已有数据。
mkdir -p /path/to/empty-restore
cat /path/to/downloaded-assets/fedpasa-server-20261031.tar.gz.part-* | tar -xz -C /path/to/empty-restore
```

若恢复到其他路径，解包后须把指向原 FedPASA 根目录的绝对符号链接映射到新根目录；不要在链接未修正时运行实验。归档保留原路径、文件权限和修改时间。Windows 下建议使用 WSL 或 Linux 恢复，以保留符号链接与权限。

备份记录会披露三张原服务器 I/O 错误图片及其按历史 SHA-256 核验的健康副本。其余文件直接读取自服务器，未重新训练或改写实验结果。
