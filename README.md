# FedPASA 服务器备份

备份日期：2026-10-31（北京时间）。

**状态：全量备份、逐文件校验及公共下载验证均通过。服务器上的实验目录、三个历史归档和本次备份暂存已删除。**

完整备份：[server-backup-20261031 Release](https://github.com/A1234-bot/FedPASA/releases/tag/server-backup-20261031)，共 241 个分卷，压缩体积 258,702,519,849 字节。每个归档文件均核对 SHA-256，每个云端附件均匹配 GitHub 返回的 SHA-256。

- `server/FedPASA/`：可浏览的代码、配置、JSON/CSV 结果及日志文本子集，保留三个工作区的相对位置。
- `latest-local-handoff/`：包含 2026-10-29 补实验的本地最新交接文档及图片。
- `backup-audit/`：完整文件清单、读取审计、恢复证据、云端校验、公共下载验证与服务器清理记录。

## 全量范围

`/root/autodl-tmp/FedPASA` 整个目录，以及同级 `fedpasa_final_20260910.tar.gz`、`fedpasa_resume_20260907.tar.gz`、`icme2027_final_20260913.tar.gz`。

范围内共 347,718 个普通文件、207 个符号链接，普通文件逻辑体积 285,906,975,894 字节。独立实验目录、共享 Conda 环境及系统文件保留。

## 图片恢复记录

服务器有 3 张 COCO 图片存在磁盘 I/O 错误。备份采用本地健康副本，其 SHA-256 全部匹配 2026-09-22 历史迁移台账；未覆盖原服务器路径。其余文件直接读取自服务器，未重新训练或改写实验结果。证据见 `backup-audit/recovered-files.json`。

## 恢复

先下载全部 Release 分卷，再依照 [RESTORE.md](RESTORE.md) 操作。只克隆 Git 仓库不能恢复完整数据。运行 `python verify_backup.py /path/to/downloaded-assets` 可只读核对全部分卷和归档内容。
