# 2026-09-19 Full 协议补实验

最新用户指令：统一协议下经过调参的强基线与以下三项全部补充，服务器运行，完成写入交接。之前10:15按用户指示停止了原队列，保留checkpoint；最新指令重新授权接续强基线。执行次序为新三项数值任务完成后自动接回原强基线队列，不能并发重复启动。原强基线COCO已完成，NUS从停止断点恢复。

## 顺序与验收

1. 四数据集 calibration：10个客户端进程经本机TCP发送各自验证图像的 QA/Ridge FP32 scores、原始标签和行序号；服务器恢复原验证顺序，按原网格求 z-score 和 classwise fusion，并向客户端下发校准参数。比较同一历史三split Full中央参考与分布式协议的参数、逐split预测、最终ensemble预测和mAP，差异如实报告。缓存预测重放仅验证calibration，不宣称重新训练。
2. VOC07/VOC12完整outer：partition seeds107/207/307；inner seeds1007/1017/1027、2007/2017/2027、3007/3017/3027；client order seeds1107/1207/1307。直接复用旧COCO/NUS `run_priority_20260913.outer` 和固定七点Ridge网格，不混用9月18日扩展正则；KMeans n_init10，从头构造客户端bank、QA、Ridge、训练侧校准与三split融合。均值及样本标准差ddof1；外层预测不再集成。
3. COCO/NUS的10客户端+1服务器独立进程实际TCP fitting/selection/calibration。统计真实序列化payload及应用帧头上下行、消息阶段、拟合墙钟、序列化反序列化、server solve；明确loopback非物理多机，预训练图像特征缓存前置。客户端持有本地特征和标签，服务器只能读text/protocol及最终评价缓存；完整QA有entropy排序上传，不能把所有QA fitting消息误称纯可加moments。

## 运行位置

本地 E:/FedPASA；服务器 root@10.69.37.195:10117；凭据仅查用户原消息。服务器根目录 `/root/autodl-tmp/FedPASA/reviewer_20260913`，Python `/root/autodl-tmp/miniconda3/envs/fbu/bin/python`。新输出 `outputs/full_protocol_20260919`，新队列使用独立tmux名字。所有实验在服务器运行。

## 设备情况

2026-09-19 10:15新进程 `torch.cuda.init()` 报No CUDA GPUs are available，nvidia-smi报NVML Unknown Error；没有修改驱动或重启宿主。先用服务器CPU完成缓存calibration；outer和多进程拟合允许CPU执行并记录设备，禁止把CPU时间与旧GPU时间当公平加速比。原core允许device=cpu，但必须记录可能的数值差异。

## 交付

同步数值证据、代码/输入SHA、预测文件并独立复核；保留历史记录，更新正式handoff Markdown与Word，Word使用documents技能及渲染验证。原强基线状态应记用户停止、未完成，不等于失败也不等于通过作者等价。用户给出的94.22等数字是示例，输出用实测；历史测试已暴露，不声称未见测试预注册。
