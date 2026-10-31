# 服务器 FedMVP 加速记录

2026-09-10，按用户“优化代码，跑快点”的要求实施。所有训练及对照均在同一RTX 5090服务器进行。

## 修改

缓存不随训练改变的类别分词、冻结token embedding，以及固定“a photo of a CLASS.”文本分支的CLIP输出。学习中的文本提示、两次图像编码、loss和FedAvg计算保留。上游模型文件不修改，运行时仅包装固定文本的重复计算。

仍为batch32、每数据集/种子200轮、10客户端全参与、原样本顺序、SGD参数和dropout种子。缓存不进入state_dict，不增加模型参数或通信量。

## 对照证据

COCO和NUS分别用真实客户端图像，测试original→cached和cached→original两个顺序。每种配置训练103步；比较前3步的logits/loss/梯度、103步后全部提示参数及CUDA RNG。全部逐位一致，最大差值0。

稳定计时：原版约61毫秒/批，缓存版约51毫秒/批。取两个顺序中较小的配对速度比，COCO为1.2067倍，NUS为1.2085倍，相当于耗时减少约17%。初次NUS读取包含冷文件缓存，原汇总3.84倍受此影响，不作为计算加速结论；原始记录保留，修正解释在timing_review.json。

## 应用与续跑

新入口于15:07在tmux会话`icme2027`恢复。保留101个完整轮和第102轮已完成的9个客户端；只重放切换时尚未完成的第10客户端。第103轮开始全部使用优化。切换前checkpoint、progress、queue及脚本备份位于`outputs/icme2027/optimization/before_deployment/`。

切换脚本曾因tmux会话名前缀匹配而提前退出，已修复为精确匹配，并立即重新启动队列。已完成客户端的原子断点未丢失。

服务器证据目录（相对独立工作目录`/root/autodl-tmp/FedPASA/icme2027_workspace`）：

- `outputs/icme2027/optimization/benchmark.json`：原始逐配置结果。
- `outputs/icme2027/optimization/timing_review.json`：排除冷读夸大后的速度结论。
- `outputs/icme2027/optimization/enabled.json`：启用版本和代码哈希。
- `outputs/icme2027/optimization/deployment.json`：切换与恢复记录。
- 每个baseline的`runtime_optimization.json`及每轮client_records：开始使用优化的位置、版本与实测计时。

整体完成时间应继续按正式轮次计时更新；微基准速度不能替代全部数据的端到端耗时。

15:10正式整轮验证：第103轮全部客户端采用优化，训练耗时133.009秒；此前稳定轮次约159.72秒，耗时减少16.72%。已进入第104轮，GPU利用率约92%。第102轮混合新旧实现并包含基准测试暂停，保留原始计时但排除出速度对比。验证记录为`optimization/full_round_verification.json`。

按已完成103轮、COCO整轮实测和NUS批次数外推，纯剩余训练约58.8小时；计入评估、IO与速度波动，暂估60–72小时。NUS尚未取得正式整轮的新计时，这一预计仍需随后校正。
