# 第二强基线配置附录

本附录在第二方法训练前固定。只有源码和训练侧数值核验通过，Fed-RAM才进入正式队列；没有根据新测试结果选择方法。

候选Fed-RAM采用RAM作者公开COCO配置的ViT-B/16、Adapter、视觉/文本层[9,10,11]、全部12层SAA、vision_adapt8、text_ctx4、kernel3、temperature0.002、OT reg0.1与teacher reg0.05、MMC损失。benchmark骨干与作者骨干只有builder重命名及无效调试导入差异；LLA相同；Sinkhorn多一个不使用的reg默认值；运行审计仍须成立。

训练前使用同一模型状态和同一训练图像，比较作者原forward与benchmark forward的训练分数、损失、梯度和评估分数；仅替换返回包装与train/req接口，不修改数学计算。短测不等于完整训练复现。

身份固定为“Fed-RAM with verified RAM core; full-label federated adaptation”。作者原始开放词汇划分改为本项目全部目标类别监督，加入10客户端按样本数FedAvg、统一中心裁剪、不使用EMA。因此不宣称完整作者实验等价。

学习率候选{3e-5,5e-5,1e-4}；保留作者双优化器结构：非embedding参数AdamW，CLIP分支lr为base的0.1，其他参数base；embedding参数SGD momentum0.9，lr=0.001*(base/5e-5)；weight_decay均1e-4。所有组余弦调度按200轮定义。各客户端每轮重置优化器，本地1epoch，batch32。BF16前向、FP32主参数、FP64服务器加权和。

所有可训练参数以及Adapter BN浮点buffer参与加权聚合，整型num_batches_tracked取max并保持整数dtype；冻结参数不上传。参数温度和embedding均纳入更新及通信，禁止只聚合名为prompt_learner的子模块而漏掉Adapter。

按主协议20/60/200轮逐级筛选，额外在2/4/6/10轮验证，以覆盖原作者较短训练配置。只在训练验证集选择，选中后全量训练3个种子，若选在预算边界则明确有限预算限制。

RAM作者源码下载归档SHA256：49bee71f0b23fd4819396afb3f925ef104fba5d48effb5c8133e6a9b613cbfd3。参考：https://github.com/EricTan7/RAM 。逐文件SHA和数值检查由服务器审计产出。
