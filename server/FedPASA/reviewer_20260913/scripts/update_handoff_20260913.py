"""Preserve prior evidence and append all server studies with verified status."""
from pathlib import Path
from datetime import datetime, timezone
import json, shutil, re
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from finalize_handoff_20260910 import replace, runfont

ROOT=Path(__file__).resolve().parents[1]
H=ROOT/'outputs/icme/handoff'; Q=ROOT/'tmp/handoff_update_20260913'
D=H/'FedPASA_ICME_experiment_handoff_zh.docx'; M=D.with_suffix('.md')
Q.mkdir(parents=True,exist_ok=True)
for f in (D,M):
    target=Q/('original_'+f.name)
    if not target.exists():shutil.copy2(f,target)
doc=Document(Q/('original_'+D.name)); md=(Q/('original_'+M.name)).read_text(encoding='utf-8')
E=ROOT/'outputs/icme2027'
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
queue=read(E/'queue.json');audit=read(E/'final_audit_20260913.json');systems=read(E/'systems_and_repeats.json')
assert queue['status']=='COMPLETE' and len(queue['jobs'])==23 and all(j['status']=='COMPLETE' for j in queue['jobs'])
assert audit['status']=='PASS' and len(audit['baselines'])==6 and systems['missing']==[]
status='原定补充实验 9/9 已归档；新增服务器队列 23/23 完成，六组基线 1200/1200 轮通过验收；校准近似边界和全部结果见第 13 节'
for p in doc.paragraphs:
    if p.text.startswith('状态：'):replace(p,p.text,'状态：'+status)
    if p.text.startswith('日期：'):replace(p,p.text,'日期：2026-09-13 服务器全量补实验执行交接更新')
    if p.text.startswith('运行交接：'):replace(p,p.text,'运行交接：原 FedMPT 修补版与新增服务器实验均已完成。末组基线于 2026-09-13 02:31 完成，08:56 补跑系统汇总后队列 23/23 COMPLETE。六组预测复算差值为 0，1328 个取回文件 SHA256 一致；所有结果、校准非等价限制和复现入口见第 13 节。')
md=re.sub(r'^\*\*状态：\*\*.*$', '**状态：**'+status, md, count=1, flags=re.M)
md=re.sub(r'^\*\*日期：\*\*.*$', '**日期：**2026-09-13 服务器全量补实验执行交接更新', md, count=1, flags=re.M)
md=re.sub(r'^运行交接：.*$', '运行交接：新增服务器队列 23/23 COMPLETE；六组预测复算差值为 0，1328 个取回文件 SHA256 一致。全部结果与校准非等价限制见第 13 节。', md, count=1, flags=re.M)
updates={
 '主交付：本 Word 文档及同名 Markdown；新增证据见 supplementary 目录':'主交付：本 Word 及同名 Markdown；新增证据见 outputs/icme2027 目录',
 '第 9-7 节：完整实验登记、复现顺序、文件索引与后续写作动作。':'第 7-13 节：实验登记、复现入口及本轮服务器最终结果。',
 '结论：原定补充实验已全部跑完并在本机归档。投稿前优先补齐 Full 校准协议、可信同协议强基线及配对统计证据；不继续增加大批敏感性设置。下列优先级是结合当前代码和结果提出的研究建议，不是 ICME 官方规定的实验数量。':'本节保留 2026-09-10 的历史计划。该计划已于 9 月 13 日完成服务器执行，最终结果及未满足的精确等价边界见第 13 节。下列优先级属于研究建议，不是 ICME 官方要求的实验数量。',
 '本次未启动新增训练。新实验另建目录并预先登记协议；现有归档可用 scripts/audit_final_fedmpt.py 复核，文档验收不能替代实验验收。':'9 月 10 日计划形成后，新增实验已在独立目录执行完成并归档，见第 13 节。原归档仍可用 scripts/audit_final_fedmpt.py 复核；文档验收不能替代实验验收。',
}
for p in doc.paragraphs:
    for a,b in updates.items():replace(p,a,b)
for a,b in updates.items():md=md.replace(a,b)
for t in doc.tables:
    for row in t.rows:
        if row.cells[0].text=='FedMVP/FedAWA':
            vals=['FedMVP/FedAWA','FedMVP 适配版六组完成','适配差异见第 13 节；FedAWA 未训练','FedMVP 完成']
            for c,v in zip(row.cells,vals):
                p=c.paragraphs[0];replace(p,p.text,v)
md=re.sub(r'\|\s*FedMVP/FedAWA\s*\|[^\n]*','| FedMVP/FedAWA | FedMVP 适配版六组完成 | 适配差异见第 13 节；FedAWA 未训练 | FedMVP 完成 |',md)
parts=[]
def para(s):
    p=doc.add_paragraph();runfont(p.add_run(s),10.5)
    p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.15
    e=OxmlElement('w:snapToGrid');e.set(qn('w:val'),'0');p._p.get_or_add_pPr().append(e)
    parts.extend([s,'']);return p
def heading(s,level=2):
    p=doc.add_heading(s,level);p.paragraph_format.space_before=Pt(10);p.paragraph_format.space_after=Pt(6)
    for r in p.runs:runfont(r,14 if level==1 else 12,True)
    parts.extend(['#'*level+' '+s,'']);return p
def table(headers,rows,widths):
    t=doc.add_table(rows=1,cols=len(headers));t.autofit=False;t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for c,w in zip(t.columns,widths):c.width=Inches(w)
    for i,row in enumerate([headers]+rows):
        cells=t.rows[0].cells if i==0 else t.add_row().cells
        for c,s,w in zip(cells,row,widths):
            c.width=Inches(w);c.text=str(s);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            for p in c.paragraphs:
                p.paragraph_format.space_before=Pt(3);p.paragraph_format.space_after=Pt(3)
                p.paragraph_format.keep_with_next=(i==0)
                for r in p.runs:runfont(r,9.5,i==0)
            if i==0:
                e=OxmlElement('w:shd');e.set(qn('w:fill'),'E7EEF5');c._tc.get_or_add_tcPr().append(e)
        t.rows[i]._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    borders=OxmlElement('w:tblBorders')
    for name in ('top','left','bottom','right','insideH','insideV'):
        e=OxmlElement('w:'+name);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
    t._tbl.tblPr.append(borders)
    parts.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
    parts.extend('| '+' | '.join(map(str,r))+' |' for r in rows);parts.append('')

heading('13 服务器补实验全量执行记录',1).paragraph_format.page_break_before=True
para('本节更新第 12 节的待补计划，覆盖 2026 年 9 月 10 日至 13 日执行的全部新增实验、加速验证与故障恢复。原 S01–S09 的 9/9 完成状态保持有效；新增队列是另一批实验，不能与原清单混为一个完成率。第 12 节中的“待补”“未启动”等文字保留为 9 月 10 日计划快照，最新执行状态以本节为准。')
para('最终验收：6 组长训练均完成固定 200 轮，末组 NUS-WIDE seed27 于北京时间 2026-09-13 02:31:33 完成。系统汇总曾因 CUDA 不可用退出；08:56 在 GPU 恢复后仅补跑汇总，23 个队列步骤全部 COMPLETE。六组最终预测复算 mAP 差值均为 0；1200 个完整轮次和 12000 个客户端轮次记录齐全，最终参数、预测与记录损失均有限。验收没有重跑图像推理。')
heading('13.1 实验清单与验收范围')
table(['编号','执行内容','数量','最后核验'],[
['N01','四数据集配对 image bootstrap','4 组','全部 COMPLETE'],
['N02','COCO 和 NUS 原始图像特征提取','2 组','全部 COMPLETE'],
['N03','两数据集分布式校准 split7/17/27','6 组','全部 COMPLETE'],
['N04','两数据集提示基线 seed7/17/27','6 组','全部 COMPLETE'],
['N05','协议测试 源码核对 两个训练短测 汇总','5 步','全部 COMPLETE'],
['N06','固定文本缓存数值一致性与吞吐对照','4 对照','全部逐位一致'],
], [.55,2.65,.65,1.45])
para('N01–N04 共 18 个实验执行单元，加上 N05 共 23 个队列步骤。N06 是独立加速验证，不占这 23 步。步骤完成只表示脚本成功退出；校准误差是否可接受、强基线的适配差异和最终重复统计仍须按研究口径验收。原 9 项补充实验及早期机制、敏感性实验已列于前文，不重复覆盖其原始数值。')
heading('13.2 六组 FedMVP 多标签适配基线')
table(['数据集','种子','轮数','完成时间 北京时间'],[
['COCO','7','200/200','09-10 18:54:04'],['NUS-WIDE','7','200/200','09-11 08:45:44'],
['COCO','17','200/200','09-11 16:19:51'],['NUS-WIDE','17','200/200','09-12 05:47:17'],
['COCO','27','200/200','09-12 13:17:43'],['NUS-WIDE','27','200/200','09-13 02:31:33'],
], [1.05,.5,.85,2.9])
table(['数据集','seed7','seed17','seed27','均值 ± 样本标准差'],[[r['dataset'].upper(),*[f'{v:.6f}' for v in r['mAP']],f"{r['mean']:.6f} ± {r['sample_std']:.6f}"] for r in systems['baseline_repeats']],[.8,.9,.9,.9,1.8])
para('每组均用保存预测复算，指标差值为 0；COCO 预测形状 40137×80，NUS 为 60260×81。标准差使用 ddof=1，反映固定 partition7 下训练随机性的差异。原 FedMPT 修补版的 69.806112/35.823527 是不同实现，继续单列，不能覆盖或混合求均值。')
para('方法名称固定为“FedMPT 发布仓库中的 FedMVP 多标签适配版”。锁定上游提交 9bb5d9f688ff37b030c76b4f5c3b168fbc08999f。model/fedmvp.py、convclip/fdmvpmodel.py、utilss/asl.py 关键源码与发布版一致；config/utils.py 的 Windows DataLoader 兼容改动不影响本次 Linux 执行。原 ICCV FedMVP 使用单标签任务与 LLM 属性，本次多标签 patch pooling 和 ASL 适配不能称为原算法严格复现。')
para('公平协议：相同图像 ID、标签顺序、ViT-B/16 CLIP 权重、224 resize 加中心裁剪；10 客户端 partition seed7 固定。训练 seed 为 7/17/27，各固定 200 轮、每轮全客户端参与、每客户端全量数据 1 epoch、batch32。SGD lr0.001、momentum0.9、weight decay1e-5、200 轮余弦调度；每客户端重置优化器。损失为 0.02 × [ASL(gamma_neg=2,gamma_pos=1) + KG 余弦一致性]。BF16 autocast、FP32 参数、FP64 加权 FedAvg；按图像数聚合。')
para('仅固定第 200 轮在测试集评估，不用测试 mAP 选模型或延长训练。每完成一个客户端原子保存断点，随机种子按训练种子、轮次和客户端确定；恢复时只重放尚未保存的客户端。两数据集短测均完成 4 批 128 张图像，梯度有限且非零，9 组可训练参数张量更新。')
heading('13.3 四数据集配对统计')
para('每个数据集 1000 次共同图像重采样；Full、标准化 Ridge 与全局融合使用同一图像抽样权重，且均采用相同三划分聚合口径。AP 实现合并同分阈值，并通过单位权重结果与 sklearn 的 1e-12 容差检查。无正样本类别按既定规则为 0。该区间描述给定已训练模型的测试样本不确定性，不是三个独立完整 Full 训练的稳定性，也不是置换检验 p 值。')
table(['数据集','Full mAP','Ridge mAP','全局融合 mAP'],[
['COCO','80.528002','79.829841','80.277318'],['NUS-WIDE','60.670673','60.193007','60.325784'],
['VOC07','94.220487','94.032605','94.314767'],['VOC12','93.982353','93.684665','93.980679'],
],[1.15,1.38,1.38,1.39])
paired_rows=[]
for d in ['coco','nus','voc07','voc12']:
    for c in read(E/f'paired/{d}/result.json')['comparisons']:
        lo,hi=c['percentile_95_CI_mAP'];paired_rows.append([d.upper(),'Ridge' if 'Ridge' in c['comparison'] else '全局融合',f"{c['point_difference_mAP']:+.6f}",f'[{lo:.6f}, {hi:.6f}]'])
table(['数据集','Full 减去','差值 mAP 点','配对 95% 区间'],paired_rows,[.9,1.0,1.15,2.25])
para('四数据集的 Full−Ridge 区间均高于 0。COCO/NUS 的 Full−全局融合区间也高于 0；VOC07 区间低于 0，说明此口径下 Full 较差；VOC12 区间跨 0，不能主张相对全局融合有明确提升。区间为未做多重比较校正的配对百分位区间；bootstrap 非正比例不是置换检验 p 值。')
heading('13.4 分布式校准与通信核验')
para('COCO/NUS 各运行训练侧 split7/17/27，固定 10 个逻辑客户端 partition7。客户端通过显式数值序列化 Wire 传输 Ridge 充分统计、重拟合统计、QA 四阶段统计、z-score 的 count/sum/sum-square，以及所有候选验证 AP 直方图。六组均完成且均值/方差在 1e-10 容差内；六组 exact_selection_equivalence 均为 false。执行完成并没有证明精确选参等价。')
calrows=[]
for d in ['coco','nus']:
    for seed in [7,17,27]:calrows.append((d,seed,read(E/f'calibration/{d}/seed{seed}_bins8192/result.json')))
table(['数据集 种子','变体 mAP','条件参考 mAP','权重不同类数','最大 AP 误差'],[[f'{d.upper()} {s}',f"{r['test_mAP']:.6f}",f"{r['central_same_ridge_candidate_test_mAP']:.6f}",str(r['weight_mismatch_classes']),f"{r['fusion_AP_max_error']:.6f}"] for d,s,r in calrows],[1.0,1.05,1.1,1.0,1.15])
para('表中 AP 误差是全部融合候选与类别上的最大验证 AP 误差，单位为百分点，不是最终平均 mAP 的误差。COCO split17 的 Ridge alpha 在直方图与精确参考中分别为 0.0003 和 0.0001，其余五组 alpha 一致。各组测试分数最大绝对差依次为 COCO 1.220001、2.033229、1.313112，NUS 1.641702、5.588432、3.860802（种子顺序 7/17/27）。即使平均 mAP 接近，也不能称预测或选择精确等价。')
para('固定 8192 bins，分数单调映射为 0.5 + atan(score)/pi；全局 AP 为合并直方图后的近似值，不能使用本地 AP 平均代替。此实现应称“直方图校准变体”，不能自动认定为原集中式 Full 的精确等价替换。记录各候选与精确集中式 AP 的误差、所选参数一致性、最终分数差、校准矩误差；均值/方差容差为 1e-10。中央融合参考以该轮所选 Ridge 候选为条件，验收时需连同候选选择差异解释。')
para('每组记录 580 条消息、40 个上行阶段，包含所有候选与重拟合。COCO 每划分上行约 3.258 GB、下行 43.788 MB；NUS 上行约 3.296 GB、下行 44.361 MB（十进制单位）。直方图通信开销不能从全流程声明中省略。仅计数值应用负载，不含封装、加密与网络头。该实验是单服务器上的客户端消息仿真，未测物理网络 RTT，未实现密码学安全聚合或正式隐私保证。直方图和统计量本身仍可能暴露信息。QA 合成消息短测为 80 条消息，仅证明序列化链路工作，不能代替真实实验结果。')
heading('13.5 特征提取 系统计时与重复')
para('同一 RTX 5090 完成 COCO/NUS 冻结 CLIP CLS 与 patch 特征提取，输出使用 memmap；缓存 manifest 保留图像和标签对应关系，提取路径修复旧 pkg_resources.packaging 导入兼容问题，不修改特征计算。冷启动指无预计算特征，未强制清空操作系统页缓存。')
table(['阶段','COCO 秒','NUS 秒'],[
['特征提取队列墙钟','467.052794','213.961800'],
['校准 split7 队列墙钟','82.358019','189.370640'],
['校准 split17 队列墙钟','47.836193','102.915344'],
['校准 split27 队列墙钟','47.694876','102.698743'],
['优化后稳定训练每轮','约 133','约 238'],
],[2.3,1.5,1.5])
para('以上队列墙钟包含启动等开销，只用于执行交接。正式系统表应使用 extraction/result.json 和 calibration/result.json 的阶段字段，分开报告无特征缓存总时间与有缓存拟合、校准、选择、推理时间、峰值显存及双向数值负载。校准运行还包括独立中央参考审计，属于保守审计成本，不能称生产延迟。')
table(['数据集','特征提取 秒','有缓存 秒','无特征缓存总秒','上行 下行 GB'],[[r['dataset'].upper(),f"{r['cold_feature_extraction_seconds']:.3f}",f"{r['warm_feature_cache_fit_calibration_selection_inference_seconds']:.3f}",f"{r['cold_total_seconds']:.3f}",f"{r['uplink_payload_bytes']/1e9:.6f} / {r['downlink_payload_bytes']/1e9:.6f}"] for r in systems['rows'][:2]],[.65,1.0,.85,1.2,1.6])
para('该表为三个划分合计，包含全部候选、重拟合、推理和中央审计计算。校准阶段峰值显存为 COCO 166.518 MiB、NUS 166.619 MiB，不包含特征提取的峰值，不能标为端到端峰值。Full 的三个训练侧划分是聚合器组成部分，不能冒充三个独立完整 Full 重复。系统汇总已于 9 月 13 日 08:56:04 完成，missing 为空；旧 FAILED 日志保留为故障证据。')
table(['数据集 种子','训练累计 秒','最终评估 秒','单向负载 GB'],[[f"{r['dataset'].upper()} {r['training_seed']}",f"{r['training_seconds']:.3f}",f"{r['evaluation_seconds']:.3f}",f"{r['uplink_payload_bytes']/1e9:.6f}"] for r in systems['rows'][2:]],[1.2,1.5,1.3,1.3])
para('提示基线每次训练上行和下行各 9.083392 GB；表中只列一个方向，双向为两倍。训练累计为客户端计时之和，评估墙钟受图像读取与 GPU 故障影响，NUS seed27 的缓存评估明显更短；不能从这些不同缓存状态的总时间直接宣称单一公平加速倍数。')
heading('13.6 加速对照与故障恢复')
para('运行时优化 immutable_text_cache_v1 缓存类别分词、冻结 token embedding 和固定 zero-shot 文本编码。学习中的提示、两次图像编码、损失、FedAvg、batch32 和所有 200 轮预算保持不变；缓存不进入 state_dict，不增加通信参数。辅助代码 SHA256 为 d852c027022142bddc24dedecba3dcb6dd5d783b1c1908093b90e83d0082f44dc。')
para('COCO 和 NUS 各做 original→cached、cached→original 两个顺序，共 4 个配对对照；每配置 103 步。前 3 步 logits/loss/梯度、103 步后提示参数、最终 CUDA RNG 均逐位一致，最大绝对差 0。稳定批耗时由约 61ms 降至约 51ms，保守配对速度比 COCO 1.2067、NUS 1.2085。整轮 103 实测 133.009 秒，相对此前 159.72 秒减少 16.72%。')
para('初次 NUS 冷文件读取使原 benchmark.json 汇总出现 3.84 倍，不用于加速结论；解释以 timing_review.json 为准。COCO seed7 第 102 轮混合新旧实现并包含暂停，不计入速度比较。seed7 的累计训练时间包含优化切换，COCO seed17 测试还遇到 GPU 故障，不能把受影响墙钟当作稳定硬件基准；正式比较需单列这些异常。')
para('9 月 10 日 15:07 加速切换保留 101 个完整轮和第 102 轮已完成的 9 个客户端。部署脚本曾因 tmux 前缀匹配提前退出，已改为精确匹配并恢复，无已保存进度丢失。9 月 11 日 16:18 GPU 查询报 NVML Unknown Error，PyTorch 无可用 CUDA；第三组训练已保存 200 轮，测试随后于 16:19:51 完成。后续 3 组及汇总在故障期间启动失败，队列退出。16:21 GPU 实际计算测试通过后，备份 queue 并重建 tmux；NUS seed17 从未完成任务继续，日志批次推进、GPU 利用率 94%。')
heading('13.7 产物路径与最终接手步骤')
para('服务器独立工作目录为 /root/autodl-tmp/FedPASA/icme2027_workspace，新产物均在其中 outputs/icme2027/。服务器 RTX 5090 显存约 32GB，主机内存 16GB；“32GB”不是主机内存。原 outputs/icme/supplementary/ 归档保留。本轮已取回 1328 个文件，包含全部结果 JSON、消息记录、日志、六组最终预测与断点、加速证据；逐文件 SHA256 一致。约 63GB 特征缓存和约 6.4GB 校准中间数组保留服务器，未纳入本次轻量归档。')
table(['相对 outputs/icme2027 的位置','内容'],[
['queue.json 和 logs/','23 步状态与原始日志'],['paired/<dataset>/','共同抽样区间与来源哈希'],
['extraction/<dataset>/','提取性能及缓存核对'],['calibration/<dataset>/seed<seed>_bins8192/','候选误差 校准选择 双向负载'],
['baselines/fedmvp_release_mlc/<dataset>/seed<seed>/','result.json 最终预测 checkpoint 与 200 个轮次记录'],
['optimization/','数值对照 加速审查 启用版本和切换备份'],['systems_and_repeats.json','统一成本与三种子统计'],
],[2.85,2.45])
para('本地归档包为 tmp/icme2027_final_20260913.tar.gz，解包结果位于 outputs/icme2027/；清单为 final_archive_manifest_20260913.json，训练与预测验收为 final_audit_20260913.json。六组均已通过 200 轮断点、完整轮次和客户端记录、有限值及保存预测复算检查。三种子统计已由最终结果重新生成。六组校准的消息数、双向负载和误差值来自原始 result/messages 文件；不将执行 COMPLETE 改写为精确等价 PASS。')
para('当前无需再启动训练。若未来核验发现遗漏且 GPU 可用，先确认没有已有队列，再在服务器工作目录执行 tmux new-session -d -s icme2027 "bash scripts/start_icme2027_server.sh"。入口持有文件锁并跳过 COMPLETE 步骤；不得删除失败日志或覆盖完成结果。SSH 断开和本机关机不影响 tmux，服务器重启会终止进程。最终完成以后 tmux 正常退出，不能据此推断任务失败。')
para('投稿写作应明确直方图校准已改变选参，采用近似变体表述，或把原 Full 集中式校准写入设定；若坚持精确全流程联邦等价主张，还需另实现并验证精确选择协议。FedMVP 适配版、FedMPT 修补版和 published external 应分列；Full 单划分和三划分聚合的计算预算应区分；保留 VOC07 反例与 VOC12 区间跨 0 的结果。作者等价强基线、完整 Full 外层独立重复与正式隐私保证不能由本轮执行完成自动推出。无需为当前任务扩大 K、temperature 或新数据集网格。')
doc.core_properties.modified=datetime.now(timezone.utc)
doc.save(D);M.write_text(md.rstrip()+'\n\n'+'\n'.join(parts)+'\n',encoding='utf-8')
manifest={'updated':'2026-09-13','last_live_verified':'2026-09-13T08:57:30+08:00','original_completed':9,'new_baseline_complete':6,'new_queue_completed':23,'new_all_complete_verified':True,'archive_files_verified':1328,'metric_error_max':0,'calibration_moment_pass':6,'calibration_exact_selection_pass':0,'pending_execution':[],'research_limits':['histogram selection is not exact','baseline is a multilabel adaptation','Full outer repeats and formal privacy not established'],'docx':str(D)}
(H/'server_handoff_update_20260913.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False))
