"""Append the September supplement handoff to the existing DOCX and Markdown."""
from __future__ import annotations
import json
import re
from datetime import datetime
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/'outputs/icme/handoff'
SUPP=ROOT/'outputs/icme/supplementary'
DOCX=HANDOFF/'FedPASA_ICME_experiment_handoff_zh.docx'
MD=HANDOFF/'FedPASA_ICME_experiment_handoff_zh.md'
HEADING='11 补充实验清单与执行交接'


def read(name):return json.loads((SUPP/name).read_text(encoding='utf-8'))


def font(run,size=10.5,bold=False):
    run.font.name='Calibri';run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor(0,0,0)
    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Microsoft YaHei')


def build():
    doc=Document(DOCX)
    # Idempotent regeneration of this append-only update.
    deleting=False
    for el in list(doc.element.body):
        if el.tag==qn('w:sectPr'):continue
        if ''.join(el.itertext()).startswith(HEADING):deleting=True
        if deleting:doc.element.body.remove(el)
    for p in doc.paragraphs:
        if p.text.startswith('状态：'):
            p.text='状态：本轮 9 项补充实验已完成 8 项；FedMPT 修补版待完成'
        elif p.text.startswith('日期：'):p.text='日期：2026-09-07 更新补充实验交接'
        elif p.text.startswith('主交付：'):p.text='主交付：本 Word 文档及同名 Markdown；新增证据见 supplementary 目录'
    # The old pages are retained as the original experimental archive.
    markdown=['\n# '+HEADING,'']
    def heading(text,level=2):
        p=doc.add_heading(text,level=level)
        for r in p.runs:font(r,14 if level==1 else 11.5,True)
        p.paragraph_format.space_before=Pt(10);p.paragraph_format.space_after=Pt(6)
        markdown.extend(['#'*level+' '+text,''])
        return p
    def para(text,bold=False):
        p=doc.add_paragraph();p.paragraph_format.space_after=Pt(6)
        p.paragraph_format.line_spacing=1.12
        font(p.add_run(text),10.5,bold)
        markdown.extend([text,''])
    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for c,w in zip(t.columns,widths):c.width=Inches(w)
        for c,text,w in zip(t.rows[0].cells,headers,widths):c.text=text;c.width=Inches(w)
        for row in rows:
            cells=t.add_row().cells
            for c,text,w in zip(cells,row,widths):c.text=str(text);c.width=Inches(w)
        pr=t._tbl.tblPr
        borders=OxmlElement('w:tblBorders')
        for edge in ('top','left','bottom','right','insideH','insideV'):
            x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');borders.append(x)
        pr.append(borders)
        repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
        for i,row in enumerate(t.rows):
            row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
            for j,c in enumerate(row.cells):
                c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tc=c._tc.get_or_add_tcPr();margin=OxmlElement('w:tcMar')
                for edge in ('top','bottom','left','right'):
                    v=OxmlElement('w:'+edge);v.set(qn('w:w'),'90');v.set(qn('w:type'),'dxa');margin.append(v)
                tc.append(margin)
                if i==0:
                    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'E8EDF2');tc.append(shade)
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(2);p.paragraph_format.space_before=Pt(2);p.paragraph_format.line_spacing=1.1
                    if j==0 or (len(headers)==4 and j>=2):p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                    for r in p.runs:font(r,9,i==0)
        doc.add_paragraph().paragraph_format.space_after=Pt(2)
        markdown.extend(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|'])
        markdown.extend('| '+' | '.join(str(x) for x in row)+' |' for row in rows);markdown.append('')
    p=doc.add_heading(HEADING,level=1);p.paragraph_format.page_break_before=True
    for r in p.runs:font(r,14,True)
    para('本节是 2026 年 9 月补充实验的执行交接，覆盖所给建议中的全部 9 项。8 项已有完整结果；同协议基线中的 FedMPT 修补版尚未完成。接手者应先核对进程和断点，再继续该项训练。前文的 PASS 仅适用于原有实验套件，不代表本节全部完成。',True)
    para('交接保留实验结论、验收条件、脚本和结果索引。未完成训练的控制台流水已清理，模型断点和已完成轮次的结构化评估仍保留；阶段性 mAP 不进入正式基线表。')
    heading('11.1 需要补充的实验总清单')
    items=[('S01','同协议强基线与原理基线','P0','FedMPT 待完成'),('S02','选样偏差与本地模型平均分解','P0','已完成'),
        ('S03','QA 与 Ridge 逐类互补和融合权重','P0','已完成'),('S04','中央目标与联邦差距','P1','已完成'),
        ('S05','Boundary tie 可控压力测试','P1','已完成'),('S06','客户端数量扩展性','P1','已完成'),
        ('S07','统一口径通信基线对比','P1','已完成'),('S08','五种融合策略消融','P2','已完成'),('S09','三视图表示消融','P2','已完成')]
    table(['编号','实验','优先级','当前结论'],items,[.52,3.22,.6,1.46])
    para('本轮没有继续扩展 K、temperature、噪声等级、参与率、量化精度、随机种子或新增数据集。已经完成的结果可以复用，无需因 FedMPT 中断而整套重跑。')
    heading('11.2 唯一尚需完成的基线训练')
    para('S01 的闭式对照已经具备：Fed3R-style CLS ridge、Naive local PIAA/GDA averaging、中央 pooled estimator、QAFedPASA、三视图 Ridge 和锁定 FedPASA-Full。待补的是 COCO 与 NUS-WIDE 的 FedMPT 修补版最终结果。')
    para('共同协议为 CLIP ViT-B/16、224 输入、相同训练和测试图像列表、相同标签顺序、10 个 KMeans 客户端、划分 seed=7 及同一 mAP 实现。闭式方法使用相同冻结缓存；FedMPT 的视觉提示改变图像编码过程，必须重新处理图像，因此只能称数据与划分等条件匹配，不能称最终特征缓存相同。')
    para('已确认采用修补版并披露差异：修补后的 vanilla 聚合、4 个内置条件、优化器动量在每个客户端轮次重置、BF16 与激活重计算等均记录在 protocol.json。该结果不是官方算法等价复现，不能与原发布值混写为严格复现结果。')
    para('当前断点绑定的训练预算为每数据集 200 轮，每轮全客户端各遍历一遍本地训练图像。200 轮是本次实现选择的预算，不是建议文本要求的最低验收轮数。预算或算法若调整，应单独登记并保留原断点；短轮数结果必须标为有限预算实验，不得直接宣称完整 200 轮基线完成。')
    para('续跑前检查实际 python 进程、checkpoint.pt 和 progress.json 的修改时间。状态文件可能残留 RUNNING，不能单独作为进程存活证据。训练曾中断且未留下异常堆栈，根因尚未确认；交接时不要把累计墙钟时间等同于持续 GPU 计算时间。')
    para('接手顺序：先核验两个数据集的断点可加载及真实进程状态；然后检查训练侧损失、梯度和吞吐，必要时验证优化前后的数值一致性；再按登记的预算完成训练与最终测试；最后刷新报告并验收。超参数、训练预算和模型选择均不能依据测试集 mAP 调优。')
    para('S01 验收条件：两个数据集都有与登记预算一致的最终 result.json、对应测试预测、完整协议与来源说明；正式表仅采用预先规定的最终 checkpoint；报告中的该项状态与真实文件一致。')
    heading('11.3 已完成实验的结果与写作要点')
    mech=read('estimator_mismatch.json')
    names=[('Centralized','中央 QA'),('GlobalQuota-GDA-Avg','对齐选样后本地 GDA 平均'),('SharedPurification-GDA-Avg','共享净化后本地 GDA 平均'),('QAFedPASA','充分统计量汇总 QA'),('Fed3R-style CLS ridge','Fed3R-style CLS ridge'),('Pooled Ridge','Pooled Ridge')]
    rows=[]
    for method,label in names:
        vals=[next(r['mAP'] for r in mech if r['dataset']==d and r['method']==method) for d in ('coco','nus')]
        rows.append([label,*[f'{v:.2f}' for v in vals]])
    table(['变体','COCO mAP','NUS mAP'],rows,[3.7,1.05,1.05])
    para('S02 直接隔离两个问题：全局选样对齐修复 patch 总体的偏差；充分统计量汇总修复本地独立求解后平均的偏差。SharedPurification 控制进一步固定净化后的 patch 集，仍出现模型平均偏差。最大权重误差采用 max-entry 绝对误差，并分别推理中央和联邦测试分数。')
    fusion=read('fusion_strategies.json')
    table(['融合策略','COCO mAP','NUS mAP'],[[name,*[f"{next(r['mAP'] for r in fusion if r['dataset']==d and r['variant']==name):.2f}" for d in ('coco','nus')]] for name in ('QA only','Ridge only','Fixed 0.5','Global learned','Classwise learned')],[3.7,1.05,1.05])
    para('S03 和 S08：alpha=1 表示 Ridge，alpha=0 表示 QA。逐类图给出三个训练侧划分的 alpha 均值和范围，以及按增益排序的 Full−Ridge AP。COCO/NUS 固定 0.5 融合低于 Ridge，学习融合有必要；VOC07 全局权重优于逐类权重的负面结果仍保留。类别增益计数沿用 ±0.05 AP 点容差，避免与旧报告口径冲突。')
    para('S04 与 S06：四个数据集、各 6 个客户端数设置，共 72 项 QA/Ridge/Full 分支比较；最大中央与联邦 mAP 差距约 5.01×10⁻⁷ 个百分点。扩展性固定中央 Top-K 并分配给所属客户端，测的是统计构建、服务器合并求解和完整测试推理，不能冒充完整局部选样的端到端运行时间。中央目标是同一估计器的参考，不是精度上界。')
    para('S05：在完整训练 patch 总体上先量化熵再选样，覆盖 FP32、1e-4、1e-3、1e-2；COCO 与 NUS 各有 local/shared 两种 tie 规则，共 16 项。1e-2 下 local ID recall 为 91.11%/94.12%，shared 在所有精度下均为 100%。共享 key 的最大测试分数误差约 1.94×10⁻⁷。')
    para('共享 tie key 必须同时用于本地截断和服务器排序，仅在服务器增加 key 不能恢复本地已丢弃的候选。该可选方案额外上传 int64 patch ID，所以不能沿用不上传 patch identity 的旧声明。普通 tie 规则下 mAP 接近不等于参数或测试分数完全相同。')
    para('S07：统一按 client→server 上传计量。COCO/NUS 的 QA FP32 + Ridge FP32 合计为 373.16/382.05 MiB；FP64 Ridge 参考口径为 612.72/627.24 MiB。原始全局选中 patch 上传是忽略选样协调成本的 oracle 下界，局部 Top-K 上传又改变了样本总体，表中分别标注。')
    para('S09：四数据集分别核验 CLS、CLS+Patch Mean、CLS+Patch-Text、CLS+Mean+Patch-Text 四种表示，共 16 行，使用训练侧验证集选正则化系数并复核保存的测试分数。')
    heading('11.4 证据与复现入口')
    para('新增结果根目录：D:/project/VScodeProject/FedPASA/outputs/icme/supplementary/。下列文件均相对该目录，PNG/PDF 投稿图位于 figures/。')
    refs=[('S01','protocol_matched_baselines.csv'),('S02','estimator_mismatch.csv'),('S03','classwise_complementarity.csv 和 complementarity_summary.csv'),('S04','federation_gap.csv'),('S05','boundary_ties.csv'),('S06','client_scaling.csv'),('S07','communication_baselines.csv'),('S08','fusion_strategies.csv'),('S09','representation_ablation.csv')]
    table(['实验','机器可读结果文件'],refs,[.6,5.2])
    para('总体状态和协议指纹：completion_audit.json、protocol_manifest.json、README.md。FedMPT 每个数据集分别保存在 coco/fedmpt_repaired/ 与 nus/fedmpt_repaired/；checkpoint.pt 是续跑入口，protocol.json 是修补和预算说明，roundNNN.json 与预测文件是阶段性证据。')
    para('实验解释器：D:/Software/anaconda3/envs/pytorch/python.exe。以下命令均在工作区 D:/project/VScodeProject/FedPASA 中执行，Python 前缀使用该解释器。')
    for command in ('python scripts/run_supplementary_experiments.py --phase all',
        'python scripts/run_matched_fedmpt.py --datasets coco nus --rounds 200 --clients 10 --seed 7 --workers 2',
        'python scripts/build_supplementary_report.py'):
        para(command)
    para('第一个命令支持复用已有结果；交接继续训练主要执行第二个命令。第三个命令刷新表格、图和完成审计。修改共享 tie key 后已有单元测试共 8 项通过；文档与结果的生成脚本保留在 scripts/。')
    para('本次补充要求来源为用户提供的 pasted-text.txt 建议；方法依据为 FedMPT 论文与发布代码、Fed3R 论文。正式文稿应区分 published external、disclosed repaired rerun、Fed3R-style adaptation 和本地机制控制实验。')
    doc.core_properties.modified=datetime.now()
    doc.save(DOCX)
    original=MD.read_text(encoding='utf-8')
    original=re.split(r'\n# 11 补充实验清单与执行交接\n',original)[0]
    original=re.sub(r'\*\*状态：\*\*[^\n]*','**状态：**本轮 9 项补充实验已完成 8 项；FedMPT 修补版待完成',original,count=1)
    original=re.sub(r'\*\*日期：\*\*[^\n]*','**日期：**2026-09-07 更新补充实验交接',original,count=1)
    MD.write_text(original.rstrip()+'\n'+'\n'.join(markdown)+'\n',encoding='utf-8')
    from revise_handoff_document import revise
    revise()
    if (SUPP/'final_result_audit_20260910.json').is_file():
        print((HANDOFF/'supplementary_handoff_update.json').read_text())
        return
    manifest={'updated':'2026-09-07','supplementary_items':9,'completed_items':8,'pending':'FedMPT repaired COCO/NUS baseline',
        'training_console_logs_removed':True,'checkpoints_and_round_metrics_retained':True,
        'docx':str(DOCX),'markdown':str(MD),'tables_added':4}
    (HANDOFF/'supplementary_handoff_update.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':build()
