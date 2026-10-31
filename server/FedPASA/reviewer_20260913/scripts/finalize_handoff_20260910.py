"""Update the existing handoff with verified final results and a bounded ICME plan."""
from pathlib import Path
from datetime import datetime, timezone
import json, re, shutil
from docx import Document
from docx.text.paragraph import Paragraph
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches, RGBColor
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from handoff_revision_content import STATUS, REVISION_DATE, SERVER_NOTE, SUPPLEMENT_COMMANDS

ROOT=Path(__file__).resolve().parents[1]
HANDOFF=ROOT/'outputs/icme/handoff'
DOCX=HANDOFF/'FedPASA_ICME_experiment_handoff_zh.docx'; MD=DOCX.with_suffix('.md')
QA=ROOT/'tmp/handoff_final_20260910'
SECTION='12 ICME 2027 投稿前证据审查与最小补实验计划'

def replace(p,old,new):
    if old==new or old not in p.text:return
    start=p.text.index(old); end=start+len(old); cursor=0; done=False
    for run in p.runs:
        a=cursor; b=a+len(run.text); cursor=b
        if b<=start or a>=end:continue
        run.text=run.text[:max(0,start-a)]+(new if not done else '')+(run.text[end-a:] if b>end else '')
        done=True

def runfont(r,size=10.5,bold=False):
    r.font.name='Calibri';r.font.size=Pt(size);r.font.bold=bold;r.font.color.rgb=RGBColor(0,0,0)
    r._element.get_or_add_rPr().get_or_add_rFonts().set(qn('w:eastAsia'),'Microsoft YaHei')

def finalize():
    audit=json.loads((ROOT/'outputs/icme/supplementary/final_result_audit_20260910.json').read_text())
    assert audit['status']=='PASS' and audit['completed_items']==9
    QA.mkdir(parents=True,exist_ok=True)
    for f in (DOCX,MD):
        backup=QA/('original_'+f.name)
        if not backup.exists():shutil.copy2(f,backup)
    doc=Document(DOCX); md=MD.read_text(encoding='utf-8')
    # Preserve every earlier section, its tables, images, evidence-scope fixes and comments.
    body=doc.element.body; remove=False
    for el in list(body):
        if el.tag==qn('w:sectPr'):continue
        if el.tag==qn('w:p') and ''.join(el.itertext()).startswith(SECTION):remove=True
        if remove:body.remove(el)
    md=md.split('\n# '+SECTION)[0].rstrip()+'\n'
    changes={
      '本地补充证据记录为 8/9 项；服务器进度待核验；本机保持暂停':STATUS,
      '本轮 9 项补充实验已完成 8 项；FedMPT 修补版待完成':STATUS,
      '2026-09-09 内容修订；训练状态依据 2026-09-07 记录':REVISION_DATE,
      '2026-09-07 更新补充实验交接':REVISION_DATE,
      '补充 S01 的修补版 COCO/NUS 训练状态须在服务器核验':'补充 S01 的修补版 COCO/NUS 已完成 200 轮，分别为 69.806/35.824 mAP；最终结果已核验归档',
      '源码已审计；完整 VOC 仅 1 轮诊断':'VOC 1 轮诊断；COCO/NUS 修补版各 200 轮完成',
      'FedMPT 待完成':'修补版已完成',
      '该状态不代表同协议外部三基线已经完成。':'该状态不代表可信同协议强基线和 Full 分布式校准证据已经齐全。',
      'P0-1 同协议外部三基线仍是明确实验缺口':'P0-1 可信同协议强基线仍有证据缺口',
      '目标 3 methods×4 data×3 seeds':'历史扩展设想；现按第12节缩减',
      'BLOCKED/未完成':'见第12节最小方案',
      '同协议外部三基线仍未完成；在获得 author-equivalent 代码与完整数据前，只能写 published external comparison，不能加 * reproduced under our protocol。':
      '修补版 COCO/NUS 已完成；作者等价强基线仍待补。具体按第12节的 COCO/NUS 最小方案执行，不要求原设想的三方法四数据集全量展开；published external 与修补版分别列示。',
      '这是对 FedPASA 更保守的比较。':'这样仅避免选取外部方法较低的公开值，不能消除协议差异。',
      '最小提升为 +2.31 mAP，达到约 +2 点目标。':'与公开最高值的最小数值差为 +2.31 mAP；它不是同协议受控增益，也不作为投稿验收目标。',
      '11.2 唯一尚需完成的基线训练':'11.2 已完成的 FedMPT 修补版基线',
      '待补的是 COCO 与 NUS-WIDE 的 FedMPT 修补版最终结果。':'COCO 与 NUS-WIDE 的 FedMPT 修补版最终结果均已完成，见第 12.1 节。',
      '本节是 2026 年 9 月补充实验的执行交接，覆盖所给建议中的全部 9 项。8 项已有完整结果；同协议基线中的 FedMPT 修补版尚未完成。接手者应先核对进程和断点，再继续该项训练。前文的 PASS 仅适用于原有实验套件，不代表本节全部完成。':
      '本节覆盖原定 9 项补充实验。2026-09-10 已核验全部完成；其中 S01 按用户确认的修补版口径验收。9/9 表示这份执行清单完成，不表示作者等价强基线、Full 全流程联邦化和全部投稿证据已齐全；剩余重点见第 12 节。',
      '接手顺序：先核验两个数据集的断点可加载及真实进程状态；然后检查训练侧损失、梯度和吞吐，必要时验证优化前后的数值一致性；再按登记的预算完成训练与最终测试；最后刷新报告并验收。超参数、训练预算和模型选择均不能依据测试集 mAP 调优。':
      '完成核验：两个 checkpoint 均为第 200 轮；result.json 与最终预测齐全，mAP 复算差值为 0。全部 400 轮、4000 个客户端轮次的记录完整，参数与预测均有限。后续新实验必须单独登记，不能依据测试集 mAP 调优或替换预先固定的最后一轮。',
      'S01 验收条件：两个数据集都有与登记预算一致的最终 result.json、对应测试预测、完整协议与来源说明；正式表仅采用预先规定的最终 checkpoint；报告中的该项状态与真实文件一致。':
      'S01 已满足本轮验收条件：两个数据集均有 200 轮 result.json、round200_test_scores.npy、checkpoint.pt、协议和修补说明。只报告固定第 200 轮；不能称官方算法等价复现。',
      '总体状态和协议指纹：completion_audit.json、protocol_manifest.json、README.md。FedMPT 每个数据集分别保存在 coco/fedmpt_repaired/ 与 nus/fedmpt_repaired/；checkpoint.pt 是续跑入口，protocol.json 是修补和预算说明，roundNNN.json 与预测文件是阶段性证据。':
      '总体状态和协议指纹：completion_audit.json、protocol_manifest.json、README.md。FedMPT 最终文件位于 coco/fedmpt_repaired/ 与 nus/fedmpt_repaired/：result.json、round200_test_scores.npy、checkpoint.pt 和 protocol.json。其余 roundNNN 文件保留训练轨迹。protocol 内嵌 RUNNING 是构建时快照，实际完成状态以最终结果、进度及断点的 200 轮交叉核验为准。',
      '补充实验续跑与报告以服务器副本为准，具体步骤如下。已有闭式补充结果无需因本次文字修订而重跑；本机仅保留迁移前断点。':
      '服务器最终结果已同步本机。以下命令仅供核验和重建报告，当前无需续跑；原迁移前文件另存 tmp/before_server_final_20260910/。',
      '200 轮是本次实现选择的预算，不是建议文本要求的最低验收轮数。':
      '200 轮是本次预先登记的预算，也与 FedMPT 原论文 COCO/NUS 的训练轮数一致；它不是建议文本要求的最低验收轮数。',
      '需要修改的是协议轮数、通信口径和证据字段，而不是重跑准确率实验。':
      '该次 PurificationPacket 重构只需修正协议轮数、通信口径和证据字段，无需重跑准确率；Full 校准协议是另外的待补项。',
    }
    paragraphs=[Paragraph(el,doc._body) for el in doc.element.body.iter(qn('w:p'))]
    for old,new in changes.items():
        for p in paragraphs:replace(p,old,new)
        md=md.replace(old,new)
    for p in paragraphs:
        if p.text.startswith('运行交接：'):replace(p,p.text,SERVER_NOTE)
        elif p.text.startswith('状态：'):replace(p,p.text,'状态：'+STATUS)
        elif p.text.startswith('日期：'):replace(p,p.text,'日期：'+REVISION_DATE)
        elif p.text.startswith('cd /root/autodl-tmp/FedPASA\n'):replace(p,p.text,SUPPLEMENT_COMMANDS)
    md=re.sub(r'^运行交接：[^\n]*',lambda m:SERVER_NOTE,md,flags=re.M)
    md=re.sub(r'\*\*状态：\*\*[^\n]*',lambda m:'**状态：**'+STATUS,md,count=1)
    md=re.sub(r'\*\*日期：\*\*[^\n]*',lambda m:'**日期：**'+REVISION_DATE,md,count=1)
    md=re.sub(r'```bash\ncd /root/autodl-tmp/FedPASA\n.*?```',lambda m:'```bash\n'+SUPPLEMENT_COMMANDS+'\n```',md,flags=re.S)
    for c in doc.comments:
        if '没有据此推断当前服务器存活' in c.text:
            c.paragraphs[0].text='2026-09-10 已登录服务器并核验最终状态，结果同步及指标复算证据见 final_result_audit_20260910.json。'

    additions=['\n# '+SECTION,'']
    def para(text):
        p=doc.add_paragraph();runfont(p.add_run(text));p.paragraph_format.space_after=Pt(6)
        p.paragraph_format.line_spacing=1.2
        grid=OxmlElement('w:snapToGrid');grid.set(qn('w:val'),'0');p._p.get_or_add_pPr().append(grid)
        additions.extend([text,'']);return p
    def heading(text,level=2):
        p=doc.add_heading(text,level=level)
        p.paragraph_format.space_before=Pt(12);p.paragraph_format.space_after=Pt(6)
        for r in p.runs:runfont(r,14 if level==1 else 12,True)
        additions.extend(['#'*level+' '+text,'']);return p
    def table(headers,rows,widths):
        t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
        for c,w in zip(t.columns,widths):c.width=Inches(w)
        for i,row in enumerate([headers]+rows):
            cells=t.rows[0].cells if i==0 else t.add_row().cells
            for c,text,w in zip(cells,row,widths):
                c.width=Inches(w);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
                c.text=str(text)
                for p in c.paragraphs:
                    p.paragraph_format.space_after=Pt(4);p.paragraph_format.space_before=Pt(4)
                    for r in p.runs:runfont(r,9.5,i==0)
                if i==0:
                    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E7EEF5');c._tc.get_or_add_tcPr().append(sh)
            trpr=t.rows[i]._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
        t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
        borders=OxmlElement('w:tblBorders')
        for name in ('top','left','bottom','right','insideH','insideV'):
            el=OxmlElement('w:'+name);el.set(qn('w:val'),'single');el.set(qn('w:sz'),'4');el.set(qn('w:color'),'D9D9D9');borders.append(el)
        t._tbl.tblPr.append(borders)
        additions.extend(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |'])
        additions.extend('| '+' | '.join(map(str,row))+' |' for row in rows);additions.append('')
    p=doc.add_heading(SECTION,level=1);p.paragraph_format.page_break_before=True
    for r in p.runs:runfont(r,14,True)
    para('结论：原定补充实验已全部跑完并在本机归档。投稿前优先补齐 Full 校准协议、可信同协议强基线及配对统计证据；不继续增加大批敏感性设置。下列优先级是结合当前代码和结果提出的研究建议，不是 ICME 官方规定的实验数量。')
    heading('12.1 完成状态和最终结果')
    table(['数据集','完成轮数','修补版 mAP','预测矩阵','指标复算差'],[
      ['COCO','200/200','69.806112','40137 × 80','0'],['NUS-WIDE','200/200','35.823527','60260 × 81','0']], [1.0,.8,1.1,1.35,1.05])
    para('总任务与 9 项完成审计均为 COMPLETE，完成时间为北京时间 2026-09-09 22:50:56。9 月 10 日核验时无训练进程或 tmux 会话，属于任务完成后的正常退出。480 个归档文件哈希一致；两个最终模型参数和预测均有限，400 个完整轮次和 4000 个客户端轮次记录通过检查。指标核验使用保存预测重新计算 mAP，没有重新执行图像推理。')
    para('验收口径延续用户确认的“相同数据、划分、骨干的修补版并披露差异”。原始 final result/protocol 内保留了早期诊断策略与 RUNNING 快照等历史字段；本节依据最终 result、progress、checkpoint 和数值审计判定完成，不改写原始证据。修补版可以单列报告，不能覆盖 published external 数字或标成作者等价复现。')
    table(['评估轮数','COCO mAP','NUS mAP'],[
      [str(a['round']),f"{a['mAP']:.3f}",f"{b['mAP']:.3f}"] for a,b in zip(audit['datasets']['coco']['evaluation_curve'],audit['datasets']['nus']['evaluation_curve'])],[1.2,2.05,2.05])
    para('NUS 从第 100 轮 33.871 提升到第 200 轮 35.824，轨迹仍在改善；这不等于已经证明收敛。其最终结果较论文公开最高值 58.36 低 22.54 点，但设置和算法存在差异，不能把差距归因于某一修补或只归因于训练时长。200 轮与论文 COCO/NUS 预算相同；现在依据测试曲线追加轮数或选取最佳测试轮均不合适。')
    heading('12.2 P0 Full 的分布式校准与选参核验')
    para('当前缺口在 run_label_patch_fusion.py：训练侧 validation 的逐样本标签和预测在中心集中计算。GDA/Ridge 充分统计量可加、没有测试集选参，都不能单独证明 Full 全流程不上传单样本。若论文保留 Full 端到端联邦化主张，这项必须先闭环；若采用公开校准集或集中式校准，应把它明确写入问题设定并限定通信声明。')
    para('最小实验：COCO 和 NUS，固定 10 个客户端，先用已有 seed=7 验证协议，再覆盖原有 7/17/27 三个训练侧划分。客户端统计 count/sum/sum-square 可以合并 z-score；全局 AP 涉及跨客户端排序，不能用“本地 AP 平均”替代。需要实现明确的排序/安全汇总协议，或将分箱 AP 标为近似变体；不得将拟议方案写成已实现。')
    para('验收：独立的集中式与分布式路径比较校准均值/方差、全部候选 lambda 和融合权重的验证 AP、最终所选参数及测试分数；给出精确或近似容差。记录客户端到服务器和服务器到客户端的实际字节、交互次数、dtype、全部候选选择与重拟合成本，以及每种消息包含哪些数据。现有“共享既定校准参数后的 Full 等价”不能替代这项验收。')
    heading('12.3 P0 强基线的有效性与公平比较')
    para('修补版 FedMPT 的 vanilla 聚合、条件提示等与论文的最优传输和条件流程不等价；现有 seed=7 结果只说明该登记实现跑完。先检查发布代码、条件文件、聚合方式、标签映射、损失归一化和训练侧诊断；能取得作者可运行配置时再建立作者等价基线。修补版 NUS 的低值应保留，不能据此声称对原方法取得 24 点以上的受控提升。')
    para('最小方案：COCO/NUS 两个数据集，保留已完成的 Fed3R-style 与机制对照，新增至少一个可信的强提示学习基线；优先解决 FedMPT 的算法一致性，无法解决时评估 FedMVP 或 FedPGP 官方实现的可运行性与多标签适配。算力允许再加第二个强基线。Fed3R-style 是 CLS 多标签 Ridge 适配，仍需明确与原 Fed3R 的任务差异。无需为了扩表强行移植不支持 VOC12 的方法。')
    para('公平口径：逐方法登记同一 CLIP 权重、图像列表、标签顺序、客户端划分、分辨率、mAP、训练侧选参预算和最后轮数。提示方法重算图像特征是算法需要，不能要求复用最终冻结缓存。单种子主比较使用已有 Full seed7 的预测；三划分 Full 聚合单列，并报告其全部选择成本，避免与单次训练基线混为相同计算预算。')
    heading('12.4 P1 配对统计和独立重复')
    para('先做无需 GPU 的复分析：从已有预测对 Full 与相同口径的 Ridge、全局融合做配对 image bootstrap。每次对所有方法使用同一组图像重采样，报告 mAP 差值及 95% 区间；至少 COCO/NUS，建议顺带覆盖已有四数据集。现有 statistics.json 是相对公开单值的条件概率，不能代替同协议方法之间的配对显著性证据。')
    para('三个训练侧划分 seed7/17/27 已存在，不应作为“缺少三种子”重新跑一遍；但它们是最终 Full 聚合器的三个组成部分，并非三个独立的完整 Full 实验。先重用这些单划分预测报告逐种子差值。基线有效性解决后，再给迭代强基线补 17/27 两个训练种子，并同步相同客户端划分；若主张完整聚合方法的跨运行稳定性，另登记独立外层重复。已验证参数恢复的确定性闭式算法无需重复相同计算。')
    para('统计实现检查：历史 bootstrap_map 对同分预测逐行计算 precision，而正式 sklearn AP 合并同分阈值。单位权重诊断中 COCO/NUS 的 mAP 差约 −2.06×10⁻⁷/−2.63×10⁻⁶ 点，影响很小，不能据此否定主结果；但生成新的配对区间前应修正 tie 处理并用小样本对照 sklearn 的重采样 AP。旧区间保留为历史记录，本次没有生成新的区间。证据：bootstrap_metric_audit_20260910.json。')
    heading('12.5 系统结果与无需扩展的实验')
    para('P1 系统核验：在同一 RTX 5090 上将冻结 CLIP 特征提取、拟合、校准/选择、最终推理分别计时，同时报告缓存已存在与从原始图像开始的总时间、峰值显存和双向通信。复用已保存的 FedMPT 训练记录，不必再跑 200 轮。训练跨本机与服务器、旧慢配置与新配置，不能用累计 53.07 小时直接当作统一硬件基准；仅用最后一轮时间也不代表全程总成本。')
    para('不再扩展：K 和 temperature 网格、更多噪声率/缺失率/参与率、更多低比特压缩、更多客户端数或新数据集。已有选样与模型平均分解、201 类互补、16 个边界 tie 设置、72 个中央/联邦比较、融合及表示消融足以支持机制分析。若论文增加跨骨干泛化或正式隐私主张，再补对应证据；当前没有必要为这两项扩展研究范围。')
    table(['优先顺序','最小交付','计算需求'],[
      ['1','Full 分布式校准协议及等价/通信核验','缓存与 CPU 为主'],
      ['2','可信强基线的配置核验及 COCO/NUS 对照','先短诊断，再决定长训练'],
      ['3','Full 对 Ridge 的配对区间与单划分差值','已有预测，无需 GPU'],
      ['4','有效基线独立种子与统一硬件成本表','仅必要新训练；复用日志']], [.8,3.1,1.4])
    heading('12.6 ICME 2027 会议信息和提交边界')
    para('截至 2026-09-10，IEEE Circuits and Systems Society 官方活动页列出 ICME 2027 于 2027 年 7 月 13–17 日在中国厦门举行，页面注明更多细节待公布。本次未核验到正式论文截止日、页数、匿名和补充材料规定；不能把其他同名 ICME 会议或往届规则当作 2027 要求。正式投稿前以当届 CFP 和作者指南为准。')
    para('论文叙事建议：主表优先展示可解释的同协议本地比较，published external 表明确为背景参照；保留校准边界、shared tie key 上传标识、固定 0.5 融合下降、VOC07 全局权重优于逐类权重等限制。现有“比公开最高值多约 2 点”不是同协议受控增益，也不应当作投稿验收标准。')
    para('核验来源：IEEE CASS，2027 IEEE International Conference on Multimedia & Expo，https://ieee-cas.org/event/conference/2027-ieee-international-conference-multimedia-expo-icme 。')
    para('方法来源：FedMPT 原论文第 5 节实现设置与表 1，https://arxiv.org/html/2605.28347v1 ；发布代码 https://github.com/xuc865/FedMPT-main 。Fed3R 原论文 https://arxiv.org/abs/2406.01116 。访问日期均为 2026-09-10。')
    heading('12.7 归档和接手入口')
    para('正式结果位于 outputs/icme/supplementary/。归档包为 tmp/fedpasa_final_20260910.tar.gz，传输清单为 tmp/server_final_manifest_20260910.json，旧文件备份位于 tmp/before_server_final_20260910/。以上路径相对本项目。')
    para('本次未启动新增训练。新实验另建目录并预先登记协议；现有归档可用 scripts/audit_final_fedmpt.py 复核，文档验收不能替代实验验收。')
    doc.core_properties.modified=datetime.now(timezone.utc)
    # Remove a forced blank-heavy page before the supplementary chapter.
    for p in doc.paragraphs:
        if p.text=='11 补充实验清单与执行交接':p.paragraph_format.page_break_before=False
        if p.style.name in ('Title','Subtitle') or p.style.name.startswith('Heading'):
            for r in p.runs:
                r.font.color.rgb=RGBColor(0,0,0)
                if p.style.name=='Title':r.font.underline=False
            pr=p._p.get_or_add_pPr()
            for border in list(pr.findall(qn('w:pBdr'))):pr.remove(border)
    for name in ('Title','Subtitle','Heading 1','Heading 2','Heading 3'):
        style=doc.styles[name];style.font.color.rgb=RGBColor(0,0,0)
        if style.element.pPr is not None:
            for border in list(style.element.pPr.findall(qn('w:pBdr'))):style.element.pPr.remove(border)
    # A table header must travel with at least its first data row.
    for t in doc.tables:
        if len(t.columns)>1 and len(t.rows)>1:
            for cell in t.rows[0].cells:
                for p in cell.paragraphs:p.paragraph_format.keep_with_next=True
    # Local comments make the completion update reviewable without rewriting old evidence.
    p=next(p for p in doc.paragraphs if p.text.startswith('运行交接：'))
    if not any(c.author=='最终核验' for c in doc.comments):
        doc.add_comment(runs=[r for r in p.runs if r.text],text='依据 2026-09-10 SSH 核验、480 文件 SHA256 校验及最终保存预测复算更新。历史实验数值和 9 月 9 日证据边界修正予以保留。',author='最终核验',initials='FIN')
    doc.save(DOCX);MD.write_text(md+'\n'.join(additions)+'\n',encoding='utf-8')
    manifest={'updated':'2026-09-10','supplementary_items':9,'completed_items':9,'pending_original_runs':[],
              'submission_gaps':['Full distributed calibration','credible matched strong baseline','paired inference and independent training seeds','systems cost scope'],
              'final_result_audit':'../supplementary/final_result_audit_20260910.json','new_training_started':False}
    (HANDOFF/'supplementary_handoff_update.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest))

if __name__=='__main__':finalize()
