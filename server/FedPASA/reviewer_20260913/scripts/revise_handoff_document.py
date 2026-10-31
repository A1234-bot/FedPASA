"""Apply evidence-scope corrections to the existing handoff, without reruns.

The historical builders call this pass as well, so rebuilding cannot silently
restore superseded wording. Experimental files and numeric table cells stay put.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor
from docx.text.paragraph import Paragraph

from handoff_revision_content import (
    CALIBRATION_SCOPE, COMMUNICATION_SCOPE, ESTIMATOR_NOTE, RECOVERY_SCOPE,
    REPRO_AFTER, REPRO_BLOCKS, REPRO_INTRO, REPRO_PREREQUISITES,
    REVISION_DATE, SERVER_NOTE, STATUS, SUPPLEMENT_COMMANDS,
)

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "outputs/icme/handoff"
DOCX = HANDOFF / "FedPASA_ICME_experiment_handoff_zh.docx"
MD = DOCX.with_suffix(".md")
QA = ROOT / "tmp/handoff_revision_20260909"

REPLACEMENTS = {
    "形成无需迭代反向传播且不传输单样本的多标签联邦适配框架。":
        "形成无需迭代反向传播的多标签适配框架；拟合阶段采用联邦充分统计，校准与选参的分布式协议尚待补充。",
    "全局分位数对齐 + 四次统计上传，精确恢复集中式 PIAA":
        "全局分位数对齐 + 四次统计上传；一致选样下恢复中央估计器（给定容差）",
    "无迭代反向传播、不传 individual example、FP32 近乎无损":
        "无迭代反向传播、拟合统计不传样本、Ridge FP32 近乎无损",
    "统计量精确可加，因此客户端划分改变时可以恢复同一个集中式目标。": RECOVERY_SCOPE,
    "No image, label, patch identity, or individual patch embedding is transmitted.":
        "Default QAFed fitting uploads no image, label, patch identity or individual embedding.\nOptional shared tie keys transmit patch IDs; Full calibration is outside this claim.",
    "QAFed 用客户端局部候选和全局分位数对齐精确恢复集中式 Top-K":
        "QAFed 用局部候选和全局分位数对齐恢复 Top-K；同一选样总体要求一致的边界 tie 规则",
    "72/72 运行恢复相同集中式目标。这里的主张是可加统计的 central-equivalence，不是“任何客户端缺失都不会掉点”。掉线压力需看第 6.2 节。":
        "72 个主实验配置的参数恢复检查均在登记容差内通过；其中 mAP 按 main.json 的 score_provenance 在参数检查后复用代表性完整测试结果，并非 72 次独立测试。该证据支持给定选样条件下的 central-equivalence；客户端缺失造成的退化见第 6.2 节。",
    "Fed score err": "记录误差*",
    "四个数据集 mAP 绝对误差均为 0；最大测试分数误差 2.38e-7。注意组合通信包含 QAFedPASA 与 Ridge 两部分。":
        "本节核验 seed=17 的最终 Ridge 拟合：四数据集 mAP 绝对误差均为 0，最大测试分数误差 2.38e-7。组合通信仅计最终拟合统计上传，采用 QA FP32 + Ridge FP64；Full 校准/选参未计入。",
    "组合通信 MiB": "拟合上传 MiB*",
    "Ridge stats": "Ridge FP64",
    "Total MiB": "拟合上传 MiB*",
    "融合受损类数": "Full<QA 类数*",
    "Estimator-preserving 对照使用真实 local Top-K bank；零熵边界会发生等价 tie 替换，因此 patch-ID recall 可略低于 100%，但适配器/分数误差才是核心恢复判据。":
        "Estimator-preserving 旧对照使用真实 local Top-K bank；patch-ID recall 不足 100% 不能自动解释为等价替换。旧表分数零误差来自中央预测复用；独立推理和 shared-key 条件分别见第 11 节，须保留各自来源和实验条件。",
    "不传 individual examples 不是差分隐私。没有隐私预算、成员推断或重构攻击实验，不写 formal privacy guarantee。":
        "拟合阶段不传 individual examples 不是差分隐私。Full 的分布式校准和选参尚未验证；没有隐私预算、成员推断或重构攻击实验，不写 formal privacy guarantee。",
    "标题/摘要避免承诺 formal privacy 或完全 annotation-free；强调 no iterative backpropagation 和 no individual-example transmission。":
        "标题/摘要避免承诺 formal privacy 或完全 annotation-free；可写 no iterative backpropagation，并将 no individual-example transmission 限定于已实现的拟合统计协议。",
    "最终提交前重新运行 `audit_icme_results.py --require-complete`，并核对 `verification_report.json` 的 reviewer_suite 三项均为 true。":
        "最终提交前分别运行 `audit_icme_results.py --require-complete` 与 `audit_handoff_document.py`。前者核验结果元数据和既有证据，reviewer_suite 当前为四项；后者核验文档结构、关键措辞及复现命令，两类验收不能相互替代。",
    "pytest: 4 passed；新增和修改脚本均通过 py_compile。":
        "2026-09-09 内容审查时，现有 pytest 为 8 passed；此记录仅指当时执行的单元测试，不代表新实验训练完成。文档修改后需另行执行文档审计。",
    "其余工作为 ICME 页数内的论文取舍、叙事压缩与正式排版。":
        "此外，Full 分布式校准/选参协议仍有证据缺口；文档修订不表示这些实验或实现已经完成。",
    "本轮 9 项补充实验已完成 8 项；FedMPT 修补版待完成": STATUS,
    "2026-09-07 更新补充实验交接": REVISION_DATE,
    "已确认采用修补版并披露差异：修补后的 vanilla 聚合、4 个内置条件、优化器动量在每个客户端轮次重置、BF16 与激活重计算等均记录在 protocol.json。该结果不是官方算法等价复现，不能与原发布值混写为严格复现结果。":
        "修补版差异包括 vanilla 聚合、4 个内置条件、每客户端轮次重置优化器动量及 BF16。9 月 7 日服务器性能记录已关闭激活重计算、启用静态文本缓存，COCO/NUS workers=4/8，cpu-threads=1；执行入口为 start_server_fedmpt.sh。具体以服务器 protocol.json 的 runtime 与实际进程为准。该实现不是官方算法等价复现。",
}


def paragraphs(doc):
    """All paragraphs, including table cells, once each."""
    return [Paragraph(el, doc._body) for el in doc.element.body.iter(qn("w:p"))]


def replace_in_paragraph(p, old, new):
    """Replace a span while preserving formatting of unrelated runs."""
    if old == new:
        return
    while old in p.text:
        start = p.text.index(old)
        end = start + len(old)
        offset = 0
        spans = []
        for run in p.runs:
            spans.append((run, offset, offset + len(run.text)))
            offset += len(run.text)
        touched = [(r, a, b) for r, a, b in spans if b > start and a < end]
        if not touched:
            raise ValueError(f"unsupported text container: {old}")
        for i, (run, a, b) in enumerate(touched):
            left = run.text[:max(0, start-a)]
            right = run.text[max(0, end-a):] if b > end else ""
            run.text = left + (new if i == 0 else "") + right


def insert_after(anchor, doc, text, *, heading=False, code=False):
    node = OxmlElement("w:p")
    anchor.addnext(node)
    p = Paragraph(node, doc._body)
    p.style = doc.styles["Heading 2"] if heading else doc.styles["Normal"]
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = heading
    p.paragraph_format.keep_together = False
    r = p.add_run(text)
    r.font.color.rgb = RGBColor(0, 0, 0)
    if code:
        r.font.name = "Consolas"
        r.font.size = Pt(9)
        r._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Microsoft YaHei")
        p.paragraph_format.line_spacing = 1.0
    return node


def revise(*, backup=False):
    if backup:
        QA.mkdir(parents=True, exist_ok=True)
        for path in (DOCX, MD):
            destination = QA / ("original_" + path.name)
            if not destination.exists():
                shutil.copy2(path, destination)
    doc = Document(DOCX)
    md = MD.read_text(encoding="utf-8").replace("\\`", "`")
    for old, new in REPLACEMENTS.items():
        for p in paragraphs(doc):
            if old in p.text:
                replace_in_paragraph(p, old, new)
        md = md.replace(old, new.replace("\n", "<br />"))

    # Replacements whose older generators produce dynamic numeric strings.
    special = [
        ("四数据集 × 4 划分 × 6 方法共", ESTIMATOR_NOTE),
        ("通信拆分与原 Table III 总量逐字节一致", COMMUNICATION_SCOPE),
        ("当前 raw data 只覆盖", "9 月 7 日服务器交接已记录 COCO/NUS 图像核验；早期仅有 VOC/COCO 诊断数据的说明已过时。FedMPT benchmark 无 VOC12 入口，官方等价三基线复现仍未完成。补充 S01 的修补版 COCO/NUS 训练状态须在服务器核验；Full 分布式校准协议另有证据缺口。"),
    ]
    for prefix, replacement in special:
        for p in paragraphs(doc):
            text = p.text
            if prefix in text:
                old = text[text.index(prefix):]
                replace_in_paragraph(p, old, replacement)
        lines = md.splitlines()
        for i, line in enumerate(lines):
            if prefix in line:
                start = line.index(prefix)
                # Preserve the closing pipe on a one-cell callout table.
                tail = " |" if line.rstrip().endswith("|") else ""
                lines[i] = line[:start] + replacement + tail
        md = "\n".join(lines) + "\n"

    for p in doc.paragraphs:
        if p.text.startswith("状态："):
            replace_in_paragraph(p, p.text, "状态：" + STATUS)
        elif p.text.startswith("日期："):
            replace_in_paragraph(p, p.text, "日期：" + REVISION_DATE)
    md = re.sub(r"\*\*状态：\*\*[^\n]*", "**状态：**" + STATUS, md, count=1)
    md = re.sub(r"\*\*日期：\*\*[^\n]*", "**日期：**" + REVISION_DATE, md, count=1)

    inserts = [
        ("主交付：", SERVER_NOTE),
        ("每个 seed 把训练集划分为", CALIBRATION_SCOPE),
        ("逐类 CSV 覆盖 201 个类别。", "* Full<QA 类数按 Full AP − QAFedPASA AP < 0 计数（严格小于零）；第 5.5 节比较 Full 与 Ridge，使用 ±0.05 AP 点的近似相等容差。两处参照分支和阈值不同。"),
    ]
    for prefix, note in inserts:
        if note not in "\n".join(p.text for p in paragraphs(doc)):
            p = next(p for p in doc.paragraphs if p.text.startswith(prefix))
            insert_after(p._p, doc, note)
        if note not in md:
            lines = md.splitlines()
            pos = next(i for i, line in enumerate(lines) if prefix in line)
            lines[pos+1:pos+1] = ["", note]
            md = "\n".join(lines) + "\n"

    # Replace just the old reproduction introduction and its command table.
    start = next(p._p for p in doc.paragraphs if p.text == "8. 复现入口与执行顺序")
    node = start.getnext()
    while node is not None:
        if node.tag == qn("w:tbl") and "特征缓存" in "".join(node.itertext()):
            break
        if node.tag == qn("w:p") and "9. 论文写作" in "".join(node.itertext()):
            raise ValueError("workflow table missing; refusing to delete section 9")
        following = node.getnext()
        node.getparent().remove(node)
        node = following
    anchor = start
    for text in (REPRO_INTRO, REPRO_PREREQUISITES):
        anchor = insert_after(anchor, doc, text)
    for title, code in REPRO_BLOCKS:
        anchor = insert_after(anchor, doc, title, heading=True)
        anchor = insert_after(anchor, doc, code, code=True)
    insert_after(anchor, doc, REPRO_AFTER)
    before, section = md.split("# 8. 复现入口与执行顺序", 1)
    table_pos = section.index("| **阶段**")
    new_section = ["# 8. 复现入口与执行顺序", "", REPRO_INTRO, "", REPRO_PREREQUISITES, ""]
    for title, code in REPRO_BLOCKS:
        new_section.extend(["## " + title, "", "```powershell", code, "```", ""])
    new_section.extend([REPRO_AFTER, ""])
    md = before + "\n".join(new_section) + "\n" + section[table_pos:]

    if "11.4 证据与复现入口" in md:
        prefix = "实验解释器：D:/Software/anaconda3/envs/pytorch/python.exe。"
        old_intro = next((p for p in doc.paragraphs if p.text.startswith(prefix)), None)
        old_commands = (
            "python scripts/run_supplementary_experiments.py --phase all",
            "python scripts/run_matched_fedmpt.py --datasets coco nus --rounds 200 --clients 10 --seed 7 --workers 2",
            "python scripts/build_supplementary_report.py",
        )
        new_intro = "补充实验续跑与报告以服务器副本为准，具体步骤如下。已有闭式补充结果无需因本次文字修订而重跑；本机仅保留迁移前断点。"
        if old_intro is not None:
            old_text = old_intro.text
            replace_in_paragraph(old_intro, old_text, new_intro)
            insert_after(old_intro._p, doc, SUPPLEMENT_COMMANDS, code=True)
            md = md.replace(old_text, new_intro + "\n\n```bash\n" + SUPPLEMENT_COMMANDS + "\n```")
        for p in list(doc.paragraphs):
            if p.text in old_commands or p.text.startswith("第一个命令支持复用已有结果；"):
                md = re.sub(r"(?m)^" + re.escape(p.text) + r"$", "", md)
                p._p.getparent().remove(p._p)
        # Keep the server block canonical, including absolute Python paths.
        if new_intro in md:
            md = re.sub(
                re.escape(new_intro) + r"\s*```bash\n.*?\n```",
                lambda _: new_intro + "\n\n```bash\n" + SUPPLEMENT_COMMANDS + "\n```",
                md, count=1, flags=re.DOTALL,
            )

    # Local comments identify the source of the substantial corrections.
    comment_sources = [
        ("校准与通信边界：", "依据 run_label_patch_fusion.py 的 validation 读取与 AP 选择代码。这里只收窄声明，不声称分布式校准已实现。"),
        ("表中记录误差为", "依据 estimator_preserving_evidence.json 中 16 行 QAFedPASA 的 score_provenance。保留原数值并披露中央预测复用。"),
        ("8.1 固定解释器", "依据 run_coco_full_pipeline.py 的七点网格、三种子路径，以及 run_label_ridge.py / run_label_patch_fusion.py 的命令参数。命令在本次修订中仅做静态/无执行校验。"),
        ("运行交接：", "依据 2026-09-07 SERVER_MIGRATION.md、SERVER_PERFORMANCE.md 和 start_server_fedmpt.sh。没有据此推断当前服务器存活。"),
    ]
    anchored_ids = {el.get(qn("w:id")) for el in doc.element.body.iter(qn("w:commentRangeStart"))}
    for comment in list(doc.comments):
        if comment.author == "内容修订" and str(comment.comment_id) not in anchored_ids:
            comment._element.getparent().remove(comment._element)
    existing_comments = "\n".join(c.text for c in doc.comments)
    for marker, source in comment_sources:
        if source not in existing_comments:
            p = next(p for p in paragraphs(doc) if marker in p.text)
            runs = [r for r in p.runs if r.text]
            doc.add_comment(runs=runs, text=source, author="内容修订", initials="REV")
    doc.core_properties.modified = datetime.now(timezone.utc)
    doc.save(DOCX)
    MD.write_text(re.sub(r"\n{4,}", "\n\n\n", md), encoding="utf-8")
    if (ROOT / "outputs/icme/supplementary/final_result_audit_20260910.json").is_file():
        from finalize_handoff_20260910 import finalize
        finalize()
    print(json.dumps({"docx": str(DOCX), "markdown": str(MD), "experimental_results_modified": False}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backup", action="store_true")
    args = parser.parse_args()
    revise(backup=args.backup)
