from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs" / "icme" / "paper"
OUT = ROOT / "outputs" / "icme" / "handoff"
FIGURES = PAPER / "figures"
DOCX_PATH = OUT / "FedPASA_ICME_experiment_handoff_zh.docx"
MD_PATH = OUT / "FedPASA_ICME_experiment_handoff_zh.md"

DATASETS = ("voc07", "voc12", "coco", "nus")
DISPLAY = {"voc07": "VOC07", "voc12": "VOC12", "coco": "COCO", "nus": "NUS-WIDE"}
T_COLUMNS = ("t10", "t20", "t40", "t60", "t80", "t100")
MAIN_TABLE_BASELINES = ("FedPGP", "FedTPG", "FedAWA", "Fed-RAM", "FedMVP", "FedMPT")
ACCENT = "2E74B5"
DARK = "1F4D78"
INK = "0B2545"
LIGHT = "E8EEF5"
LIGHTER = "F4F6F9"
GRAY = "666666"
GREEN = "1F6B45"
GOLD = "7A5A00"
RED = "9B1C1C"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def f(value: float | str | None, digits: int = 2) -> str:
    if value in (None, ""):
        return "n/a"
    return f"{float(value):.{digits}f}"


def pct_range(values: Iterable[float]) -> str:
    values = list(values)
    return f"{min(values):.2f}-{max(values):.2f}"


def published_best(main: Sequence[dict[str, str]], dataset: str, method: str) -> float:
    row = next(
        item for item in main
        if item["dataset"] == dataset
        and item["result_type"] == "published_external"
        and item["method"].partition(" (")[0] == method
    )
    return max(float(row[column]) for column in T_COLUMNS)


def local_method_map(data: dict, dataset: str, method: str) -> float:
    matches = [
        row for row in data["main"]
        if row["dataset"] == dataset and row["method"] == method
    ]
    if matches:
        return float(matches[0]["t10"])
    if method == "FedPASA-Full":
        return float(data["stats"]["datasets"][dataset]["observed_mAP"])
    if method == "QAFedPASA":
        row = next(
            value for value in data["ablation"]
            if value["dataset"] == dataset and value["method"] == "topk_512"
        )
        return float(row["fixed_0.9_mAP"])
    raise KeyError((dataset, method))


def set_cell_margins(cell, top: int = 80, start: int = 120, bottom: int = 80, end: int = 120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for tag, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def shade_cell(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_repeat_table_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:tblHeader")
    marker.set(qn("w:val"), "true")
    tr_pr.append(marker)


def keep_row_together(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    marker = OxmlElement("w:cantSplit")
    tr_pr.append(marker)


def set_run_font(run, size: float | None = None, bold: bool | None = None, color: str | None = None,
                 italic: bool | None = None, latin: str = "Calibri", east_asia: str = "Microsoft YaHei") -> None:
    run.font.name = latin
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), latin)
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), east_asia)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic
    if color is not None:
        run.font.color.rgb = RGBColor.from_string(color)


def set_table_geometry(table, widths_dxa: Sequence[int], indent_dxa: int = 120) -> None:
    total = sum(widths_dxa)
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    table.autofit = False
    tbl_pr = table._tbl.tblPr
    for tag in ("tblW", "tblInd", "tblLayout"):
        existing = tbl_pr.find(qn(f"w:{tag}"))
        if existing is not None:
            tbl_pr.remove(existing)
    width = OxmlElement("w:tblW")
    width.set(qn("w:w"), str(total))
    width.set(qn("w:type"), "dxa")
    tbl_pr.append(width)
    indent = OxmlElement("w:tblInd")
    indent.set(qn("w:w"), str(indent_dxa))
    indent.set(qn("w:type"), "dxa")
    tbl_pr.append(indent)
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for value in widths_dxa:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(value))
        grid.append(col)
    for row in table.rows:
        keep_row_together(row)
        for cell, value in zip(row.cells, widths_dxa):
            cell.width = Inches(value / 1440)
            tc_pr = cell._tc.get_or_add_tcPr()
            tc_w = tc_pr.find(qn("w:tcW"))
            if tc_w is None:
                tc_w = OxmlElement("w:tcW")
                tc_pr.append(tc_w)
            tc_w.set(qn("w:w"), str(value))
            tc_w.set(qn("w:type"), "dxa")
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER


def add_page_field(paragraph, field: str) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = field
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend((begin, instr, separate, text, end))
    set_run_font(run, size=8.5, color=GRAY)


def add_numbering(doc: Document, *, ordered: bool) -> int:
    numbering = doc.part.numbering_part.element
    abstract_ids = [int(x.get(qn("w:abstractNumId"))) for x in numbering.findall(qn("w:abstractNum"))]
    num_ids = [int(x.get(qn("w:numId"))) for x in numbering.findall(qn("w:num"))]
    abstract_id = max(abstract_ids, default=0) + 1
    num_id = max(num_ids, default=0) + 1
    abstract = OxmlElement("w:abstractNum")
    abstract.set(qn("w:abstractNumId"), str(abstract_id))
    multi = OxmlElement("w:multiLevelType")
    multi.set(qn("w:val"), "singleLevel")
    abstract.append(multi)
    lvl = OxmlElement("w:lvl")
    lvl.set(qn("w:ilvl"), "0")
    start = OxmlElement("w:start")
    start.set(qn("w:val"), "1")
    lvl.append(start)
    num_fmt = OxmlElement("w:numFmt")
    num_fmt.set(qn("w:val"), "decimal" if ordered else "bullet")
    lvl.append(num_fmt)
    lvl_text = OxmlElement("w:lvlText")
    lvl_text.set(qn("w:val"), "%1." if ordered else "•")
    lvl.append(lvl_text)
    jc = OxmlElement("w:lvlJc")
    jc.set(qn("w:val"), "left")
    lvl.append(jc)
    p_pr = OxmlElement("w:pPr")
    tabs = OxmlElement("w:tabs")
    tab = OxmlElement("w:tab")
    tab.set(qn("w:val"), "num")
    tab.set(qn("w:pos"), "540")
    tabs.append(tab)
    p_pr.append(tabs)
    ind = OxmlElement("w:ind")
    ind.set(qn("w:left"), "540")
    ind.set(qn("w:hanging"), "270")
    p_pr.append(ind)
    lvl.append(p_pr)
    abstract.append(lvl)
    first_num = numbering.find(qn("w:num"))
    if first_num is None:
        numbering.append(abstract)
    else:
        numbering.insert(numbering.index(first_num), abstract)
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(num_id))
    abstract_ref = OxmlElement("w:abstractNumId")
    abstract_ref.set(qn("w:val"), str(abstract_id))
    num.append(abstract_ref)
    numbering.append(num)
    return num_id


def add_list_item(doc: Document, text: str, num_id: int) -> None:
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.25
    num_pr = p._p.get_or_add_pPr().get_or_add_numPr()
    ilvl = OxmlElement("w:ilvl")
    ilvl.set(qn("w:val"), "0")
    num = OxmlElement("w:numId")
    num.set(qn("w:val"), str(num_id))
    num_pr.append(ilvl)
    num_pr.append(num)
    run = p.add_run(text)
    set_run_font(run, size=10.5)


def add_paragraph(doc: Document, text: str, *, bold_prefix: str | None = None,
                  color: str | None = None, italic: bool = False, after: float = 6) -> None:
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.space_after = Pt(after)
    p.paragraph_format.line_spacing = 1.25
    if bold_prefix and text.startswith(bold_prefix):
        first = p.add_run(bold_prefix)
        set_run_font(first, size=10.5, bold=True, color=color)
        rest = p.add_run(text[len(bold_prefix):])
        set_run_font(rest, size=10.5, color=color, italic=italic)
    else:
        run = p.add_run(text)
        set_run_font(run, size=10.5, color=color, italic=italic)


def add_callout(doc: Document, label: str, text: str, *, fill: str = LIGHTER, color: str = INK) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade_cell(cell, fill)
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r1 = p.add_run(label + "  ")
    set_run_font(r1, size=10.5, bold=True, color=color)
    r2 = p.add_run(text)
    set_run_font(r2, size=10.5, color=color)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_table(doc: Document, headers: Sequence[str], rows: Sequence[Sequence[object]], widths: Sequence[int],
              *, font_size: float = 8.2, header_fill: str = LIGHT):
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        shade_cell(cell, header_fill)
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(1)
        p.paragraph_format.space_after = Pt(1)
        p.paragraph_format.keep_with_next = True
        run = p.add_run(str(header))
        set_run_font(run, size=font_size, bold=True, color=INK)
    set_repeat_table_header(table.rows[0])
    for values in rows:
        row = table.add_row()
        for i, value in enumerate(values):
            cell = row.cells[i]
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(0)
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.05
            run = p.add_run(str(value))
            set_run_font(run, size=font_size)
    set_table_geometry(table, widths)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(2)
    return table


def emphasize_table_row(table, row_index: int, *, bold: bool = False, underline_numbers: bool = False,
                        fill: str | None = None) -> None:
    row = table.rows[row_index]
    for column_index, cell in enumerate(row.cells):
        if fill is not None:
            shade_cell(cell, fill)
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                if bold:
                    run.bold = True
                if underline_numbers and column_index > 0:
                    run.underline = True


def add_code(doc: Document, lines: Sequence[str]) -> None:
    table = doc.add_table(rows=1, cols=1)
    set_table_geometry(table, [9360])
    cell = table.cell(0, 0)
    shade_cell(cell, "F2F4F7")
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.0
    for i, line in enumerate(lines):
        if i:
            p.add_run().add_break()
        run = p.add_run(line)
        set_run_font(run, size=8.3, latin="Consolas", east_asia="Microsoft YaHei")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def add_figure(doc: Document, path: Path, caption: str, *, width: float = 6.2) -> None:
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_before = Pt(6)
    cap.paragraph_format.space_after = Pt(4)
    cap.paragraph_format.keep_with_next = True
    run = cap.add_run(caption)
    set_run_font(run, size=9, bold=True, color=DARK)
    image_p = doc.add_paragraph()
    image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    image_p.paragraph_format.space_after = Pt(7)
    inline_shape = image_p.add_run().add_picture(str(path), width=Inches(width))
    # Keep figures accessible in Word/PDF readers.  The visible caption is also
    # a concise, content-specific alternative description for the inline image.
    doc_pr = inline_shape._inline.docPr
    doc_pr.set("descr", caption)
    doc_pr.set("title", caption)


def configure_styles(doc: Document) -> None:
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(10.5)
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    normal.paragraph_format.space_before = Pt(0)
    normal.paragraph_format.space_after = Pt(6)
    normal.paragraph_format.line_spacing = 1.25
    for name, size, color, before, after in (
        ("Title", 23, INK, 0, 6),
        ("Subtitle", 12.5, GRAY, 0, 16),
        ("Heading 1", 16, ACCENT, 18, 10),
        ("Heading 2", 13, ACCENT, 14, 7),
        ("Heading 3", 12, DARK, 10, 5),
    ):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor.from_string(color)
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
        style.paragraph_format.space_before = Pt(before)
        style.paragraph_format.space_after = Pt(after)
        style.paragraph_format.keep_with_next = True
        if name != "Subtitle":
            style.font.bold = True


def configure_document(doc: Document) -> tuple[int, int]:
    configure_styles(doc)
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.header_distance = Inches(0.492)
    section.footer_distance = Inches(0.492)
    header = section.header
    p = header.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run("FedPASA / ICME experiment handoff")
    set_run_font(run, size=8.5, color=GRAY)
    footer = section.footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p.add_run("Page ")
    set_run_font(r, size=8.5, color=GRAY)
    add_page_field(p, "PAGE")
    r = p.add_run(" of ")
    set_run_font(r, size=8.5, color=GRAY)
    add_page_field(p, "NUMPAGES")
    return add_numbering(doc, ordered=False), add_numbering(doc, ordered=True)


def build_markdown(data: dict) -> str:
    main = data["main"]
    stats = data["stats"]["datasets"]
    verification = data["verification"]
    lines = [
        "# FedPASA ICME 实验与创新交接文档",
        "",
        "- 日期：2026-08-23",
        "- 工作区：`D:\\project\\VScodeProject\\FedPASA`",
        f"- 最终验收：`overall_pass = {str(verification['overall_pass']).lower()}`",
        "- 锁定模型：FedPASA-Full",
        "",
        "> 一句话贡献：将 PIAA 的集中式 patch 分类器估计改写为全局分位数对齐的联邦充分统计问题，并将其与多视图、闭式、标签感知联邦 Ridge 通过严格训练侧逐类别融合，得到无需迭代反向传播、无需传输单样本的多标签联邦适配框架。",
        "",
        "## 0. 先看结论",
        "",
        "| 数据集 | FedPASA-Full | 最强公开外部结果 | 提升 | 95% bootstrap CI |",
        "|---|---:|---:|---:|---|",
    ]
    for dataset in DATASETS:
        ours = local_method_map(data, dataset, "FedPASA-Full")
        external = [row for row in main if row["dataset"] == dataset and row["result_type"] == "published_external"]
        strongest = max((float(row[column]) for row in external for column in T_COLUMNS), default=None)
        margin = None if strongest is None else ours - strongest
        ci = stats[dataset]["ci95_mAP"]
        lines.append(
            f"| {DISPLAY[dataset]} | {ours:.2f} | "
            f"{('n/a' if strongest is None else f'{strongest:.2f}')} | "
            f"{('n/a' if margin is None else f'{margin:+.2f}')} | [{ci[0]:.2f}, {ci[1]:.2f}] |"
        )
    lines.extend([
        "",
        "在存在 FedMPT 外部表项的 VOC07、COCO、NUS-WIDE 上，最小优势为 +2.31 mAP；VOC12 没有对应联邦外部表项，不能写成超过外部基线 2 点。",
        "",
        "## 1. 创新点与证据",
        "",
        "### 创新点 1：QAFedPASA - 全局分位数对齐的联邦 Patch 充分统计",
        "",
        "PIAA 的集中式每类 top-K patch bank 不能通过每个客户端各取 K 后直接合并，否则会扩大为 K×clients 并引入客户端偏差。QAFedPASA 先交换类别熵顺序统计量，恢复全局 top-K 配额；再上传初始 GDA moments、净化置信度 moments {count, sum(q), sum(q^2)} 和净化后的 GDA moments，精确重构 PIAA 的初始 GDA、全局净化阈值和最终 GDA。",
        "",
        "核心证据：72 组 KMeans 客户端规模/种子和 36 组 Dirichlet beta/种子均保持与集中式适配器数值等价；去除 quantile alignment 在 VOC07/VOC12/COCO/NUS 上分别损失 3.58/4.11/7.19/3.77 mAP。",
        "",
        "### 创新点 2：多视图标签感知联邦 Ridge 充分统计",
        "",
        "客户端只上传 n、sum_x、sum_y、X^T X 和 X^T Y；服务器用 float64 合并并闭式求解。冻结特征拼接 CLS、patch mean 和 patch-to-text max 三种互补视图。四个数据集上联邦解与集中式解 mAP 完全一致，最大测试分数误差不超过 2.38e-7。",
        "",
        "### 创新点 3：无测试泄漏的逐类别双分支融合",
        "",
        "训练侧 80/20 划分同时产生真正 out-of-sample 的 QAFedPASA 与 Ridge 验证预测，按类别选择 21 个候选权重。三个训练侧划分的逐类别仿射函数取均值后仍可折叠为单个仿射评分函数，不增加推理前向次数。",
        "",
        "### 已探索但未锁定为贡献的方案",
        "",
        "初始备忘中的 entropy/consensus reliability weighting 与 personalized shrinkage 已实现为 `fit_fedpasa()` 分支，但四个数据集均明显弱于 quantile-aligned 分支；不能在论文中写成最终核心模块。`FedPASA-Full` 也不是 annotation-free：QAFedPASA 分支无标签，但 Ridge 分支使用训练标签。",
        "",
        "## 2. 完整实验矩阵",
        "",
        "| 实验族 | 规模 | 支撑内容 | 主要产物 |",
        "|---|---:|---|---|",
        "| 公开基线 | 主文 6 个最佳公开单值；补充 11 方法 × 3 数据集 × 6 t | 外部竞争力 | `paper/main_comparison.csv` |",
        "| KMeans 异质性 | 4 数据集 × 6 客户端规模 × 3 种子 = 72 | 全局恢复与规模稳定性 | `*/main.json`, `paper/systems.csv` |",
        "| Dirichlet 异质性 | 4 × 3 beta × 3 种子 = 36 | 强 Non-IID | `paper/dirichlet.csv` |",
        "| 组件消融 | 4 × 10 = 40 | K、分位数、净化、协方差、可靠性 | `paper/ablation.csv` |",
        "| 特征消融 | 4 × 7 = 28 | CLS/patch mean/patch max-text | `paper/label_feature_ablation.csv` |",
        "| 三划分与 bootstrap | 12 seed runs + 4,000 bootstrap | 方差与置信区间 | `paper/full_seed_summary.csv`, `statistics.json` |",
        "| 缺失正标签 | 4 × 5 × 3 = 60 | 不完整标注 | `paper/partial_annotation.json` |",
        "| 成对标签噪声 | 4 × 4 × 3 = 48 | 错误标注 | `paper/label_noise.json` |",
        "| 客户端参与率 | 4 × 4 × 3 = 48 | 25%-100% 掉线压力 | `paper/client_participation.json` |",
        "| 温度敏感性 | 4 × 5 = 20 | 净化温度 | `paper/temperature_sensitivity.csv` |",
        "| 通信压缩 | 4 × 4 = 16 | FP64/32/16/INT8 | `paper/communication_compression.csv` |",
        "| 逐类失败分析 | 201 类 | 长尾和负增益 | `paper/per_class_failure_analysis.csv` |",
        "| 定性失败案例 | 4 × 3 = 12 | 边界案例/漏标线索 | `paper/qualitative_failure_examples.json` |",
        "| 官方 FedMPT 审计 | 完整 VOC 1 轮 | 可运行性、时间、显存、通信 | `paper/official_fedmpt_runnability.json` |",
        "",
        "## 3. 关键鲁棒性结果",
        "",
        "- 20% 成对标签噪声：VOC07 93.86、VOC12 93.61、COCO 80.14、NUS 60.24 mAP。",
        "- 75% 客户端可用：VOC07 93.91、VOC12 93.49、COCO 79.95、NUS 59.89 mAP。",
        "- 90% 正标签缺失：VOC07 90.99、VOC12 92.37、COCO 77.92、NUS 56.17 mAP。",
        "- 温度倍率 0.25-4.0：VOC07/VOC12/NUS 完全不变；COCO 全范围仅 0.0398 mAP。",
        "- FP32 将 Ridge 包通信量减半，四个数据集绝对 mAP 变化均小于 0.00002；FP16 在 COCO/NUS 有损，朴素 INT8 失败。",
        "",
        "## 4. 必须保留的声明边界",
        "",
        "1. `FedPASA-Full` 是标签感知、无迭代反向传播的闭式适配，不得写成完全 training-free 或 annotation-free。",
        "2. 未传输单样本不等于差分隐私；没有 DP 预算或攻击评估，不能声称正式隐私保证。",
        "3. 外部基线值来自 FedMPT Table 1，属于 published external，不是本地复现；主文仅放 6 个代表方法的最佳公开值，11 方法完整范围/逐 t 结果放补充材料。",
        "4. FedMPT 发布代码需 10 个修复才能跑通完整 VOC；其中有预测语义改变，所以 47.42 mAP 仅作诊断，不能进入主准确率对比。",
        "5. 25% 客户端参与率不稳定；朴素 INT8 失败；定性高分负例中存在疑似漏标。负结果必须保留。",
        "6. Dirichlet/KMeans 下 mAP 不变主要来自充分统计的精确可加性，应表述为 central-equivalence，而不是把它夸大为普遍鲁棒性。",
        "",
        "## 5. 复现顺序",
        "",
        "```powershell",
        r"$PY = 'D:\Software\anaconda3\envs\pytorch\python.exe'",
        r"& $PY scripts\run_icme_experiments.py --dataset <voc07|voc12|coco|nus> --phase all",
        r"& $PY scripts\run_label_ridge.py --dataset <dataset> --features cls patch_mean patch_max_text",
        r"& $PY scripts\run_label_patch_fusion.py --dataset <dataset> --ridge-name label_ridge_cls_patch_mean_patch_max_text",
        r"& $PY scripts\build_cv_fusion_ensemble.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\run_partial_annotation.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\run_reviewer_robustness.py",
        r"& $PY scripts\run_reviewer_federated_stress.py",
        r"& $PY scripts\verify_federated_full.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\build_icme_artifacts.py --bootstrap 1000",
        r"& $PY scripts\audit_icme_results.py --require-complete",
        "```",
        "",
        "COCO 的完整可恢复流水线见 `scripts/run_coco_full_pipeline.py`。",
        "",
        "## 6. 交接验收",
        "",
        f"- `verification_report.json`: overall_pass = {verification['overall_pass']}。",
        "- pytest: 4 passed。",
        "- 9 组投稿图均同时存在 PNG 与 PDF。",
        "- 论文写作剩余工作：按 ICME 页数选择主文图表，将完整审计与逐类表放补充材料，并在 limitations 中保留上述边界。",
        "",
        "## 7. 来源",
        "",
        "- 原始 PIAA：`C:\\Users\\17597\\Downloads\\CLS_is_Not_Enough_ICML2026_PIAA.pdf`。",
        "- 审稿实验清单：https://maxwellforbes.com/posts/how-to-get-a-paper-accepted/ 。",
        "- 外部联邦结果：https://arxiv.org/html/2605.28347 。",
    ])
    return "\n".join(lines) + "\n"


def build_docx(data: dict) -> None:
    doc = Document()
    bullet_id, number_id = configure_document(doc)
    main = data["main"]
    ablation = data["ablation"]
    feature = data["feature"]
    stats = data["stats"]["datasets"]
    full_verify = data["full_verify"]["rows"]
    partial = data["partial"]
    noise = data["noise"]
    participation = data["participation"]
    temperature = data["temperature"]
    compression = data["compression"]
    failures = data["failures"]
    official = data["official"]
    verification = data["verification"]
    estimator = data["estimator"]
    complementarity = data["complementarity"]
    communication = data["communication"]

    title = doc.add_paragraph(style="Title")
    title.alignment = WD_ALIGN_PARAGRAPH.LEFT
    r = title.add_run("FedPASA ICME 实验与创新交接文档")
    set_run_font(r, size=23, bold=True, color=INK)
    subtitle = doc.add_paragraph(style="Subtitle")
    r = subtitle.add_run("创新定义、证据链、复现入口与论文声明边界")
    set_run_font(r, size=12.5, color=GRAY)
    for label, value in (
        ("状态", "P0-2/P0-3/P1 证据实验 PASS；P0-1 同协议外部复现未闭环"),
        ("锁定方法", "FedPASA-Full"),
        ("日期", "2026-08-23"),
        ("工作区", str(ROOT)),
        ("主交付", str(PAPER / "README.md")),
    ):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(2)
        r1 = p.add_run(label + "：")
        set_run_font(r1, size=10.5, bold=True, color=INK)
        r2 = p.add_run(value)
        set_run_font(r2, size=10.5)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)
    add_callout(
        doc,
        "一句话贡献",
        "将 PIAA 的集中式 patch 分类器估计改写为全局分位数对齐的联邦充分统计问题，并与多视图、闭式、标签感知联邦 Ridge 通过严格训练侧逐类别融合，形成无需迭代反向传播且不传输单样本的多标签联邦适配框架。",
        fill=LIGHT,
    )

    doc.add_heading("0. 交接摘要", level=1)
    result_rows = []
    for dataset in DATASETS:
        ours = local_method_map(data, dataset, "FedPASA-Full")
        external = [row for row in main if row["dataset"] == dataset and row["result_type"] == "published_external"]
        strongest = max((float(row[column]) for row in external for column in T_COLUMNS), default=None)
        margin = None if strongest is None else ours - strongest
        ci = stats[dataset]["ci95_mAP"]
        result_rows.append((DISPLAY[dataset], f(ours), f(strongest), f"{margin:+.2f}" if margin is not None else "n/a", f"[{ci[0]:.2f}, {ci[1]:.2f}]"))
    add_table(doc, ("数据集", "FedPASA-Full", "最强公开外部", "提升", "95% bootstrap CI"), result_rows,
              (1350, 1700, 1700, 1200, 3410), font_size=8.6)
    add_callout(
        doc,
        "目标结论",
        "在 FedMPT Table 1 覆盖的 VOC07、COCO、NUS-WIDE 上，最小提升为 +2.31 mAP，达到约 +2 点目标。VOC12 没有对应联邦外部表项，只报告 93.98 mAP，不写 +2 结论。",
        fill="EAF4ED",
        color=GREEN,
    )
    add_paragraph(doc, "当前机器验收器已将主实验、无测试集选参、96 行 estimator-preserving 对照、201 类互补性分析、通信拆分、官方基线审计、12 个定性样例以及 11 组 PNG/PDF 双格式图全部纳入；本地证据套件 `overall_pass = true`。该状态不代表同协议外部三基线已经完成。")

    doc.add_heading("阅读导航", level=2)
    for item in (
        "第 1-2 节：从原始 PIAA 到最终锁定创新，明确哪些能写、哪些不能写。",
        "第 3-6 节：所有支持创新点的主结果、消融、统计、鲁棒性、系统和失败分析。",
        "第 7-9 节：完整实验登记、复现顺序、文件索引与后续写作动作。",
    ):
        add_list_item(doc, item, bullet_id)
    doc.add_heading("1. 原始 PIAA 血统与问题定义", level=1)
    add_paragraph(doc, "原始 PIAA 将多标签识别重写为 Patch-level Inference + Adaptive Aggregation。PVCL 从未标注 patch bank 估计类别均值与共享协方差，并用 GDA 闭式获得视觉分类器；PAA 再以固定 alpha=0.9 融合 patch 与 CLS 分数。原稿第 5-6 页给出 PVCL/PAA 公式，第 9 页 limitations 明确指出初始 patch 预测和 patch bank 的 noisy/biased 风险，并提出 reliability-aware calibration 作为未来方向。")
    add_code(doc, [
        "PIAA / GDA: w_c = Sigma^{-1} mu_c",
        "            b_c = -0.5 * mu_c^T Sigma^{-1} mu_c",
        "PIAA / PAA: S_f,c = alpha * S_patch,c + (1-alpha) * S_cls,c, alpha=0.9",
    ])
    add_paragraph(doc, "联邦场景放大了该风险：各客户端类别覆盖、patch 数量、纯度和目标尺度不同。若每个客户端独立取 K 个 patch 后再合并，会把集中式每类 K=512 的 bank 变成最多 K×clients 的偏置 bank；若直接平均分类器参数，也不能保证恢复集中式闭式解。")
    add_callout(doc, "切入点", "不是 PIAA + FedAvg，而是把集中式 patch distribution estimation 改造成可合并、可审计、无需传单样本的联邦充分统计。")

    doc.add_heading("2. 最终锁定的创新点", level=1)
    innovation_rows = [
        ("I1", "QAFedPASA", "全局分位数对齐 + 四次统计上传，精确恢复集中式 PIAA", "强；直接消融、central-equivalence、Non-IID"),
        ("I2", "多视图联邦 Ridge", "CLS/patch mean/patch max-text 的标签感知闭式适配，仅传可加统计", "强；特征消融、集中式等价、通信"),
        ("I3", "逐类别双分支融合", "训练侧 out-of-sample 校准与选权；三划分均值折叠为一次仿射评分", "强；三种子、bootstrap、逐类增益"),
        ("S1", "系统与协议性质", "无迭代反向传播、不传 individual example、FP32 近乎无损", "支撑项；不是正式 DP 保证"),
    ]
    add_table(doc, ("编号", "名称", "核心机制", "证据强度"), innovation_rows, (650, 1900, 4700, 2110), font_size=8.1)

    doc.add_heading("2.1 I1 - 全局分位数对齐的 QAFedPASA", level=2)
    add_paragraph(doc, "客户端首先保留本地候选 patch，但只交换每类熵顺序统计量；服务器据此下发全局 top-K 配额。客户端随后上传初始 GDA moment packet；收到初始全局 GDA 后，再上传每类 {count, sum(q), sum(q^2)} 以构造全局 mean+sample-std 净化阈值；最后上传净化后的 GDA moments。共四次客户端上传，其中 sum(q^2) 不能由另两项重构。")
    add_code(doc, [
        "Upload 1: entropy order statistics -> global per-class top-K quota",
        "Upload 2: {count, sum_x, mass, weighted_sum_x, second_moment} -> preliminary GDA",
        "Upload 3: {count, sum(q), sum(q^2)} -> global mean + sample-std threshold",
        "Upload 4: GDA moments after global purification -> final GDA",
        "No image, label, patch identity, or individual patch embedding is transmitted.",
    ])
    for item in (
        "直接解决 local top-K bank 过大和客户端偏置，而不是依赖启发式模型平均。",
        "统计量精确可加，因此客户端划分改变时可以恢复同一个集中式目标。",
        "标签完全不进入 QAFedPASA bank 构建和适配器拟合。",
    ):
        add_list_item(doc, item, bullet_id)

    doc.add_heading("2.1.1 投稿前实现—公式一致性核对（已闭环）", level=3)
    add_callout(
        doc,
        "结论",
        "显式 PurificationPacket 是对既有计算的数据边界重构，不改变 purification threshold、保留 patch、最终 GDA 权重/偏置或 mAP；需要修改的是协议轮数、通信口径和证据字段，而不是重跑准确率实验。",
        fill="EAF5EE",
        color=GREEN,
    )
    add_paragraph(
        doc,
        "实际 q_x 是客户端收到 preliminary global GDA 后，对已选 patch 计算的预测类别 refined confidence。由于 q_x 依赖该全局 GDA，它不能并入此前的 preliminary GDA moment packet；而 sum(q_x^2) 也不能由 count 与 sum(q_x) 重构。因此客户端必须在收到 preliminary GDA 后显式上传每类 {count, sum(q), sum(q^2)}。",
    )
    add_code(doc, [
        "r_kc = |S_kc|,  a_kc = sum_{x in S_kc} q_x,  b_kc = sum_{x in S_kc} q_x^2",
        "r_c = sum_k r_kc,  A_c = sum_k a_kc,  B_c = sum_k b_kc",
        "theta_c = A_c/r_c + sqrt(max(B_c - A_c^2/r_c, 0)/(r_c-1))  # sample std",
        "Comm_up = 4M[CK + 2(D^2 + 2CD + 2C) + 3C] bytes  # dense FP32 uplink",
    ])
    consistency_rows = [
        ("数值路径", "旧实现直接累加 count/sum(q)/sum(q^2)；新实现先形成客户端 packet 再逐项求和", "逐位等价；阈值最大绝对差 0.0"),
        ("模型与指标", "计算仍使用 float64；聚合公式、样本标准差及边界处理不变", "保留 patch、GDA weight/bias、AP/mAP 不变；无需重跑准确率"),
        ("协议元数据", "原文将 preliminary GDA 与其后才能计算的 q-moments 合并称为第 2 阶段", "修正为 4 次客户端上传；仅记录和叙述变化"),
        ("通信量", "旧公式已用 3C 计入 count/sum(q)/sum(q^2)，但没有独立命名", "总 MiB 不变；新增 purification_moments_MiB 分项"),
    ]
    add_table(
        doc,
        ("核对项", "确认结果", "对既有实验的影响"),
        consistency_rows,
        (1650, 4700, 3010),
        font_size=8.0,
        header_fill="EAF5EE",
    )
    add_paragraph(
        doc,
        "验证证据：随机多客户端旧/新阈值逐位相同（bitwise_equal=true，max_abs_diff=0.0）；核心测试 5/5 通过；投稿证据审计 overall_pass=true 且 communication_breakdown_exact=true。VOC07 的 t=60% 配置中，12 个客户端、20 类的独立净化 packet 仅为 0.0027466 MiB。",
    )
    add_callout(
        doc,
        "通信口径",
        "当前表格统计 dense FP32 client-to-server uplink upper bound；代码中的实际统计聚合仍为 float64 联邦逻辑模拟。论文表头应写 Uplink communication。若未来真实序列化为 FP32 或统计双向通信，则必须另做数值回归并加入 quota、preliminary GDA、threshold 等下行载荷。",
        fill="FFF5E5",
        color=GOLD,
    )

    doc.add_heading("2.2 I2 - 多视图标签感知联邦 Ridge", level=2)
    add_paragraph(doc, "为了利用训练标签但避免样本级传输，客户端对冻结图像特征构造 RidgePacket。服务器对 n、sum_x、sum_y、X^T X、X^T Y 求和，以 float64 解正规方程。特征视图为全局 CLS、patch mean 和对文本原型的 patch max 响应，三者互补。")
    add_code(doc, [
        "RidgePacket_k = {n_k, sum_x_k, sum_y_k, X_k^T X_k, X_k^T Y_k}",
        "W = (X_c^T X_c + lambda * scale * I)^{-1} X_c^T Y_c",
        "b = mean_y - mean_x * W",
    ])
    add_callout(doc, "术语边界", "该分支使用训练标签，因此 FedPASA-Full 应写成 label-aware, backpropagation-free closed-form federated adaptation；不能写成完全 annotation-free。", fill="FFF5E5", color=GOLD)

    doc.add_heading("2.3 I3 - 无测试泄漏的逐类别融合", level=2)
    add_paragraph(doc, "每个 seed 把训练集划分为 80% fit 与 20% validation。Ridge 与 PIAA 两个分支都只在 fit 子集拟合，再在 validation 上产生真正 out-of-sample 预测；按类别从 0.0-1.0 的 21 个权重中选取融合系数。三个 seed 的逐类别得分函数取平均后仍是单个仿射函数，可在部署时预先合并。")
    add_code(doc, [
        "score_c(x) = a_c * z_ridge,c(x) + (1-a_c) * z_piaa,c(x)",
        "score_CV,c(x) = mean_seed score_seed,c(x)  # still one affine class score",
        "Selection data: training-side validation only; test labels never used.",
    ])

    doc.add_heading("2.4 已验证但未锁定的早期方案", level=2)
    add_paragraph(doc, "初始创新备忘提出 client-class entropy/consensus reliability weighting 与 empirical-Bayes personalized shrinkage。代码中对应 `fit_fedpasa()` / `fit_fedpasa_from_packets()`，但 `reliability_*` 消融在四个数据集均弱于 quantile-aligned top-K=512。因此它们应作为探索性负结果或未来工作，不应继续作为最终主贡献。")
    reliability_rows = []
    by_ablation = {(r["dataset"], r["method"]): r for r in ablation}
    for dataset in DATASETS:
        base = float(by_ablation[(dataset, "topk_512")]["fixed_0.9_mAP"])
        rel = float(by_ablation[(dataset, "reliability_consensus_shrinkage")]["fixed_0.9_mAP"])
        reliability_rows.append((DISPLAY[dataset], f"{base:.2f}", f"{rel:.2f}", f"{rel-base:+.2f}"))
    add_table(doc, ("数据集", "QAFed top-K=512", "reliability+shrinkage", "差值"), reliability_rows, (1800, 2500, 3000, 2060), font_size=8.6, header_fill="FFF5E5")

    doc.add_heading("3. 实验协议与外部基线", level=1)
    protocol_rows = [
        ("骨干/输入", "冻结 CLIP ViT-B/16；PIAA/SC-CLIP patch 与 CLS 缓存"),
        ("数据集", "VOC07 20 类、VOC12 20 类、COCO 80 类、NUS-WIDE 81 类"),
        ("指标", "class-wise Average Precision 的均值 mAP (%)"),
        ("KMeans 划分", "最终 ICME 流水线对 L2-normalized ViT-B/16 CLS 与文本特征的零样本概率向量做聚类；t={10,20,40,60,80,100}%，seeds={7,17,27}，n_init=10，Lloyd"),
        ("Dirichlet 划分", "原等价性实验 beta={0.1,0.3,0.5}、10 clients、3 seeds；naive 证据实验为相同 beta 加 KMeans、seed=7"),
        ("选参", "训练侧 80/20；Ridge alpha、分支校准、融合权重均不读 test label"),
        ("外部结果", "FedMPT Table 1 的 11 个 published_external 方法；官方源码审计不能产生 author-equivalent controlled rerun，禁止把修复版诊断值写入准确率主表"),
        ("主方法", "FedPASA-Full = QAFedPASA + multi-view federated Ridge + classwise CV fusion"),
    ]
    add_table(doc, ("项目", "锁定设置"), protocol_rows, (1900, 7460), font_size=8.8)
    add_callout(doc, "比较范围", "“超过最强外部基线约 2 点”只适用于 FedMPT 表中有可比数值的 VOC07、COCO、NUS-WIDE；不延伸到 VOC12，也不延伸到不同监督级别的集中式全监督模型。")

    doc.add_heading("3.1 四类可复现性与实验协议核对（投稿锁定版）", level=2)
    add_callout(
        doc,
        "核对结论",
        "数据划分、Ridge、QAFed/Patch 与鲁棒性四类信息均已从最终结果文件及其生成脚本回查。现有结果不需要因为补写这些参数而重跑；其中客户端参与率实验本身已经包含 101/202/303 三个预声明子集种子。只有当论文改称“每轮重新采样客户端”或“逐标签独立 Bernoulli 噪声”时，才会与当前实现不一致并需要新实验。",
        fill="EAF5EE",
        color=GREEN,
    )

    doc.add_heading("3.1.1 数据划分与随机性设置", level=3)
    add_paragraph(
        doc,
        "最终 ICME 多数据集协议以 `scripts/run_icme_experiments.py` 和各数据集 `main.json` 为准。先分别 L2 归一化冻结的 CLIP ViT-B/16 全局 CLS 特征与单模板文本特征，再计算 q_i=softmax(z_i T^T)，得到每幅训练图像的 C 维零样本类别概率向量。K-means 直接作用于 q_i：没有使用标签、没有 PCA，也没有额外 logit temperature。",
    )
    add_code(doc, [
        "partition_feature_i = softmax(L2(cls_i) @ L2(text)^T)  # C-dimensional",
        "KMeans(n_clusters=N, init='k-means++', n_init=10, algorithm='lloyd', random_state=seed)",
        "partition seeds = {7, 17, 27}; feature extraction DataLoader shuffle=False",
    ])
    partition_rows = [
        ("VOC07", "20", "2/4/8/12/16/20", "7/17/27", "18"),
        ("VOC12", "20", "2/4/8/12/16/20", "7/17/27", "18"),
        ("COCO", "80", "8/16/32/48/64/80", "7/17/27", "18"),
        ("NUS-WIDE", "81", "8/16/32/49/65/81", "7/17/27", "18"),
    ]
    add_table(
        doc,
        ("数据集", "类别 C", "t=10/20/40/60/80/100% 的客户端数", "partition seeds", "runs"),
        partition_rows,
        (1500, 1000, 3860, 1650, 1350),
        font_size=7.9,
        header_fill="EAF5EE",
    )
    add_paragraph(
        doc,
        "客户端数由 max(2, round(C×t)) 得到。`init` 未在构造器中显式传入，因而采用 scikit-learn KMeans 的默认 `k-means++`；`n_init=10` 与 `algorithm='lloyd'` 是显式锁定项。旧脚本 `scripts/run_voc07_fedpasa.py` 曾直接对归一化 CLS 聚类并使用 n_init=20，但它是遗留的 VOC-only 路径，不是本文最终多数据集主结果协议，投稿文字不得混用。",
        color=GRAY,
    )

    doc.add_heading("3.1.2 Ridge baseline 的超参数与选择规则", level=3)
    add_paragraph(
        doc,
        "最终 Ridge 输入为 `cls + patch_mean + patch_max_text`：CLS 与 patch mean 各 512 维，patch max-text 为 C 维；拼接完成后对整条向量做一次 L2 归一化。每个客户端只上传 {n, sum_x, sum_y, X^T X, X^T Y}，服务器在 float64 中聚合并求解带截距的中心化正规方程。这里的 lambda 是按训练特征能量缩放的系数，不等同于未经缩放的 sklearn Ridge alpha。",
    )
    add_code(doc, [
        "scale = trace(X_centered^T X_centered) / D",
        "W = (X_centered^T X_centered + lambda * scale * I)^(-1) X_centered^T Y_centered",
        "lambda grid = {1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}",
    ])
    ridge_rows = [
        ("VOC07", "0.03", "0.03", "0.03"),
        ("VOC12", "0.10", "0.10", "0.03"),
        ("COCO", "0.0001", "0.0001", "0.0001"),
        ("NUS-WIDE", "0.0001", "0.0001", "0.0001"),
    ]
    add_table(
        doc,
        ("数据集", "seed 7 选中 lambda", "seed 17 选中 lambda", "seed 27 选中 lambda"),
        ridge_rows,
        (2100, 2420, 2420, 2420),
        font_size=8.4,
        header_fill="EAF5EE",
    )
    add_paragraph(
        doc,
        "对每个 seed，`np.random.default_rng(seed).permutation` 将训练集随机分为 80% fit/20% validation；lambda 按 validation mAP 最大值独立选择，随后用所选 lambda 在完整训练集上重拟合，测试标签从不参与选择。三视图先拼接后共用一个 lambda，并非每个视图单独设正则。特征消融使用 split seed=2026，并对每个视图组合单独选 lambda。需要特别注意：`run_label_ridge.py` 当前解析器默认网格仍是较旧的 {0.01,...,10}，最终结果实际使用的是上面的七点小正则网格；论文与复现命令必须显式传 `--alphas`，不能依赖脚本默认值。",
    )
    add_paragraph(
        doc,
        "FedPASA-Full 的融合选择同样只读训练侧 validation：Ridge 与 QAFedPASA 都在不含 validation 图像的 80% fit 上拟合；两分支分别按类别用 validation 预测的均值/标准差做 z-score（std 下限 1e-6），再从 0.00 到 1.00、步长 0.05 的 21 个 Ridge 权重中逐类别选择。三组 seed 的最终逐类得分取平均。",
    )

    doc.add_heading("3.1.3 QAFedPASA / Patch 分支的具体实现参数", level=3)
    patch_rows = [
        ("冻结骨干", "OpenAI-style CLIP ViT-B/16 提取全局 CLS；PIAA 的 SC-CLIP ViT-B/16 分支提取 patch token；同一 `ViT-B-16.pt`，SHA256=5806e77c...88df416f"),
        ("输入与 patch", "官方 CLIP preprocess，224×224；patch size=16，因此每图 14×14=196 个 patch token，每个 512 维；缓存为 float16，计算时转 float32 并 L2 归一化"),
        ("Prompt", "每类仅一个模板：`a photo of a {class_name}`；无 prompt ensemble"),
        ("初始置信度/候选", "logit=x·text；confidence=softmax(60×logit)，不是 sigmoid/raw logit；按 argmax 类别分组，每类保留最低 entropy 的全局 Top-K，K=512"),
        ("全局估计", "QAFed 用客户端局部候选和全局分位数对齐精确恢复集中式 Top-K；GDA 使用原始 softmax 类别置信度加权原型、合并 full covariance 与 trace shrinkage"),
        ("Purification", "初步 GDA 后 refined=softmax(T_pur×logits)；q 为预测类别 refined confidence；每类阈值为全局 mean(q)+unbiased sample std(q)，由显式 {count,sum_q,sum_q2} packet 聚合；保留 q≥threshold"),
        ("Patch aggregation", "最终每个 patch 先计算 softmax((xW+b)/500)，再对 196 个 patch 按类别取 max，最后 softmax(patch_scale×max)；不是 mean pooling"),
        ("固定分支融合", "QAFedPASA 分支内部为 0.1×CLS probability + 0.9×patch probability；entropy-adaptive 融合只做过消融，未用于最终结果"),
        ("Full 校准", "QAFed 内部尺度为固定 PIAA 参数；与 Ridge 组成 FedPASA-Full 时，才使用训练侧 validation 的逐类 z-score 与逐类融合权重"),
    ]
    add_table(doc, ("实现项", "投稿锁定设置"), patch_rows, (1900, 7460), font_size=7.8, header_fill="EAF5EE")
    temperature_rows = [
        ("VOC07", "100", "80", "200"),
        ("VOC12", "100", "80", "200"),
        ("COCO", "0.01", "80", "400"),
        ("NUS-WIDE", "500", "10", "400"),
    ]
    add_table(
        doc,
        ("数据集", "purification T_pur", "CLS scale", "patch scale"),
        temperature_rows,
        (2400, 2320, 2320, 2320),
        font_size=8.4,
        header_fill="F5F7FA",
    )
    add_paragraph(
        doc,
        "温度敏感性实验只围绕上述数据集基准值乘以 {0.25,0.5,1,2,4}，不读取测试标签选点。净化阈值、四次客户端上传与 `PurificationPacket` 的公式—实现一致性详见 2.1.1；这次显式 packet 重构保持阈值逐位一致，因此不会改变已报告的 patch、adapter、AP 或 mAP。",
    )

    doc.add_heading("3.1.4 鲁棒性实验：label noise 与 client participation", level=3)
    add_paragraph(
        doc,
        "Label noise 采用 `balanced_pair_noise`，不是每个 multi-label 位独立 Bernoulli 翻转。对每个类别 c，在整个训练集的正样本和负样本中分别无放回抽取 m_c=min(round(rho×n_c^+), n_c^-) 个，将所选正标签置 0、等量负标签置 1，因此每类正例数严格不变。噪声率为 {0,5,10,20}%，基础 seeds={7,17,27}，实际 RNG seed=seed×100000+round(rho×10000)。每个 dataset/rate/seed 只生成一张全局训练标签噪声矩阵，再按既定数据使用；不是每客户端独立重新加噪。",
    )
    add_paragraph(
        doc,
        "噪声条件下，Ridge lambda 与融合权重仍只用干净的训练侧 20% validation 标签选择；选中 lambda 后在全部带噪训练标注上重拟合。QAFedPASA 为 label-free 分支，保持固定。最终均值/样本标准差来自三个 seeds，test label 不参与调参。",
    )
    add_code(doc, [
        "m_c = min(round(rho * positives_c), negatives_c)",
        "choose m_c positives without replacement: 1 -> 0",
        "choose m_c negatives without replacement: 0 -> 1  # prevalence preserved per class",
    ])
    add_paragraph(
        doc,
        "Client participation 使用固定的 t=40%、partition seed=7 的 K-means 划分作为母划分：VOC07/VOC12 各 8 个客户端，COCO/NUS-WIDE 各 32 个。请求比例为 {25,50,75,100}%；每个条件在聚合/重拟合前，以 uniform sampling without replacement 一次性选择 max(1,round(fraction×N)) 个客户端。subset seeds={101,202,303}，实际 RNG seed=subset_seed+10000×N+round(100×fraction)。同一子集同时用于 QAFedPASA 与 Ridge；不存在逐轮重新采样，因为当前方法是一次性充分统计聚合/闭式拟合而非迭代训练。",
    )
    participation_rows = [
        ("VOC07", "8", "101:1/7; 202:4/7; 303:0/6", "101:3/4; 202:4/7; 303:2/7", "18.66-24.73 / 75.17-82.20"),
        ("VOC12", "8", "101:1/7; 202:4/7; 303:0/6", "101:3/4; 202:4/7; 303:2/7", "22.44-39.23 / 77.37-80.92"),
        ("COCO", "32", "101:2/7/13/14/22/29/30/31; 202:5/7/10/19/24/25/27/28; 303:2/4/6/7/9/11/21/26", "101:1/3/8/10/17/25/27/30; 202:0/3/6/7/13/19/20/29; 303:2/10/11/20/22/23/27/30", "25.20-28.08 / 74.67-76.77"),
        ("NUS-WIDE", "32", "101:2/7/13/14/22/29/30/31; 202:5/7/10/19/24/25/27/28; 303:2/4/6/7/9/11/21/26", "101:1/3/8/10/17/25/27/30; 202:0/3/6/7/13/19/20/29; 303:2/10/11/20/22/23/27/30", "22.71-26.17 / 70.39-78.21"),
    ]
    add_table(
        doc,
        ("数据集", "母客户端数", "25% 实际选中 ID", "75% 实际排除 ID", "训练图像保留率范围：25% / 75%"),
        participation_rows,
        (1250, 950, 2580, 2580, 2000),
        font_size=6.9,
        header_fill="EAF5EE",
    )
    add_paragraph(
        doc,
        "表中 75% 为节省篇幅列出“排除 ID”；其余客户端即实际参与集合。K-means 客户端大小不均衡，所以客户端比例不等于训练图像保留比例，结果文件同时记录两者。Ridge lambda、两分支逐类校准参数和逐类融合权重均在 availability shock 之前冻结自 seed-7 训练侧 validation，不针对任何参与率条件重调；100% 条件三个子集种子得到同一完整集合，故标准差为 0。全部 25/50/75/100% 的逐条件集合与结果保存在 `outputs/icme/paper/client_participation.json`。",
    )
    rerun_rows = [
        ("数据划分/K-means", "无需", "参数与 72 个主实验 run 已写入 main.json；只需准确补文"),
        ("Ridge", "无需", "实际网格、split seed 与每 seed 选中 lambda 均已有结果证据"),
        ("QAFed/Patch", "无需", "实现参数已锁定；显式 PurificationPacket 与旧阈值逐位相同"),
        ("Label noise", "无需", "0/5/10/20% × 3 seeds 已完成；正文必须称 prevalence-preserving paired flips"),
        ("Client participation", "无需", "25/50/75/100% × 3 subset seeds 已完成；正文必须称 one-shot availability shock"),
    ]
    add_table(doc, ("核对项", "是否补跑", "原因/投稿边界"), rerun_rows, (2100, 1500, 5760), font_size=8.2, header_fill="FFF5E5")

    doc.add_heading("3.2 ICME 主文 SOTA 表：单值、三数据集、六个代表方法", level=2)
    add_paragraph(
        doc,
        "主文不再并列“baseline 范围”与“FedPASA 单点”。对每个已发表方法，统一取其 t=10%-100% 公开结果中的最高 mAP；这是对 FedPASA 更保守的比较。主文仅保留 2024-2026 年的六个代表方法，删除没有外部可比项的 VOC12 列，并显式给出相对最强公开方法 FedMPT 的 Gain。",
    )
    submission_rows = []
    for method in MAIN_TABLE_BASELINES:
        submission_rows.append((
            method,
            f"{published_best(main, 'voc07', method):.2f}",
            f"{published_best(main, 'coco', method):.2f}",
            f"{published_best(main, 'nus', method):.2f}",
        ))
    ours_values = tuple(local_method_map(data, dataset, "FedPASA-Full") for dataset in ("voc07", "coco", "nus"))
    strongest_values = tuple(published_best(main, dataset, "FedMPT") for dataset in ("voc07", "coco", "nus"))
    submission_rows.append(("FedPASA (Ours)", *(f"{value:.2f}" for value in ours_values)))
    submission_rows.append(("Gain", *(f"{ours - baseline:+.2f}" for ours, baseline in zip(ours_values, strongest_values))))
    submission_table = add_table(
        doc,
        ("Method", "VOC07", "COCO", "NUS-WIDE"),
        submission_rows,
        (3600, 1920, 1920, 1920),
        font_size=8.8,
    )
    emphasize_table_row(submission_table, 6, underline_numbers=True)
    emphasize_table_row(submission_table, 7, bold=True, fill="EAF4ED")
    emphasize_table_row(submission_table, 8, bold=True, fill="EAF4ED")
    add_callout(
        doc,
        "主表视觉规则",
        "按 ICME/IEEE 常见紧凑表格处理：使用单栏 table[t]、lccc、small、tabcolsep=4.5pt，不画竖线；最佳结果加粗，第二名（FedMPT）数值加下划线，Gain 行加粗。caption 必须说明：For methods reported under multiple settings, we conservatively use their best published result.",
    )

    doc.add_heading("3.3 VOC12 与组件表的安排", level=2)
    add_paragraph(
        doc,
        "VOC12 不从论文删除，但不进入 Published Federated SOTA Comparison。主结果表后紧接以下英文说明；四数据集结果继续保留在 component analysis、overall results、鲁棒性和消融表中。",
    )
    add_code(doc, [
        "FedPASA further achieves 93.98 mAP on VOC12; we report this",
        "result for completeness but exclude it from the direct SOTA comparison",
        "because the adopted published federated benchmark does not provide a",
        "corresponding VOC12 entry.",
    ])
    component_rows = []
    component_specs = (
        ("w/o Quantile Alignment", "local_topk_no_quantile_alignment"),
        ("w/o Global Purification", "no_global_purification"),
        ("Diagonal Covariance", "diagonal_covariance"),
        ("QAFedPASA", "topk_512"),
    )
    for label, method in component_specs:
        component_rows.append((
            label,
            *(f"{float(by_ablation[(dataset, method)]['fixed_0.9_mAP']):.2f}" for dataset in DATASETS),
        ))
    for label, features in (("Ridge (CLS only)", "cls"), ("Ridge (Three Views)", "cls+patch_mean+patch_max_text")):
        component_rows.append((
            label,
            *(f"{float(next(row for row in feature if row['dataset'] == dataset and row['features'] == features)['test_mAP']):.2f}" for dataset in DATASETS),
        ))
    component_rows.append((
        "FedPASA-Full",
        *(f"{local_method_map(data, dataset, 'FedPASA-Full'):.2f}" for dataset in DATASETS),
    ))
    component_table = add_table(
        doc,
        ("Variant", "VOC07", "VOC12", "COCO", "NUS-WIDE"),
        component_rows,
        (3440, 1480, 1480, 1480, 1480),
        font_size=8.4,
    )
    emphasize_table_row(component_table, 7, bold=True, fill="EAF4ED")
    add_paragraph(doc, "该组件表不是外部 SOTA 横向表，因此保留 VOC12 合理；它同时把 QAFedPASA、三视图 Ridge 与逐类别融合的贡献放在同一统一口径下。四数据集宽表可继续使用双栏 table*[t]，同样不使用竖线。")

    doc.add_heading("3.4 补充材料：全部 11 个公开外部基线", level=2)
    external_methods = []
    seen = set()
    for row in main:
        if row["result_type"] != "published_external" or row["method"] in seen:
            continue
        seen.add(row["method"])
        ranges = []
        for dataset in ("voc07", "coco", "nus"):
            rr = next(x for x in main if x["method"] == row["method"] and x["dataset"] == dataset)
            ranges.append(pct_range(float(rr[k]) for k in T_COLUMNS))
        external_methods.append((row["method"], *ranges))
    add_table(doc, ("方法", "VOC07 范围", "COCO 范围", "NUS 范围"), external_methods, (3600, 1920, 1920, 1920), font_size=7.8)
    add_paragraph(doc, "本表仅作为 supplementary full comparison 的交接底稿：范围为 t=10%-100% 的最小-最大 mAP；完整逐 t 数字在 `outputs/icme/paper/main_comparison.csv`。不得再把这些范围直接复制回 ICME 主表。", italic=True, color=GRAY)

    doc.add_heading("3.5 同协议外部 baseline 的实际状态", level=2)
    controlled_rows = [
        ("FedMPT", "源码已审计；完整 VOC 仅 1 轮诊断", "需要改变预测语义的修复", "不能作为 controlled accuracy rerun"),
        ("FedMVP/FedAWA", "源码存在于同一 benchmark", "共享数据/协议缺陷，未完成 3 seeds 全训练", "未完成"),
        ("FedPGP/FedTPG", "源码存在于同一 benchmark", "共享数据/协议缺陷，未完成 3 seeds 全训练", "未完成"),
    ]
    add_table(doc, ("方法组", "已完成", "阻塞", "交接口径"), controlled_rows, (1700, 2550, 3200, 1910), font_size=8.0, header_fill="FFF5E5")
    add_callout(doc, "P0-1 边界", "当前 raw data 只覆盖可诊断的 VOC07/COCO，FedMPT benchmark 没有 VOC12 对应入口；更关键的是官方 launcher 需要改变预测语义的修复。故本轮没有伪造“统一协议三基线表”，而是把它保留为唯一未闭环的实验缺口。", fill="FFF5E5", color=GOLD)

    doc.add_heading("4. 支撑 I1 的实验：全局恢复、消融与 Non-IID", level=1)
    doc.add_heading("4.1 K 与关键组件消融", level=2)
    k_rows = []
    component_rows = []
    for dataset in DATASETS:
        vals = {m: float(by_ablation[(dataset, m)]["fixed_0.9_mAP"]) for m in ("topk_128", "topk_256", "topk_512", "topk_1024")}
        k_rows.append((DISPLAY[dataset], *(f"{vals[m]:.2f}" for m in ("topk_128", "topk_256", "topk_512", "topk_1024"))))
        full = vals["topk_512"]
        local = float(by_ablation[(dataset, "local_topk_no_quantile_alignment")]["fixed_0.9_mAP"])
        no_purify = float(by_ablation[(dataset, "no_global_purification")]["fixed_0.9_mAP"])
        diagonal = float(by_ablation[(dataset, "diagonal_covariance")]["fixed_0.9_mAP"])
        component_rows.append((DISPLAY[dataset], f"{full:.2f}", f"{local:.2f}", f"{full-local:+.2f}", f"{no_purify:.2f}", f"{diagonal:.2f}"))
    add_table(doc, ("数据集", "K=128", "K=256", "K=512", "K=1024"), k_rows, (1760, 1900, 1900, 1900, 1900), font_size=8.6)
    add_table(doc, ("数据集", "完整", "无分位数对齐", "对齐增益", "无全局净化", "对角协方差"), component_rows,
              (1350, 1400, 1900, 1500, 1700, 1510), font_size=8.1)
    add_paragraph(doc, "K=512 在四个数据集均最优。Quantile alignment 是一致且幅度最大的关键消融。全局净化在 COCO 上贡献显著，但在 VOC/NUS 上接近中性；对角协方差在 COCO 略高、其余数据集略低，不能写成所有数据集一致增益。")
    add_figure(doc, FIGURES / "ablation.png", "图 1  QAFedPASA 关键消融（源：paper/ablation.csv）")

    doc.add_heading("4.2 KMeans 客户端规模与集中式等价", level=2)
    main_rows = defaultdict(list)
    for dataset in DATASETS:
        payload = read_json(ROOT / "outputs" / "icme" / dataset / "main.json")
        main_rows[dataset] = payload["runs"]
    kmeans_summary = []
    for dataset in DATASETS:
        runs = main_rows[dataset]
        kmeans_summary.append((
            DISPLAY[dataset], len(runs), f"{min(r['clients'] for r in runs)}-{max(r['clients'] for r in runs)}",
            f"{min(r['mAP'] for r in runs):.5f}-{max(r['mAP'] for r in runs):.5f}",
            f"{max(r['central_recovery_max_abs_weight'] for r in runs):.2e}",
            f"{min(r['communication_MiB'] for r in runs):.2f}-{max(r['communication_MiB'] for r in runs):.2f}",
        ))
    add_table(doc, ("数据集", "runs", "clients", "mAP 范围", "最大权重误差", "通信 MiB"), kmeans_summary,
              (1350, 800, 1100, 1900, 1900, 2310), font_size=8.0)
    add_paragraph(doc, "72/72 运行恢复相同集中式目标。这里的主张是可加统计的 central-equivalence，不是“任何客户端缺失都不会掉点”。掉线压力需看第 6.2 节。")
    add_figure(doc, FIGURES / "heterogeneity.png", "图 2  外部基线与客户端规模异质性对比")

    doc.add_heading("4.3 Dirichlet 强 Non-IID", level=2)
    dir_rows = []
    for dataset in DATASETS:
        rows = [r for r in data["dirichlet"] if r["dataset"] == dataset]
        dir_rows.append((DISPLAY[dataset], "0.1/0.3/0.5", len(rows), f"{float(rows[0]['mAP']):.5f}", f"{max(float(r['max_abs_weight_error']) for r in rows):.2e}", str(all(r["equivalent"] == "True" for r in rows))))
    add_table(doc, ("数据集", "beta", "runs", "mAP", "最大权重误差", "等价"), dir_rows, (1500, 1700, 900, 1800, 1900, 1560), font_size=8.5)

    doc.add_heading("4.4 Naive federation 与 estimator-preserving federation", level=2)
    estimator_rows = []
    estimator_summary = estimator["summary"]
    for dataset in ("coco", "nus"):
        for partition in ("dirichlet_beta_0.1", "dirichlet_beta_0.3", "dirichlet_beta_0.5", "kmeans"):
            lookup = {
                row["method"]: row
                for row in estimator_summary
                if row["dataset"] == dataset and row["partition"] == partition
            }
            label = partition.replace("dirichlet_beta_", "beta=").replace("kmeans", "KMeans")
            estimator_rows.append((
                DISPLAY[dataset],
                label,
                f"{lookup['Naive-A local-TopK union']['mAP_mean']:.2f}",
                f"{lookup['Naive-B local-GDA average']['mAP_mean']:.2f}",
                f"{lookup['FedPASA quantile-aligned']['mAP_mean']:.2f}",
                f"{lookup['FedPASA quantile-aligned']['max_abs_score_error_mean']:.1e}",
            ))
    add_table(doc, ("数据集", "划分", "Naive-A", "Naive-B", "FedPASA", "Fed score err"), estimator_rows,
              (1350, 1650, 1500, 1500, 1500, 1860), font_size=8.0)
    fed_evidence = [row for row in estimator["rows"] if row["method"] == "FedPASA quantile-aligned"]
    naive_evidence = [row for row in estimator["rows"] if row["method"].startswith("Naive-")]
    add_paragraph(doc, f"四数据集 × 4 划分 × 6 方法共 {len(estimator['rows'])} 行。FedPASA 的最大分数误差为 {max(row['max_abs_score_error'] for row in fed_evidence):.1e}；Naive 方法的最大相对权重误差达到 {max(row['relative_weight_l2_error'] for row in naive_evidence):.3f}。精确 patch-ID recall 为 {min(row['central_topk_recall'] for row in fed_evidence):.3%}-{max(row['central_topk_recall'] for row in fed_evidence):.3%}，不足 100% 的部分来自零熵边界 tie 的等价替换，而适配器与测试分数仍恢复集中式目标。")
    add_paragraph(doc, "Naive-A 为 local Top-K union 后 pooled solve；Naive-B 为 sample-weighted local GDA 参数平均；Naive-C 为 sample-weighted local Ridge 参数平均。为避免利用数值崩溃夸大差距，naive PIAA 的 mean+std 净化阈值在饱和 tie 时保守夹到该类观测最大值。")
    add_figure(doc, FIGURES / "estimator_preserving_non_iid.png", "图 3  Naive local-union 与 FedPASA 的 Non-IID estimator 证据")

    doc.add_heading("5. 支撑 I2/I3 的实验：多视图闭式适配与融合", level=1)
    doc.add_heading("5.1 七种冻结特征组合", level=2)
    feature_names = ["cls", "patch_mean", "patch_max_text", "cls+patch_mean", "cls+patch_max_text", "patch_mean+patch_max_text", "cls+patch_mean+patch_max_text"]
    feature_rows = []
    for name in feature_names:
        values = []
        for dataset in DATASETS:
            row = next(r for r in feature if r["dataset"] == dataset and r["features"] == name)
            values.append(f"{float(row['test_mAP']):.2f}")
        feature_rows.append((name, *values))
    add_table(doc, ("特征", "VOC07", "VOC12", "COCO", "NUS"), feature_rows, (3360, 1500, 1500, 1500, 1500), font_size=8.3)
    add_paragraph(doc, "三视图 `cls+patch_mean+patch_max_text` 在 VOC07、VOC12、COCO 最优，并在 NUS 上接近最优；其选定 alpha 和完整验证分数保存在 `label_feature_ablation.csv`。")

    doc.add_heading("5.2 联邦 Ridge 与集中式闭式解一致", level=2)
    verify_rows = [(
        DISPLAY[row["dataset"]], row["clients"], f"{row['central_ridge_mAP']:.5f}", f"{row['federated_ridge_mAP']:.5f}",
        f"{row['max_abs_test_score_error']:.2e}", f"{row['combined_communication_MiB']:.2f}") for row in full_verify]
    add_table(doc, ("数据集", "clients", "集中式 mAP", "联邦 mAP", "最大分数误差", "组合通信 MiB"), verify_rows,
              (1350, 1000, 1600, 1600, 1900, 1910), font_size=8.2)
    add_paragraph(doc, "四个数据集 mAP 绝对误差均为 0；最大测试分数误差 2.38e-7。注意组合通信包含 QAFedPASA 与 Ridge 两部分。")
    communication_rows = []
    runtime_rows = []
    for row in communication["rows"]:
        communication_rows.append((
            DISPLAY[row["dataset"]],
            f"{row['quantile_MiB']:.2f}",
            f"{row['gda_moments_MiB']:.2f}",
            f"{row['ridge_stats_MiB']:.2f}",
            f"{row['total_MiB']:.2f}",
        ))
        runtime_rows.append((
            DISPLAY[row["dataset"]],
            row["runtime_clients"],
            f"{row['client_bank_seconds_sequential']:.2f}",
            f"{row['client_ridge_packet_seconds_sequential']:.2f}",
            f"{row['total_condition_wall_clock_seconds']:.2f}",
        ))
    add_table(doc, ("数据集", "Quantile", "GDA moments", "Ridge stats", "Total MiB"), communication_rows,
              (1800, 1800, 2200, 2100, 1460), font_size=8.4)
    add_table(doc, ("数据集", "clients", "client bank s", "ridge packet s", "condition wall s"), runtime_rows,
              (1800, 1200, 2200, 2100, 2060), font_size=8.2)
    add_paragraph(doc, "通信拆分与原 Table III 总量逐字节一致：QAFedPASA 为 4 次客户端上传，其中净化 packet 为每客户端 3C 个 FP32 标量；Ridge 为 1 次统计上传，可与现有同步阶段合并而不增加额外屏障。表中 QAFedPASA 数字是 client-to-server uplink upper bound，不包含服务器下发。运行时来自 10-client KMeans evidence condition，按单机顺序模拟客户端，因此 client-bank 时间是保守串行值。")

    doc.add_heading("5.3 三训练划分与 bootstrap", level=2)
    seed_rows = []
    for dataset in DATASETS:
        value = stats[dataset]
        seed_rows.append((DISPLAY[dataset], "/".join(f"{x:.2f}" for x in value["observed_mAP_per_seed"]), f"{value['between_seed_std_mAP']:.3f}", f"{value['observed_mAP']:.2f}", f"[{value['ci95_mAP'][0]:.2f}, {value['ci95_mAP'][1]:.2f}]"))
    add_table(doc, ("数据集", "3 seed mAP", "seed std", "CV 聚合", "95% bootstrap CI"), seed_rows,
              (1400, 2550, 1300, 1550, 2560), font_size=8.2)
    add_paragraph(doc, "每个数据集 1000 次 image-level bootstrap，seed=2026。VOC07/COCO 对所有 t 的 P(margin>2)=1；NUS 在最强 t10 基线上为 0.929，其余 t 为 1.0。外部基线没有方差，因此该概率条件于公开 mAP 标量。")

    doc.add_heading("5.4 QAFedPASA 到 FedPASA-Full 的增益", level=2)
    gain_rows = []
    for dataset in DATASETS:
        q = local_method_map(data, dataset, "QAFedPASA")
        full = local_method_map(data, dataset, "FedPASA-Full")
        gain_rows.append((DISPLAY[dataset], f(q), f(full), f"{full-q:+.2f}"))
    add_table(doc, ("数据集", "QAFedPASA", "FedPASA-Full", "融合增益"), gain_rows, (2200, 2300, 2600, 2260), font_size=8.8)

    doc.add_heading("5.5 Ridge-only 与 QAFedPASA 的逐类互补", level=2)
    complementarity_rows = []
    fusion_rows = []
    for dataset in DATASETS:
        row = complementarity["datasets"][dataset]["summary"]
        complementarity_rows.append((
            DISPLAY[dataset], row["full_gt_ridge"], row["full_approx_ridge"],
            row["full_lt_ridge"], f"{row['average_positive_gain_AP']:.2f}",
        ))
        fusion_rows.append((
            DISPLAY[dataset], f"{row['ridge_only_mAP']:.2f}", f"{row['global_fusion_mAP']:.2f}",
            f"{row['classwise_full_mAP']:.2f}", f"{row['classwise_gain_over_global_mAP']:+.2f}",
        ))
    add_table(doc, ("数据集", "Full>Ridge", "Full≈Ridge", "Full<Ridge", "正增益均值"), complementarity_rows,
              (1800, 1800, 1800, 1800, 2160), font_size=8.4)
    add_table(doc, ("数据集", "Ridge", "global fusion", "classwise full", "classwise-global"), fusion_rows,
              (1800, 1700, 2100, 2200, 1560), font_size=8.3)
    add_paragraph(doc, "COCO 有 59/80 类明确受益、平均正增益 0.96 AP；NUS-WIDE 有 42/81 类受益、平均正增益 1.06 AP。互补不是逐类单调：NUS 有 20 类下降，VOC07 的 global fusion 还比 classwise 高 0.09 mAP，均应如实保留。核心结论是 QAFedPASA 在 Ridge 失效的部分类别提供补充证据，而不是保证所有类别提高。")
    add_figure(doc, FIGURES / "classwise_complementarity.png", "图 4  COCO/NUS-WIDE 的 Full-Ridge 逐类 AP 差")

    doc.add_heading("6. 鲁棒性、系统与失败分析", level=1)
    doc.add_heading("6.1 缺失正标签（60 个原始配置）", level=2)
    partial_rows = []
    for rate in (10, 30, 50, 70, 90):
        values = []
        for dataset in DATASETS:
            row = next(r for r in partial if r["dataset"] == dataset and float(r["masked_positive_annotations_percent"]) == rate)
            values.append(f"{float(row['global_fusion_mAP_mean']):.2f}")
        partial_rows.append((f"{rate}%", *values))
    add_table(doc, ("缺失率", "VOC07", "VOC12", "COCO", "NUS"), partial_rows, (1600, 1940, 1940, 1940, 1940), font_size=8.6)
    add_paragraph(doc, "表中为 3 seed 全局融合均值。90% 正标签缺失时仍为 90.99/92.37/77.92/56.17 mAP。QAFedPASA 分支理论上不读取训练标签，`annotation_mask_invariance.csv` 只记录该解析不变性，不把它伪装成额外经验运行。")
    add_figure(doc, FIGURES / "partial_annotation.png", "图 5  不完整正标签鲁棒性")

    doc.add_heading("6.2 成对标签噪声与客户端参与率", level=2)
    noise_rows = []
    for rate in (0, 5, 10, 20):
        values = []
        for dataset in DATASETS:
            row = next(r for r in noise if r["dataset"] == dataset and float(r["pair_noise_percent"]) == rate)
            values.append(f"{float(row['global_fusion_mAP_mean']):.2f}±{float(row['global_fusion_mAP_std']):.2f}")
        noise_rows.append((f"{rate}%", *values))
    add_table(doc, ("pair noise", "VOC07", "VOC12", "COCO", "NUS"), noise_rows, (1600, 1940, 1940, 1940, 1940), font_size=8.2)
    participation_rows = []
    for frac in (25, 50, 75, 100):
        values = []
        for dataset in DATASETS:
            row = next(r for r in participation if r["dataset"] == dataset and float(r["requested_fraction_percent"]) == frac)
            values.append(f"{float(row['frozen_fusion_mAP_mean']):.2f}±{float(row['frozen_fusion_mAP_std']):.2f}")
        participation_rows.append((f"{frac}%", *values))
    add_table(doc, ("可用客户端", "VOC07", "VOC12", "COCO", "NUS"), participation_rows, (1600, 1940, 1940, 1940, 1940), font_size=8.2)
    add_paragraph(doc, "20% 标签噪声下结果仍稳定；75% 客户端可用时仅小幅下降。25% 可用时 VOC07/NUS 方差与下降明显，应作为部署下限而非稳健性亮点。所有融合/校准参数在参与率冲击前冻结。")
    add_figure(doc, FIGURES / "label_noise_robustness.png", "图 6  成对标签噪声鲁棒性")
    add_figure(doc, FIGURES / "client_participation.png", "图 7  客户端参与率压力测试")

    doc.add_heading("6.3 净化温度敏感性", level=2)
    temp_rows = []
    for dataset in DATASETS:
        rows = [r for r in temperature if r["dataset"] == dataset]
        maps = [float(r["mAP"]) for r in rows]
        temp_rows.append((DISPLAY[dataset], "/".join(r["temperature_multiplier"] for r in rows), f"{min(maps):.4f}", f"{max(maps):.4f}", f"{max(maps)-min(maps):.4f}"))
    add_table(doc, ("数据集", "倍率", "min", "max", "range"), temp_rows, (1700, 2700, 1600, 1600, 1760), font_size=8.5)
    add_paragraph(doc, "温度最大变化仅 0.04 mAP。投稿主文优先使用图 3 的 naive-vs-FedPASA 证据，本表和原温度图移至补充材料。", italic=True, color=GRAY)

    doc.add_heading("6.4 通信压缩与系统边界", level=2)
    comp_rows = []
    for codec in ("fp64", "fp32", "fp16", "int8"):
        values = []
        ratios = []
        for dataset in DATASETS:
            row = next(r for r in compression if r["dataset"] == dataset and r["codec"] == codec)
            values.append(f"{float(row['mAP_delta_vs_fp64']):+.3f}")
            ratios.append(float(row["compression_ratio_vs_fp64"]))
        comp_rows.append((codec.upper(), f"{sum(ratios)/len(ratios):.1f}x", *values))
    add_table(doc, ("codec", "平均压缩比", "VOC07 Δ", "VOC12 Δ", "COCO Δ", "NUS Δ"), comp_rows,
              (1200, 1700, 1600, 1600, 1600, 1660), font_size=8.4)
    add_paragraph(doc, "FP32 将 Ridge packet 减半，绝对 mAP 变化均 <0.00002。FP16 在 COCO/NUS 分别下降 1.376/1.934；朴素 per-tensor INT8 在四个数据集均失败，不能作为可用压缩方案。")
    add_figure(doc, FIGURES / "communication_compression.png", "图 8  通信量-精度前沿（保留失败的低比特边界）")

    doc.add_heading("6.5 长尾、逐类别与定性失败", level=2)
    tail_rows = []
    for dataset in DATASETS:
        value = failures[dataset]
        worst = value["five_lowest_AP"][0]
        tail_rows.append((DISPLAY[dataset], f"{value['rare_quartile_full_AP_mean']:.2f}", f"{value['common_quartile_full_AP_mean']:.2f}", value["classes_hurt_by_fusion"], f"{worst['class_name']} ({worst['fedpasa_full_AP']:.2f})"))
    add_table(doc, ("数据集", "稀有四分位 AP", "常见四分位 AP", "融合受损类数", "最低 AP 类"), tail_rows,
              (1450, 1850, 1900, 1600, 2560), font_size=8.2)
    add_paragraph(doc, "逐类 CSV 覆盖 201 个类别。COCO 的 toaster 最低（14.97 AP），NUS 的 town 最低（25.01 AP）；NUS 稀有类均值明显低于常见类。12 个定性例子分别选择每个数据集最弱类的高分负例、低分正例和高分正例。部分高分负例画面可见目标，提示公开多标签数据可能漏标。")
    add_figure(doc, FIGURES / "tail_failure_analysis.png", "图 9  逐类 AP、训练频次与融合增益")
    add_figure(doc, FIGURES / "qualitative_failure_examples.png", "图 10  四个数据集最弱类别的定性边界案例", width=5.3)

    doc.add_heading("6.6 官方 FedMPT 同机审计", level=2)
    runtime = official["repaired_one_round_diagnostic"]
    audit_rows = [(x["stage"], "是" if x["changes_predictions"] else "否", x["repair"]) for x in official["upstream_blockers"]]
    add_table(doc, ("阻塞阶段", "改变预测", "修复"), audit_rows, (2200, 1400, 5760), font_size=7.8, header_fill="FFF5E5")
    add_paragraph(doc, f"修复后完整 VOC 5,011/4,952 协议完成 1 个诊断轮次：mAP={runtime['global_metrics']['mAP']:.2f}，训练+最终评估 {runtime['training_plus_final_evaluation_seconds']:.2f}s，峰值 CUDA 显存 {runtime['peak_cuda_memory_MiB']:.1f} MiB，双向 prompt 通信 {runtime['estimated_bidirectional_prompt_communication_MiB']:.2f} MiB/round。由于存在改变预测语义的修复，该 mAP 不能作为官方复现进入准确率表；主表继续使用 published external 数值。")

    doc.add_heading("7. 全部实验登记与证据路径", level=1)
    registry = [
        ("E01", "外部比较", "主文 6 个单值；补充 11×3×6", "main_comparison.csv", "PASS"),
        ("E02", "KMeans 异质性", "4×6×3=72", "*/main.json; systems.csv", "PASS"),
        ("E03", "Dirichlet Non-IID", "4×3×3=36", "dirichlet.csv", "PASS"),
        ("E04", "PIAA/QAFed 消融", "4×10=40", "ablation.csv", "PASS"),
        ("E05", "标签特征消融", "4×7=28", "label_feature_ablation.csv", "PASS"),
        ("E06", "三训练划分", "4×3=12", "full_seed_summary.csv", "PASS"),
        ("E07", "image bootstrap", "4×1000", "statistics.json", "PASS"),
        ("E08", "联邦闭式恢复", "4 个数据集", "federated_full_verification.json", "PASS"),
        ("E09", "缺失正标签", "4×5×3=60", "partial_annotation.json/csv", "PASS"),
        ("E10", "成对标签噪声", "4×4×3=48", "label_noise.json/csv", "PASS"),
        ("E11", "客户端参与率", "4×4×3=48", "client_participation.json/csv", "PASS/25% 边界"),
        ("E12", "净化温度", "4×5=20", "temperature_sensitivity.json/csv", "PASS"),
        ("E13", "通信压缩", "4×4=16", "communication_compression.json/csv", "FP32 PASS; INT8 FAIL"),
        ("E14", "逐类别失败", "20+20+80+81=201", "per_class_failure_analysis.json/csv", "PASS"),
        ("E15", "定性失败", "4×3=12", "qualitative_failure_examples.json", "PASS"),
        ("E16", "FedMPT 官方代码", "完整 VOC 1 轮", "official_fedmpt_runnability.json", "诊断完成"),
        ("E17", "最终机器验收", "4 数据集+投稿图", "verification_report.json", "overall_pass=true"),
        ("E18", "Estimator-preserving 对照", "4×4×6=96", "estimator_preserving_evidence.json/csv", "PASS"),
        ("E19", "逐类互补性", "20+20+80+81=201", "classwise_complementarity.json/csv", "PASS"),
        ("E20", "通信/运行时拆分", "4 数据集", "communication_runtime_breakdown.json/csv", "PASS"),
        ("E21", "同协议外部三基线", "目标 3 methods×4 data×3 seeds", "official_fedmpt_runnability.json", "BLOCKED/未完成"),
    ]
    add_table(doc, ("ID", "实验", "规模", "主证据（均在 paper/）", "状态"), registry, (650, 1900, 1500, 3710, 1600), font_size=7.7)
    add_callout(doc, "总量口径", "`annotation_mask_invariance.csv` 是因为 QAFedPASA 不读标签而成立的解析不变性记录，不应重复计入 20 次经验实验。外部 11 个方法均为已发表数值，不是本地训练运行；主文展示其中 6 个代表方法的最佳公开值，完整 11×3×6 放补充材料。")

    doc.add_heading("8. 复现入口与执行顺序", level=1)
    add_paragraph(doc, "推荐使用现有 PyTorch 环境。COCO 的端到端恢复脚本最完整，可作为其他数据集的命令模板。大缓存已存在时脚本会复用产物。")
    add_code(doc, [
        r"$PY = 'D:\Software\anaconda3\envs\pytorch\python.exe'",
        r"& $PY scripts\run_icme_experiments.py --dataset <dataset> --phase all",
        r"& $PY scripts\run_label_ridge.py --dataset <dataset> --features cls patch_mean patch_max_text",
        r"& $PY scripts\run_label_patch_fusion.py --dataset <dataset> --ridge-name label_ridge_cls_patch_mean_patch_max_text",
        r"& $PY scripts\build_cv_fusion_ensemble.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\run_partial_annotation.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\run_reviewer_robustness.py",
        r"& $PY scripts\run_reviewer_federated_stress.py",
        r"& $PY scripts\run_reviewer_evidence.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\verify_federated_full.py --datasets voc07 voc12 coco nus",
        r"& $PY scripts\build_icme_artifacts.py --bootstrap 1000",
        r"& $PY scripts\audit_icme_results.py --require-complete",
        r"& $PY -m pytest -q",
    ])
    workflow_rows = [
        ("特征缓存", "extract_benchmark_features.py; extract_patch_summaries.py", "outputs/icme/cache/<dataset>/"),
        ("QAFed 主实验", "run_icme_experiments.py", "outputs/icme/<dataset>/{main,dirichlet,ablation}.json"),
        ("多视图 Ridge", "run_label_ridge.py", "outputs/icme/<dataset>/label_ridge*/"),
        ("逐类融合", "run_label_patch_fusion.py; build_cv_fusion_ensemble.py", "outputs/icme/<dataset>/cv_fusion/"),
        ("审稿补充实验", "run_partial_annotation.py; run_reviewer_*.py", "estimator/complementarity/communication 等 JSON|CSV|figures"),
        ("汇总验收", "build_icme_artifacts.py; audit_icme_results.py", "artifact_manifest.json; verification_report.json"),
    ]
    add_table(doc, ("阶段", "脚本", "产物"), workflow_rows, (1700, 3600, 4060), font_size=8.2)

    doc.add_heading("9. 论文写作声明边界与后续动作", level=1)
    boundaries = [
        "FedPASA-Full 使用训练标签；只有 QAFedPASA 分支是 label-free。整体可写 backpropagation-free / closed-form，不写 fully training-free。",
        "不传 individual examples 不是差分隐私。没有隐私预算、成员推断或重构攻击实验，不写 formal privacy guarantee。",
        "外部 11 个方法来自 FedMPT Table 1；必须明确 published external。主文 6 个代表方法统一取各自公开设置中的最高 mAP，完整范围与逐 t 数字放补充材料。FedMPT 修复版 47.42 mAP 只放补充材料的可运行性审计。",
        "VOC12 无对应联邦外部表项：不能进入 Published Federated SOTA Comparison，也不能写“所有四个数据集均超过最强外部基线 2 点”；93.98 mAP 仅作完整性报告，并保留在四数据集组件/消融表中。",
        "25% 客户端可用、FP16/INT8 和少数负增益类别是已观测边界，limitations 中保留。",
        "部分 qualitative high-score negatives 实际含有目标，优先解释为预测错误与潜在漏标的混合边界。",
        "QAFed 的 Non-IID 不变性源于全局充分统计恢复；真实客户端掉线的退化由 participation 实验描述。",
        "Estimator-preserving 对照使用真实 local Top-K bank；零熵边界会发生等价 tie 替换，因此 patch-ID recall 可略低于 100%，但适配器/分数误差才是核心恢复判据。",
        "同协议外部三基线仍未完成；在获得 author-equivalent 代码与完整数据前，只能写 published external comparison，不能加 * reproduced under our protocol。",
    ]
    for text in boundaries:
        add_list_item(doc, text, number_id)

    doc.add_heading("9.1 投稿版建议取舍", level=2)
    actions = [
        "主文 SOTA 表：单栏 table[t]，仅 VOC07/COCO/NUS-WIDE，保留 FedPGP、FedTPG、FedAWA、Fed-RAM、FedMVP、FedMPT；每个 baseline 取公开范围最大值；FedMPT 下划线，FedPASA 与 Gain 加粗。",
        "主文组件表：双栏 table*[t]，保留 VOC07/VOC12/COCO/NUS-WIDE 和七个关键变体；这是 VOC12 93.98 的主要表格位置。另保留 central-equivalence/通信拆分表。",
        "主文 Fig. 3(a) 改为 COCO/NUS 的 Naive local-union vs FedPASA under Non-IID；Fig. 3(b) 保留客户端可用率。温度图移至补充材料。",
        "正文增加一小段 classwise complementarity：报告 Full>Ridge/≈/<Ridge 数量和平均正增益，不写所有类别单调受益。",
        "补充材料保留：全部 11 外部基线的范围与逐 t 表、Dirichlet、完整 label/noise/participation、压缩负结果、逐类 CSV 和 FedMPT 审计。",
        "方法段明确三层结构 I1/I2/I3；不要继续把失败的 reliability-only 分支放进贡献列表。",
        "标题/摘要避免承诺 formal privacy 或完全 annotation-free；强调 no iterative backpropagation 和 no individual-example transmission。",
        "最终提交前重新运行 `audit_icme_results.py --require-complete`，并核对 `verification_report.json` 的 reviewer_suite 三项均为 true。",
    ]
    for text in actions:
        add_list_item(doc, text, bullet_id)

    doc.add_heading("10. 来源与交接验收", level=1)
    source_rows = [
        ("PIAA 原稿", str(Path(r"C:\Users\17597\Downloads\CLS_is_Not_Enough_ICML2026_PIAA.pdf"))),
        ("初始创新备忘", str(Path(r"C:\Users\17597\.codex\attachments\529b259a-b6c4-4d4e-96ba-6ca0a7640a22\pasted-text-1.txt"))),
        ("审稿实验原则", "https://maxwellforbes.com/posts/how-to-get-a-paper-accepted/"),
        ("外部联邦表", "https://arxiv.org/html/2605.28347"),
        ("完整实验报告", str(PAPER / "README.md")),
        ("最终验收", str(PAPER / "verification_report.json")),
        ("审稿清单", str(PAPER / "REVIEWER_CHECKLIST.md")),
        ("本轮补实验要求", str(Path(r"C:\Users\17597\.codex\attachments\0fbe7f76-b202-4327-9520-0b12203b685c\pasted-text.txt"))),
    ]
    add_table(doc, ("来源", "位置"), source_rows, (2100, 7260), font_size=8.0)
    final_checks = [
        f"机器验收 overall_pass = {verification['overall_pass']}。",
        "pytest: 4 passed；新增和修改脚本均通过 py_compile。",
        "11 组投稿图均有非空 PNG/PDF；12 个定性样例通过无测试集选参检查。",
        "96 行 estimator-preserving 对照、201 类互补性分析与 4 行通信拆分均通过机器验收。",
        "文档中的关键数字均来自 outputs/icme/paper 下的机器可读 JSON/CSV。",
    ]
    for text in final_checks:
        add_list_item(doc, text, bullet_id)
    add_callout(doc, "交接状态", "P0-2 estimator-preserving、P0-3 classwise complementarity、P1-1 Non-IID 曲线与 P1-2 通信/运行时拆分已经闭环。P0-1 同协议外部三基线仍是明确实验缺口；其余工作为 ICME 页数内的论文取舍、叙事压缩与正式排版。", fill="FFF5E5", color=GOLD)

    doc.core_properties.title = "FedPASA ICME 实验与创新交接文档"
    doc.core_properties.subject = "Innovation, evidence, reproduction, and submission boundaries"
    doc.core_properties.author = "FedPASA project team"
    doc.core_properties.keywords = "FedPASA, QAFedPASA, ICME, multi-label, federated learning, handoff"
    doc.save(DOCX_PATH)


def build_full_markdown_from_docx() -> None:
    """Convert the verified long-form DOCX into the primary Markdown handoff."""
    pandoc = shutil.which("pandoc")
    if pandoc is None:
        raise RuntimeError("pandoc is required to build the complete Markdown handoff")
    subprocess.run(
        [
            pandoc,
            str(DOCX_PATH),
            "--to=gfm",
            "--wrap=none",
            "--extract-media=assets",
            f"--output={MD_PATH.name}",
        ],
        cwd=OUT,
        check=True,
    )
    markdown = MD_PATH.read_text(encoding="utf-8")
    markdown = markdown.replace("assets\\media\\", "assets/media/")
    markdown = markdown.replace("\\_", "_")
    markdown = re.sub(r"\[([^\]]+)\]\{\.underline\}", r"<u>\1</u>", markdown)
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("**主交付：**"):
            lines[index] = (
                "**主交付：**本 Markdown 文件；机器可读证据位于 "
                "`D:\\project\\VScodeProject\\FedPASA\\outputs\\icme\\paper\\`。"
            )
            break
    lines[0:0] = ["# FedPASA ICME 创新与全量实验交接文档", ""]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {
        "main": read_csv(PAPER / "main_comparison.csv"),
        "ablation": read_csv(PAPER / "ablation.csv"),
        "feature": read_csv(PAPER / "label_feature_ablation.csv"),
        "dirichlet": read_csv(PAPER / "dirichlet.csv"),
        "partial": read_csv(PAPER / "partial_annotation.csv"),
        "noise": read_csv(PAPER / "label_noise.csv"),
        "participation": read_csv(PAPER / "client_participation.csv"),
        "temperature": read_csv(PAPER / "temperature_sensitivity.csv"),
        "compression": read_csv(PAPER / "communication_compression.csv"),
        "stats": read_json(PAPER / "statistics.json"),
        "full_verify": read_json(PAPER / "federated_full_verification.json"),
        "failures": read_json(PAPER / "per_class_failure_analysis.json"),
        "official": read_json(PAPER / "official_fedmpt_runnability.json"),
        "verification": read_json(PAPER / "verification_report.json"),
        "estimator": read_json(PAPER / "estimator_preserving_evidence.json"),
        "complementarity": read_json(PAPER / "classwise_complementarity.json"),
        "communication": read_json(PAPER / "communication_runtime_breakdown.json"),
    }
    build_docx(data)
    build_full_markdown_from_docx()
    print(DOCX_PATH)
    print(MD_PATH)


if __name__ == "__main__":
    main()
