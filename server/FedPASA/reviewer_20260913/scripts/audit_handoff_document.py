"""Structural/content checks for the ICME experiment handoff artifacts."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile
import re

from docx import Document
from docx.oxml.ns import qn
from handoff_revision_content import (
    CALIBRATION_SCOPE, COMMUNICATION_SCOPE, ESTIMATOR_NOTE, RECOVERY_SCOPE,
    REPRO_BLOCKS, SERVER_NOTE, STATUS, SUPPLEMENT_COMMANDS,
)


ROOT = Path(__file__).resolve().parents[1]
DOCX = ROOT / "outputs/icme/handoff/FedPASA_ICME_experiment_handoff_zh.docx"
MARKDOWN = ROOT / "outputs/icme/handoff/FedPASA_ICME_experiment_handoff_zh.md"


def main() -> None:
    assert DOCX.is_file() and DOCX.stat().st_size > 100_000
    assert MARKDOWN.is_file() and MARKDOWN.stat().st_size > 30_000

    doc = Document(DOCX)
    paragraph_text = "\n".join(p.text for p in doc.paragraphs)
    table_text = "\n".join(cell.text for table in doc.tables for row in table.rows for cell in row.cells)
    all_text = paragraph_text + "\n" + table_text
    markdown_text = MARKDOWN.read_text(encoding="utf-8")

    required = (
        "2.1 I1 - 全局分位数对齐的 QAFedPASA", "2.2 I2 - 多视图标签感知联邦 Ridge",
        "2.3 I3 - 无测试泄漏的逐类别融合",
        "ICME 主文 SOTA 表：单值、三数据集、六个代表方法",
        "VOC12 与组件表的安排",
        "补充材料：全部 11 个公开外部基线",
        "同协议外部 baseline 的实际状态",
        "4.4 Naive federation 与 estimator-preserving federation",
        "5.5 Ridge-only 与 QAFedPASA 的逐类互补",
        "94.22", "80.53", "60.67", "+4.09", "+13.16", "+2.31",
        "For methods reported under multiple settings, we conservatively use their best published result.",
        "label-aware", "published external", "formal privacy guarantee",
        "overall_pass", "reliability+shrinkage",
        "P0-1 可信同协议强基线仍有证据缺口",
        "96 行 estimator-preserving 对照",
    )
    missing = [token for token in required if token not in all_text]
    assert not missing, f"missing DOCX content: {missing}"
    for token in (CALIBRATION_SCOPE, COMMUNICATION_SCOPE, ESTIMATOR_NOTE,
                  RECOVERY_SCOPE, SERVER_NOTE, STATUS):
        assert token in all_text, f"missing DOCX evidence boundary: {token}"
        assert token in markdown_text, f"missing Markdown evidence boundary: {token}"
    for title, code in REPRO_BLOCKS:
        assert title in paragraph_text and title in markdown_text
        assert code in paragraph_text, f"incomplete Word commands: {title}"
        assert f"```powershell\n{code}\n```" in markdown_text, f"incomplete Markdown commands: {title}"
    assert SUPPLEMENT_COMMANDS in paragraph_text and SUPPLEMENT_COMMANDS in markdown_text
    for stale in ("--dataset <dataset>", "pytest: 4 passed", "reviewer_suite 三项",
                  "零熵边界 tie 的等价替换", "--seed 7 --workers 2", "FedMPT 待完成", "服务器进度待核验"):
        assert stale not in all_text and stale not in markdown_text, f"stale handoff content: {stale}"
    for token in ("69.806112", "35.823527", "配对统计和独立重复", "12 ICME 2027", "9/9"):
        assert token in all_text and token in markdown_text, f"missing final handoff update: {token}"

    for idx in range(1, 22):
        token = f"E{idx:02d}"
        assert token in table_text, f"missing registry row {token} in DOCX"
        assert token in markdown_text, f"missing registry row {token} in Markdown"
    for idx in range(1, 10):
        token = f"S{idx:02d}"
        assert token in table_text and token in markdown_text, f"missing supplement: {token}"

    markdown_evidence = (
        "main_comparison.csv", "systems.csv", "dirichlet.csv", "ablation.csv",
        "label_feature_ablation.csv", "full_seed_summary.csv", "statistics.json",
        "partial_annotation.json", "label_noise.json", "client_participation.json",
        "temperature_sensitivity", "communication_compression",
        "per_class_failure_analysis", "qualitative_failure_examples.json",
        "official_fedmpt_runnability.json", "verify_federated_full.py",
        "estimator_preserving_evidence", "classwise_complementarity",
        "communication_runtime_breakdown", "run_reviewer_evidence.py",
        "audit_icme_results.py",
    )
    missing_markdown = [token for token in markdown_evidence if token not in markdown_text]
    assert not missing_markdown, f"missing Markdown evidence: {missing_markdown}"
    assert markdown_text.startswith("# FedPASA ICME 创新与全量实验交接文档\n")
    for idx in range(1, 11):
        relative = f"assets/media/image{idx}.png"
        assert relative in markdown_text, f"missing Markdown image reference: {relative}"
        assert (MARKDOWN.parent / relative).is_file(), f"missing Markdown image asset: {relative}"

    placeholders = ("TODO", "TBD", "FIXME", "XXX")
    found_placeholders = [token for token in placeholders if token in all_text.upper()]
    assert not found_placeholders, f"placeholders found: {found_placeholders}"

    assert len(doc.inline_shapes) == 10
    for shape in doc.inline_shapes:
        doc_pr = shape._inline.docPr
        assert doc_pr.get("descr") and doc_pr.get("title"), "inline figure missing alt text"

    data_tables = 0
    layout_tables = 0
    for table in doc.tables:
        if len(table.columns) == 1:
            layout_tables += 1
            continue
        data_tables += 1
        tr_pr = table.rows[0]._tr.get_or_add_trPr()
        assert tr_pr.find(qn("w:tblHeader")) is not None, "data table missing repeating header"
    # Check tables by schema and their evidence rows, not a brittle total count.
    registry = next(t for t in doc.tables if t.rows[0].cells[0].text == "ID")
    assert {row.cells[0].text for row in registry.rows[1:]} == {f"E{i:02d}" for i in range(1, 22)}
    supplements = next(t for t in doc.tables if [c.text for c in t.rows[0].cells] == ["编号", "实验", "优先级", "当前结论"])
    assert {row.cells[0].text for row in supplements.rows[1:]} == {f"S{i:02d}" for i in range(1, 10)}

    # A zero in the legacy table must remain visibly identified as reused data.
    legacy = next(t for t in doc.tables if t.rows[0].cells[-1].text == "记录误差*")
    assert len(legacy.rows) == 9
    assert all(row.cells[-1].text == "0.0e+00" for row in legacy.rows[1:])

    comments = list(doc.comments)
    anchors = {int(el.get(qn("w:id"))) for el in doc.element.body.iter(qn("w:commentRangeStart"))}
    for comment in comments:
        assert comment.comment_id in anchors, f"unanchored comment: {comment.comment_id}"

    sota_table = next(
        table for table in doc.tables
        if [cell.text for cell in table.rows[0].cells] == ["Method", "VOC07", "COCO", "NUS-WIDE"]
    )
    assert len(sota_table.rows) == 9
    assert [cell.text for cell in sota_table.rows[6].cells] == ["FedMPT", "90.13", "67.37", "58.36"]
    for cell in sota_table.rows[6].cells[1:]:
        assert any(run.underline for paragraph in cell.paragraphs for run in paragraph.runs)
    for row_index in (7, 8):
        for cell in sota_table.rows[row_index].cells:
            assert all(run.bold for paragraph in cell.paragraphs for run in paragraph.runs if run.text)

    component_table = next(
        table for table in doc.tables
        if [cell.text for cell in table.rows[0].cells] == ["Variant", "VOC07", "VOC12", "COCO", "NUS-WIDE"]
    )
    assert len(component_table.rows) == 8
    assert [cell.text for cell in component_table.rows[-1].cells] == ["FedPASA-Full", "94.22", "93.98", "80.53", "60.67"]

    with ZipFile(DOCX) as archive:
        xml = archive.read("word/document.xml")
    assert xml.count(b"<wp:inline") == 10
    assert b"<wp:anchor" not in xml

    print(
        "PASS | paragraphs=%d tables=%d data_tables=%d layout_tables=%d "
        "inline_figures=%d markdown_images=10 registry=E01-E21"
        % (len(doc.paragraphs), len(doc.tables), data_tables, layout_tables, len(doc.inline_shapes))
    )


if __name__ == "__main__":
    main()
