"""Structural/content checks for the ICME experiment handoff artifacts."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml.ns import qn


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
        "3.1 ICME 主文 SOTA 表：单值、三数据集、六个代表方法",
        "3.2 VOC12 与组件表的安排",
        "3.3 补充材料：全部 11 个公开外部基线",
        "3.4 同协议外部 baseline 的实际状态",
        "4.4 Naive federation 与 estimator-preserving federation",
        "5.5 Ridge-only 与 QAFedPASA 的逐类互补",
        "94.22", "80.53", "60.67", "+4.09", "+13.16", "+2.31",
        "For methods reported under multiple settings, we conservatively use their best published result.",
        "label-aware", "published external", "formal privacy guarantee",
        "overall_pass", "reliability+shrinkage",
        "P0-1 同协议外部三基线仍是明确实验缺口",
        "96 行 estimator-preserving 对照",
    )
    missing = [token for token in required if token not in all_text]
    assert not missing, f"missing DOCX content: {missing}"

    for idx in range(1, 22):
        token = f"E{idx:02d}"
        assert token in table_text, f"missing registry row {token} in DOCX"
        assert token in markdown_text, f"missing registry row {token} in Markdown"

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
    assert data_tables == 31 and layout_tables == 15

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
