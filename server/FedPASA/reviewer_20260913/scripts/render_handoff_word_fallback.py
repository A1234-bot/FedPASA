"""Use the packaged DOCX rasterizer with a Word-exported PDF on Windows.

The package lacks LibreOffice here. Its PDF-to-PNG and page naming logic remain
unchanged; only the conversion stage is supplied by Word's native PDF export.
"""
import importlib.util
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SKILL=Path('C:/Users/17597/.codex/plugins/cache/openai-primary-runtime/documents/26.905.11957/skills/documents')
QA=ROOT/'tmp/handoff_supplement_20260907'
os.environ['PATH']='C:/Users/17597/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin'+os.pathsep+os.environ.get('PATH','')
spec=importlib.util.spec_from_file_location('packaged_docx_renderer',SKILL/'render_docx.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)


def word_pdf(input_path,user_profile,convert_tmp_dir,stem,**kwargs):
    target=Path(convert_tmp_dir)/(stem+'.pdf')
    shutil.copy2(QA/'word_render.pdf',target)
    return str(target),'Word native ExportAsFixedFormat; packaged LibreOffice unavailable'


renderer.convert_to_pdf=word_pdf
renderer.rasterize(str(ROOT/'outputs/icme/handoff/FedPASA_ICME_experiment_handoff_zh.docx'),str(QA/'pages'),120,verbose=False,emit_pdf=False)
print('Rendered',len(list((QA/'pages').glob('page-*.png'))),'pages')
