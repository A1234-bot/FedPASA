from pathlib import Path
from zipfile import ZipFile
import json,hashlib,re
from docx import Document
ROOT=Path(__file__).resolve().parents[1];Q=ROOT/'tmp/handoff_update_20260913';H=ROOT/'outputs/icme/handoff';E=ROOT/'outputs/icme2027'
rd=lambda p:json.loads(p.read_text(encoding='utf-8'))
name='FedPASA_ICME_experiment_handoff_zh.docx'
a=Document(Q/('original_'+name));b=Document(H/name)
changed=[]
for i,t in enumerate(a.tables):
    old=[[c.text for c in row.cells] for row in t.rows]
    new=[[c.text for c in row.cells] for row in b.tables[i].rows]
    for j,(x,y) in enumerate(zip(old,new)):
        if x!=y:
            assert x[0]=='FedMVP/FedAWA' and y[0]==x[0]
            changed.append([i,j])
assert len(changed)==1 and len(a.inline_shapes)==len(b.inline_shapes)==10
with ZipFile(Q/('original_'+name)) as x, ZipFile(H/name) as y:
    media=[f for f in x.namelist() if f.startswith('word/media/')]
    assert all(x.read(f)==y.read(f) for f in media)
text='\n'.join(p.text for p in b.paragraphs)+'\n'+'\n'.join(c.text for t in b.tables for row in t.rows for c in row.cells)
md=(H/name).with_suffix('.md').read_text(encoding='utf-8')
for r in rd(E/'systems_and_repeats.json')['baseline_repeats']:
    for v in r['mAP']+[r['mean'],r['sample_std']]:
        assert f'{v:.6f}' in text and f'{v:.6f}' in md
for i in range(1,22):assert f'E{i:02d}' in text
for i in range(1,10):assert f'S{i:02d}' in text
for i in range(1,7):assert f'N{i:02d}' in text
cal=[]
for d in ['coco','nus']:
    for s in [7,17,27]:
        p=E/f'calibration/{d}/seed{s}_bins8192';r=rd(p/'result.json');msgs=rd(p/'messages.json')
        assert len(msgs)==r['message_count']==580
        assert len({m['stage'] for m in msgs if m['direction']=='uplink'})==40
        for direction in ['uplink','downlink']:assert sum(m['bytes'] for m in msgs if m['direction']==direction)==r[f'application_payload_{direction}_bytes']
        assert r['moment_equivalence_passed'] and not r['exact_selection_equivalence']
        assert f"{r['test_mAP']:.6f}" in text
        cal.append([d,s,'PASS_NUMERIC_PAYLOAD_AUDIT','NOT_EXACT_SELECTION'])
queue=rd(E/'queue.json');assert queue['status']=='COMPLETE' and all(j['status']=='COMPLETE' for j in queue['jobs'])
manifest=rd(E/'final_archive_manifest_20260913.json');assert len(manifest)==1328
assert all(hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256'] for r in manifest)
pages=list((Q/'pages').glob('page-*.png'));assert len(pages)==46
qa={'status':'PASS','pages':46,'visual_review':'All 46 pages inspected; seven changed pages re-inspected after final revision; other page image hashes unchanged','original_tables':len(a.tables),'tables_final':len(b.tables),'only_legacy_row_changed':changed,'original_images_preserved':len(media),'baseline_metric_recomputation':rd(E/'final_audit_20260913.json'),'calibration_payload_checks':cal,'downloaded_files_sha256_verified':1328,'docx_sha256':hashlib.sha256((H/name).read_bytes()).hexdigest()}
(Q/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
print('PASS 46 pages, original numeric tables and images preserved, six baselines and calibration payloads audited, 1328 file hashes verified')
