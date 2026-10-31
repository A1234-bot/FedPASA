"""Read-only numerical and completion audit of the downloaded 200-round runs."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
import torch
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
SUPP = ROOT / 'outputs/icme/supplementary'
HASHES = {
 'coco': {'train_targets.npy':'4b07523744570723c4dedf96288cbf1e0c173604209b8320aafc4ba1a2d720d9',
          'test_targets.npy':'ab75d66a93121f256887cd1e24027b94e0d7b0f18ddec13a8bc17b1f3aa1b250',
          'train_ids.txt':'2aee04f29ea9962f5988daddf8fe74e355a8db22164b54b9454bde04d96c9369',
          'test_ids.txt':'440f6bdb3cb4157de1c0a2eaf642b8734a442889c1f041a3f8e8c0b7ad9af334'},
 'nus': {'train_targets.npy':'9aed687b4f3fda9a65ab6697789a38928e0b3db5967c6743336732359f6ee74d',
         'test_targets.npy':'4667e6497058961bad145ea9278e9301bd13d2d4a9611563054c266db08d8d8d',
         'train_ids.txt':'2612ee1a37ceb23c962050c3e0ef3ad9362d95221b93c68cc1b9f1f21459eaa8',
         'test_ids.txt':'2466894f88abcae7f2b73063f8262960116aaf117c860723dc81958b1ecc5940'}}

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    report = {'checked_at': datetime.datetime.now().astimezone().isoformat(), 'datasets': {},
              'scope': 'Saved-score metric recomputation and checkpoint/round integrity; no new inference or training.'}
    for d in ('coco', 'nus'):
        p = SUPP/d/'fedmpt_repaired'; cache = ROOT/'outputs/icme/cache'/d
        for name, expected in HASHES[d].items():
            assert hashlib.sha256((cache/name).read_bytes()).hexdigest() == expected, (d,name)
        s = torch.load(p/'checkpoint.pt', map_location='cpu', weights_only=False)
        r = read(p/'result.json'); progress = read(p/'progress.json')
        assert s['completed_round'] == s['target_rounds'] == r['round'] == progress['completed_round'] == 200
        assert r['status'] == progress['status'] == 'COMPLETE'
        assert all(torch.isfinite(v).all() for v in s['global'].values())
        targets = np.load(cache/'test_targets.npy'); scores = np.load(p/'round200_test_scores.npy')
        assert scores.shape == targets.shape and np.isfinite(scores).all()
        ap = np.array([100*average_precision_score(targets[:,c],scores[:,c]) for c in range(scores.shape[1])])
        assert abs(ap.mean()-r['mAP']) < 1e-10
        partition = ROOT/'outputs/icme'/d/'partitions/clients10_seed7.npy'
        assert hashlib.sha256(partition.read_bytes()).hexdigest() == r['protocol']['partition_sha256']
        assignment = np.load(partition); train_count = len(np.load(cache/'train_targets.npy'))
        curve = []; total_seconds = 0.; total_steps = 0
        for i in range(1,201):
            row = read(p/f'round{i:03d}.json'); clients=row['client_losses']
            assert row['round'] == i and len(clients) == 10 and {x['client'] for x in clients} == set(range(10))
            assert sum(x['images'] for x in clients) == train_count
            assert all(x['images'] == int((assignment==x['client']).sum()) for x in clients)
            assert all(np.isfinite(x['loss_sum']) and x['finite_optimizer_steps'] > 0 for x in clients)
            total_seconds += row['train_seconds']; total_steps += sum(x['finite_optimizer_steps'] for x in clients)
            if 'mAP' in row: curve.append({'round':i,'mAP':row['mAP']})
        report['datasets'][d] = {'status':'COMPLETE', 'round':200, 'mAP':r['mAP'],
          'recomputed_mAP':float(ap.mean()), 'metric_error':float(ap.mean()-r['mAP']),
          'prediction_shape':list(scores.shape), 'round_records':200, 'client_round_records':2000,
          'finite_optimizer_steps':total_steps, 'recorded_train_hours':total_seconds/3600,
          'all_parameters_and_predictions_finite':True,'labels_ids_partition_hashes_match_server':True,
          'evaluation_curve':curve, 'official_reproduction':False,
          'protocol_status_note':'Nested protocol status RUNNING is a construction-time snapshot; final result/progress/checkpoint are COMPLETE/200.'}
    completion=read(SUPP/'completion_audit.json')
    assert completion['status']=='COMPLETE' and len(completion['items'])==9
    assert all(x['status']=='COMPLETE' for x in completion['items'])
    report.update(status='PASS', completed_items=9, completed_at=completion['updated_at'])
    (SUPP/'final_result_audit_20260910.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
