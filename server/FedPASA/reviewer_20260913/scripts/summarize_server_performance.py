"""Summarize measured performance without treating cache effects as code speedup."""
import json
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'tmp/server_performance'


def main():
    warm=json.loads((P/'end_to_end_control_reverse.json').read_text())
    first=json.loads((P/'end_to_end_control.json').read_text())
    facts=[]
    for dataset in ('coco','nus'):
        base=next(r for r in warm if r['dataset']==dataset and r['variant']=='baseline')
        opt=next(r for r in warm if r['dataset']==dataset and r['variant']=='optimized')
        cold=next(r for r in first if r['dataset']==dataset and r['variant']=='baseline')
        variants=json.loads((P/f'{dataset}_benchmark.json').read_text())
        facts.append(dict(dataset=dataset,warm_baseline_seconds_per_batch=base['seconds_per_step'],
            warm_optimized_seconds_per_batch=opt['seconds_per_step'],warm_speedup=base['seconds_per_step']/opt['seconds_per_step'],
            first_pass_baseline_seconds_per_batch=cold['seconds_per_step'],
            workers=opt['workers'],cpu_threads=opt['cpu_threads'],peak_allocated_MiB=max(x['peak_allocated_MiB'] for x in variants),
            controls_102_steps_bitwise_equal=all(r['final_parameters_bitwise_equal'] for r in warm+first if r['dataset']==dataset)))
    result=dict(updated_at=datetime.now().astimezone().isoformat(),status='DEPLOYED',facts=facts,
        preserved=['200 rounds','batch size 32','10 clients','partition seed 7','original image IDs, targets and partitions','BF16','SGD and learning-rate schedule'],
        changes=['disable activation recomputation','cache frozen text tokenization/embeddings only','combine finite-gradient checks','one compute thread','4 COCO / 8 NUS loader workers'],
        limitation='First-pass 4-5x comparison includes cache warming and is not a causal code speedup. Reversed-order warm controls establish about 1.43x. Full-client production observations are required for a final ETA.',
        log='outputs/server_logs/fedmpt_optimized_20260907.log',session='fedpasa_fedmpt',
        checkpoint_protection='Formal checkpoint hashes unchanged across benchmarks; pre-optimization copies retained in tmp/server_performance/checkpoints_before.')
    output=ROOT/'outputs/icme/supplementary/server_performance.json'
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
