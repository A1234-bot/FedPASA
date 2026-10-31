"""Collect measured system costs and independent baseline training repeats."""
import platform,sys
import numpy as np
import torch
from icme2027_common import *

def main():
    rows=[];missing=[]
    for d in ['coco','nus']:
        ep=OUT/'extraction'/d/'result.json'
        cpaths=[OUT/'calibration'/d/f'seed{s}_bins8192/result.json' for s in [7,17,27]]
        if not ep.exists() or not all(p.exists() for p in cpaths):
            missing.append(f'{d}: extraction/calibration');continue
        e=read(ep);cal=[read(p) for p in cpaths]
        rows.append({'dataset':d,'method':'Full histogram-calibration variant, three training splits',
            'cold_feature_extraction_seconds':e['seconds_wall'],
            'warm_feature_cache_fit_calibration_selection_inference_seconds':sum(r['seconds'] for r in cal),
            'cold_total_seconds':e['seconds_wall']+sum(r['seconds'] for r in cal),
            'peak_training_cuda_MiB':max(r['peak_cuda_MiB'] for r in cal),
            'uplink_payload_bytes':sum(r['application_payload_uplink_bytes'] for r in cal),
            'downlink_payload_bytes':sum(r['application_payload_downlink_bytes'] for r in cal),
            'stage_seconds':{k:sum(r.get('timing_seconds',{}).get(k,0.) for r in cal) for k in cal[0].get('timing_seconds',{})},
            'scope':'Measured single-server protocol simulation with cached CLIP arrays; all three splits/candidates/refits/inference and independent central audit computations included; conservative audit-run cost, not production latency or physical network RTT'})
    repeats=[]
    for d in ['coco','nus']:
        paths=[OUT/'baselines/fedmvp_release_mlc'/d/f'seed{s}/result.json' for s in [7,17,27]]
        if not all(p.exists() for p in paths):missing.append(f'{d}: 3 baseline seeds');continue
        v=[read(p) for p in paths];acc=[r['mAP'] for r in v]
        repeats.append({'dataset':d,'training_seeds':[7,17,27],'fixed_partition_seed':7,'mAP':acc,'mean':float(np.mean(acc)),'sample_std':float(np.std(acc,ddof=1))})
        for seed,r in zip([7,17,27],v):rows.append({'dataset':d,'method':'FedMVP released multilabel adaptation','training_seed':seed,
            'training_seconds':r['training_seconds'],'evaluation_seconds':r['evaluation_seconds'],'uplink_payload_bytes':r['uplink_bytes'],'downlink_payload_bytes':r['downlink_bytes']})
    save(OUT/'systems_and_repeats.json',{'status':'COMPLETE' if not missing else 'PARTIAL','rows':rows,'baseline_repeats':repeats,'missing':missing,
        'hardware':{'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'python':sys.version,'platform':platform.platform()},
        'comparability':'Cold and warm Full costs reported separately. Existing repaired FedMPT mixed-hardware historical runtime is excluded from the uniform benchmark.',
        'completed':stamp()})
    lines=['# ICME 2027 服务器补实验状态','',f'更新时间：{stamp()}','',f'系统与重复统计：{"完成" if not missing else "等待队列完成"}。',
        '','所有新产物位于独立 outputs/icme2027；原9项补充实验不覆盖。',
        '','- 分布式校准：10个逻辑客户端、固定8192 bins近似AP，必须披露近似与隐私边界。',
        '- 配对统计：四数据集、1000次共同图像重采样，合并同分阈值。',
        '- 强基线：FedMPT发布仓库内FedMVP多标签适配版，源码核对后在COCO/NUS各跑seed7/17/27、200轮。',
        '- 系统计时：同一5090上的无特征缓存提取与有缓存拟合/校准/推理、双向消息负载。',
        '- 完成条件：queue.json全部COMPLETE且实验报告逐项验收；进程退出不能单独证明成功。',
        '','## 实现边界','',
        'FedMPT原论文与当前发布代码在条件、视觉适配器、OT、概率/损失定义上不一致；没有把改动OT开关视为作者等价复现。',
        '新增FedMVP明确称FedMPT发布仓库的多标签适配基线，不称原ICCV算法严格复现。分布式校准是单机客户端消息仿真，不是密码学安全聚合。']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
if __name__=='__main__':main()
