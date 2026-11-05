"""Paired results, target costs, and measured resource frontiers for the queued jobs."""
from pathlib import Path
import json,csv,collections
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from strengthen_queue import ROOT,OUT,spec,save

def csvout(p,rows):
    with p.open("w",newline="") as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    groups=collections.defaultdict(list);ops=[];costs=[]
    for j in spec():
        val=json.loads((OUT/j["id"]/"result.json").read_text());d=j["d"]
        if j["kind"]=="fedavg":groups[(d,j["condition"],"strong FedAvg - Full1")].append(val["delta_vs_full1"])
        elif j["kind"]=="rf":
            hist=json.loads((ROOT/"outputs/supplement_20261029/runs"/f'{d}_iid_m10_s{j["seed"]}'/"result.json").read_text())
            ref=next(q["mAP"] for q in hist["rows"] if q["method"]=="Full_1")
            for q in val["rows"]:groups[(d,q["view"],"RF - Full1")].append(q["mAP"]-ref)
        elif j["kind"]=="partial":
            q=val["rows"][0]
            for label,value in [("Full3 mAP",val["global_0_three_mAP"]),("Full3 - active central3",val["global_0_three_mAP"]-val["central_three_mAP"]),("Full3 - active Ridge3",val["global_0_three_mAP"]-val["ridge_three_mAP"])]:groups[(d,str(j["fraction"]),label)].append(value)
            for label,value in [("Full mAP",q["mAP"]),("Full - active central",q["mAP"]-val["central_full_mAP"]),("Full - active Ridge",q["mAP"]-val["ridge_mAP"])]:groups[(d,str(j["fraction"]),label)].append(value)
        elif j["kind"]=="union":
            ref=next(q["mAP"] for q in val["rows"] if q["key"]=="global_0")
            for q in val["rows"]:groups[(d,q["key"],"fusion - global unclamped")].append(q["mAP"]-ref)
        else:
            net=val["network"]
            base=dict(dataset=d,network=j["network"],mode=j["mode"],method=j["method"],repeat=j["repeat"])
            for q in val.get("points",[]):ops.append(dict(**base,stage=str(q["round"]),mAP=q["mAP"],seconds=q["seconds"],MiB=q["framed_bytes"]/2**20))
            ops.append(dict(**base,stage="final",mAP=val["mAP"],seconds=val["total_fitting_wall_seconds"],MiB=(net["uplink_framed_bytes"]+net["downlink_framed_bytes"])/2**20))
    out=OUT/"report";out.mkdir(exist_ok=True)
    summary=[]
    for (d,cond,metric),values in groups.items():
        mean=float(np.mean(values));n=len(values);sd=float(np.std(values,ddof=1));half=float(t.ppf(.975,n-1)*sd/np.sqrt(n))
        summary.append(dict(dataset=d,condition=cond,metric=metric,n=n,mean=mean,sd=sd,ci95_lo=mean-half,ci95_hi=mean+half))
    csvout(out/"accuracy.csv",summary);csvout(out/"operating_points.csv",ops)
    for d in ("coco","nus"):
        for network in ("loopback","100Mbps_10ms","10Mbps_50ms"):
            for mode in ("inclusive","locked"):
                rows=[q for q in ops if (q["dataset"],q["network"],q["mode"])==(d,network,mode)]
                # Fixed checkpoints only; aggregate repeat timing before comparing methods.
                by=collections.defaultdict(list)
                for q in rows:by[q["method"],q["stage"]].append(q)
                avg=[dict(method=m,stage=s,**{k:float(np.mean([q[k] for q in rr])) for k in ("seconds","MiB","mAP")}) for (m,s),rr in by.items()]
                full=next(q for q in avg if q["method"]=="full1" and q["stage"]=="final")
                fig,axes=plt.subplots(1,2,figsize=(8,3.4))
                for method in ("full1","full3","ridge","fedavg"):
                    points=[p for p in avg if p["method"]==method]
                    for axis,key in zip(axes,("MiB","seconds")):
                        ordered=sorted(points,key=lambda p:p[key]);axis.plot([p[key] for p in ordered],[p["mAP"] for p in ordered],".-",label=method)
                        axis.set_xlabel(key);axis.set_ylabel("mAP")
                        feasible=[p for p in points if p[key]<=full[key]]
                        target=[p for p in points if p["mAP"]>=full["mAP"]]
                        costs.append(dict(dataset=d,network=network,mode=mode,method=method,resource=key,full1_budget=full[key],target_mAP=full["mAP"],best_mAP_within_budget=max((p["mAP"] for p in feasible),default=None),least_cost_attaining_target=min((p[key] for p in target),default=None)))
                axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/f"{d}_{network}_{mode}.pdf");plt.close(fig)
    csvout(out/"matched_budgets_targets.csv",costs)

if __name__=="__main__":main()
