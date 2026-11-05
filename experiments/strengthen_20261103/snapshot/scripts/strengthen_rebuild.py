"""Rebuild the exact four archived tensors on original hardware; publish only matching hashes."""
from pathlib import Path
import hashlib,json,os,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"third_party/PIAA"))
import packaging,packaging.version,pkg_resources
pkg_resources.packaging=packaging
import scclip
import torch
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from extract_benchmark_features import extract_array,sha256

class Images(Dataset):
    def __init__(self,paths,transform):self.paths=paths;self.transform=transform
    def __len__(self):return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im:return self.transform(im.convert("RGB"))

def resolve(d,part):
    ids=(ROOT/"outputs/icme/cache"/d/f"{part}_ids.txt").read_text().splitlines()
    if d=="nus":
        base=Path("/root/BCP/data/raw/nuswide/kaggle_tc21/NUS-WIDE/images")
        return [base/(Path(i).name+".jpg") for i in ids]
    base=Path("/root/BCP/data/raw/coco2017")
    paths=[]
    for i in ids:
        n=i.split("_")[-1]+".jpg"
        candidates=[base/"train2017"/n,base/"val2017"/n,ROOT/"data/coco2014"/("train2014" if part=="train" else "val2014")/(i+".jpg")]
        paths.append(next((p for p in candidates if p.is_file()),candidates[-1]))
    return paths

def main():
    inv=json.loads((ROOT/"inputs.json").read_text())
    wt=Path("/root/BCP/external/FedMPT-main/ViT-B-16.pt")
    assert sha256(wt)==json.loads((ROOT/"outputs/icme/cache/coco/manifest.json").read_text())["weight_sha256"]
    model,transform=scclip.load(str(wt),device="cuda",jit=False);model.eval()
    for d in ("nus","coco"):
        for part in ("test","train"):
            rel=f"outputs/icme/cache/{d}/{part}_patch.npy";expected=inv[rel]["sha256"];p=ROOT/rel
            if p.is_file() and sha256(p)==expected:continue
            paths=resolve(d,part);missing=[str(p) for p in paths if not p.is_file()]
            if missing:
                (ROOT/f"missing_{d}_{part}.json").write_text(json.dumps(missing))
                print("MISSING",d,part,len(missing),flush=True);continue
            tmp=ROOT/"rebuild"/d/p.name;tmp.parent.mkdir(parents=True,exist_ok=True)
            data=Images(paths,transform)
            if not tmp.exists():extract_array(model,DataLoader(data,batch_size=64,num_workers=4,pin_memory=True,shuffle=False),tmp,"patch","cuda")
            actual=sha256(tmp)
            if actual==expected:tmp.replace(p);print("VERIFIED",rel,flush=True)
            else:print("MISMATCH",rel,actual,expected,flush=True)
    if all((ROOT/k).is_file() and sha256(ROOT/k)==v["sha256"] for k,v in inv.items() if k.endswith("_patch.npy")):
        (ROOT/"patch.ready").write_text("All four archive SHA-256 verified by reconstruction.")
        print("ALL_PATCH_READY",flush=True)

if __name__=="__main__":main()
