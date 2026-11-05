"""Resume original B/16 patch extraction at 512-image boundaries on the data volume."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/"rebuild_blocks_v1"
INVENTORY=json.loads((ROOT/"inputs.json").read_text())

def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest() if hasattr(hashlib,"file_digest") else stream_hash(stream)
def stream_hash(stream):
    value=hashlib.sha256()
    for data in iter(lambda:stream.read(4<<20),b""):value.update(data)
    return value.hexdigest()

def recover(dataset,part):
    import numpy as np
    import strengthen_rebuild as original
    from torch.utils.data import DataLoader
    from torch.nn import functional as F
    import torch
    paths=original.resolve(dataset,part)
    weights=Path("/root/BCP/external/FedMPT-main/ViT-B-16.pt")
    assert sha(weights)==json.loads((ROOT/"outputs/icme/cache"/dataset/"manifest.json").read_text())["weight_sha256"]
    rel=f"outputs/icme/cache/{dataset}/{part}_patch.npy"
    output=ROOT/rel
    expected=INVENTORY[rel]["sha256"]
    if output.exists() and sha(output)==expected:return
    work=STORE/dataset/part
    work.mkdir(exist_ok=True,parents=True)
    model,transform=original.scclip.load(str(weights),device="cuda",jit=False)
    model.eval()
    count=len(paths);block_size=512
    with torch.inference_mode():
        for start in range(0,count,block_size):
            end=min(start+block_size,count)
            target=work/f"{start:07d}.raw"
            if target.is_file() and target.stat().st_size==(end-start)*196*512*2:continue
            dataset_slice=original.Images(paths[start:end],transform)
            scratch=target.with_suffix(".writing")
            with scratch.open("wb") as handle:
                for images in DataLoader(dataset_slice,batch_size=64,shuffle=False,num_workers=0):
                    features=model.encode_image(images.to("cuda"),return_all=True).float()
                    arr=F.normalize(features,dim=-1).cpu().half().numpy()
                    handle.write(arr.tobytes(order="C"))
                handle.flush();os.fsync(handle.fileno())
            assert scratch.stat().st_size==(end-start)*196*512*2
            scratch.replace(target)
            print("CHECKPOINT",dataset,part,end,count,flush=True)
    import shutil
    scratch=work/"assembled.npy"
    with scratch.open("wb") as out:
        np.lib.format.write_array_header_1_0(out,dict(shape=(count,196,512),fortran_order=False,descr=np.dtype(np.float16).str))
        for start in range(0,count,block_size):
            with (work/f"{start:07d}.raw").open("rb") as inp:shutil.copyfileobj(inp,out,4<<20)
    actual=sha(scratch)
    if actual!=expected:
        print("HASH_MISMATCH",dataset,part,actual,expected,flush=True)
        sys.exit(8)
    scratch.replace(output)
    print("VERIFIED",rel,flush=True)

def main():
    STORE.mkdir(exist_ok=True)
    lock=(STORE/"lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for d,p in [("nus","train"),("coco","test"),("coco","train"),("nus","test")]:
        for attempt in range(3):
            run=subprocess.run([sys.executable,"-X","faulthandler",__file__,d,p],cwd=ROOT)
            if run.returncode==0:break
            print("ATTEMPT_FAILED",d,p,run.returncode,flush=True)
            if run.returncode==8:break
            time.sleep(3)
    for key,v in INVENTORY.items():
        if key.endswith("_patch.npy"):
            if not (ROOT/key).exists() or sha(ROOT/key)!=v["sha256"]:raise RuntimeError("Await archive for "+key)
    temp=ROOT/"patch.ready.blocks";temp.write_text("All four tensors verified against release SHA256");temp.replace(ROOT/"patch.ready")
    print("ALL_VERIFIED",flush=True)
    import psutil
    for proc in psutil.process_iter(["pid","cmdline"]):
        try:
            cmd=proc.info["cmdline"] or []
            if cmd[-1:]==["restore.py"] and Path(proc.cwd()).resolve()==ROOT and "python" in Path(cmd[0]).name:proc.terminate()
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass

if __name__=="__main__":
    if len(sys.argv)==3:recover(*sys.argv[1:])
    else:main()
