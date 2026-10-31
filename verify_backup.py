"""Verify downloaded FedPASA release volumes and every archived file. No writes."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tarfile


def digest(stream):
    h=hashlib.sha256()
    for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


class Volumes(io.RawIOBase):
    def __init__(self,paths):self.paths=iter(paths);self.current=None
    def readable(self):return True
    def read(self,n=-1):
        if n<0:raise ValueError('Use bounded reads')
        out=bytearray()
        while len(out)<n:
            if self.current is None:
                try:self.current=next(self.paths).open('rb')
                except StopIteration:break
            data=self.current.read(n-len(out));out.extend(data)
            if not data:self.current.close();self.current=None
        return bytes(out)


def verify(folder):
    snapshot=json.loads((folder/'snapshot.json').read_text(encoding='utf-8'))
    with (folder/'files.jsonl.gz').open('rb') as f:assert digest(f)==snapshot['manifest_sha256'],'Manifest digest mismatch'
    paths=[]
    for row in snapshot['parts']:
        p=folder/row['name'];assert p.name==row['name'] and p.parent==folder,'Unsafe volume name'
        assert p.stat().st_size==row['size'],p.name
        with p.open('rb') as f:assert digest(f)==row['sha256'],p.name
        paths.append(p);print('Volume verified:',p.name,flush=True)
    with gzip.open(folder/'files.jsonl.gz','rt',encoding='utf-8') as f:expected={r['path']:r for r in map(json.loads,f)}
    seen={};count=total=0
    with gzip.GzipFile(fileobj=Volumes(paths),mode='rb') as gz,tarfile.open(fileobj=gz,mode='r|',bufsize=1024**2) as tf:
        for member in tf:
            row=expected.pop(member.name)
            if member.isfile():
                assert row['type']=='file' and member.size==row['size'],member.name
                actual=digest(tf.extractfile(member));assert actual==row['sha256'],member.name
                seen[member.name]=actual;count+=1;total+=member.size
            elif member.islnk():
                assert row['type']=='file' and seen[member.linkname]==row['sha256'],member.name
                seen[member.name]=row['sha256'];count+=1;total+=row['size']
            elif member.issym():assert row['type']=='symlink' and member.linkname==row['target'],member.name
            else:assert member.isdir() and row['type']=='dir',member.name
    assert not expected,'Missing archive members'
    assert count==snapshot['files'] and total==snapshot['logical_bytes']
    print(json.dumps(dict(status='PASS',files=count,bytes=total,volumes=len(paths))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('folder',type=Path)
    verify(parser.parse_args().folder.resolve())
