import pathlib,time,os,signal,json
root=pathlib.Path('/root/autodl-tmp/FedPASA'); pid=473
assert 'scripts/run_matched_fedmpt.py' in pathlib.Path(f'/proc/{pid}/cmdline').read_text()
paths=[root/'outputs/icme/supplementary'/d/'fedmpt_repaired/progress.json' for d in ('coco','nus')]
initial=[p.stat().st_mtime_ns for p in paths]; deadline=time.monotonic()+900
while time.monotonic()<deadline:
 if [p.stat().st_mtime_ns for p in paths]!=initial:
  os.kill(pid,signal.SIGTERM)
  print('PAUSED_AFTER_SAVED_CLIENT',[(str(p),json.loads(p.read_text())) for p in paths],flush=True)
  break
 time.sleep(.2)
else: raise RuntimeError('No checkpoint boundary in 15 minutes; trainer left running')
