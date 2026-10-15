from pathlib import Path
import json,sys
R=Path(__file__).resolve().parent
for folder in (R/'models').iterdir():
    if not folder.is_dir():continue
    rows=[json.loads(p.read_text()) for p in folder.glob('*/training.json')]
    print(folder.name,len(rows),'runs completed',flush=True)
    if rows:
        print('train/tune/cal counts',rows[0]['n'],'parameters',rows[0]['parameters'],flush=True)
        if any(v%256==1 for r in rows for v in [r['n']['train']]):print('SINGLETON LAST BATCH',flush=True)
