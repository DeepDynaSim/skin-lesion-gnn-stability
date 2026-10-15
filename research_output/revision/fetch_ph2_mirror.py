"""Retrieve a public PH2 mirror for research; never include its images in a redistributed package."""
from pathlib import Path
import urllib.request,json,hashlib,os,zipfile
os.environ.pop('SSLKEYLOGFILE',None)
R=Path(__file__).resolve().parent;out=R/'data/PH2_mirror.zip'
url='https://www.kaggle.com/api/v1/datasets/download/spacesurfer/ph2-dataset'
try:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'})
    with urllib.request.urlopen(req,timeout=90) as response,out.open('wb') as f:
        while b:=response.read(1048576):f.write(b)
    with zipfile.ZipFile(out) as z:
        names=z.namelist()
        # Dataset extraction only, with resolved path containment checks.
        destination=R/'data/ph2';destination.mkdir(exist_ok=True)
        for name in names:
            target=(destination/name).resolve();assert destination.resolve() in target.parents or target==destination.resolve()
            if name.lower().endswith(('.bmp','.png','.jpg','.jpeg','.txt','.csv','.xls','.xlsx')):z.extract(name,destination)
    result=dict(status='downloaded_public_mirror',url=url,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),n_files=len(names),first_files=names[:20],original_source='https://www.fc.up.pt/addi/ph2%20database.html',terms='PH2 research/educational use only; redistribution and commercial use prohibited; images excluded from deliverable archive')
except Exception as ex:result=dict(status='unavailable',url=url,error=str(ex))
(R/'data/ph2_mirror_manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
