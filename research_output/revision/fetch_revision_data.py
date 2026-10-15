from pathlib import Path
import urllib.request,json,hashlib,sys,os
os.environ.pop('SSLKEYLOGFILE',None)
ROOT=Path(__file__).resolve().parent;ROOT.mkdir(exist_ok=True)
DATA=ROOT/'data';DATA.mkdir(exist_ok=True)
urls={
 'dermamnist_extended_28.npz':'https://zenodo.org/records/11101338/files/dermamnist_extended_28.npz?download=1',
 'DermaMNIST-E.csv':'https://zenodo.org/records/11101338/files/DermaMNIST-E.csv?download=1'}
manifest=[]
for name,url in urls.items():
    path=DATA/name
    if not path.exists():
        print('Downloading',name,flush=True)
        req=urllib.request.Request(url,headers={'User-Agent':'ResearchRevision/1.0'})
        with urllib.request.urlopen(req,timeout=120) as r,path.open('wb') as f:
            while b:=r.read(1024*1024):f.write(b)
    record=dict(path=name,url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest());manifest.append(record);print(record,flush=True)
(DATA/'download_manifest.json').write_text(json.dumps(manifest,indent=2))
