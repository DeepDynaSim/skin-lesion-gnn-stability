"""Retrieve the four frozen C/E files, enforcing the measured source hashes."""
from pathlib import Path
import urllib.request,hashlib,os,json
R=Path(__file__).resolve().parent;os.environ.pop('SSLKEYLOGFILE',None)
for folder in [R.parent/'data',R/'data']:
    records=json.loads((folder/'download_manifest.json').read_text())
    for record in records:
        path=folder/record['path']
        if not path.exists():
            req=urllib.request.Request(record['url'],headers={'User-Agent':'ScientificReplication/1.0'})
            with urllib.request.urlopen(req,timeout=60) as response,path.open('wb') as output:
                while chunk:=response.read(1024*1024):output.write(chunk)
        actual=hashlib.sha256(path.read_bytes()).hexdigest();assert actual==record['sha256'],f'Unexpected source hash: {path}'
        print(path.name,'hash verified',flush=True)
