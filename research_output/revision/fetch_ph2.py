from pathlib import Path
import os,urllib.request,hashlib,json
os.environ.pop('SSLKEYLOGFILE',None)
R=Path(__file__).resolve().parent;out=R/'data/PH2Dataset.rar'
url='https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar?dl=1'
try:
    req=urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 (research dataset retrieval)'})
    with urllib.request.urlopen(req,timeout=40) as response:
        if 'html' in response.headers.get('Content-Type',''):raise RuntimeError('Official download returned a web page rather than dataset')
        with out.open('wb') as f:
            while b:=response.read(1048576):f.write(b)
    assert out.read_bytes()[:4]==b'Rar!'
    report=dict(status='downloaded',url=url,bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),terms='Research and educational use; redistribution and commercial use not allowed',homepage='https://www.fc.up.pt/addi/ph2%20database.html')
except Exception as ex:
    report=dict(status='official_download_unavailable',url=url,error=str(ex),homepage='https://www.fc.up.pt/addi/ph2%20database.html')
(R/'data/ph2_retrieval.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
