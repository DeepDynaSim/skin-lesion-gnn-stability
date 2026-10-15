from pathlib import Path
import os,urllib.request,sys
os.environ.pop('SSLKEYLOGFILE',None)
R=Path(__file__).resolve().parent;out=R/'bibliography';out.mkdir(exist_ok=True)
sys.path.insert(0,str(R/'python_packages'))
for file,url in [('CLAIM_2024.html','https://rsna.github.io/claim/checklist.html'),('TRIPOD_AI_official.pdf','https://www.tripod-statement.org/wp-content/uploads/2024/04/TRIPODAI-Supplement.pdf')]:
    req=urllib.request.Request(url,headers={'User-Agent':'SkinLesionResearch/1.0'});data=urllib.request.urlopen(req,timeout=60).read();(out/file).write_bytes(data);print(file,len(data))
import fitz
doc=fitz.open(out/'TRIPOD_AI_official.pdf');(out/'TRIPOD_AI_official.txt').write_text('\n'.join(p.get_text() for p in doc),encoding='utf-8')
