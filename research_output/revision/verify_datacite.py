from pathlib import Path
import urllib.request,urllib.parse,json,os
os.environ.pop('SSLKEYLOGFILE',None)
R=Path(__file__).resolve().parent;OUT=R/'bibliography';rows=[]
for doi in ['10.5281/zenodo.11101338','10.48550/arXiv.1609.02907','10.48550/arXiv.1710.10903','10.48550/arXiv.1810.05997']:
    url='https://api.datacite.org/dois/'+urllib.parse.quote(doi,safe='')
    with urllib.request.urlopen(url,timeout=45) as response:data=json.load(response)['data']['attributes']
    assert data['doi'].lower()==doi.lower()
    record=dict(doi=doi,titles=data['titles'],creators=data['creators'],year=data['publicationYear'],type=data['types'],url=data.get('url'),status='verified_DataCite_metadata',source=url,accessed='2026-10-14')
    rows.append(record);print(doi,data['titles'],flush=True)
(OUT/'datacite_audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
