from pathlib import Path
import urllib.request,urllib.parse,json,concurrent.futures,time,os
os.environ.pop('SSLKEYLOGFILE',None)
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'bibliography';OUT.mkdir(exist_ok=True)
dois=['10.1016/j.cmpb.2020.105351','10.1016/j.compbiomed.2021.104458','10.3390/diagnostics11081390','10.1016/j.compbiomed.2018.12.007','10.32604/cmc.2025.069162','10.32604/cmes.2026.080335','10.1038/s41597-025-04382-5','10.1038/sdata.2018.161','10.1038/s41597-022-01721-8','10.1109/TPAMI.2012.120','10.1109/ICDM58522.2023.00086','10.1136/bmj-2023-078378','10.1148/ryai.240300','10.1109/CVPR.2016.90','10.1109/ICCV.2019.00140','10.1109/EMBC.2013.6610779']
def fetch(doi):
    url='https://api.crossref.org/works/'+urllib.parse.quote(doi,safe='')
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'SkinLesionResearchReferenceAudit/1.0','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=45) as response:record=json.load(response)['message']
            assert record['DOI'].lower()==doi.lower()
            raw_path=OUT/(doi.replace('/','_')+'.json');raw_path.write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
            return dict(doi=doi,status='registered_title_author_metadata_verified',title=record.get('title'),authors=record.get('author'),journal=record.get('container-title'),volume=record.get('volume'),issue=record.get('issue'),page=record.get('page'),article_number=record.get('article-number'),published=record.get('published'),URL=record.get('URL'),metadata_source=url,accessed='2026-10-14')
        except Exception as e:
            if attempt==2:return dict(doi=doi,status='unverified',error=str(e),metadata_source=url)
            time.sleep(2)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:results=list(pool.map(fetch,dois))
(OUT/'doi_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
for r in results:print(r['doi'],r['status'],r.get('title'),flush=True)
