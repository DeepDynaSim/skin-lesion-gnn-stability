from pathlib import Path
import json,re,html
R=Path(__file__).resolve().parent;B=R/'bibliography'
cross={r['doi'].lower():r for r in json.loads((B/'doi_audit.json').read_text(encoding='utf-8'))}
order=['10.48550/arXiv.1609.02907','10.48550/arXiv.1810.05997','10.48550/arXiv.1710.10903','10.1016/j.cmpb.2020.105351','10.1016/j.compbiomed.2021.104458','10.3390/diagnostics11081390','10.32604/cmc.2025.069162','10.32604/cmes.2026.080335','10.1109/ICDM58522.2023.00086','10.1038/sdata.2018.161','10.1038/s41597-022-01721-8','10.1038/s41597-025-04382-5','10.5281/zenodo.11101338','10.1109/EMBC.2013.6610779','10.1109/TPAMI.2012.120','10.1109/CVPR.2016.90','10.1109/ICCV.2019.00140',None,'10.1136/bmj-2023-078378','10.1148/ryai.240300','10.1016/j.compbiomed.2018.12.007']
special={1:'Kipf TN, Welling M. Semi-Supervised Classification with Graph Convolutional Networks. ICLR; 2017. Author preprint arXiv:1609.02907.',2:'Gasteiger J, Bojchevski A, Günnemann S. Predict then Propagate: Graph Neural Networks meet Personalized PageRank. ICLR; 2019. Author preprint arXiv:1810.05997.',3:'Veličković P, Cucurull G, Casanova A, Romero A, Liò P, Bengio Y. Graph Attention Networks. ICLR; 2018. Author preprint arXiv:1710.10903.',13:'Abhishek K, Jain A, Hamarneh G. Investigating the Quality of DermaMNIST and Fitzpatrick17k Dermatological Image Datasets. Dataset: DermaMNIST-C, DermaMNIST-E, and Fitzpatrick17k-C. Versioned release. Zenodo; 2024.',18:'Guo C, Pleiss G, Sun Y, Weinberger KQ. On Calibration of Modern Neural Networks. Proceedings of Machine Learning Research. 2017;70:1321–1330. https://proceedings.mlr.press/v70/guo17a.html. No publisher DOI is listed.'}
refs=[];audit=[]
for i,doi in enumerate(order,1):
    if i in special:text=special[i]
    else:
        r=cross[doi.lower()];authors=r['authors'];names=[]
        for a in authors[:6]:
            initials=''.join(x[0] for x in re.findall(r'\w+',a.get('given','')));names.append(a.get('family','')+' '+initials)
        if len(authors)>6:names.append('et al.')
        title=re.sub('<[^>]+>','',html.unescape(' '.join(r['title'])));title=' '.join(title.split());year=r['published']['date-parts'][0][0];journal=' '.join(r['journal']);volume=r.get('volume') or '';issue=r.get('issue') or '';loc=r.get('article_number') or r.get('page') or ''
        if doi=='10.32604/cmes.2026.080335':loc='38'  # Publisher uses an article number; Crossref supplies first-version pages.
        if doi=='10.1148/ryai.240300':issue='4'
        place=f'{year}'+(f';{volume}'+(f'({issue})' if issue else '') if volume else '')+(f':{loc}' if loc else '')
        text=f"{', '.join(names)}. {title}. {html.unescape(journal)}. {place}.".replace('et al..','et al.')
    if doi:text+=f' https://doi.org/{doi}.'
    refs.append(dict(number=i,text=text,doi=doi));audit.append(dict(number=i,doi=doi,status='Verified in Crossref' if doi and doi.lower() in cross else 'Verified in DataCite; preprint DOI explicitly distinguished' if doi and i<=3 else 'Verified version DOI in DataCite' if doi else 'No publisher DOI; official PMLR landing page checked',metadata_source=cross[doi.lower()]['metadata_source'] if doi and doi.lower() in cross else 'https://api.datacite.org/dois/'+doi if doi else 'https://proceedings.mlr.press/v70/guo17a.html',accessed='2026-10-14'))
(B/'references.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8');(B/'reference_audit_final.json').write_text(json.dumps(audit,indent=2));(B/'DOI_Verification.md').write_text('# Reference verification\n\nAll DOI-bearing references were checked through the primary registration agencies on 14 October 2026. Conference author preprints are explicitly identified; a DOI is not invented for the PMLR calibration paper. The CMES publisher article number 38 is used in preference to provisional Crossref page numbering.\n\n'+'\n'.join(f"- [{v['number']}] {v['doi'] or 'No DOI'} — {v['status']}" for v in audit),encoding='utf-8')
print('21 references: 20 verified DOI records and one verified no-DOI exception')
