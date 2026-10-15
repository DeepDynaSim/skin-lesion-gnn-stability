"""Check the final revision's equations, content, source matches and rendered geometry."""
from pathlib import Path
import sys,json,zipfile,re,hashlib
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent/'python_packages'))
from lxml import etree
import pymupdf

docx=HERE/'final/Skin_Lesion_GNN_Final_Manuscript.docx';pdf=docx.with_suffix('.pdf');manifest=json.loads((HERE/'final/document_manifest.json').read_text());ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
with zipfile.ZipFile(docx) as z:
    root=etree.fromstring(z.read('word/document.xml'));text=' '.join(root.xpath('.//w:t/text()',namespaces=ns));math_count=len(root.xpath('.//m:oMath',namespaces=ns));assert math_count==manifest['native_math'] and manifest['equations']==29 and manifest['figures']==14 and manifest['tables']==16
    for marker in ['{{','[TABLE:','[FIG:','[AUTHOR]','[REFERENCES]','\\frac','\\sum','$$','DRAFT','audit still running','Ä°','Ã¼']:assert marker not in text,marker
    for phrase in ['Caglar Uyulan','İzmir','Türkiye','caglar.uyulan@ikcu.edu.tr','no external financial support','no competing interests','ISIC 2018','222 actual fits','Persistent public deposition is pending','same-host clean-installation audit','not fifty independent biological samples']:assert phrase in text,phrase
    assert len(root.xpath('.//w:drawing',namespaces=ns))==14
    for i in range(1,22):assert f'[{i}]' in text
source=(HERE/'final/Skin_Lesion_GNN_Final_Manuscript.md').read_text(encoding='utf-8');abstract=source.split('## Abstract',1)[1].split('Keywords:',1)[0];words=len(abstract.split());assert words<=300
with pymupdf.open(pdf) as document:
    pages=[]
    for number,page in enumerate(document,1):
        words_on_page=page.get_text('words');assert words_on_page,(number,'Empty page')
        outside=[w for w in words_on_page if w[0]<27 or w[1]<24 or w[2]>page.rect.width-24 or w[3]>page.rect.height-18];assert not outside,(number,outside[:2]);pages.append({'page':number,'words':len(words_on_page),'images':len(page.get_images())})
    pdf_text='\n'.join(page.get_text() for page in document)
    for number in range(1,30):assert f'({number})' in pdf_text,str(number)
for name in ['clean_replication','clean_new_cnn','descriptor_noise']:assert json.loads((HERE/'analysis'/(name+'.json')).read_text())['status']=='passed'
report={'status':'passed','numbered_equations':29,'native_math':math_count,'figures':14,'tables':16,'abstract_words':words,'pdf_pages':len(pages),'page_geometry_audit':pages,'docx_sha256':hashlib.sha256(docx.read_bytes()).hexdigest(),'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),'scientific_fits':222,'clean_installation_scope':'All saved states and analyses checked; three representative refits; same host, not independent investigator','visual_QA':'Separate manual page inspection required'}
(HERE/'final/artifact_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='page_geometry_audit'},indent=2),flush=True)
