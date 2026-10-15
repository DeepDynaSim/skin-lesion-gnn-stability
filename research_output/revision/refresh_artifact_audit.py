from pathlib import Path
import sys,json,hashlib,zipfile
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'python_packages'))
from lxml import etree
import pymupdf
old=json.loads((R/'final/verification_report.json').read_text());assert old['status']=='passed' and old['primary_fits']==150 and old['additional_transfer_heads']==10 and old['sensitivity_fits']==42
manifest=json.loads((R/'final/document_manifest.json').read_text());path=R/'final/Skin_Lesion_GNN_Revised_Manuscript.docx';pdf=path.with_suffix('.pdf')
with zipfile.ZipFile(path) as z:
    root=etree.fromstring(z.read('word/document.xml'));ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};texts=' '.join(root.xpath('.//w:t/text()',namespaces=ns));assert len(root.xpath('.//m:oMath',namespaces=ns))==51==manifest['native_math'];assert manifest['equations']==27 and manifest['inline_equations']==24;assert len(root.xpath('.//w:drawing',namespaces=ns))==12
    for marker in ['{{','[TABLE:','[FIG:','[AUTHOR]','[REFERENCES]','\\frac','Q_K','PSP^T','||W_']:assert marker not in texts,marker
    for value in ['Caglar Uyulan','caglar.uyulan@ikcu.edu.tr','no competing interests','no external financial support']:assert value in texts
doc=pymupdf.open(pdf);pages=[]
for i,page in enumerate(doc):
    words=page.get_text('words');assert words;assert not [w for w in words if w[0]<28 or w[1]<25 or w[2]>page.rect.width-25 or w[3]>page.rect.height-20];pages.append({'page':i+1,'words':len(words),'images':len(page.get_images())})
old.update(docx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),native_math=51,inline_equations=24,pdf_pages=len(doc),page_audit=pages,artifact_refresh='Scientific measurements and checkpoint validation passed the full audit; this final pass checks document changes only.')
(R/'final/verification_report.json').write_text(json.dumps(old,indent=2));print('Final artifact audit passed:',len(doc),'pages; 51 native math structures')
