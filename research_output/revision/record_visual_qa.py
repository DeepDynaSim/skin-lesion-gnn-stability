"""Record the completed manual inspection of final rendered pages."""
from pathlib import Path
import hashlib,json,datetime

root=Path(__file__).resolve().parent
pdf=root/'final/Skin_Lesion_GNN_Revised_Manuscript.pdf'
docx=root/'final/Skin_Lesion_GNN_Revised_Manuscript.docx'
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pages=[]
for n in range(1,29):
    p=root/f'qa/page-{n}.png'
    assert p.is_file()
    pages.append({'page':n,'rendered_page_sha256':digest(p),'manual_visual_status':'passed'})
report={
    'status':'passed',
    'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'method':'Manual visual inspection of every final page PNG at 160 dpi; separate from automated artifact and scientific validation.',
    'pdf_sha256':digest(pdf),
    'docx_sha256':digest(docx),
    'page_count':len(pages),
    'checks':['Readable native equations and numbering','Figure and caption placement','Complete tables and repeated continuation headers','Margins and page numbering','No clipped or overlapping document content'],
    'pages':pages,
}
(root/'final/visual_QA.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Manual visual QA recorded for all 28 final pages.')
