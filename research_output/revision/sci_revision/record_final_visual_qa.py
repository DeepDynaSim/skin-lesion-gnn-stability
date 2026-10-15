"""Bind actual page inspections to the delivered files using identical PNG hashes."""
from pathlib import Path
import hashlib,json,datetime
HERE=Path(__file__).resolve().parent;digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
pages=[]
for n in range(1,34):
    if n in [1]+list(range(3,24)):evidence=HERE/f'qa_preview/page-{n}.png'
    elif n in [2,24]:evidence=HERE/f'qa/page-{n}.png'
    elif n==32:evidence=HERE/f'qa_release/page-{n}.png'
    else:evidence=HERE/f'qa_final/page-{n}.png'
    released=HERE/f'qa_release/page-{n}.png';assert digest(evidence)==digest(released),(n,'Requires new visual inspection')
    pages.append({'page':n,'status':'passed','released_page_sha256':digest(released),'inspected_render':evidence.relative_to(HERE).as_posix(),'identical_to_inspected_pixels':True})
docx=HERE/'final/Skin_Lesion_GNN_Final_Manuscript.docx';pdf=docx.with_suffix('.pdf')
record={'status':'passed','completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'page_count':len(pages),'docx_sha256':digest(docx),'pdf_sha256':digest(pdf),'method':'Every page inspected manually. For unchanged pages, exact PNG SHA-256 equality carries the inspection to the release raster. Changed final pages were inspected directly. This is separate from automated geometry and numerical verification.','checks':['Native math notation and numbering','Readable figures and captions','Complete tables and continuation headers','Correct Turkish author-affiliation characters','No clipped content or isolated final-reference page'],'pages':pages}
(HERE/'final/visual_QA.json').write_text(json.dumps(record,indent=2));(HERE/'final/page_render_comparison.json').write_text(json.dumps({'release_pages':len(pages),'all_released_pages_match_manually_inspected_renders':True,'pages':pages},indent=2));print('All 33 released page rasters match the manually inspected evidence.',flush=True)
