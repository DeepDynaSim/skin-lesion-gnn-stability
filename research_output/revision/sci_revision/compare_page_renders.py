from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
same=[];changed=[]
for p in sorted((HERE/'qa').glob('page-*.png'),key=lambda p:int(p.stem.split('-')[1])):
    n=int(p.stem.split('-')[1]);previous=HERE/'qa_preview'/p.name
    (same if previous.exists() and hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256(previous.read_bytes()).digest() else changed).append(n)
report={'identical_page_images':same,'changed_or_new_page_images':changed,'manually_inspected_preview_pages':[1]+list(range(3,24))}
(HERE/'final/page_render_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
