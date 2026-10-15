"""Validate the public archive bytes and scientific state inventory (stdlib only)."""
from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
m=json.loads((root/'release_manifest.json').read_text(encoding='utf-8'))
for row in m['files']:
 p=(root/row['path']).resolve()
 assert p.is_relative_to(root),row['path']
 assert p.stat().st_size==row['bytes'],row['path']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['path']
assert sum(m['scientific_fitted_states'].values())==222
assert sum(1 for p in (root/'research_output').rglob('*.pt'))==222
print(f"Verified {len(m['files'])} files and all 222 scientific fitted states.")
