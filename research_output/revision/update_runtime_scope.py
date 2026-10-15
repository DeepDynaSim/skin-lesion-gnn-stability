from pathlib import Path
import json
R=Path(__file__).resolve().parent;scope='Native 28x28 PNG decoding; official resize 256 / center crop 224 / ImageNet normalization included in preparation; frozen backbone and learned head included in forward. Source clinical-image downsampling, disk I/O and ensembling excluded. Peak process memory includes imports/data/runtime.'
p=R/'analysis/runtime_FrozenMobileNetV3.json';row=json.loads(p.read_text());row['scope']=scope;p.write_text(json.dumps(row,indent=2));p=R/'analysis/runtime_summary.json';rows=json.loads(p.read_text())
for row in rows:
    if row['name']=='FrozenMobileNetV3':row['scope']=scope
p.write_text(json.dumps(rows,indent=2));print('Runtime scope clarified; numerical timings unchanged')
