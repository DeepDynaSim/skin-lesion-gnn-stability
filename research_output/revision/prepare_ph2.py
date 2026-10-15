from pathlib import Path
import sys,json,hashlib,re,csv
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'python_packages'))
import numpy as np
from PIL import Image
base=R/'data/ph2/PH2Dataset';text=(base/'PH2_dataset.txt').read_text(encoding='latin-1')
records=[]
for line in text.splitlines():
    if not re.match(r'\|\|\s+IMD\d+',line):continue
    parts=[p.strip() for p in line.split('||')]
    identifier=parts[1];diagnosis=int(parts[3].split('|')[0].strip())
    records.append(dict(image_id=identifier,clinical_diagnosis=diagnosis,binary_melanoma=int(diagnosis==2),seven_class_mapping=4 if diagnosis==2 else 5,histology=parts[2]))
assert len(records)==200;assert [sum(r['clinical_diagnosis']==c for r in records) for c in range(3)]==[80,80,40]
images=[];hashes=[]
for r in records:
    p=base/'PH2 Dataset images'/r['image_id']/(r['image_id']+'_Dermoscopic_Image')/(r['image_id']+'.bmp')
    assert p.exists()
    with Image.open(p) as im:images.append(np.asarray(im.convert('RGB').resize((28,28),Image.Resampling.BICUBIC)))
    hashes.append(dict(image_id=r['image_id'],raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),resized_sha256=hashlib.sha256(images[-1].tobytes()).hexdigest()))
np.savez_compressed(R/'data/ph2_28.npz',images=np.array(images),labels=np.array([r['binary_melanoma'] for r in records]),image_ids=np.array([r['image_id'] for r in records]))
with (R/'data/ph2_metadata.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
c=np.load(R.parent/'data/dermamnist_corrected_28.npz');cim=np.concatenate([c[s+'_images'] for s in ['train','val','test']])
ch={hashlib.sha256(im.tobytes()).hexdigest() for im in cim};overlap=[r for r in hashes if r['resized_sha256'] in ch]
assert not overlap
# Normalized cross-correlation is a near-image screen, not a patient identity guarantee.
x=cim.reshape(len(cim),-1).astype(np.float32);x-=x.mean(1,keepdims=True);x/=np.linalg.norm(x,axis=1,keepdims=True).clip(1e-6)
z=np.array(images).reshape(200,-1).astype(np.float32);z-=z.mean(1,keepdims=True);z/=np.linalg.norm(z,axis=1,keepdims=True).clip(1e-6)
corr=z@x.T;best=corr.argmax(1);maximum=corr.max(1)
audit=dict(n=200,class_counts=[80,80,40],binary_counts=[160,40],reference_standard='Original PH2 clinical diagnosis field; histology when available',resize='PIL bicubic RGB to 28 by 28; no crop or mask used',exact_resized_overlap_C_all=len(overlap),unique_raw_hashes=len(set(r['raw_sha256'] for r in hashes)),maximum_cross_dataset_NCC=float(maximum.max()),patient_ids_available=False,source_external='Hospital Pedro Hispano, Matosinhos, Portugal',diagnosis_metadata_sha256=hashlib.sha256((base/'PH2_dataset.txt').read_bytes()).hexdigest(),hashes=hashes,closest_pairs=[dict(ph2_id=records[i]['image_id'],C_index=int(best[i]),NCC=float(maximum[i])) for i in np.argsort(maximum)[-10:][::-1]])
(R/'data/ph2_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps({k:v for k,v in audit.items() if k!='hashes'},indent=2))
addendum=dict(frozen_before_PH2_predictions=True,model_settings='Unchanged from protocol.json',PH2_evaluation='Apply frozen seven-class networks to all 200 externally acquired images. Use the unmodified MEL probability for binary ROC and the seven-class argmax for melanoma sensitivity; common and atypical nevi are negatives. Threshold tradeoffs use only C development calibration labels. No PH2 fitting.',PH2_source='Original PH2 images and diagnosis table from a public mirror, matched to original documentation and counts; no redistribution of images.')
(R/'protocol_external_addendum.json').write_text(json.dumps(addendum,indent=2))
