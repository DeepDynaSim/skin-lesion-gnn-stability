from pathlib import Path
import sys,csv,json,hashlib
R=Path(__file__).resolve().parent;OLD=R.parent;sys.path.insert(0,str(OLD/'python_packages'))
import numpy as np
c=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz')
cr=list(csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()));er=list(csv.DictReader((R/'data/DermaMNIST-E.csv').open()))
print('E arrays',{k:e[k].shape for k in e.files},flush=True);print('E columns',list(er[0]),flush=True)
print('E first rows',er[:2],flush=True)
report={'C_shapes':{k:list(c[k].shape) for k in c.files},'E_shapes':{k:list(e[k].shape) for k in e.files},'E_columns':list(er[0])}
for source_name,source_splits in [('C_development',['train','val']),('C_all',['train','val','test'])]:
    crows=[r for r in cr if r['split'] in source_splits];etest=[r for r in er if r['split']=='test']
    image_overlap=set(r['image_id'] for r in crows)&set(r['image'] for r in etest)
    idcol='lesion_id' if 'lesion_id' in etest[0] else None
    lesion_overlap=set(r[idcol] for r in crows if r.get(idcol))&set(r[idcol] for r in etest if r.get(idcol)) if idcol else None
    ch=set(hashlib.sha256(im.tobytes()).hexdigest() for s in source_splits for im in c[s+'_images']);eh=set(hashlib.sha256(im.tobytes()).hexdigest() for im in e['test_images'])
    report[source_name+'_vs_E_test']={'image_id_overlap':len(image_overlap),'lesion_id_overlap':len(lesion_overlap) if lesion_overlap is not None else None,'exact_pixel_overlap':len(ch&eh),'image_ids':sorted(image_overlap)}
classes=['akiec','bcc','bkl','df','mel','nv','vasc']
for s in ['train','val','test']:
    m=[r for r in er if r['split']==s]
    report['E_'+s]={'n':len(m),'counts':np.bincount(e[s+'_labels'].ravel(),minlength=7).tolist(),'label_order_match':bool(np.array_equal([int(r['diag_int']) for r in m],e[s+'_labels'].ravel()))}
(R/'data/overlap_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
