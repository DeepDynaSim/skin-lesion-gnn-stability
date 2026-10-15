from experiment_core import *
def main():
    data=np.load(OLD/'data/dermamnist_corrected_28.npz');rows=list(csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()));splits={};sets={}
    for s in ['train','val','test']:
        images=data[s+'_images'];y=data[s+'_labels'].ravel();meta=[r for r in rows if r['split']==s];mapping={v:i for i,v in enumerate(['akiec','bcc','bkl','df','mel','nv','vasc'])};assert np.array_equal(y,np.array([mapping[r['dx']] for r in meta]));hashes={hashlib.sha256(im.tobytes()).hexdigest() for im in images};lesions={r['lesion_id'] for r in meta};sets[s]=(hashes,lesions);splits[s]=dict(images=len(y),lesions=len(lesions),class_counts=np.bincount(y,minlength=7).tolist(),metadata_labels_match=True)
    overlaps={}
    for a,b in [('train','val'),('train','test'),('val','test')]:overlaps[a+'_'+b]=dict(exact_hashes=len(sets[a][0]&sets[b][0]),lesion_ids=len(sets[a][1]&sets[b][1]))
    assert all(v['exact_hashes']==0 and v['lesion_ids']==0 for v in overlaps.values());im,y,g,_=load_development();part={}
    for seed in CFG['main_seeds']:
        p=partitions(y,g,seed);part[seed]={s:dict(n=len(ix),lesions=len(set(g[ix])),counts=np.bincount(y[ix],minlength=7).tolist()) for s,ix in p.items()};assert all(not set(g[ix])&sets['test'][1] for ix in p.values())
    result=dict(splits=splits,overlaps=overlaps,development_partitions=part,metadata_sha256=hashlib.sha256((OLD/'data/DermaMNIST-C.csv').read_bytes()).hexdigest(),npz_sha256=hashlib.sha256((OLD/'data/dermamnist_corrected_28.npz').read_bytes()).hexdigest());(R/'analysis/identity_reaudit.json').write_text(json.dumps(result,indent=2));print('Independent identity and partition audit passed',flush=True)
if __name__=='__main__':main()
