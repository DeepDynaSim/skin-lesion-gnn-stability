from experiment_core import *
from summarize_revision import fixed_metrics_fast
from corruption_revision import all_cases
from transfer_reference import TR
data=np.load(OLD/'data/dermamnist_corrected_28.npz');y=data['test_labels'].ravel();meta=[r for r in csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()) if r['split']=='test'];ids=np.array([r['lesion_id'] for r in meta]);groups=[np.flatnonzero(ids==v) for v in np.unique(ids)];result=json.loads((R/'analysis/transfer_reference.json').read_text());cases=list(all_cases());families={}
for key,family,value,seed in cases:families.setdefault(f'{family}_{value}',[]).append(key)
for group,keys in families.items():
    p=np.stack([np.load(TR/'corruptions'/(key+'.npz'))['raw'] for key in keys]);rng=np.random.default_rng(48271);values=[]
    for _ in range(1000):
        idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);trials=rng.integers(0,len(keys),len(keys)) if len(keys)>1 else [0];values.append(np.mean([fixed_metrics_fast(y[idx],p[t,idx])['macro_f1'] for t in trials]))
    result['corruption_mean'][group]['macro_f1_ci95']=np.quantile(values,[.025,.975]).tolist()
(R/'analysis/transfer_reference.json').write_text(json.dumps(result,indent=2));print('Additional transfer-reference corruption intervals complete')
