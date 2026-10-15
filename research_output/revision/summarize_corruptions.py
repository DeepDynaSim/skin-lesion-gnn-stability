from experiment_core import *
from summarize_revision import fixed_metrics_fast
from corruption_revision import all_cases
import argparse

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='*');args=parser.parse_args()
    dest=R/'corruptions';names=args.models or CFG['central_models']+CFG['augmentation_controls']+['RGBOnlyGraph'];cases=list(all_cases());output=R/'analysis/corruption_summary.json';records=json.loads(output.read_text())['models'] if output.exists() else {};d=np.load(OLD/'data/dermamnist_corrected_28.npz');y=d['test_labels'].ravel()
    meta=[r for r in csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()) if r['split']=='test'];ids=np.array([r['lesion_id'] for r in meta]);groups=[np.flatnonzero(ids==v) for v in np.unique(ids)]
    families={}
    for key,family,value,seed in cases:families.setdefault(f'{family}_{value}',[]).append(key)
    for name in names:
        if name in records and len(records[name])==len(families):continue
        rows=[];records[name]={}
        for key,_,_,_ in cases:
            p=dest/name/(key+'.json');assert p.exists(),p
            rows.append(json.loads(p.read_text()))
        for group,keys in families.items():
            selected=[r for r in rows if r['case'] in keys];conditions=list(selected[0]['conditions']);arrays={label:np.stack([np.load(dest/name/(key+'.npz'))[label] for key in keys]) for label in conditions}
            record={'realizations':len(keys),'conditions':{},'structure':{}}
            for label in conditions:
                vals=[r['conditions'][label] for r in selected];record['conditions'][label]={metric:float(np.mean([v[metric] for v in vals])) for metric in vals[0]};record['conditions'][label]['noise_macro_f1_sd']=float(np.std([v['macro_f1'] for v in vals],ddof=1)) if len(vals)>1 else 0.
            structure=[json.loads((dest/(key+'_structure.json')).read_text()) for key in keys]
            for metric in ['spectral_delta_mean','spectral_delta_p95','candidate_jaccard_mean','candidate_edge_changed_fraction','raw_descriptor_relative']:record['structure'][metric]=float(np.mean([s[metric] for s in structure]))
            for metric in ['encoded_feature_relative_mean','standardized_descriptor_relative_mean','conditional_logit_bound_ratio_max']:
                if metric in selected[0]:record['structure'][metric]=float(np.mean([s[metric] for s in selected])) if metric!='conditional_logit_bound_ratio_max' else max(s[metric] for s in selected)
            rng=np.random.default_rng(48271);bs={label:[] for label in conditions};diff={label:[] for label in conditions if label!='A'}
            for _ in range(1000):
                idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);trials=rng.integers(0,len(keys),len(keys)) if len(keys)>1 else [0];v={}
                for label in conditions:
                    sample_values=[fixed_metrics_fast(y[idx],arrays[label][t,idx]) for t in trials];v[label]={metric:float(np.mean([one[metric] for one in sample_values])) for metric in ['macro_f1','balanced_accuracy']};bs[label].append(v[label])
                for label in diff:diff[label].append({metric:v[label][metric]-v['A'][metric] for metric in v[label]})
            for label in conditions:record['conditions'][label]['ci95']={metric:np.quantile([b[metric] for b in bs[label]],[.025,.975]).tolist() for metric in bs[label][0]}
            record['paired_change_ci95']={label:{metric:np.quantile([b[metric] for b in diff[label]],[.025,.975]).tolist() for metric in diff[label][0]} for label in diff}
            records[name][group]=record
        print('Corruption uncertainty',name,flush=True)
        output.write_text(json.dumps({'models':records,'bootstrap_replicates':1000,'unit':'C lesion cluster; noise additionally resamples five realizations shared across models','conditional_scope':'fixed ten-model ensemble, not retraining uncertainty'},indent=2))
    (R/'analysis/corruption_summary.json').write_text(json.dumps({'models':records,'bootstrap_replicates':1000,'unit':'C lesion cluster; noise additionally resamples five realizations shared across models','conditional_scope':'fixed ten-model ensemble, not retraining uncertainty'},indent=2))
if __name__=='__main__':main()
