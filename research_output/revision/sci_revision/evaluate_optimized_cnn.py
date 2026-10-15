"""Frozen CNN evaluation after all development selections; paired exploratory intervals."""
from optimized_cnn import *
from evaluate_revision import binary_metrics
from summarize_revision import fixed_metrics_fast,class_metrics
from corruption_revision import all_cases,corrupted

def main():
    assert len(list((HERE/'optimized_cnn/selections').glob('*.json')))==10
    c=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz');ph=np.load(R/'data/ph2_28.npz')
    data={'C_test':(c['test_images'],c['test_labels'].ravel()),'E_test':(e['test_images'],e['test_labels'].ravel()),'PH2':(ph['images'],ph['labels'])};models=[selected_model(seed) for seed in CFG['main_seeds']];arrays={};result={'parameters':16647,'actual_candidate_fits':20,'development_selected_states':10,'datasets':{},'selections':[r for _,r in models]}
    dest=HERE/'optimized_cnn/evaluation';dest.mkdir(exist_ok=True)
    for ds,(im,y) in data.items():
        x=torch.tensor(im.transpose(0,3,1,2).astype(np.float32)/255);logs=np.array([inference(model,x) for model,_ in models]);seedp=softmax(logs,2);seedcal=np.array([softmax(logs[j]/rec['temperature'],1) for j,(_,rec) in enumerate(models)]);p=seedp.mean(0);cal=seedcal.mean(0)
        arrays.update({ds+'_seed_logits':logs,ds+'_seed_raw':seedp,ds+'_raw':p,ds+'_calibrated':cal})
        measure=binary_metrics if ds=='PH2' else metrics;rows=[measure(y,v) for v in seedp];record={'raw':measure(y,p),'calibrated':measure(y,cal),'seed_metrics':rows,'seed_mean':{},'seed_sd':{}}
        for key,value in rows[0].items():
            if isinstance(value,float):record['seed_mean'][key]=float(np.mean([v[key] for v in rows]));record['seed_sd'][key]=float(np.std([v[key] for v in rows],ddof=1))
        if ds!='PH2':record['classwise']=class_metrics(y,p)
        result['datasets'][ds]=record
    np.savez_compressed(dest/'predictions.npz',**arrays)
    meta=[r for r in csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()) if r['split']=='test'];ids=np.array([r['lesion_id'] for r in meta]);clusters=[np.flatnonzero(ids==v) for v in np.unique(ids)];rng=np.random.default_rng(20261015);boot={ds:[] for ds in data};paired=[];prior=np.load(R/'analysis/ensemble_predictions.npz')
    for _ in range(1000):
        indices={'C_test':np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]),'E_test':rng.integers(0,1511,1511),'PH2':rng.integers(0,200,200)}
        for ds,ix in indices.items():boot[ds].append(binary_metrics(data[ds][1][ix],arrays[ds+'_raw'][ix]) if ds=='PH2' else fixed_metrics_fast(data[ds][1][ix],arrays[ds+'_raw'][ix]))
        ix=indices['C_test'];cp=fixed_metrics_fast(data['C_test'][1][ix],prior['ContrastPenalty_C_test_raw'][ix]);paired.append(cp['macro_f1']-boot['C_test'][-1]['macro_f1'])
    for ds in data:result['datasets'][ds]['ci95']={k:np.quantile([v[k] for v in boot[ds]],[.025,.975]).tolist() for k in boot[ds][0] if k not in ['n','tp','tn','fn','fp']}
    result['paired_C_penalized_minus_tuned_CNN_macro_f1_ci95']=np.quantile(paired,[.025,.975]).tolist();result['bootstrap_scope']='1000 draws; C lesion clusters; E/PH2 image units; conditional on fitted ensemble; unadjusted exploratory comparison'
    corruptdest=HERE/'optimized_cnn/corruptions';corruptdest.mkdir(exist_ok=True);families={};corruptions={}
    for key,family,value,seed in all_cases():
        path=corruptdest/(key+'.npz')
        if path.exists():p=np.load(path)['raw']
        else:
            im=corrupted(c['test_images'],family,value,seed);x=torch.tensor(im.transpose(0,3,1,2).astype(np.float32)/255);p=np.mean([softmax(inference(model,x),1) for model,_ in models],0);np.savez_compressed(path,raw=p)
        corruptions[key]={'family':family,'value':value,'seed':seed,'metrics':metrics(data['C_test'][1],p)};families.setdefault(f'{family}_{value}',[]).append(key);print('Tuned CNN corruption',key,round(corruptions[key]['metrics']['macro_f1'],4),flush=True)
    result['corruptions']=corruptions;result['corruption_mean']={}
    for group,keys in families.items():
        ps=np.stack([np.load(corruptdest/(key+'.npz'))['raw'] for key in keys]);rng=np.random.default_rng(48271);values=[]
        for _ in range(1000):
            ix=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]);trials=rng.integers(0,len(keys),len(keys)) if len(keys)>1 else [0];values.append(np.mean([fixed_metrics_fast(data['C_test'][1][ix],ps[t,ix])['macro_f1'] for t in trials]))
        result['corruption_mean'][group]={'macro_f1':float(np.mean([corruptions[key]['metrics']['macro_f1'] for key in keys])),'macro_f1_ci95':np.quantile(values,[.025,.975]).tolist()}
    (HERE/'analysis/optimized_cnn.json').write_text(json.dumps(result,indent=2));print('Tuned CNN evaluation complete',json.dumps({ds:result['datasets'][ds]['raw'] for ds in data}),flush=True)

if __name__=='__main__':main()
