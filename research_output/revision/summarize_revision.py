from experiment_core import *
from evaluate_revision import binary_metrics
from sklearn.metrics import average_precision_score,roc_auc_score

def class_metrics(y,p):
    pred=p.argmax(1);cm=np.zeros((7,7),dtype=int);np.add.at(cm,(y,pred),1);tp=cm.diagonal();fn=cm.sum(1)-tp;fp=cm.sum(0)-tp;tn=cm.sum()-tp-fn-fp
    records=[]
    for c in range(7):
        pc=p[:,c];truth=y==c;ece=0
        for i in range(10):
            mask=(pc>=i/10)&(pc<(i+1)/10 if i<9 else pc<=1)
            if mask.any():ece+=mask.mean()*abs(pc[mask].mean()-truth[mask].mean())
        records.append(dict(class_index=c,n=int(truth.sum()),sensitivity=float(tp[c]/(tp[c]+fn[c])),specificity=float(tn[c]/(tn[c]+fp[c])),precision=float(tp[c]/(tp[c]+fp[c])) if tp[c]+fp[c] else 0.,f1=float(2*tp[c]/(2*tp[c]+fp[c]+fn[c])),auc=float(roc_auc_score(truth,pc)),average_precision=float(average_precision_score(truth,pc)),binary_ece=float(ece),binary_brier=float(np.mean((pc-truth)**2))))
    return records

def fixed_metrics_fast(y,p):
    pred=p.argmax(1);cm=np.zeros((7,7),dtype=int);np.add.at(cm,(y,pred),1);tp=cm.diagonal();den=cm.sum(1)+cm.sum(0)
    f1=np.divide(2*tp,den,out=np.zeros(7),where=den>0).mean();recall=np.divide(tp,cm.sum(1),out=np.zeros(7),where=cm.sum(1)>0)
    return dict(macro_f1=float(f1),balanced_accuracy=float(recall.mean()),accuracy=float((pred==y).mean()))

def finite_interval(values):
    values=np.asarray(values);values=values[np.isfinite(values)]
    return np.quantile(values,[.025,.975]).tolist() if len(values) else [None,None]

def main():
    out=R/'analysis';out.mkdir(exist_ok=True)
    allrows=[json.loads(p.read_text()) for p in (R/'evaluation').glob('*/*/metrics.json')];names=CFG['central_models']+CFG['augmentation_controls']+['RGBOnlyGraph']
    c=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz');ph=np.load(R/'data/ph2_28.npz')
    ys={'C_test':c['test_labels'].ravel(),'E_test':e['test_labels'].ravel(),'PH2':ph['labels']}
    meta=list(csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()));ct=[r for r in meta if r['split']=='test'];cg=np.array([r['lesion_id'] for r in ct]);unique=np.unique(cg);groups=[np.flatnonzero(cg==v) for v in unique]
    ensemble={};summary={};used=[]
    for name in names:
        rows=sorted([r for r in allrows if r['name']==name],key=lambda r:CFG['main_seeds'].index(r['seed']))
        if len(rows)!=10:continue
        files=[np.load(R/'evaluation'/name/str(r['seed'])/'predictions.npz') for r in rows];model={};summary[name]=dict(parameters=rows[0]['parameters'],runs=10,datasets={},seed_mean={},seed_sd={},temperatures=[r['temperature'] for r in rows],partitions=[r['n'] for r in rows])
        for ds in ys:
            raw=np.mean([f[ds+'_raw'] for f in files],0);cal=np.mean([f[ds+'_calibrated'] for f in files],0);model[ds]=(raw,cal)
            summary[name]['datasets'][ds]={'raw':binary_metrics(ys[ds],raw) if ds=='PH2' else metrics(ys[ds],raw),'calibrated':binary_metrics(ys[ds],cal) if ds=='PH2' else metrics(ys[ds],cal)}
            summary[name]['seed_mean'][ds]={};summary[name]['seed_sd'][ds]={}
            for metric,value in rows[0]['datasets'][ds]['raw'].items():
                if not isinstance(value,(float,int)) or metric in ['n','tp','tn','fn','fp']:continue
                vals=[r['datasets'][ds]['raw'][metric] for r in rows];summary[name]['seed_mean'][ds][metric]=float(np.mean(vals));summary[name]['seed_sd'][ds][metric]=float(np.std(vals,ddof=1))
            if ds!='PH2':summary[name]['datasets'][ds]['classwise_raw']=class_metrics(ys[ds],raw);summary[name]['datasets'][ds]['classwise_calibrated']=class_metrics(ys[ds],cal)
        calibration_probs=np.concatenate([f['calibration_probs'] for f in files]);calibration_labels=np.concatenate([f['calibration_labels'] for f in files]);reject=float(np.quantile(calibration_probs.max(1),.2));mel95=float(np.quantile(calibration_probs[calibration_labels==4,4],.05,method='lower'))
        summary[name]['pooled_calibration_thresholds']={'reject':reject,'mel95':mel95,'construction':'Each probability was predicted by a model excluding its image from training and tuning. Pooled predictions combine repeated calibration cohorts; no test fitting.'}
        for ds in ys:
            raw,cal=model[ds];summary[name]['datasets'][ds]['mel95']=binary_metrics(ys[ds] if ds=='PH2' else ys[ds]==4,cal,mel95)
            if ds!='PH2':
                keep=cal.max(1)>=reject;summary[name]['datasets'][ds]['selective']={'n_accepted':int(keep.sum()),'coverage':float(keep.mean()),'accuracy':float((cal[keep].argmax(1)==ys[ds][keep]).mean()),'mel_rejected':int(np.sum(~keep&(ys[ds]==4)))}
        ensemble[name]=model;used.append(name)
    assert len(used)==len(names),'All 150 fitted models must be evaluated before final summarization'
    # Shared cluster draws across models preserve pairing; image units for E and PH2 where group identities are absent.
    rng=np.random.default_rng(20261014);boot={name:{ds:[] for ds in ys} for name in used};classboot={name:[] for name in ['ContrastPenalty','ContrastNoPenalty','SmallCNN','ResNet18','MobileNetV3Small']}
    for b in range(1000):
        indices={'C_test':np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]),'E_test':rng.integers(0,len(ys['E_test']),len(ys['E_test'])),'PH2':rng.integers(0,200,200)}
        for name in used:
            for ds,idx in indices.items():
                p=ensemble[name][ds][0][idx];y=ys[ds][idx]
                vals=binary_metrics(y,p) if ds=='PH2' else fixed_metrics_fast(y,p)
                boot[name][ds].append(vals)
            if name in classboot:
                idx=indices['C_test'];y=ys['C_test'][idx];p=ensemble[name]['C_test'][0][idx];pred=p.argmax(1)
                cm=np.zeros((7,7),dtype=int);np.add.at(cm,(y,pred),1);tp=cm.diagonal();fn=cm.sum(1)-tp;fp=cm.sum(0)-tp;tn=cm.sum()-tp-fn-fp
                with np.errstate(divide='ignore',invalid='ignore'):
                    sens=tp/(tp+fn);spec=tn/(tn+fp);prec=tp/(tp+fp);f1=2*tp/(2*tp+fp+fn)
                classboot[name].append(np.stack([sens,spec,prec,f1],1))
        if b%200==0:print('Bootstrap',b,flush=True)
    for name in used:
        for ds in ys:
            summary[name]['datasets'][ds]['ci95']={metric:np.nanquantile([v[metric] for v in boot[name][ds]],[.025,.975]).tolist() for metric in boot[name][ds][0] if metric not in ['n','tp','tn','fp','fn']}
        if name in classboot:
            cb=np.array(classboot[name]);summary[name]['datasets']['C_test']['classwise_ci95']=[{metric:finite_interval(cb[:,c,j]) for j,metric in enumerate(['sensitivity','specificity','precision','f1'])} for c in range(7)]
    paired={name:{metric:np.quantile([a[metric]-b[metric] for a,b in zip(boot['ContrastPenalty']['C_test'],boot[name]['C_test'])],[.025,.975]).tolist() for metric in ['macro_f1','balanced_accuracy']} for name in used if name!='ContrastPenalty'}
    # Conditional calibration uncertainty, including class-conditional ECE, from the same lesion resampling.
    calibration={};rng=np.random.default_rng(7301)
    for name in ['ContrastPenalty','ContrastNoPenalty','SmallCNN','ResNet18','MobileNetV3Small']:
        vals=[];classvals=[];cal=ensemble[name]['C_test'][1]
        for _ in range(1000):
            idx=np.concatenate([groups[i] for i in rng.integers(0,len(groups),len(groups))]);y=ys['C_test'][idx];p=cal[idx];conf=p.max(1);correct=p.argmax(1)==y;ece=0
            for i in range(10):
                mask=(conf>=i/10)&(conf<(i+1)/10 if i<9 else conf<=1)
                if mask.any():ece+=mask.mean()*abs(conf[mask].mean()-correct[mask].mean())
            vals.append(ece);ec=[]
            for c in range(7):
                score=p[:,c];truth=y==c;v=0
                for i in range(10):
                    mask=(score>=i/10)&(score<(i+1)/10 if i<9 else score<=1)
                    if mask.any():v+=mask.mean()*abs(score[mask].mean()-truth[mask].mean())
                ec.append(v)
            classvals.append(ec)
        calibration[name]={'top_label_ece_ci95':np.quantile(vals,[.025,.975]).tolist(),'class_binary_ece_ci95':np.quantile(classvals,[.025,.975],axis=0).T.tolist()}
    payload=dict(models=summary,paired_C_test=paired,calibration_uncertainty=calibration,bootstrap={'replicates':1000,'C_unit':'lesion_id','E_unit':'image; no patient or lesion metadata provided','PH2_unit':'image; no patient identifiers provided'},protocol_sha256=hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest())
    (out/'summary.json').write_text(json.dumps(payload,indent=2));np.savez_compressed(out/'ensemble_predictions.npz',**{name+'_'+ds+'_'+mode:ensemble[name][ds][j] for name in used for ds in ys for j,mode in enumerate(['raw','calibrated'])})
    print('Summary completed',flush=True)
if __name__=='__main__':main()
