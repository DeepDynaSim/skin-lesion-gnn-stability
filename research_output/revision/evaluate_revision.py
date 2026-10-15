from experiment_core import *
from sklearn.metrics import roc_auc_score,precision_score,recall_score,accuracy_score,confusion_matrix

def binary_metrics(y,probs,threshold=None):
    score=probs[:,4];pred=(probs.argmax(1)==4) if threshold is None else score>=threshold
    cm=confusion_matrix(y,pred,labels=[0,1]);tn,fp,fn,tp=cm.ravel()
    return dict(n=len(y),auc=float(roc_auc_score(y,score)),accuracy=float(accuracy_score(y,pred)),sensitivity=float(tp/(tp+fn)),specificity=float(tn/(tn+fp)),precision=float(tp/(tp+fp)) if tp+fp else 0.,f1=float(2*tp/(2*tp+fp+fn)),tp=int(tp),fn=int(fn),fp=int(fp),tn=int(tn))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='*');args=parser.parse_args()
    names=args.models or CFG['central_models']+CFG['augmentation_controls']+['RGBOnlyGraph']
    c=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz');ph=np.load(R/'data/ph2_28.npz')
    data={'C_test':(c['test_images'],c['test_labels'].ravel()),'E_test':(e['test_images'],e['test_labels'].ravel()),'PH2':(ph['images'],ph['labels'])}
    prepared={ds:tensors(v[0]) for ds,v in data.items()};out=R/'evaluation';out.mkdir(exist_ok=True)
    results=[]
    for name in names:
        kind=name.replace('_NoAug','')
        qs={ds:None if kind in ['NoGraph','RGBOnly','SmallCNN','ResNet18','MobileNetV3Small'] else torch.tensor(graph_matrix(f,kind)) for ds,(_,_,f) in prepared.items()}
        for seed in CFG['main_seeds']:
            folder=R/'models'/name/str(seed);dest=out/name/str(seed);dest.mkdir(exist_ok=True,parents=True)
            if (dest/'metrics.json').exists():results.append(json.loads((dest/'metrics.json').read_text()));continue
            checkpoint_path=folder/'checkpoint.pt'
            if not checkpoint_path.exists() or not (folder/'training.json').exists():print('Not yet complete',name,seed,flush=True);continue
            ck=torch.load(checkpoint_path,weights_only=False,map_location='cpu');model=make_model(kind);model.load_state_dict(ck['state']);model.eval()
            trainlog=json.loads((folder/'training.json').read_text());cal=np.load(folder/'development.npz');cal_logits=cal['calibration_logits'];cal_prob=softmax(cal_logits/ck['temperature'],axis=1)
            # Calibration indices refer to the pooled original train + validation dataset.
            _,dy,_,_=load_development();cal_y=dy[cal['calibration_index']]
            reject_threshold=float(np.quantile(cal_prob.max(1),.2));mel95=float(np.quantile(cal_prob[cal_y==4,4],.05,method='lower'))
            row=dict(name=name,kind=kind,seed=seed,parameters=trainlog['parameters'],temperature=ck['temperature'],n=trainlog['n'],best_epoch=trainlog['best_epoch'],tuning_macro_f1=trainlog['tuning_macro_f1'],reject_threshold=reject_threshold,mel95_threshold=mel95,datasets={})
            arrays={}
            for ds,(imgs,raw,f) in prepared.items():
                logits=predict(model,imgs,raw,qs[ds],ck['mu'],ck['sd'],kind);rawp=softmax(logits,1);calp=softmax(logits/ck['temperature'],1);y=data[ds][1]
                arrays[ds+'_logits']=logits;arrays[ds+'_raw']=rawp;arrays[ds+'_calibrated']=calp
                if ds=='PH2':row['datasets'][ds]={'raw':binary_metrics(y,rawp),'calibrated':binary_metrics(y,calp),'mel95':binary_metrics(y,calp,mel95)}
                else:
                    keep=calp.max(1)>=reject_threshold
                    row['datasets'][ds]={'raw':metrics(y,rawp),'calibrated':metrics(y,calp),'mel95':binary_metrics(y==4,calp,mel95),'selective':{'n_accepted':int(keep.sum()),'coverage':float(keep.mean()),'accuracy':float((calp[keep].argmax(1)==y[keep]).mean()),'mel_rejected':int(np.sum(~keep&(y==4)))}}
            np.savez_compressed(dest/'predictions.npz',**arrays,calibration_probs=cal_prob,calibration_labels=cal_y)
            (dest/'metrics.json').write_text(json.dumps(row,indent=2));results.append(row)
            (out/'all_run_metrics.json').write_text(json.dumps(results,indent=2));print('Evaluated',name,seed,flush=True)
    # Preserve outputs from any other model-specific evaluation job.
    allrows=[json.loads(p.read_text()) for p in out.glob('*/*/metrics.json')]
    (out/'all_run_metrics.json').write_text(json.dumps(allrows,indent=2));print('Evaluated runs',len(allrows),flush=True)
if __name__=='__main__':
    import argparse
    main()
