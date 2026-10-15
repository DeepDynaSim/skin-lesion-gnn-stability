"""Executed only in the extracted S1 tree with the clean Python installation."""
from experiment_core import *
from corruption_revision import all_cases,corrupted
from transfer_reference import backbone,embed,probs
from sklearn.metrics import f1_score

c=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz');ph=np.load(R/'data/ph2_28.npz');dev,dy,dg,_=load_development();dev_im,dev_raw,dev_f=tensors(dev)
cohorts={'C_test':c['test_images'],'E_test':e['test_images'],'PH2':ph['images']};prepared={ds:tensors(im) for ds,im in cohorts.items()};largest=0.;count=0
for name in CFG['central_models']+CFG['augmentation_controls']+['RGBOnlyGraph']:
    kind=name.replace('_NoAug','');qs={ds:None if kind in ['NoGraph','RGBOnly','SmallCNN','ResNet18','MobileNetV3Small'] else torch.tensor(graph_matrix(f,kind)) for ds,(_,_,f) in prepared.items()}
    for seed in CFG['main_seeds']:
        ck=torch.load(R/'models'/name/str(seed)/'checkpoint.pt',weights_only=False,map_location='cpu');model=make_model(kind);model.load_state_dict(ck['state']);saved=np.load(R/'evaluation'/name/str(seed)/'predictions.npz')
        for ds,(im,raw,f) in prepared.items():
            logits=predict(model,im,raw,qs[ds],ck['mu'],ck['sd'],kind);error=float(np.max(abs(logits-saved[ds+'_logits'])));assert error<1e-5,(name,seed,ds,error);largest=max(largest,error)
        count+=1
    print('Fresh three-cohort checkpoint inference',name,flush=True)
sens=json.loads((R/'sensitivity/sensitivity_results.json').read_text());assert len(sens)==42
for row in sens:
    prefix={'k':'neighbors','K':'steps'}.get(row['parameter'],row['parameter']);p=R/'sensitivity'/f"{prefix}_{row['value']}_{row['seed']}";ck=torch.load(p/'checkpoint.pt',weights_only=False,map_location='cpu');model=make_model('ContrastPenalty');model.load_state_dict(ck['state']);split=partitions(dy,dg,row['seed']);ix=split['tune'];f=features(dev[ix]);q=torch.tensor(graph_matrix(f,config=row['configuration']));logits=predict(model,dev_im[ix],dev_raw[ix],q,ck['mu'],ck['sd'],'ContrastPenalty');score=float(f1_score(dy[ix],logits.argmax(1),labels=range(7),average='macro',zero_division=0));assert abs(score-row['tuning_macro_f1'])<1e-12
print('Fresh sensitivity checkpoint inference',len(sens),flush=True)
model,transform,_=backbone();heads=[]
for seed in CFG['main_seeds']:
    ck=torch.load(R/'transfer_reference/models'/str(seed)/'checkpoint.pt',weights_only=False,map_location='cpu');head=nn.Linear(576,7);head.load_state_dict(ck['state']);head.eval();heads.append((head,ck))
for key,family,value,seed in all_cases():
    z=embed(model,transform,corrupted(c['test_images'],family,value,seed),key);fresh=probs(heads,z)[0];saved=np.load(R/'transfer_reference/corruptions'/(key+'.npz'))['raw'];error=float(np.max(abs(fresh-saved)));assert error<2e-6,(key,error);largest=max(largest,error)
print('Fresh frozen-reference corruption predictions',23,flush=True)
(R/'clean_forward_audit.json').write_text(json.dumps({'status':'passed','primary_states_all_three_cohorts':count,'sensitivity_states_tuning_inference':len(sens),'frozen_heads':10,'fresh_frozen_corruption_ensembles':23,'largest_logits_or_probability_error':largest},indent=2))
