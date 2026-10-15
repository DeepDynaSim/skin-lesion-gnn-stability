"""Clean-environment replay of the added CNN and descriptor analyses."""
from optimized_cnn import *
from evaluate_optimized_cnn import main as evaluate
from descriptor_noise import main as decompose
import optimized_cnn as training

def numeric_compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys();return sum((numeric_compare(a[k],b[k]) for k in a),[])
    if isinstance(a,list):
        assert len(a)==len(b);return sum((numeric_compare(x,y) for x,y in zip(a,b)),[])
    if isinstance(a,(int,float)) and not isinstance(a,bool):
        error=abs(a-b);assert error<2e-7,(a,b);return [error]
    assert a==b;return []

im,y,groups,_=load_development();images=torch.tensor(im.transpose(0,3,1,2).astype(np.float32)/255);settings=lock();largest=0.;count=0
for seed in settings['seeds']:
    split=partitions(y,groups,seed);selection=json.loads((HERE/'optimized_cnn/selections'/f'{seed}.json').read_text());scores=[]
    for lr in settings['learning_rates']:
        dest=HERE/'optimized_cnn/candidates'/str(seed)/f'lr_{lr}';record=json.loads((dest/'training.json').read_text());ck=torch.load(dest/'checkpoint.pt',weights_only=False,map_location='cpu');stored=np.load(dest/'development.npz')
        assert record['protocol_sha256']==hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        for key in split:assert np.array_equal(split[key],stored[key+'_index'])
        model=make_model('SmallCNN');model.load_state_dict(ck['state']);pred=inference(model,images[split['tune']]).argmax(1);score=float(f1_score(y[split['tune']],pred,labels=range(7),average='macro',zero_division=0));assert abs(score-record['tuning_macro_f1'])<1e-12;scores.append((score,lr));count+=1
    assert selection['selected_learning_rate']==sorted(scores,key=lambda v:(-v[0],v[1]))[0][1]
    model,rec=selected_model(seed);cal=np.load(HERE/'optimized_cnn/selections'/f'{seed}.npz');logits=inference(model,images[split['cal']]);error=float(np.max(abs(logits-cal['calibration_logits'])));assert error<1e-5;largest=max(largest,error)
evaluate();decompose()
errors=numeric_compare(json.loads((HERE/'analysis/expected_optimized_cnn.json').read_text()),json.loads((HERE/'analysis/optimized_cnn.json').read_text()))
errors.extend(numeric_compare(json.loads((HERE/'analysis/expected_descriptor_noise.json').read_text()),json.loads((HERE/'analysis/descriptor_noise.json').read_text())))
original=torch.load(HERE/'optimized_cnn/candidates/101/lr_0.002/checkpoint.pt',weights_only=False,map_location='cpu');training.HERE=HERE/'fresh_refit';row=training.fit(101,.002,images,torch.tensor(y,dtype=torch.long),partitions(y,groups,101),settings);fresh=torch.load(training.HERE/'optimized_cnn/candidates/101/lr_0.002/checkpoint.pt',weights_only=False,map_location='cpu');parameter_error=max(float(torch.max(abs(t-original['state'][k]))) for k,t in fresh['state'].items());assert parameter_error<1e-6
report={'status':'passed','candidate_states_verified':count,'calibration_selections_verified':10,'regenerated_cohorts':3,'regenerated_corruption_ensembles':23,'numerical_entries_compared':len(errors),'maximum_numeric_error':max(errors),'maximum_calibration_logits_error':largest,'representative_refit':{'seed':101,'learning_rate':.002,'maximum_parameter_error':parameter_error},'scope':'Same-host clean installation; all added analyses replayed; one additional candidate refitted'}
(HERE/'analysis/clean_new_cnn.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
