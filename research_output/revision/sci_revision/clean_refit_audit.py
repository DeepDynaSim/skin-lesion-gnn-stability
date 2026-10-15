"""Two representative deterministic refits; never described as all-fit replication."""
from experiment_core import *
im,y,groups,_=load_development();images,raw,f=tensors(im);yt=torch.tensor(y,dtype=torch.long);results=[]
for name in ['ContrastPenalty','SmallCNN_NoAug']:
    kind=name.replace('_NoAug','');seed=101;split=partitions(y,groups,seed);q=None if kind=='SmallCNN' else torch.tensor(graph_matrix(f,kind));model,mu,sd,result=train(kind,seed,images,yt,raw,q,split,augment=not name.endswith('_NoAug'),save_dir=R/'fresh_refits'/name)
    original=torch.load(R/'models'/name/str(seed)/'checkpoint.pt',weights_only=False,map_location='cpu');difference=max(float(torch.max(abs(t-original['state'][k]))) for k,t in model.state_dict().items());assert difference<1e-6,(name,difference);assert abs(result['temperature']-original['temperature'])<1e-7
    results.append({'model':name,'seed':seed,'maximum_parameter_error':difference,'temperature_error':abs(result['temperature']-original['temperature']),'best_epoch':result['best_epoch']})
(R/'clean_refit_audit.json').write_text(json.dumps({'status':'passed','refits':results,'scope':'Two representative original recipes, not every model retrained'},indent=2));print('Representative fresh refits passed',flush=True)
