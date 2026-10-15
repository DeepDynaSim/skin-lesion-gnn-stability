from experiment_core import *
def main():
    im,y,groups,meta=load_development();images,raw,f=tensors(im);yt=torch.tensor(y,dtype=torch.long)
    out=R/'sensitivity';out.mkdir(exist_ok=True);rows=[]
    cases=[('default',0,dict(DEFAULT))]
    for parameter,values in CFG['sensitivity_ranges'].items():
        for value in values:
            if value==DEFAULT[parameter]:continue
            conf=dict(DEFAULT);conf[parameter]=value;cases.append((parameter,value,conf))
    for parameter,value,conf in cases:
        q=torch.tensor(graph_matrix(f,'ContrastPenalty',conf))
        for seed in CFG['sensitivity_seeds']:
            prefix={'k':'neighbors','K':'steps'}.get(parameter,parameter)
            key=f'{prefix}_{value}_{seed}';folder=out/key
            if (folder/'sensitivity.json').exists():
                record=json.loads((folder/'sensitivity.json').read_text());assert record['parameter']==parameter and record['value']==value and record['configuration']==conf
                checkpoint=torch.load(folder/'checkpoint.pt',weights_only=False,map_location='cpu')
                if checkpoint.get('actual_graph_configuration')!=conf:checkpoint['actual_graph_configuration']=conf;torch.save(checkpoint,folder/'checkpoint.pt')
                rows.append(record);continue
            split=partitions(y,groups,seed);m,mu,sd,result=train('ContrastPenalty',seed,images,yt,raw,q,split,True,save_dir=folder,sensitivity=True)
            ix=split['tune'];logits=predict(m,images[ix],raw[ix],q[ix],mu,sd,'ContrastPenalty');p=softmax(logits,1)
            select=ix[:64];rng=np.random.default_rng(20261014);pert=np.clip(im[select].astype(float)+rng.normal(0,.02*255,im[select].shape),0,255).astype(np.uint8)
            fp=features(pert);Sc=graph_matrix(tuple(x[select] if isinstance(x,np.ndarray) and x.shape[0]==len(im) else x for x in f),'ContrastPenalty',conf,True)
            Sp=graph_matrix(fp,'ContrastPenalty',conf,True)
            deltas=[float(np.linalg.norm(a-b,2)) for a,b in zip(Sc,Sp)]
            result.update(parameter=parameter,value=value,configuration=conf,tuning_metrics=metrics(y[ix],p),graph_spectral_delta_mean=float(np.mean(deltas)),graph_spectral_delta_sd=float(np.std(deltas,ddof=1)))
            checkpoint=torch.load(folder/'checkpoint.pt',weights_only=False,map_location='cpu');checkpoint['actual_graph_configuration']=conf;torch.save(checkpoint,folder/'checkpoint.pt')
            (folder/'sensitivity.json').write_text(json.dumps(result,indent=2));rows.append(result)
            (out/'sensitivity_results.json').write_text(json.dumps(rows,indent=2));print('Sensitivity complete',parameter,value,seed,flush=True)
    (out/'sensitivity_results.json').write_text(json.dumps(rows,indent=2))
if __name__=='__main__':main()
