"""Prespecified corruptions, frozen-weight factorial substitutions and paired intervals."""
from experiment_core import *
from summarize_revision import fixed_metrics_fast
from scipy.ndimage import gaussian_filter
from PIL import Image
import argparse,io

def corrupted(images,family,value,seed=0):
    x=images.astype(np.float32)/255
    if family=='noise':x+=np.random.default_rng(seed).normal(0,value,x.shape).astype(np.float32)
    elif family=='blur':x=gaussian_filter(x,sigma=(0,value,value,0))
    elif family=='brightness':x*=value
    elif family=='contrast':
        mean=x.mean((1,2),keepdims=True);x=mean+value*(x-mean)
    elif family=='jpeg':
        out=[]
        for im in images:
            stream=io.BytesIO();Image.fromarray(im).save(stream,format='JPEG',quality=int(value));stream.seek(0);out.append(np.array(Image.open(stream).convert('RGB')))
        return np.array(out)
    return np.rint(np.clip(x,0,1)*255).astype(np.uint8)

def all_cases():
    for value in CFG['corruptions']['noise_sigma']:
        for seed in CFG['corruption_seeds']:yield f'noise_{value}_{seed}','noise',value,seed
    for key,family in [('blur_sigma','blur'),('jpeg_quality','jpeg'),('brightness','brightness'),('contrast','contrast')]:
        for value in CFG['corruptions'][key]:yield f'{family}_{value}',family,value,0

def topology(f):
    _,mean,coord,_=f;dd=((mean[:,:,None]-mean[:,None,:])**2).sum(-1);dp=((coord[:,None]-coord[None,:])**2).sum(-1)
    order=np.argsort(dd+100*np.eye(49)[None],axis=-1)[:,:,:4];mask=np.broadcast_to(np.isclose(dp,(1/6)**2,atol=1e-5),(len(mean),49,49)).copy();np.put_along_axis(mask,order,True,-1);mask|=mask.transpose(0,2,1);return mask

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='*');args=parser.parse_args()
    names=args.models or CFG['central_models']+CFG['augmentation_controls']+['RGBOnlyGraph']
    dest=R/'corruptions';dest.mkdir(exist_ok=True);d=np.load(OLD/'data/dermamnist_corrected_28.npz');images=d['test_images'];y=d['test_labels'].ravel();clean_im,clean_raw,clean_f=tensors(images)
    cases=list(all_cases());prepared={};clean_mask=topology(clean_f);sample=np.arange(64)
    for key,family,value,seed in cases:
        im=corrupted(images,family,value,seed);prepared[key]=tensors(im)
        fp=prepared[key][2];mask=topology(fp);upper=np.triu(np.ones((49,49),bool),1);intersection=(mask&clean_mask)[:,upper].sum(1);union=(mask|clean_mask)[:,upper].sum(1)
        p=dest/(key+'_structure.json')
        if not p.exists():
            s=graph_matrix(clean_f,return_s=True);sp=graph_matrix(fp,return_s=True);ds=np.linalg.svd((s-sp)[sample].astype(np.float64),compute_uv=False)[:,0]
            p.write_text(json.dumps(dict(case=key,family=family,value=value,seed=seed,spectral_delta_mean=float(ds.mean()),spectral_delta_p95=float(np.quantile(ds,.95)),candidate_jaccard_mean=float((intersection/union).mean()),candidate_edge_changed_fraction=float((intersection<union).mean()),raw_descriptor_relative=float(np.linalg.norm(fp[0]-clean_f[0])/np.linalg.norm(clean_f[0])),spectral_sample_indices=sample.tolist()),indent=2))
    for name in names:
        paths=[R/'models'/name/str(seed)/'training.json' for seed in CFG['main_seeds']]
        if not all(p.exists() for p in paths):print('Waiting for complete fits',name,flush=True);continue
        kind=name.replace('_NoAug','');graph_kind=kind not in ['NoGraph','RGBOnly','SmallCNN','ResNet18','MobileNetV3Small'];folder=dest/name;folder.mkdir(exist_ok=True)
        pending=[case for case in cases if not (folder/(case[0]+'.json')).exists()]
        if not pending:continue
        models=[]
        for seed in CFG['main_seeds']:
            ck=torch.load(R/'models'/name/str(seed)/'checkpoint.pt',map_location='cpu',weights_only=False);model=make_model(kind);model.load_state_dict(ck['state']);model.eval();models.append((model,ck))
        clean_q=torch.tensor(graph_matrix(clean_f,kind)) if graph_kind else None
        clean_probs=[]
        for model,ck in models:clean_probs.append(softmax(predict(model,clean_im,clean_raw,clean_q,ck['mu'],ck['sd'],kind),1))
        base=np.mean(clean_probs,0)
        for key,family,value,seed in pending:
            dirty_im,dirty_raw,dirty_f=prepared[key];dirty_q=torch.tensor(graph_matrix(dirty_f,kind)) if graph_kind else None
            conditions={'D':(dirty_im,dirty_raw,dirty_q)}
            if graph_kind:conditions.update(B=(clean_im,clean_raw,dirty_q),C=(dirty_im,dirty_raw,clean_q))
            arrays={'A':base};row=dict(name=name,case=key,family=family,value=value,seed=seed,conditions={})
            for label,(im,raw,q) in conditions.items():
                probs=[softmax(predict(model,im,raw,q,ck['mu'],ck['sd'],kind),1) for model,ck in models];arrays[label]=np.mean(probs,0)
                m=metrics(y,arrays[label]);m.pop('confusion');row['conditions'][label]=m
            row['conditions']['A']=metrics(y,base);row['conditions']['A'].pop('confusion')
            if graph_kind:
                changes=[];norms=[];ratios=[]
                with torch.no_grad():
                    for model,ck in models:
                        x=(clean_raw[sample]-ck['mu'])/ck['sd'];xp=(dirty_raw[sample]-ck['mu'])/ck['sd']
                        if kind=='RGBOnlyGraph':x=x.clone();xp=xp.clone();x[:,:,6:12]=0;xp[:,:,6:12]=0
                        z=model.encoder(x);zp=model.encoder(xp);changes.append(float(torch.linalg.vector_norm(z-zp)/torch.linalg.vector_norm(z)));norms.append(float(torch.linalg.vector_norm(x-xp)/torch.linalg.vector_norm(x)))
                        ds=torch.tensor(np.linalg.svd((graph_matrix(clean_f,kind,return_s=True)-graph_matrix(dirty_f,kind,return_s=True))[sample].astype(np.float64),compute_uv=False)[:,0],dtype=torch.float32)
                        bound=torch.linalg.vector_norm(z-zp,dim=(1,2))+2.3616*ds*torch.linalg.vector_norm(zp,dim=(1,2))
                        lf=float(torch.linalg.matrix_norm(model.head[0].weight,2)*torch.linalg.matrix_norm(model.head[3].weight,2));logits=model(x,clean_q[sample]);logitsp=model(xp,dirty_q[sample]);rhs=lf*np.sqrt(1+1/49)*bound;ratios.extend((torch.linalg.vector_norm(logits-logitsp,dim=1)/rhs.clamp_min(1e-8)).tolist())
                row['encoded_feature_relative_mean']=float(np.mean(changes));row['standardized_descriptor_relative_mean']=float(np.mean(norms));row['conditional_logit_bound_ratio_max']=max(ratios)
            np.savez_compressed(folder/(key+'.npz'),**arrays);(folder/(key+'.json')).write_text(json.dumps(row,indent=2));print('Corruption',name,key,round(row['conditions']['D']['macro_f1'],4),flush=True)
    rows=[json.loads(p.read_text()) for p in dest.glob('*/*.json')];(dest/'all_metrics.json').write_text(json.dumps(rows,indent=2));print('Corruption ensembles',len(rows),flush=True)

if __name__=='__main__':main()
