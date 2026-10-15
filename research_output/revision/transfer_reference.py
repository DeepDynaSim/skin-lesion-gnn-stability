from experiment_core import *
from torchvision.models import MobileNet_V3_Small_Weights
from evaluate_revision import binary_metrics
from summarize_revision import fixed_metrics_fast,class_metrics
from corruption_revision import all_cases,corrupted
import urllib.request,argparse

TR=R/'transfer_reference';TR.mkdir(exist_ok=True);WEIGHT=R/'data/mobilenet_v3_small-047dcff4.pth'

def backbone():
    weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1;url=weights.url;os.environ.pop('SSLKEYLOGFILE',None)
    if not WEIGHT.exists():
        req=urllib.request.Request(url,headers={'User-Agent':'ScientificReference/1.0'})
        with urllib.request.urlopen(req,timeout=60) as response,WEIGHT.open('wb') as output:
            while chunk:=response.read(1024*1024):output.write(chunk)
    sha=hashlib.sha256(WEIGHT.read_bytes()).hexdigest();assert sha.startswith('047dcff4')
    (TR/'weight_provenance.json').write_text(json.dumps({'url':url,'sha256':sha,'bytes':WEIGHT.stat().st_size,'initialization':'Official IMAGENET1K_V1; no skin-lesion pretraining','preprocessing':str(weights.transforms())},indent=2))
    model=mobilenet_v3_small(weights=None);model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True));model.classifier=nn.Identity();model.requires_grad_(False);model.eval();return model,weights.transforms(),sha

def embed(model,transform,images,key):
    cache=TR/'cache';cache.mkdir(exist_ok=True);path=cache/(key+'.npy')
    if path.exists():return np.load(path)
    out=[];tick=time.time()
    with torch.no_grad():
        for i in range(0,len(images),64):
            x=torch.tensor(images[i:i+64].transpose(0,3,1,2).copy());out.append(model(transform(x)).numpy())
            if i%2048==0:print('Frozen embeddings',key,i,'/',len(images),'seconds',round(time.time()-tick,1),flush=True)
    features=np.concatenate(out);assert features.shape==(len(images),576);np.save(path,features);return features

def fit(seed,x,y,split,sha):
    folder=TR/'models'/str(seed);folder.mkdir(exist_ok=True,parents=True)
    if (folder/'training.json').exists():return
    torch.manual_seed(seed);model=nn.Linear(576,7);mu=x[split['train']].mean(0);sd=x[split['train']].std(0,correction=0).clamp_min(1e-5);xx=(x-mu)/sd;freq=np.bincount(y[split['train']].numpy(),minlength=7);weight=torch.tensor(np.sqrt(len(split['train'])/(7*freq)),dtype=torch.float32);optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4);best=-1;wait=0;history=[];tick=time.time()
    for epoch in range(60):
        model.train();order=torch.tensor(split['train'])[torch.randperm(len(split['train']))]
        for idx in order.split(256):optimizer.zero_grad();loss=F.cross_entropy(model(xx[idx]),y[idx],weight=weight);loss.backward();optimizer.step()
        model.eval()
        with torch.no_grad():score=float(f1_score(y[split['tune']],model(xx[split['tune']]).argmax(1),labels=range(7),average='macro',zero_division=0))
        history.append({'epoch':epoch+1,'tuning_macro_f1':score})
        if score>best+1e-6:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;wait=0
        else:wait+=1
        if wait>=10:break
    model.load_state_dict(state);ci=split['cal']
    with torch.no_grad():logits=model(xx[ci]).numpy()
    objective=lambda lt:float(np.mean(logsumexp(logits/np.exp(lt),1)-logits[np.arange(len(ci)),y[ci].numpy()]/np.exp(lt)));temp=float(np.exp(minimize_scalar(objective,bounds=(np.log(.1),np.log(10)),method='bounded').x))
    row={'seed':seed,'learned_parameters':4039,'best_epoch':best_epoch,'tuning_macro_f1':best,'temperature':temp,'train_seconds':time.time()-tick,'history':history,'n':{k:len(ix) for k,ix in split.items()},'augmentation':False,'weight_sha256':sha}
    torch.save({'state':state,'mu':mu,'sd':sd,'seed':seed,'temperature':temp,'weight_sha256':sha},folder/'checkpoint.pt');np.savez_compressed(folder/'development.npz',train_index=split['train'],tuning_index=split['tune'],calibration_index=ci,calibration_logits=logits);(folder/'training.json').write_text(json.dumps(row,indent=2));print('Transfer head',seed,best_epoch,round(best,4),flush=True)

def probs(heads,embedding):
    raw=[];cal=[];log=[];x=torch.tensor(embedding)
    with torch.no_grad():
        for model,ck in heads:
            logits=model((x-ck['mu'])/ck['sd']).numpy();log.append(logits);raw.append(softmax(logits,1));cal.append(softmax(logits/ck['temperature'],1))
    return np.mean(raw,0),np.mean(cal,0),raw,cal,log

def main():
    model,transform,sha=backbone();images,y,groups,_=load_development();embedding=embed(model,transform,images,'development');x=torch.tensor(embedding);yt=torch.tensor(y,dtype=torch.long)
    for seed in CFG['main_seeds']:fit(seed,x,yt,partitions(y,groups,seed),sha)
    heads=[];rows=[];calp=[];caly=[]
    for seed in CFG['main_seeds']:
        folder=TR/'models'/str(seed);ck=torch.load(folder/'checkpoint.pt',weights_only=False,map_location='cpu');head=nn.Linear(576,7);head.load_state_dict(ck['state']);head.eval();heads.append((head,ck));rows.append(json.loads((folder/'training.json').read_text()));dev=np.load(folder/'development.npz');calp.append(softmax(dev['calibration_logits']/ck['temperature'],1));caly.append(y[dev['calibration_index']])
    d=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz');ph=np.load(R/'data/ph2_28.npz');data={'C_test':(d['test_images'],d['test_labels'].ravel()),'E_test':(e['test_images'],e['test_labels'].ravel()),'PH2':(ph['images'],ph['labels'])};arrays={};result={'learned_parameters':4039,'fixed_backbone_parameters':sum(p.numel() for p in model.parameters()),'head_fits':10,'training':rows,'datasets':{},'corruptions':{},'weight_sha256':sha}
    cp=np.concatenate(calp);cy=np.concatenate(caly);reject=float(np.quantile(cp.max(1),.2));mel95=float(np.quantile(cp[cy==4,4],.05,method='lower'));result['thresholds']={'reject':reject,'mel95':mel95}
    for ds,(im,yy) in data.items():
        z=embed(model,transform,im,ds);rp,tcp,sraw,scal,logs=probs(heads,z);arrays[ds+'_raw']=rp;arrays[ds+'_calibrated']=tcp;arrays[ds+'_seed_raw']=np.array(sraw);arrays[ds+'_seed_calibrated']=np.array(scal);arrays[ds+'_seed_logits']=np.array(logs);v={'raw':binary_metrics(yy,rp) if ds=='PH2' else metrics(yy,rp),'calibrated':binary_metrics(yy,tcp) if ds=='PH2' else metrics(yy,tcp),'mel95':binary_metrics(yy if ds=='PH2' else yy==4,tcp,mel95)}
        if ds!='PH2':v['classwise']=class_metrics(yy,rp);keep=tcp.max(1)>=reject;v['selective']={'coverage':float(keep.mean()),'accuracy':float((tcp[keep].argmax(1)==yy[keep]).mean()),'mel_rejected':int(np.sum(~keep&(yy==4)))}
        v['seed_mean']={};v['seed_sd']={}
        seed_metrics=[binary_metrics(yy,p) if ds=='PH2' else metrics(yy,p) for p in sraw]
        for k,val in seed_metrics[0].items():
            if isinstance(val,float):v['seed_mean'][k]=float(np.mean([r[k] for r in seed_metrics]));v['seed_sd'][k]=float(np.std([r[k] for r in seed_metrics],ddof=1))
        result['datasets'][ds]=v
    np.savez_compressed(TR/'predictions.npz',**arrays)
    for key,family,value,seed in all_cases():
        saved=TR/'corruptions';saved.mkdir(exist_ok=True);path=saved/(key+'.npz')
        if path.exists():p=np.load(path)['raw']
        else:z=embed(model,transform,corrupted(d['test_images'],family,value,seed),key);p=probs(heads,z)[0];np.savez_compressed(path,raw=p)
        result['corruptions'][key]={'family':family,'value':value,'seed':seed,'metrics':metrics(data['C_test'][1],p)};print('Transfer corruption',key,round(result['corruptions'][key]['metrics']['macro_f1'],4),flush=True)
    # Conditional ensemble uncertainty; same C cluster structure as the core study.
    meta=[r for r in csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()) if r['split']=='test'];ids=np.array([r['lesion_id'] for r in meta]);clusters=[np.flatnonzero(ids==v) for v in np.unique(ids)];rng=np.random.default_rng(20261014);boot={ds:[] for ds in data}
    for _ in range(1000):
        indices={'C_test':np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))]),'E_test':rng.integers(0,1511,1511),'PH2':rng.integers(0,200,200)}
        for ds,ix in indices.items():boot[ds].append(binary_metrics(data[ds][1][ix],arrays[ds+'_raw'][ix]) if ds=='PH2' else fixed_metrics_fast(data[ds][1][ix],arrays[ds+'_raw'][ix]))
    for ds in data:result['datasets'][ds]['ci95']={k:np.quantile([v[k] for v in boot[ds]],[.025,.975]).tolist() for k in boot[ds][0] if k not in ['n','tp','tn','fn','fp']}
    result['corruption_mean']={}
    for family,value in [(a,v) for _,a,v,_ in all_cases()]:
        group=f'{family}_{value}'
        if group in result['corruption_mean']:continue
        rr=[r for r in result['corruptions'].values() if r['family']==family and r['value']==value];result['corruption_mean'][group]={k:float(np.mean([r['metrics'][k] for r in rr])) for k in ['macro_f1','balanced_accuracy','macro_ap']}
    (R/'analysis/transfer_reference.json').write_text(json.dumps(result,indent=2));print('Frozen ImageNet reference complete',flush=True)
if __name__=='__main__':main()
