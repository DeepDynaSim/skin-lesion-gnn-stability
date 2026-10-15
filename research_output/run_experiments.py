"""Auditable CPU experiment. No label is used to construct an image graph."""
from pathlib import Path
import sys,os,json,csv,time,hashlib,platform,copy
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'python_packages'))
os.environ['MPLCONFIGDIR']=str(ROOT/'tmp/mpl')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score,roc_auc_score,confusion_matrix,average_precision_score
from scipy.optimize import minimize_scalar
from scipy.special import softmax,logsumexp
torch.set_num_threads(4)
torch.set_num_interop_threads(2)
torch.use_deterministic_algorithms(True)
OUT=ROOT/'experiments'; OUT.mkdir(exist_ok=True)
CLASSES=['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
SEEDS=[11,29,47]
CONFIG=dict(seeds=SEEDS,epochs=60,patience=10,batch_size=256,lr=0.002,weight_decay=1e-4,hidden=32,alpha=0.2,K=4,beta=1.5,k=4,sigma_color=0.25,sigma_position=0.4,threads=4)

def features(images):
    im=images.astype(np.float32)/255
    p=im.reshape(-1,7,4,7,4,3).transpose(0,1,3,2,4,5).reshape(-1,49,4,4,3)
    mean=p.mean((2,3)); sd=p.std((2,3))
    gx=np.abs(np.diff(p,axis=3)).mean((2,3)); gy=np.abs(np.diff(p,axis=2)).mean((2,3))
    yy,xx=np.mgrid[0:7,0:7]; coord=np.c_[xx.ravel()/6,yy.ravel()/6].astype(np.float32)
    radius=np.linalg.norm(coord-.5,axis=1,keepdims=True)
    pos=np.broadcast_to(np.c_[coord,radius],(len(im),49,3))
    x=np.concatenate([mean,sd,gx,gy,pos],axis=-1).astype(np.float32)
    boundary=np.clip((gx+gy).mean(-1)/0.15,0,1)
    return x,mean,coord,boundary

def graph(mean,coord,boundary,mode='contrast',beta=1.5):
    dd=((mean[:,:,None,:]-mean[:,None,:,:])**2).sum(-1)
    dp=((coord[:,None,:]-coord[None,:,:])**2).sum(-1)
    local=np.isclose(dp,(1/6)**2,atol=1e-5)
    if mode=='uniform':
        w=np.broadcast_to(local,(len(mean),49,49)).copy().astype(np.float32)
    else:
        order=np.argsort(dd+np.eye(49,dtype=np.float32)[None]*100,axis=-1)[:,:,:4]
        mask=np.broadcast_to(local,(len(mean),49,49)).copy()
        np.put_along_axis(mask,order,True,axis=-1)
        mask=mask|mask.transpose(0,2,1)
        w=mask*np.exp(-dd/(2*.25**2)-dp[None]/(2*.4**2)-beta*(boundary[:,:,None]+boundary[:,None,:])/2)
    w=w+np.eye(49,dtype=np.float32)[None]
    degree=w.sum(-1); s=w/np.sqrt(degree[:,:,None]*degree[:,None,:])
    return s.astype(np.float32)

def restart(s,alpha=.2,K=4):
    # Q_K Z is exactly equivalent to K successive restart updates.
    q=np.broadcast_to(np.eye(49,dtype=np.float32),(len(s),49,49)).copy()
    for _ in range(K): q=(1-alpha)*(s@q)+alpha*np.eye(49,dtype=np.float32)[None]
    return q.astype(np.float32)

class GraphModel(nn.Module):
    def __init__(self,kind):
        super().__init__(); self.kind=kind
        self.encoder=nn.Sequential(nn.Linear(15,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        if kind=='GCN': self.g1=nn.Linear(32,32); self.g2=nn.Linear(32,32)
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,7))
    def forward(self,x,s):
        h=self.encoder(x)
        if self.kind=='GCN': h=F.relu(self.g2(s@F.relu(self.g1(s@h))))
        elif self.kind!='NoGraph': h=s@h
        return self.head(torch.cat([h.mean(1),h.amax(1)],-1))

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(3,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(16,32,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),nn.Conv2d(32,32,3,padding=1),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,7))
    def forward(self,x,s):
        h=self.net(x); return self.head(torch.cat([h.mean((2,3)),h.amax((2,3))],-1))

def predict(model,x,s):
    model.eval()
    with torch.no_grad(): return torch.cat([model(x[i:i+256],s[i:i+256]) for i in range(0,len(x),256)]).numpy()

def metrics(y,p):
    pred=p.argmax(-1); cm=confusion_matrix(y,pred,labels=np.arange(7)); tp=cm.diagonal(); fn=cm.sum(1)-tp; fp=cm.sum(0)-tp; tn=cm.sum()-tp-fn-fp
    conf=p.max(1); bins=np.linspace(0,1,11); ece=0
    for j in range(10):
        mask=(conf>=bins[j])&(conf<(bins[j+1]) if j<9 else conf<=1)
        if mask.any(): ece+=mask.mean()*abs((pred[mask]==y[mask]).mean()-conf[mask].mean())
    return dict(accuracy=float(accuracy_score(y,pred)),balanced_accuracy=float(balanced_accuracy_score(y,pred)),macro_f1=float(f1_score(y,pred,average='macro',zero_division=0)),macro_auc=float(roc_auc_score(y,p,multi_class='ovr',average='macro')),macro_ap=float(average_precision_score(np.eye(7)[y],p,average='macro')),nll=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-12,1)).mean()),brier=float(((p-np.eye(7)[y])**2).sum(1).mean()),ece=float(ece),mel_sensitivity=float(tp[4]/(tp[4]+fn[4])),mel_specificity=float(tn[4]/(tn[4]+fp[4])),confusion=cm.tolist())

def main():
    start=time.time(); data=np.load(ROOT/'data/dermamnist_corrected_28.npz')
    audit=json.loads((ROOT/'data/audit.json').read_text()); assert all(v['metadata_label_match'] for v in audit['splits'].values()); assert all(v['lesion_ids']==0 and v['exact_images']==0 for v in audit['overlap'].values())
    fs={s:features(data[s+'_images']) for s in ['train','val','test']}
    mu=fs['train'][0].mean((0,1)); sd=fs['train'][0].std((0,1)); sd=np.maximum(sd,1e-5)
    xs={s:torch.tensor((f[0]-mu)/sd) for s,f in fs.items()}; ys={s:torch.tensor(data[s+'_labels'].ravel(),dtype=torch.long) for s in fs}
    imgs={s:torch.tensor(data[s+'_images'].astype(np.float32).transpose(0,3,1,2)/255) for s in fs}
    matrices={}
    for kind in ['GCN','UniformRWR','ContrastNoBoundary','Proposed']:
        matrices[kind]={}
        for split,f in fs.items():
            s=graph(*f[1:],mode='uniform' if kind in ['GCN','UniformRWR'] else 'contrast',beta=0 if kind=='ContrastNoBoundary' else 1.5)
            matrices[kind][split]=torch.tensor(s if kind=='GCN' else restart(s))
    identity={s:torch.eye(49).repeat(len(xs[s]),1,1) for s in fs}
    freq=np.bincount(ys['train'].numpy(),minlength=7)
    weights=torch.tensor(np.sqrt(len(ys['train'])/(7*freq)),dtype=torch.float32)
    results=[]; histories=json.loads((OUT/'histories.json').read_text()) if (OUT/'histories.json').exists() else []
    for kind in ['NoGraph','GCN','UniformRWR','ContrastNoBoundary','Proposed','CNN']:
        for seed in SEEDS:
            key=f'{kind}_{seed}'; path=OUT/(key+'.npz'); model_path=OUT/(key+'.pt')
            if path.exists() and (OUT/(key+'.json')).exists():
                results.append(json.loads((OUT/(key+'.json')).read_text())); continue
            torch.manual_seed(seed); np.random.seed(seed)
            model=CNNModel() if kind=='CNN' else GraphModel(kind)
            x=imgs if kind=='CNN' else xs; s=matrices.get(kind,identity)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
            best=-1; patience=0; history=[]; tick=time.time()
            for epoch in range(60):
                model.train(); order=torch.randperm(len(ys['train'])); loss_sum=0
                for idx in order.split(256):
                    optimizer.zero_grad(); logits=model(x['train'][idx],s['train'][idx]); loss=F.cross_entropy(logits,ys['train'][idx],weight=weights); loss.backward(); optimizer.step(); loss_sum+=float(loss.detach())*len(idx)
                val_logits=predict(model,x['val'],s['val']); vf=f1_score(ys['val'],val_logits.argmax(-1),average='macro',zero_division=0)
                history.append(dict(epoch=epoch+1,loss=loss_sum/len(order),val_macro_f1=float(vf)))
                if vf>best+1e-6: best=float(vf); state=copy.deepcopy(model.state_dict()); best_epoch=epoch+1; patience=0
                else: patience+=1
                if epoch%10==0: print(key,epoch+1,round(vf,4),'seconds',round(time.time()-tick,1),flush=True)
                if patience>=10: break
            model.load_state_dict(state)
            val_logits=predict(model,x['val'],s['val']); test_logits=predict(model,x['test'],s['test'])
            objective=lambda logt:float(np.mean(logsumexp(val_logits/np.exp(logt),axis=1)-val_logits[np.arange(len(val_logits)),ys['val'].numpy()]/np.exp(logt)))
            opt=minimize_scalar(objective,bounds=(np.log(.1),np.log(10)),method='bounded'); temperature=float(np.exp(opt.x))
            probs=softmax(test_logits,axis=1); cal=softmax(test_logits/temperature,axis=1)
            row=dict(kind=kind,seed=seed,best_epoch=best_epoch,val_macro_f1=best,temperature=temperature,parameters=sum(p.numel() for p in model.parameters()),train_seconds=time.time()-tick,raw=metrics(ys['test'].numpy(),probs),calibrated=metrics(ys['test'].numpy(),cal))
            np.savez_compressed(path,val_logits=val_logits,test_logits=test_logits,probs=probs,calibrated_probs=cal)
            torch.save(dict(state=state,mu=mu.tolist(),sd=sd.tolist(),config=CONFIG),model_path)
            (OUT/(key+'.json')).write_text(json.dumps(row,indent=2)); results.append(row); histories.append(dict(kind=kind,seed=seed,history=history))
            print('DONE',key,{k:round(row['raw'][k],4) for k in ['accuracy','macro_f1','macro_auc']},flush=True)
            (OUT/'results.json').write_text(json.dumps(results,indent=2))
            (OUT/'histories.json').write_text(json.dumps(histories,indent=2))
    env=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),processor=platform.processor(),elapsed_seconds=time.time()-start)
    (OUT/'environment.json').write_text(json.dumps(env,indent=2)); (OUT/'config.json').write_text(json.dumps(CONFIG,indent=2))
    (OUT/'results.json').write_text(json.dumps(results,indent=2)); np.savez(OUT/'normalization.npz',mu=mu,sd=sd)

if __name__=='__main__': main()
