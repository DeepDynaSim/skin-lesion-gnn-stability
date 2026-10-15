"""Revision experiment primitives; all selection uses development labels only."""
from pathlib import Path
import os,sys,json,csv,copy,time,hashlib
R=Path(__file__).resolve().parent;OLD=R.parent
sys.path.insert(0,str(OLD/'python_packages'));sys.path.insert(0,str(OLD))
sys.path.insert(0,str(R/'python_packages'))
os.environ['MPLCONFIGDIR']=str(R/'tmp/mpl')
import numpy as np,torch
from torch import nn
from torch.nn import functional as F
from scipy.special import softmax,logsumexp
from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from torchvision.models import resnet18,mobilenet_v3_small
from run_experiments import features,metrics,CNNModel
sys.path.insert(0,str(R));sys.path.insert(0,str(R/'python_packages'))
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
CFG=json.loads((R/'protocol.json').read_text(encoding='utf-8'))
DEFAULT=CFG['default_graph']

def load_development():
    d=np.load(OLD/'data/dermamnist_corrected_28.npz')
    images=np.concatenate([d['train_images'],d['val_images']]); y=np.concatenate([d['train_labels'],d['val_labels']]).ravel()
    rows=list(csv.DictReader((OLD/'data/DermaMNIST-C.csv').open()));meta=[r for s in ['train','val'] for r in rows if r['split']==s]
    groups=np.array([r['lesion_id'] for r in meta]); assert len(groups)==len(y)
    return images,y,groups,meta

def partitions(y,groups,seed):
    pool,tune=next(StratifiedGroupKFold(n_splits=7,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
    t,c=next(StratifiedGroupKFold(n_splits=6,shuffle=True,random_state=seed+31).split(np.zeros(len(pool)),y[pool],groups[pool]))
    train,cal=pool[t],pool[c]
    sets=[set(groups[idx]) for idx in [train,tune,cal]]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return {'train':train,'tune':tune,'cal':cal}

def graph_matrix(f,kind='ContrastPenalty',config=None,return_s=False):
    conf=dict(DEFAULT);conf.update(config or {})
    _,mean,coord,b=f
    dd=((mean[:,:,None,:]-mean[:,None,:,:])**2).sum(-1)
    dp=((coord[:,None,:]-coord[None,:,:])**2).sum(-1)
    local=np.isclose(dp,(1/6)**2,atol=1e-5)
    order=np.argsort(dd+np.eye(49,dtype=np.float32)[None]*100,axis=-1)[:,:,:int(conf['k'])]
    knn=np.zeros((len(mean),49,49),dtype=bool);np.put_along_axis(knn,order,True,axis=-1);knn|=knn.transpose(0,2,1)
    mask=np.broadcast_to(local,(len(mean),49,49)).copy() if kind=='GridOnly' else knn if kind=='KNNOnly' else (knn|local[None])
    beta=0 if kind=='ContrastNoPenalty' else conf['beta']
    if kind=='UniformTopology':w=mask.astype(np.float32)
    else:w=mask*np.exp(-dd/(2*conf['sigma_color']**2)-dp[None]/(2*conf['sigma_position']**2)-beta*(b[:,:,None]+b[:,None,:])/2)
    w+=np.eye(49,dtype=np.float32)[None];degree=w.sum(-1)
    s=(w/np.sqrt(degree[:,:,None]*degree[:,None,:])).astype(np.float32)
    if return_s:return s
    alpha,K=conf['alpha'],int(conf['K']);q=np.broadcast_to(np.eye(49,dtype=np.float32),s.shape).copy()
    for _ in range(K):q=(1-alpha)*(s@q)+alpha*np.eye(49,dtype=np.float32)[None]
    return q.astype(np.float32)

class RegionalModel(nn.Module):
    def __init__(self,no_graph=False):
        super().__init__();self.no_graph=no_graph
        self.encoder=nn.Sequential(nn.Linear(15,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Dropout(.15),nn.Linear(32,7))
    def forward(self,x,q):
        z=self.encoder(x);h=z if self.no_graph else q@z
        return self.head(torch.cat([h.mean(1),h.amax(1)],-1))

class ImageModel(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='SmallCNN':self.model=CNNModel()
        elif kind=='ResNet18':self.model=resnet18(weights=None,num_classes=7)
        elif kind=='MobileNetV3Small':self.model=mobilenet_v3_small(weights=None,num_classes=7)
        else:raise ValueError(kind)
    def forward(self,x,q=None):return self.model(x,q) if self.kind=='SmallCNN' else self.model(x)

def make_model(kind):
    return ImageModel(kind) if kind in ['SmallCNN','ResNet18','MobileNetV3Small'] else RegionalModel(no_graph=kind in ['NoGraph','RGBOnly'])

def augment_region(raw,q,rotation,flip):
    grid=np.arange(49).reshape(7,7);grid=np.rot90(grid,rotation)
    if flip:grid=np.fliplr(grid)
    perm=torch.tensor(grid.ravel().copy(),dtype=torch.long)
    x=raw[:,perm].clone()
    if rotation%2:x[:,:,6:9],x[:,:,9:12]=x[:,:,9:12].clone(),x[:,:,6:9].clone()
    x[:,:,12:]=raw[:,: ,12:]
    matrix=q[:,perm][:,:,perm] if q is not None else None
    return x,matrix

def predict(model,images,raw,q,mu,sd,kind,batch=256):
    model.eval();outs=[]
    with torch.no_grad():
        for i in range(0,len(raw),batch):
            x=images[i:i+batch] if isinstance(model,ImageModel) else (raw[i:i+batch]-mu)/sd
            if kind in ['RGBOnly','RGBOnlyGraph']:x=x.clone();x[:,:,6:12]=0
            outs.append(model(x,None if q is None else q[i:i+batch]))
    return torch.cat(outs).numpy()

def train(kind,seed,images,y,raw,q,split,augment=True,save_dir=None,sensitivity=False):
    torch.manual_seed(seed);np.random.seed(seed)
    model=make_model(kind);mu=raw[split['train']].mean((0,1));sd=raw[split['train']].std((0,1),correction=0).clamp_min(1e-5)
    freq=np.bincount(y[split['train']].numpy(),minlength=7);weights=torch.tensor(np.sqrt(len(split['train'])/(7*freq)),dtype=torch.float32)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=1e-4)
    best=-1;wait=0;history=[];start=time.time();aug_rng=np.random.default_rng(seed+900001)
    for epoch in range(60):
        model.train();order=torch.tensor(split['train'])[torch.randperm(len(split['train']))];loss_sum=0
        for idx in order.split(256):
            rot=int(aug_rng.integers(0,4)) if augment else 0;flip=bool(aug_rng.integers(0,2)) if augment else False
            if isinstance(model,ImageModel):
                x=images[idx];x=torch.rot90(x,rot,(2,3))
                if flip:x=torch.flip(x,(3,))
                matrix=None
            else:
                xx,matrix=augment_region(raw[idx],None if q is None else q[idx],rot,flip)
                x=(xx-mu)/sd
                if kind in ['RGBOnly','RGBOnlyGraph']:x=x.clone();x[:,:,6:12]=0
            optimizer.zero_grad();logits=model(x,matrix);loss=F.cross_entropy(logits,y[idx],weight=weights);loss.backward();optimizer.step();loss_sum+=float(loss.detach())*len(idx)
        ti=split['tune'];val=predict(model,images[ti],raw[ti],None if q is None else q[ti],mu,sd,kind)
        score=float(f1_score(y[ti],val.argmax(1),average='macro',labels=range(7),zero_division=0));history.append(dict(epoch=epoch+1,loss=loss_sum/len(order),tuning_macro_f1=score))
        if score>best+1e-6:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;wait=0
        else:wait+=1
        if epoch%10==0:print(kind,seed,'epoch',epoch+1,'tuneF1',round(score,4),'elapsed',round(time.time()-start,1),flush=True)
        if wait>=10:break
    model.load_state_dict(state);model.eval();ci=split['cal']
    cal_logits=predict(model,images[ci],raw[ci],None if q is None else q[ci],mu,sd,kind)
    obj=lambda logt:float(np.mean(logsumexp(cal_logits/np.exp(logt),1)-cal_logits[np.arange(len(ci)),y[ci].numpy()]/np.exp(logt)))
    temperature=float(np.exp(minimize_scalar(obj,bounds=(np.log(.1),np.log(10)),method='bounded').x))
    result=dict(kind=kind,seed=seed,augmented=augment,parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,tuning_macro_f1=best,temperature=temperature,train_seconds=time.time()-start,split_counts={s:np.bincount(y[ix].numpy(),minlength=7).tolist() for s,ix in split.items()},n={s:len(ix) for s,ix in split.items()},history=history)
    if save_dir:
        save_dir=Path(save_dir);save_dir.mkdir(exist_ok=True,parents=True)
        torch.save(dict(state=state,mu=mu,sd=sd,seed=seed,kind=kind,augment=augment,temperature=temperature,protocol=CFG),save_dir/'checkpoint.pt')
        np.savez_compressed(save_dir/'development.npz',train_index=split['train'],tuning_index=split['tune'],calibration_index=ci,calibration_logits=cal_logits)
        (save_dir/'training.json').write_text(json.dumps(result,indent=2))
    return model,mu,sd,result

def tensors(images):
    f=features(images);return torch.tensor(images.astype(np.float32).transpose(0,3,1,2)/255),torch.tensor(f[0]),f
