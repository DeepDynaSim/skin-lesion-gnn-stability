"""Modest per-partition development tuning; calibration and tests never select a recipe."""
from pathlib import Path
import sys,argparse,datetime
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from experiment_core import *

PROTOCOL=HERE/'optimized_cnn_protocol.json'

def lock():
    settings=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),purpose='Reviewer-requested development-tuned compact CNN reference; exploratory extension after previous tests were examined',architecture='Existing SmallCNN, 16647 parameters; native 28x28 RGB; no external pretraining',learning_rates=[0.0005,0.002],augmentation=False,batch_size=128,max_epochs=80,patience=12,weight_decay=0.0001,schedule='CosineAnnealingLR over 80 epochs, eta_min=initial_lr*0.05',seeds=CFG['main_seeds'],selection='For each seed, choose the larger best tuning macro F1 across two learning rates; exact ties choose lower learning rate. Calibration is excluded from all candidate fitting and selection; fit temperature only for selected candidate.',uncertainty='Separate across-partition SD and conditional paired lesion bootstrap; no confirmatory superiority claim',fit_count=20)
    if not PROTOCOL.exists():PROTOCOL.write_text(json.dumps(settings,indent=2))
    return json.loads(PROTOCOL.read_text())

def inference(model,images,batch=256):
    model.eval()
    with torch.no_grad():return torch.cat([model(x,None) for x in images.split(batch)]).numpy()

def fit(seed,lr,images,y,split,settings):
    dest=HERE/'optimized_cnn/candidates'/str(seed)/f'lr_{lr}'
    if (dest/'training.json').exists():return json.loads((dest/'training.json').read_text())
    dest.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(seed);np.random.seed(seed);model=make_model('SmallCNN')
    freq=np.bincount(y[split['train']].numpy(),minlength=7)
    weights=torch.tensor(np.sqrt(len(split['train'])/(7*freq)),dtype=torch.float32)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=settings['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=settings['max_epochs'],eta_min=lr*.05)
    best=-1.;wait=0;history=[];started=time.time()
    for epoch in range(settings['max_epochs']):
        model.train();order=torch.tensor(split['train'])[torch.randperm(len(split['train']))];loss_sum=0.
        for ix in order.split(settings['batch_size']):
            opt.zero_grad();loss=F.cross_entropy(model(images[ix],None),y[ix],weight=weights);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(ix)
        logits=inference(model,images[split['tune']]);score=float(f1_score(y[split['tune']],logits.argmax(1),labels=range(7),average='macro',zero_division=0))
        history.append(dict(epoch=epoch+1,loss=loss_sum/len(order),learning_rate=opt.param_groups[0]['lr'],tuning_macro_f1=score))
        if score>best+1e-6:best=score;state=copy.deepcopy(model.state_dict());best_epoch=epoch+1;wait=0
        else:wait+=1
        scheduler.step()
        if epoch%10==0:print('CNN tuning',seed,lr,epoch+1,round(score,4),round(time.time()-started,1),flush=True)
        if wait>=settings['patience']:break
    record=dict(seed=seed,learning_rate=lr,parameters=sum(p.numel() for p in model.parameters()),best_epoch=best_epoch,tuning_macro_f1=best,seconds=time.time()-started,history=history,protocol_sha256=hashlib.sha256(PROTOCOL.read_bytes()).hexdigest())
    torch.save(dict(state=state,seed=seed,learning_rate=lr,protocol= settings),dest/'checkpoint.pt')
    np.savez_compressed(dest/'development.npz',**{k+'_index':v for k,v in split.items()})
    (dest/'training.json').write_text(json.dumps(record,indent=2));return record

def selected_model(seed):
    rec=json.loads((HERE/'optimized_cnn/selections'/f'{seed}.json').read_text())
    ck=torch.load(HERE/'optimized_cnn'/rec['checkpoint_relative'],weights_only=False,map_location='cpu')
    model=make_model('SmallCNN');model.load_state_dict(ck['state']);model.eval();return model,rec

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',type=int,nargs='*');args=parser.parse_args();settings=lock()
    im,labels,groups,_=load_development();images=torch.tensor(im.transpose(0,3,1,2).astype(np.float32)/255);y=torch.tensor(labels,dtype=torch.long)
    for seed in args.seeds or settings['seeds']:
        selection=HERE/'optimized_cnn/selections'/f'{seed}.json'
        if selection.exists():continue
        split=partitions(labels,groups,seed);rows=[fit(seed,lr,images,y,split,settings) for lr in settings['learning_rates']]
        winner=sorted(rows,key=lambda r:(-r['tuning_macro_f1'],r['learning_rate']))[0]
        ckrel=f"candidates/{seed}/lr_{winner['learning_rate']}/checkpoint.pt";ck=torch.load(HERE/'optimized_cnn'/ckrel,weights_only=False,map_location='cpu');model=make_model('SmallCNN');model.load_state_dict(ck['state'])
        ci=split['cal'];logits=inference(model,images[ci]);cy=labels[ci]
        obj=lambda logt:float(np.mean(logsumexp(logits/np.exp(logt),1)-logits[np.arange(len(ci)),cy]/np.exp(logt)))
        temperature=float(np.exp(minimize_scalar(obj,bounds=(np.log(.1),np.log(10)),method='bounded').x))
        selection.parent.mkdir(exist_ok=True,parents=True)
        record=dict(seed=seed,checkpoint_relative=ckrel,selected_learning_rate=winner['learning_rate'],best_epoch=winner['best_epoch'],tuning_macro_f1=winner['tuning_macro_f1'],temperature=temperature,candidates=[{k:v for k,v in r.items() if k!='history'} for r in rows])
        np.savez_compressed(selection.with_suffix('.npz'),calibration_logits=logits,calibration_labels=cy,calibration_index=ci)
        selection.write_text(json.dumps(record,indent=2));print('SELECTED',seed,record['selected_learning_rate'],round(winner['tuning_macro_f1'],4),flush=True)

if __name__=='__main__':main()
