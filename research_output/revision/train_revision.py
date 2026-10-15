from pathlib import Path
import sys,json,argparse,hashlib,time
R=Path(__file__).resolve().parent
from experiment_core import *
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--models',nargs='*');parser.add_argument('--seeds',nargs='*',type=int);args=parser.parse_args()
    im,y,groups,meta=load_development();images,raw,f=tensors(im);yt=torch.tensor(y,dtype=torch.long)
    kinds=args.models or CFG['central_models']+CFG['augmentation_controls']
    # Save the protocol fingerprint before fitting any new model.
    (R/'protocol.sha256').write_text(hashlib.sha256((R/'protocol.json').read_bytes()).hexdigest())
    for name in kinds:
        kind=name.replace('_NoAug','');augment=not name.endswith('_NoAug')
        q=None if kind in ['NoGraph','RGBOnly','SmallCNN','ResNet18','MobileNetV3Small'] else torch.tensor(graph_matrix(f,kind))
        for seed in args.seeds or CFG['main_seeds']:
            folder=R/'models'/name/str(seed)
            if (folder/'training.json').exists() and (folder/'checkpoint.pt').exists():continue
            split=partitions(y,groups,seed)
            model,mu,sd,result=train(kind,seed,images,yt,raw,q,split,augment=augment,save_dir=folder)
            print('FINISHED',name,seed,result['best_epoch'],round(result['tuning_macro_f1'],4),flush=True)
    print('Training job completed',flush=True)
if __name__=='__main__':main()
