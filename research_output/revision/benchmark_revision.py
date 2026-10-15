"""Run only after fitting jobs finish; one fresh process for every architecture."""
from experiment_core import *
import subprocess,platform,io,ctypes,argparse,winreg
from PIL import Image

class Counters(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('PageFaultCount',ctypes.c_ulong),('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]

def peak_memory():
    m=Counters();m.cb=ctypes.sizeof(m);kernel=ctypes.WinDLL('kernel32');kernel.GetCurrentProcess.restype=ctypes.c_void_p;psapi=ctypes.WinDLL('psapi');psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_ulong];assert psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(m),m.cb);return m.PeakWorkingSetSize/1024**2

def one(name):
    kind=name;transfer=name=='FrozenMobileNetV3'
    if transfer:
        from transfer_reference import backbone,TR
        bb,transform,_=backbone();ck=torch.load(TR/'models/101/checkpoint.pt',map_location='cpu',weights_only=False);head=nn.Linear(576,7);head.load_state_dict(ck['state']);head.eval()
        class Reference(nn.Module):
            def __init__(self):super().__init__();self.backbone=bb;self.head=head
            def forward(self,x,q=None):return self.head((self.backbone(x)-ck['mu'])/ck['sd'])
        model=Reference();model.eval()
    else:
        ck=torch.load(R/'models'/name/'101/checkpoint.pt',map_location='cpu',weights_only=False);model=make_model(kind);model.load_state_dict(ck['state']);model.eval()
    d=np.load(OLD/'data/dermamnist_corrected_28.npz');idx=np.random.default_rng(9471).choice(1227,120,replace=False);encoded=[]
    for image in d['test_images'][idx]:
        b=io.BytesIO();Image.fromarray(image).save(b,format='PNG');encoded.append(b.getvalue())
    graph_kind=kind not in ['NoGraph','RGBOnly','SmallCNN','ResNet18','MobileNetV3Small'];times=[]
    with torch.no_grad():
        for i,b in enumerate(encoded):
            t=time.perf_counter_ns();im=np.array(Image.open(io.BytesIO(b)).convert('RGB'))[None];t1=time.perf_counter_ns()
            if transfer:x=transform(torch.tensor(im.transpose(0,3,1,2).copy()));q=None
            elif isinstance(model,ImageModel):x=torch.tensor(im.astype(np.float32).transpose(0,3,1,2)/255);q=None
            else:
                f=features(im);x=(torch.tensor(f[0])-ck['mu'])/ck['sd'];q=torch.tensor(graph_matrix(f,kind)) if graph_kind else None
                if kind in ['RGBOnly','RGBOnlyGraph']:x[:,:,6:12]=0
            t2=time.perf_counter_ns();model(x,q);t3=time.perf_counter_ns()
            if i>=20:times.append([(t1-t)/1e6,(t2-t1)/1e6,(t3-t2)/1e6,(t3-t)/1e6])
    a=np.array(times);cpu=winreg.QueryValueEx(winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0'),'ProcessorNameString')[0].strip()
    row=dict(name=name,parameters=sum(p.numel() for p in model.parameters()),measurements=100,warmups=20,threads=torch.get_num_threads(),processor=cpu,os=platform.platform(),torch_version=torch.__version__,python=platform.python_version(),peak_process_MiB=peak_memory(),median_ms=dict(zip(['decode','preparation','forward','total'],np.median(a,0).tolist())),p95_ms=dict(zip(['decode','preparation','forward','total'],np.quantile(a,.95,axis=0).tolist())),raw_times_ms=a.tolist(),scope='PNG decoding and native 28x28 input; single fitted model, batch one. Graph polynomial constructed within preparation; no model ensembling, resizing or file I/O. Process memory includes imports/data/runtime.')
    if transfer:row['scope']='Native 28x28 PNG decoding; official resize 256 / center crop 224 / ImageNet normalization included in preparation; frozen backbone and learned head included in forward. Source clinical-image downsampling, disk I/O and ensembling excluded. Peak process memory includes imports/data/runtime.'
    (R/'analysis'/('runtime_'+name+'.json')).write_text(json.dumps(row,indent=2));print(name,row['median_ms']['total'],row['peak_process_MiB'],flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model');args=parser.parse_args()
    if args.model:one(args.model)
    else:
        for name in CFG['central_models']+['RGBOnlyGraph','FrozenMobileNetV3']:
            p=subprocess.run([sys.executable,__file__,'--model',name]);assert p.returncode==0
        rows=[json.loads((R/'analysis'/('runtime_'+name+'.json')).read_text()) for name in CFG['central_models']+['RGBOnlyGraph','FrozenMobileNetV3']];(R/'analysis/runtime_summary.json').write_text(json.dumps(rows,indent=2))
