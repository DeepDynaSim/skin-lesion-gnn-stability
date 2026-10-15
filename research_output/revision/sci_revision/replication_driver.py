"""Run the archived package in a clean installation, retaining complete command logs."""
from pathlib import Path
import subprocess,sys,json,os,shutil,math
HERE=Path(__file__).resolve().parent;RR=HERE/'replication_workspace/research_output/revision';ENV=HERE/'replication_env';PY=ENV/'Scripts/python.exe';EXPECTED=HERE/'replication_expected';LOG=HERE/'replication_logs';LOG.mkdir(exist_ok=True)

def run(name,script):
    print('CLEAN REPLICATION START',name,flush=True)
    env=dict(os.environ);env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None);env.pop('SSLKEYLOGFILE',None);env['OMP_NUM_THREADS']='4'
    with (LOG/(name+'.log')).open('w',encoding='utf-8') as output:
        result=subprocess.run([str(PY),str(script)],cwd=RR,env=env,stdout=output,stderr=subprocess.STDOUT,text=True)
    if result.returncode:print((LOG/(name+'.log')).read_text()[-4000:],flush=True);raise RuntimeError(name)
    print('CLEAN REPLICATION PASS',name,flush=True)

def compare(a,b,path='root'):
    errors=[]
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:errors.extend(compare(a[k],b[k],path+'/'+k))
    elif isinstance(a,list):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):errors.extend(compare(x,y,path+'/'+str(i)))
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert isinstance(b,(float,int));error=abs(a-b);assert error<2e-7,(path,a,b);errors.append(error)
    else:assert a==b,(path,a,b)
    return errors

def main():
    # This worker deliberately imports no modules from the authoring package directories.
    proof=RR/'clean_environment_proof.py'
    proof.write_text("import sys,json,numpy,torch,torchvision,scipy,sklearn,importlib.metadata\nfrom pathlib import Path\nmodules={m.__name__:m.__file__ for m in [numpy,torch,torchvision,scipy,sklearn]}\nassert sys.prefix!=sys.base_prefix\nassert all(Path(p).is_relative_to(Path(sys.prefix)) for p in modules.values())\nPath('clean_environment.json').write_text(json.dumps({'python':sys.version,'prefix':sys.prefix,'base_prefix':sys.base_prefix,'modules':modules,'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}},indent=2))\n")
    run('00_environment',proof)
    shutil.copy2(RR/'transfer_reference/predictions.npz',EXPECTED/'transfer_predictions.npz')
    for label,name in [('01_PH2_processing','prepare_ph2.py'),('02_identity','reaudit_identity.py'),('03_finite_theory','verify_finite_theory.py'),('04_augmentation','verify_augmentation.py'),('05_all_primary_bootstraps','summarize_revision.py'),('06_all_corruption_bootstraps','summarize_corruptions.py'),('07_fresh_backbone_embeddings','transfer_reference.py'),('08_transfer_corruption_bootstraps','transfer_uncertainty.py'),('09_archived_full_audit','verify_revision.py')]:run(label,RR/name)
    errors=[]
    for name in ['summary.json','corruption_summary.json','transfer_reference.json','finite_theory.json','augmentation_verification.json']:
        errors.extend(compare(json.loads((EXPECTED/name).read_text()),json.loads((RR/'analysis'/name).read_text()),name))
    errors.extend(compare(json.loads((EXPECTED/'ph2_audit.json').read_text()),json.loads((RR/'data/ph2_audit.json').read_text()),'PH2_processing'))
    for worker in ['clean_forward_audit.py','clean_refit_audit.py']:
        shutil.copy2(HERE/worker,RR/worker);run(worker.removesuffix('.py'),RR/worker)
    report={'status':'passed','scope':'Same host, fresh isolated Python installation and extracted S1 tree; not an independent investigator and not a full retraining of all 202 original fits','original_manifest_verified':True,'clean_environment':json.loads((RR/'clean_environment.json').read_text()),'regenerated_numeric_entries_compared':len(errors),'maximum_regenerated_numeric_error':max(errors),'PH2_original_table_and_200_images_reprocessed':True,'all_primary_calibration_classwise_external_and_corruption_bootstraps_regenerated':True,'fresh_frozen_ImageNet_embeddings_and_predictions_regenerated':True,'forward_audit':json.loads((RR/'clean_forward_audit.json').read_text()),'representative_refits':json.loads((RR/'clean_refit_audit.json').read_text()),'commands':[p.name for p in LOG.glob('*.log')]}
    (HERE/'analysis/clean_replication.json').write_text(json.dumps(report,indent=2));print('CLEAN REPLICATION COMPLETE',len(errors),max(errors),flush=True)

if __name__=='__main__':main()
