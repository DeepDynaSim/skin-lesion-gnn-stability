from pathlib import Path
import shutil,subprocess,os
HERE=Path(__file__).resolve().parent;target=HERE/'replication_workspace/research_output/revision/sci_revision';target.mkdir(exist_ok=True)
for name in ['optimized_cnn.py','evaluate_optimized_cnn.py','descriptor_noise.py','clean_new_cnn_audit.py','optimized_cnn_protocol.json']:
    shutil.copy2(HERE/name,target/name)
for folder in ['candidates','selections']:shutil.copytree(HERE/'optimized_cnn'/folder,target/'optimized_cnn'/folder,dirs_exist_ok=True)
(target/'analysis').mkdir(exist_ok=True)
for name in ['optimized_cnn','descriptor_noise']:shutil.copy2(HERE/'analysis'/(name+'.json'),target/'analysis'/('expected_'+name+'.json'))
environment=dict(os.environ);environment.pop('PYTHONPATH',None);environment.pop('PYTHONHOME',None);environment.pop('SSLKEYLOGFILE',None)
log=HERE/'replication_logs/10_added_CNN_and_descriptor.log'
with log.open('w',encoding='utf-8') as output:result=subprocess.run([str(HERE/'replication_env/Scripts/python.exe'),str(target/'clean_new_cnn_audit.py')],cwd=target,env=environment,stdout=output,stderr=subprocess.STDOUT)
if result.returncode:print(log.read_text()[-4000:],flush=True);raise RuntimeError('Added clean audit')
shutil.copy2(target/'analysis/clean_new_cnn.json',HERE/'analysis/clean_new_cnn.json');print((HERE/'analysis/clean_new_cnn.json').read_text(),flush=True)
