from pathlib import Path
import shutil,json
R=Path(__file__).resolve().parent;out=R/'sensitivity';before=json.loads((out/'sensitivity_results.json').read_text());audit={'issue':'Windows case-insensitive k/K paths collided for value 8; three steps=8 configurations were not fitted and lower-case k=8 records appeared twice','original_records':len(before),'corrected_prefixes':{'k':'neighbors','K':'steps'},'reuse':[]}
for parameter,value,prefix in [('k',2,'neighbors'),('k',8,'neighbors'),('K',1,'steps')]:
    for seed in [1201,1301,1409]:
        source=out/f'{parameter}_{value}_{seed}';dest=out/f'{prefix}_{value}_{seed}';record=json.loads((source/'sensitivity.json').read_text());assert record['parameter']==parameter and record['value']==value and record['seed']==seed
        assert R.resolve() in source.resolve().parents and R.resolve() in dest.resolve().parents
        if not dest.exists():shutil.copytree(source,dest)
        audit['reuse'].append({'source':source.name,'canonical':dest.name})
(R/'analysis/sensitivity_path_correction.json').write_text(json.dumps(audit,indent=2));print('Nine valid states reused under unambiguous paths; missing steps=8 states will be fitted')
