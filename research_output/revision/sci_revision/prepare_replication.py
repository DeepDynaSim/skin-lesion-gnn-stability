"""Extract S1 to an isolated tree; restricted inputs stay local and outside delivered archives."""
from pathlib import Path
import zipfile,json,hashlib,shutil,argparse
HERE=Path(__file__).resolve().parent;R=HERE.parent
parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=R/'Supplementary_Reproducibility_S1.zip');args=parser.parse_args()
target=HERE/'replication_workspace';target.mkdir(exist_ok=True)
with zipfile.ZipFile(args.archive) as z:
    for row in z.read('research_output/revision/manifest.sha256').decode().splitlines():
        expected,relative=row.split('  ',1);assert hashlib.sha256(z.read('research_output/'+relative)).hexdigest()==expected
    for member in z.infolist():
        destination=(target/member.filename).resolve();assert destination.is_relative_to(target.resolve())
    z.extractall(target)
rr=target/'research_output/revision';data=rr/'data';data.mkdir(exist_ok=True)
for source,dest in [(R.parent/'data/dermamnist_corrected_28.npz',rr.parent/'data/dermamnist_corrected_28.npz'),(R/'data/dermamnist_extended_28.npz',data/'dermamnist_extended_28.npz')]:
    shutil.copy2(source,dest);assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256(dest.read_bytes()).digest()
with zipfile.ZipFile(R/'data/PH2_mirror.zip') as z:
    selected=[n for n in z.namelist() if n.endswith('PH2_dataset.txt') or ('_Dermoscopic_Image/' in n and n.endswith('.bmp'))]
    assert len(selected)==201
    for name in selected:
        relative=name[name.index('PH2Dataset/'):];dest=(data/'ph2'/relative).resolve();assert dest.is_relative_to((data/'ph2').resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
expected=HERE/'replication_expected';expected.mkdir(exist_ok=True)
for relative in ['analysis/summary.json','analysis/corruption_summary.json','analysis/transfer_reference.json','analysis/finite_theory.json','analysis/augmentation_verification.json','data/ph2_audit.json']:
    source=rr/relative;dest=expected/Path(relative).name;shutil.copy2(source,dest)
for relative in ['analysis/corruption_summary.json']:(rr/relative).unlink()
# Full dependency lock records all packages used in the prior verified installation.
packages=json.loads((R/'environment.json').read_text())['packages']
(HERE/'requirements_full.lock').write_text('--extra-index-url https://download.pytorch.org/whl/cpu\n'+'\n'.join(f'{n}=={v}' for n,v in sorted(packages.items(),key=lambda x:x[0].lower()))+'\n')
print('Verified S1 manifest and extracted an isolated replication tree; PH2 local inputs are excluded from final packaging.',flush=True)
