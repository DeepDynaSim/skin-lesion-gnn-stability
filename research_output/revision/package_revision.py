"""Local reproduction archive; restricted PH2 source data are never included."""
from experiment_core import *
import zipfile,platform,importlib.metadata,re

def main():
    ver=json.loads((R/'final/verification_report.json').read_text());assert ver['status']=='passed' and ver['primary_fits']==150 and ver['sensitivity_fits']==42
    packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions(path=[str(R/'python_packages')])}
    required=['torch','torchvision','numpy','scipy','scikit-learn','matplotlib','pillow','python-docx','lxml','latex2mathml','pymupdf','pdf2image','beautifulsoup4']
    normalized_packages={n.lower().replace('_','-'):v for n,v in packages.items()}
    for name in required:
        if name not in normalized_packages:
            version=importlib.metadata.version(name)
            packages[name]=version
            normalized_packages[name]=version
    (R/'environment.json').write_text(json.dumps({'python':platform.python_version(),'os':platform.platform(),'packages':packages},indent=2))
    requirements=['--extra-index-url https://download.pytorch.org/whl/cpu']+[f'{n}=={normalized_packages[n]}' for n in required]
    (R/'requirements.txt').write_text('\n'.join(requirements)+'\n')
    readme='''# Supplementary Reproducibility Archive S1

This archive accompanies the revised English manuscript by Caglar Uyulan. It contains 150 fitted primary/control states, 42 development sensitivity states, saved logits and probabilities, complete development split indices and fitting histories, corruption/factorial outputs, uncertainty analyses, figure sources and verification records. It is a local submission archive; no persistent public URL or archive DOI is claimed.

## Data and directory structure

Keep the `research_output/revision` and parent `research_output/run_experiments.py` layout. Install the pinned scientific dependencies from `revision/requirements.txt` into a Python 3.12 environment (CPU PyTorch index is specified). The scripts also look for an optional local `python_packages` directory; none of those installed libraries is redistributed.

Retrieve the corrected C data and metadata into `research_output/data`, and E into `revision/data`, from the versioned record https://doi.org/10.5281/zenodo.11101338. Source URLs, exact hashes and label/overlap audits are included. `fetch_revision_data.py` obtains the E files; the parent `data/README_sources.txt` lists C provenance when available. The original C test is historical and was examined in the preliminary study. E testing and PH2 testing use no model or threshold refitting.

PH2 source images, original annotation tables, mirror ZIP, prepared image NPZ and derived diagnosis CSV are excluded. Obtain PH2 separately under its research/education terms, see https://www.fc.up.pt/addi/ph2%20database.html. The official legacy download failed; `fetch_ph2_mirror.py` records the public mirror used. `prepare_ph2.py` verifies all 200 identities and the original 80 common-nevus / 80 atypical-nevus / 40 melanoma table before resizing. Its hashes and retrieval/processing audits are provided. Exact raw-pixel overlap is checked, but no patient-level independence or complete near-duplicate exclusion is certified. The PH2 images may not be redistributed or commercially reused.

## Reproducing measurements

From the archive root, run in order:

1. `python research_output/revision/reaudit_identity.py`
2. `python research_output/revision/train_revision.py`
3. `python research_output/revision/train_revision.py --models RGBOnlyGraph`
4. `python research_output/revision/development_sensitivity.py`
5. `python research_output/revision/evaluate_revision.py`
6. `python research_output/revision/summarize_revision.py`
7. `python research_output/revision/corruption_revision.py`
8. `python research_output/revision/summarize_corruptions.py`
9. With fitting/inference jobs stopped, `python research_output/revision/benchmark_revision.py`
10. `python research_output/revision/verify_finite_theory.py` and `verify_augmentation.py`
11. `python research_output/revision/make_references.py` and `reporting_maps.py`
12. `python research_output/revision/make_figures.py` and `build_revision.py`

Existing completed checkpoints/results are reused. For a fresh replication, copy the source/protocol/data into a new output directory and leave `models`, `sensitivity`, `evaluation`, `corruptions` and derived `analysis` outputs absent. All model state files in this archive are generated here; use a trusted local archive when loading PyTorch checkpoints. Ten main seeds, three sensitivity seeds and five noise seeds are fixed in `protocol.json` and its addenda. The full 28-pixel, from-scratch fitting recipe is disclosed; MobileNet learning failures under this shared recipe are not replaced by an optimized or pretrained system.

## Documents and visual checks

Equations are native Office Math (OMML), not bitmaps. `build_revision.py` uses the MathML-to-OMML XSL included with licensed Microsoft Word. This authoring path and native Word PDF export require Windows/Office; the scientific fitting and plotting code uses ordinary Python libraries. `render_revision.py` uses the Codex document skill's canonical rasterizer on the authoring host; for another machine use native Word PDF export and Poppler page rasterization. PDF and Word page content were inspected; automated artifact checks are in `final/verification_report.json`, with the separate visual page log. Eleven figures have 400-dpi PNG and editable text/vector SVG/PDF sources.

The registry audit covers 20 DOI-bearing references and one official PMLR paper with no publisher DOI. Three ICLR citations explicitly use author-preprint DOIs. The original source PDFs and unpublished project materials are not redistributed as bibliographic evidence. Review response and full item-number maps disclose clinical reporting limitations and the pending persistent public deposit.

`manifest.sha256` lists every included file. This archive has no bundled clinical PH2 image data and no installed third-party libraries. Dataset and third-party software rights remain with their respective owners; source URLs and versions allow independent retrieval. No prospective registration, new ethics-approval number, mobile-device energy test or clinical workflow validation is claimed.
'''
    readme=readme.replace('It contains 150 fitted primary/control states, 42','It contains 150 fitted primary/control states, ten additional frozen-ImageNet head states, 42').replace('Eleven figures have','Twelve figures have').replace('4. `python research_output/revision/development_sensitivity.py`','4. `python research_output/revision/development_sensitivity.py`\n\nAdditional separate comparator: `python research_output/revision/transfer_reference.py` then `python research_output/revision/transfer_uncertainty.py`. The official ImageNet source weights and source hash are included; the clinical-image embedding cache is excluded and is regenerated. This comparator uses official resize/crop/normalization and no head augmentation; it does not replace the prespecified comparisons.').replace('`fetch_revision_data.py` obtains the E files','`retrieve_benchmarks.py` obtains/verifies all four frozen files; `fetch_revision_data.py` obtains the E files').replace('memory test or clinical workflow','memory test or clinical workflow')
    (R/'README_Reproducibility.md').write_text(readme,encoding='utf-8')
    # Include generated scientific artifacts and source, excluding runtime caches and source image datasets.
    included=[]
    for file in R.iterdir():
        if file.is_file() and file.suffix in ['.py','.md','.json','.txt','.sha256'] and file.name not in ['manifest.sha256','archive_manifest.json']:included.append(file)
    for folder in ['models','sensitivity','evaluation','corruptions','analysis','figures','final','transfer_reference/models','transfer_reference/corruptions']:
        included.extend(p for p in (R/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix in ['.pt','.npz','.json','.png','.svg','.pdf','.docx','.md'] and not (folder=='sensitivity' and p.parent.name.lower().startswith('k_')))
    included.extend(p for p in (R/'bibliography').iterdir() if p.is_file() and p.suffix in ['.json','.md'])
    for name in ['overlap_audit.json','ph2_audit.json','ph2_retrieval.json','ph2_mirror_manifest.json','download_manifest.json','DermaMNIST-E.csv']:
        p=R/'data'/name
        if p.exists():included.append(p)
    included.append(OLD/'run_experiments.py');included.append(OLD/'export_word.ps1')
    included.extend([R/'transfer_reference/predictions.npz',R/'transfer_reference/weight_provenance.json',R/'data/mobilenet_v3_small-047dcff4.pth'])
    for name in ['audit.json','README_sources.txt','source_provenance.json','download_manifest.json','DermaMNIST-C.csv']:
        p=OLD/'data'/name
        if p.exists():included.append(p)
    included=sorted(set(included));assert all('python_packages' not in p.parts for p in included);assert all('ph2' not in p.parts for p in included);assert all(p.name not in ['PH2_mirror.zip','ph2_28.npz','ph2_metadata.csv'] for p in included)
    manifest='\n'.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(OLD).as_posix() for p in included)+'\n';(R/'manifest.sha256').write_text(manifest);archive=R/'Supplementary_Reproducibility_S1.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in included:z.write(p,'research_output/'+p.relative_to(OLD).as_posix())
        z.write(R/'manifest.sha256','research_output/revision/manifest.sha256')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None;names=z.namelist();assert not any('PH2_mirror.zip' in n or n.endswith('ph2_28.npz') or '/python_packages/' in n for n in names)
    record={'archive':archive.name,'files':len(included)+1,'bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'restricted_PH2_source_files_included':False,'public_persistent_deposit':'Pending author publication; no fabricated identifier'};(R/'archive_manifest.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2),flush=True)
if __name__=='__main__':main()
