"""Local, traceable narrative changes; the previous final manuscript is preserved."""
from pathlib import Path
import shutil,json,re
HERE=Path(__file__).resolve().parent;R=HERE.parent
TITLE='Stable graph propagation does not ensure image robustness: finite-step analysis and component evaluation in dermoscopic classification'

def main():
    text=(R/'manuscript_template.md').read_text(encoding='utf-8')
    text=TITLE.join(text.split(text.splitlines()[0][2:],1))
    start=text.index('Purpose:');end=text.index('\n\nKeywords:',start)
    text=text[:start]+'{{revised_abstract}}'+text[end:]
    anchor='We address three questions.'
    pos=text.index(anchor)
    hypothesis='We organize this exploratory study around a bounded working hypothesis: for the investigated regional descriptor family, image noise can change standardized local-difference inputs enough to degrade prediction even when graph-only substitution has a small conditional effect. This hypothesis was refined from the exploratory component results; it is not a prospectively registered confirmatory claim. The experimental object is a GNN family whose topology, weighting and descriptor assumptions are tested, rather than a final superior classifier. The central penalty setting is retained as a reference intervention, not selected as the best design.\n\n'
    text=text[:pos]+hypothesis+text[pos:]
    text=text.replace('The official DermaMNIST-E test set contains 1511 images with the same seven-label convention.','The official DermaMNIST-E test set contains 1511 images with the same seven-label convention. These are resized images from the ISIC 2018 challenge testing partition after exclusion of the documented easter-egg image [12,13].')
    text=text.replace('## 3 Regional graph models','## 3 Controlled regional graph family')
    anchor='After checkpoint selection, each network\'s temperature'
    pos=text.index(anchor)
    extra='An additional development-tuned compact CNN addresses the need for a functioning small image reference. It retains the 16647-parameter Small CNN, native inputs and no external pretraining. For each of the same ten lesion-grouped partitions, two candidates use initial learning rates 0.0005 and 0.002, batch size 128, AdamW weight decay 0.0001, cosine decay to five percent of the initial rate over at most 80 epochs, patience 12 and no augmentation. The candidate with highest best tuning macro F1 is selected within that partition; an exact tie selects the lower rate. Only the selected candidate receives temperature calibration. Both candidates exclude that partition\'s calibration images from fitting and tuning, giving twenty actual additional fits and ten selected states. This is a modest development search, not exhaustive optimization or a capacity-matched comparison with the regional network. The extension was defined after the prior results had been examined; all new test comparisons remain exploratory.\n\n'
    text=text[:pos]+extra+text[pos:]
    text=text.replace('### 5 5 Computational measurements','### 5 6 Computational measurements')
    anchor='### 5 6 Computational measurements'
    pos=text.index(anchor)
    mechanism=r'''### 5 5 Descriptor noise and standardization

For a constant underlying pair of neighboring pixels with independent Gaussian noise before clipping or quantization, the noise difference has variance twice the pixel-noise variance. Integrating its half-normal density gives the elementary relation

$$\eta_a,\eta_b\sim\mathcal{N}(0,\sigma^2)\ \mathrm{independently},\qquad\mathbb{E}|\eta_a-\eta_b|=\frac{2\sigma}{\sqrt{\pi}}$$

The expected sum of horizontal and vertical absolute differences is therefore four times the noise standard deviation divided by the square root of pi in this ideal constant-patch setting. The resulting variation-proxy expectation is approximately 0.752 at noise standard deviation 0.05 before the cap at one. Nonconstant image structure, clipping, quantization and capping change the expectation for actual images; shared pixels additionally make local differences dependent. This explanatory idealization is not a robustness certificate. Monte Carlo checks use one million independently sampled differences at each of the three noise severities.

Training standardization subtracts a fixed mean and divides by the training population scale, with the implemented floor of 0.00001. Consequently, perturbations obey the following exact algebraic identity. We also define each feature group's fraction of squared standardized perturbation energy:

$$\Delta\widehat{x}_{Iij}=\frac{x'_{Iij}-x_{Iij}}{s_j},\qquad R_g=\frac{\sum_{I,i}\sum_{j\in g}(\Delta\widehat{x}_{Iij})^2}{\sum_{I,i,j}(\Delta\widehat{x}_{Iij})^2}$$

Here I indexes images, i nodes and j descriptor coordinates. The decomposition uses all 1227 historical C test images, five fixed noise realizations at standard deviation 0.05 and each of the ten penalized-network training scales. RGB means and RGB standard deviations each contain three coordinates, local differences six and position/radius three. Group fractions depend on dimensionality as well as per-coordinate perturbation size. We average the crossed noise/scale combinations descriptively; they are not fifty independent biological samples. The learned encoder can attenuate or amplify these input changes; its observed relative change on the previously defined 64-image subset is reported separately from this full-cohort descriptor decomposition.

'''
    text=text[:pos]+mechanism+text[pos:]
    # Put the factorial experiment first without altering later result section numbers.
    resultstart=text.index('### 6 1 Component');b=text.index('### 6 2 Classwise');c=text.index('### 6 3 Separation');d=text.index('### 6 4 Multiple')
    architecture=text[resultstart:b].replace('### 6 1','### 6 2',1)
    errors=text[b:c].replace('### 6 2','### 6 3',1)
    factorial=text[c:d].replace('### 6 3','### 6 1',1)
    factorial+='{{descriptor_results_text}}\n\n[TABLE:descriptor_noise]\n\n[FIG:fig_descriptor_noise|Descriptor-group changes under five moderate-noise realizations and ten fixed training scales. RMS values are per coordinate; energy shares also reflect the number of coordinates. These descriptive crossed combinations do not represent independent patient samples.]\n\n'
    architecture+='{{tuned_cnn_results_text}}\n\nThe scratch MobileNetV3 is retained as a failed fitting control. It is excluded from any claim about competitive lightweight performance or architectural inferiority. Its constant predictions diagnose this executed fitting/resolution recipe.\n\n'
    text=text[:resultstart]+factorial+architecture+errors+text[d:]
    text=text.replace('### 6 6 Benchmark transfer and source-external testing','### 6 6 Benchmark transfer and source-external exploratory evaluation')
    text=text.replace('{{external_results_text}}','{{external_results_text}}\n\n{{tuned_cnn_external_text}}\n\nThe PH2 result is a joint transfer failure of the complete executed system: downsampling to 28 by 28, the regional descriptor representation, learned fitting, a seven-class source task transferred to melanoma versus nevi, and a different acquisition/cohort distribution all remain possible contributors. These factors were not crossed in a high-resolution binary-task experiment, so the external loss cannot be causally assigned to graph propagation alone.')
    text=text.replace('The central distinction is between three levels of stability.','The factorial result is the central empirical finding: graph-only substitution has a small conditional performance effect at the tested moderate noise, while noisy node descriptors yield a large loss. This supports the bounded working hypothesis for the measured regional family and perturbations; it does not establish the same dominant mechanism in every GNN. The central distinction is between three levels of stability.')
    text=text.replace('{{discussion_results_text}}','{{discussion_results_text}}\n\n{{descriptor_discussion_text}}')
    old='The added ResNet-18 and MobileNetV3 comparisons make the architectural reference set stronger under a shared fitting protocol.'
    pos=text.index(old);end=text.index('\n\nSeparate calibration',pos)
    text=text[:pos]+'''ResNet-18 provides the stronger measured image reference: its internal ensemble macro F1 is 0.649 versus 0.586 for the penalized regional network, and its PH2 melanoma AUC is 0.806 versus 0.507. Its 11180103 learned parameters differ substantially from the regional network's 3879, so this is a system comparison rather than a matched-capacity experiment. Neither a parameter-efficiency advantage in diagnostic accuracy nor superiority over an optimized image backbone follows. The added development-tuned 16647-parameter CNN supplies a functioning compact comparison with an explicit, calibration-separated search. The scratch MobileNet collapse identifies failure of its executed training and native-resolution recipe; it does not establish inferiority of the architecture. The separate frozen ImageNet reference differs in pretraining, preprocessing and effective representation, and is interpreted accordingly. Learned regional features and higher-resolution superpixels remain future alternatives rather than substitutions for the tested family.'''+text[end:]
    text=text.replace('External PH2 testing and the independent E benchmark protocol','Source-external exploratory PH2 evaluation and E benchmark transfer')
    text=text.replace('## 8 Conclusion\n\n{{conclusion_results_text}}','## 8 Conclusion\n\n{{revised_conclusion}}')
    text=text.replace('Data and code availability: Exact','Data and code availability: The complete local S1 package and its manifest are supplied for reviewer inspection. A same-host clean-installation audit regenerated numerical analyses and saved-state inferences, with its precise scope and representative refits reported in the supplement. Exact')
    text=text.replace('No fictitious public repository, archive DOI or prospective registration number is provided; a public code deposit is a separate author-controlled publication step.','Persistent public deposition is pending; no public repository or archive DOI is claimed. Restricted PH2 source images and original annotation files are not part of S1. The local package supports inspection and rerunning, while public deposition remains a separate publication step.')
    text=text.replace('There are 160 primary/control/additional head fits plus 42 development-only sensitivity fits in the complete package.','Together with the twenty compact-CNN tuning fits, the complete scientific package contains 222 actual fits: 150 primary/control fits, ten frozen-ImageNet heads, 42 graph sensitivity fits and twenty compact-CNN candidates. Ten CNN selections are pointers to candidate states, not ten additional fits.')
    text=text.replace('[REPORTING_MAP]','''### Compact CNN development search and clean-installation audit

[TABLE:cnn_search]

[FIG:fig_tuned_cnn|Development-selected compact CNN histories and frozen corruption comparison. Every learning-rate selection uses its own tuning partition; calibration and test images do not choose a candidate. Training histories are descriptive and all twenty candidate histories are supplied.]

{{clean_replication_text}}

[REPORTING_MAP]''')
    (HERE/'manuscript_template.md').write_text(text,encoding='utf-8')
    (HERE/'bibliography').mkdir(exist_ok=True);shutil.copy2(R/'bibliography/references.json',HERE/'bibliography/references.json')
    for p in (R/'figures').iterdir():
        if p.is_file() and p.suffix in ['.png','.svg','.pdf','.json']:shutil.copy2(p,HERE/'figures'/p.name)
    maps=json.loads((R/'analysis/reporting_maps.json').read_text())
    for rows in [maps['TRIPOD_AI'],maps['CLAIM_2024']]:
        for row in rows:
            row[3]=row[3].replace('5.1–5.5','5.1–5.6')
            if row[0]=='31':row[3]=row[3].replace('6.3','6.1')
            if row[0]=='38':row[3]=row[3].replace('6.2','6.3')
            if row[0]=='30':row[3]='5.3–5.5, 6.1, 6.4, 6.7'
    (HERE/'analysis/reporting_maps.json').write_text(json.dumps(maps,indent=2))
    source=(R/'build_revision.py').read_text(encoding='utf-8')
    source=source.replace("sys.path.insert(0,str(R/'python_packages'))","sys.path.insert(0,str(R.parent/'python_packages'));sys.path.insert(0,str(R.parent))")
    source=source.replace('from manuscript_results import build','from manuscript_results_second import build')
    source=source.replace('Finite step propagation stability and image robustness in graph neural networks for skin lesion classification',TITLE)
    source=source.replace('Skin_Lesion_GNN_Revised_Manuscript','Skin_Lesion_GNN_Final_Manuscript')
    source=source.replace('pp.paragraph_format.line_spacing=1;pp.paragraph_format.space_after=Pt(5)','pp.paragraph_format.line_spacing=1;pp.paragraph_format.space_after=Pt(3.5)')
    source=source.replace("st.font.size=Pt(8.8 if len(headers)>5 else 9.2)","st.font.size=Pt(8.8 if len(headers)>5 else 9.2)")
    (HERE/'build_manuscript.py').write_text(source,encoding='utf-8')
    print('Prepared the final manuscript source with local, traceable changes.',flush=True)

if __name__=='__main__':main()
