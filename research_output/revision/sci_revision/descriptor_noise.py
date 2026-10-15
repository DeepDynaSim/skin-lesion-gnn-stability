"""Descriptor-group perturbation decomposition; no refitting or label selection."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
from experiment_core import *
from corruption_revision import corrupted
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    d=np.load(OLD/'data/dermamnist_corrected_28.npz');images=d['test_images'];raw=features(images)[0]
    feature_groups=[('RGB means',slice(0,3)),('RGB standard deviations',slice(3,6)),('Local differences',slice(6,12)),('Position and radius',slice(12,15))]
    records=[]
    for noise_seed in CFG['corruption_seeds']:
        dirty=features(corrupted(images,'noise',.05,noise_seed))[0];delta=dirty-raw
        for seed in CFG['main_seeds']:
            ck=torch.load(R/'models/ContrastPenalty'/str(seed)/'checkpoint.pt',map_location='cpu',weights_only=False);scale=ck['sd'].numpy();standardized=delta/scale
            energy=float(np.sum(standardized.astype(np.float64)**2))
            row={'noise_seed':noise_seed,'training_seed':seed,'groups':[]}
            for name,ix in feature_groups:
                row['groups'].append(dict(name=name,coordinates=ix.stop-ix.start,raw_RMS=float(np.sqrt(np.mean(delta[:,:,ix].astype(np.float64)**2))),standardized_RMS=float(np.sqrt(np.mean(standardized[:,:,ix].astype(np.float64)**2))),squared_perturbation_share=float(np.sum(standardized[:,:,ix].astype(np.float64)**2)/energy),training_scale_median=float(np.median(scale[ix]))))
            records.append(row)
    means=[]
    for i,(name,ix) in enumerate(feature_groups):
        means.append({'name':name,'coordinates':ix.stop-ix.start,**{metric:float(np.mean([row['groups'][i][metric] for row in records])) for metric in ['raw_RMS','standardized_RMS','squared_perturbation_share','training_scale_median']}})
    rng=np.random.default_rng(15102026);checks=[]
    for sigma in [.02,.05,.1]:
        differences=rng.normal(0,np.sqrt(2)*sigma,1000000);expected=2*sigma/np.sqrt(np.pi);observed=float(np.mean(np.abs(differences)));relative=abs(observed/expected-1);assert relative<.004
        checks.append(dict(sigma=sigma,analytic_mean_absolute_difference=expected,monte_carlo_mean_absolute_difference=observed,relative_error=relative,unclipped_uncapped_expected_proxy=4*sigma/(.15*np.sqrt(np.pi))))
    result=dict(status='passed',cohort='All 1227 historical C test images; 49 nodes; five noise realizations and ten fixed training scales',scope='Descriptive decomposition of standardized perturbation energy; dimensions differ across groups; model partitions and noise draws are crossed, not 50 independent biological samples',sigma=.05,groups=means,records=records,gaussian_checks=checks,formula_scope='Constant underlying neighboring pixels, independent zero-mean Gaussian noise, before clipping/uint8 quantization; analytical proxy expectation is before the cap at one.')
    (HERE/'analysis').mkdir(exist_ok=True);(HERE/'analysis/descriptor_noise.json').write_text(json.dumps(result,indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(9.5,3.8));short=['RGB mean','RGB SD','Local differences','Position'];colors=['#2d6f9c','#689b7a','#b74528','#999999']
    axes[0].bar(short,[r['standardized_RMS'] for r in means],color=colors);axes[0].set_ylabel('Standardized descriptor change, RMS');axes[0].set_title('A  Actual image perturbations, σ = 0.05',loc='left',weight='bold')
    axes[1].bar(short,[100*r['squared_perturbation_share'] for r in means],color=colors);axes[1].set_ylabel('Squared perturbation energy share (%)');axes[1].set_ylim(0,100);axes[1].set_title('B  Contribution to descriptor change',loc='left',weight='bold')
    for ax in axes:
        ax.tick_params(axis='x',labelrotation=28)
        for spine in ['top','right']:ax.spines[spine].set_visible(False)
    fig.tight_layout();(HERE/'figures').mkdir(exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(HERE/'figures'/('fig_descriptor_noise.'+suffix),dpi=400,bbox_inches='tight')
    plt.close(fig);print(json.dumps({'groups':means,'maximum_gaussian_check_relative_error':max(r['relative_error'] for r in checks)},indent=2),flush=True)

if __name__=='__main__':main()
