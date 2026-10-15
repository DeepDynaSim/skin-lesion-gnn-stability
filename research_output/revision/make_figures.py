from experiment_core import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from sklearn.metrics import roc_curve
from corruption_revision import corrupted,topology

F=R/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
colors={'ContrastPenalty':'#B04A28','ContrastNoPenalty':'#2E6A93','SmallCNN':'#32845B','ResNet18':'#7B57A1','MobileNetV3Small':'#B58A25','RGBOnlyGraph':'#555555'}
labels={'NoGraph':'Graph-free','GridOnly':'Grid only','KNNOnly':'Color kNN only','UniformTopology':'Constant weights','ContrastNoPenalty':'Contrast, beta = 0','ContrastPenalty':'Contrast, beta = 1.5','RGBOnly':'No gradients / no graph','RGBOnlyGraph':'No gradients / full graph','SmallCNN':'Small CNN','ResNet18':'ResNet-18','MobileNetV3Small':'MobileNetV3 Small'}
labels['FrozenMobileNetV3']='ImageNet MobileNetV3'
classes=['AKIEC','BCC','BKL','DF','MEL','NV','VASC']
def save(fig,name):
    fig.savefig(F/(name+'.png'),dpi=400,bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.svg'),bbox_inches='tight',facecolor='white');fig.savefig(F/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
def tag(ax,text):ax.set_title(text,loc='left',fontweight='bold',pad=8)
def box(ax,xy,w,h,text,color='#F0F4F7'):
    ax.add_patch(FancyBboxPatch(xy,w,h,boxstyle='round,pad=0.01',facecolor=color,edgecolor='#77818A',linewidth=.8));ax.text(xy[0]+w/2,xy[1]+h/2,text,ha='center',va='center',fontsize=8)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#444444','lw':1})

def main():
    s=json.loads((R/'analysis/summary.json').read_text());m=s['models'];rob=json.loads((R/'analysis/corruption_summary.json').read_text())['models'];ep=np.load(R/'analysis/ensemble_predictions.npz');c=np.load(OLD/'data/dermamnist_corrected_28.npz');y=c['test_labels'].ravel();p=ep['ContrastPenalty_C_test_raw'];pred=p.argmax(1)
    fig,ax=plt.subplots(figsize=(7,3.5));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    box(ax,(.03,.76),.94,.19,'8788 development images  →  lesion-grouped training | tuning | separate calibration\nTen matched repetitions; C / E / PH2 tests excluded from all fitting',color='#EAF1ED')
    box(ax,(.03,.36),.16,.23,'28 × 28 RGB\n49 fixed regions');box(ax,(.26,.52),.18,.17,'Raw descriptors\nGrid + color kNN');box(ax,(.26,.16),.18,.22,'15 descriptors\nTrain-only scaling\nShared MLP → Z')
    box(ax,(.51,.52),.16,.17,'Edge weights\n+ self-loops\nNormalize → S');box(ax,(.51,.16),.16,.22,'Four-step\npropagation\nH = Q₄(S)Z')
    box(ax,(.75,.36),.21,.23,'Mean + max pooling\nClassification head\n7 probabilities')
    arrow(ax,(.19,.50),(.26,.59));arrow(ax,(.19,.40),(.26,.28));arrow(ax,(.44,.61),(.51,.61));arrow(ax,(.44,.27),(.51,.27));arrow(ax,(.59,.52),(.59,.38));arrow(ax,(.67,.27),(.75,.44));ax.text(.51,.045,'Factorial tests substitute S and Z independently; weights remain fixed.',ha='center',fontsize=8.5)
    save(fig,'fig01_pipeline')
    th=json.loads((R/'analysis/finite_theory.json').read_text());a=.8;lam=np.linspace(0,2,401);q=a**4*(1-lam)**4+.2*sum(a**t*(1-lam)**t for t in range(4))
    fig,axs=plt.subplots(1,3,figsize=(7,2.55),layout='constrained');axs[0].plot(lam,q,label='K = 4',color=colors['ContrastPenalty']);axs[0].plot(lam,.2/(.2+.8*lam),'--',label='Fixed point',color=colors['ContrastNoPenalty']);axs[0].set(xlabel='Laplacian eigenvalue',ylabel='Filter response');axs[0].legend();tag(axs[0],'A  Spectral response')
    kk=np.arange(1,21);axs[1].plot(kk,4*(1-.8**kk),color=colors['ContrastNoPenalty']);axs[1].scatter([4],[2.3616],color=colors['ContrastPenalty']);axs[1].axhline(4,ls='--',color='grey',lw=.8);axs[1].set(xlabel='Propagation steps K',ylabel='Graph bound coefficient');tag(axs[1],'B  Finite-step sensitivity')
    rec=th['records'];axs[2].boxplot([[r[k] for r in rec] for k in ['finite_graph_bound_ratio','joint_bound_ratio','pooling_lipschitz_ratio']],tick_labels=['Graph','Joint','Pool']);axs[2].axhline(1,color='grey',ls='--');axs[2].set(ylabel='Observed / bound',ylim=(0,1.08));tag(axs[2],'C  Numerical agreement');save(fig,'fig02_theory')
    names=CFG['central_models'][:7]+['RGBOnlyGraph']+CFG['central_models'][7:];yy=np.arange(len(names));fig,axs=plt.subplots(1,2,figsize=(7,4.3),sharey=True,layout='constrained')
    for ax,metric,title in zip(axs,['macro_f1','balanced_accuracy'],['A  Macro F1','B  Balanced accuracy']):
        for i,n in enumerate(names):ax.errorbar(m[n]['seed_mean']['C_test'][metric],i,xerr=m[n]['seed_sd']['C_test'][metric],fmt='o',capsize=2,color=colors.get(n,'#4E6473'),markersize=4)
        ax.set_yticks(yy,[labels[n] for n in names]);ax.invert_yaxis() if ax==axs[0] else None;ax.set_xlabel('Mean ± sample SD, ten fits');ax.grid(axis='x',alpha=.2);tag(ax,title)
    save(fig,'fig03_components')
    fig,axs=plt.subplots(1,2,figsize=(7,3.65),layout='constrained');cm=np.array(m['ContrastPenalty']['datasets']['C_test']['raw']['confusion']);im=axs[0].imshow(cm/cm.sum(1,keepdims=True),vmin=0,vmax=1,cmap='Blues');axs[0].set(xticks=range(7),yticks=range(7),xticklabels=classes,yticklabels=classes,xlabel='Predicted class',ylabel='Reference class');axs[0].tick_params(axis='x',rotation=55)
    for i in range(7):
        for j in range(7):axs[0].text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=7,color='white' if cm[i,j]/cm[i].sum()>.6 else 'black')
    tag(axs[0],'A  Contrast graph confusion (counts)')
    for j,n in enumerate(['ContrastPenalty','SmallCNN','ResNet18']):
        vals=np.array([r['sensitivity'] for r in m[n]['datasets']['C_test']['classwise_raw']]);ci=np.array([r['sensitivity'] for r in m[n]['datasets']['C_test']['classwise_ci95']]);xx=np.arange(7)+(j-1)*.16;axs[1].errorbar(xx,vals,yerr=np.maximum(0,np.stack([vals-ci[:,0],ci[:,1]-vals])),fmt='o',capsize=2,color=colors[n],label=labels[n],markersize=4)
    axs[1].set(xticks=range(7),xticklabels=classes,ylim=(0,1.03),ylabel='Class sensitivity');axs[1].tick_params(axis='x',rotation=55);axs[1].legend(loc='lower left');tag(axs[1],'B  Lesion-cluster 95% intervals');save(fig,'fig04_errors')
    fig,axs=plt.subplots(1,2,figsize=(7,3),layout='constrained')
    for j,n in enumerate(['ContrastPenalty','ContrastNoPenalty']):
        r=rob[n]['noise_0.05']['conditions'];vv=np.array([r[v]['macro_f1'] for v in ['A','B','C','D']]);ci=np.array([r[v]['ci95']['macro_f1'] for v in ['A','B','C','D']]);xx=np.arange(4)+(j-.5)*.25;axs[0].bar(xx,vv,width=.23,color=colors[n],label=labels[n]);axs[0].errorbar(xx,vv,yerr=np.maximum(0,np.stack([vv-ci[:,0],ci[:,1]-vv])),fmt='none',ecolor='black',capsize=2)
    axs[0].set(xticks=range(4),xticklabels=['A: clean','B: graph','C: nodes','D: both'],ylabel='Macro F1',ylim=(0,.75));axs[0].legend(loc='upper right',fontsize=7);tag(axs[0],'A  Fixed-weight substitutions, σ = 0.05')
    levels=[.02,.05,.1]
    for metric,label,color in [('spectral_delta_mean','Mean spectral graph distance','#2E6A93'),('encoded_feature_relative_mean','Relative encoded-feature change','#B04A28'),('candidate_jaccard_mean','Candidate-edge Jaccard','#32845B')]:axs[1].plot(levels,[rob['ContrastPenalty'][f'noise_{v}']['structure'][metric] for v in levels],'-o',label=label,color=color,ms=4)
    axs[1].set(xlabel='Gaussian noise σ',ylabel='Dimensionless change / similarity');axs[1].legend(fontsize=7);tag(axs[1],'B  Graph and representation changes');save(fig,'fig05_isolation')
    fig,axs=plt.subplots(2,2,figsize=(7,4.8),layout='constrained');show=['ContrastPenalty','RGBOnlyGraph','SmallCNN','ResNet18','MobileNetV3Small']
    for n in show:
        v=[m[n]['datasets']['C_test']['raw']['macro_f1']]+[rob[n][f'noise_{lev}']['conditions']['D']['macro_f1'] for lev in levels];ci=np.array([[v[0],v[0]]]+[rob[n][f'noise_{lev}']['conditions']['D']['ci95']['macro_f1'] for lev in levels]);axs[0,0].errorbar([0]+levels,v,yerr=np.maximum(0,np.stack([np.array(v)-ci[:,0],ci[:,1]-np.array(v)])),fmt='-o',label=labels[n],color=colors[n],ms=3,capsize=2)
    axs[0,0].set(xlabel='Noise σ',ylabel='Macro F1');tag(axs[0,0],'A  Five noise realizations')
    for ax,keys,ticks,title in [(axs[0,1],['blur_0.5','blur_1.0','jpeg_40','jpeg_80'],['Blur .5','Blur 1','JPEG 40','JPEG 80'],'B  Blur and compression'),(axs[1,0],['brightness_0.75','brightness_1.25','contrast_0.75','contrast_1.25'],['B .75','B 1.25','C .75','C 1.25'],'C  Brightness and contrast')]:
        for n in show:ax.plot(range(4),[rob[n][key]['conditions']['D']['macro_f1'] for key in keys],'-o',color=colors[n],ms=3)
        ax.set(xticks=range(4),xticklabels=ticks,ylabel='Macro F1');tag(ax,title)
    for n in ['ContrastPenalty','ContrastNoPenalty','NoGraph','SmallCNN']:
        ax=axs[1,1];ax.plot([0]+levels,[m[n]['datasets']['C_test']['raw']['macro_f1']]+[rob[n][f'noise_{lev}']['conditions']['D']['macro_f1'] for lev in levels],'-o',color=colors.get(n,'#444'),label=labels[n],ms=3);ax.plot([0]+levels,[m[n+'_NoAug']['datasets']['C_test']['raw']['macro_f1']]+[rob[n+'_NoAug'][f'noise_{lev}']['conditions']['D']['macro_f1'] for lev in levels],'--',color=colors.get(n,'#444'),lw=1)
    axs[1,1].set(xlabel='Noise σ (solid: augmented; dashed: none)',ylabel='Macro F1');tag(axs[1,1],'D  Matched augmentation controls');handles,legend_labels=axs[0,0].get_legend_handles_labels()
    for handle,label in zip(*axs[1,1].get_legend_handles_labels()):
        if label not in legend_labels:handles.append(handle);legend_labels.append(label)
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=3,fontsize=7);save(fig,'fig06_robustness')
    fig,axs=plt.subplots(2,4,figsize=(7,4.2),layout='constrained');cal=ep['ContrastPenalty_C_test_calibrated']
    for ax,cl in zip(axs.flat,[-1]+list(range(7))):
        score=cal.max(1) if cl<0 else cal[:,cl];truth=(cal.argmax(1)==y) if cl<0 else y==cl;xx=[];yy=[];sz=[]
        for lo,hi in zip(np.arange(10)/10,np.arange(1,11)/10):
            mask=(score>=lo)&(score<hi if hi<1 else score<=1)
            if mask.any():xx.append(score[mask].mean());yy.append(truth[mask].mean());sz.append(8+30*mask.mean())
        ax.plot([0,1],[0,1],ls='--',c='grey',lw=.8);ax.plot(xx,yy,c=colors['ContrastPenalty'],lw=1);ax.scatter(xx,yy,s=sz,c=colors['ContrastPenalty']);ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1]);tag(ax,'Top label' if cl<0 else f'{classes[cl]} (n = {(y==cl).sum()})')
    fig.supxlabel('Mean predicted probability in equal-width bins');fig.supylabel('Observed frequency');save(fig,'fig07_calibration')
    fig,axs=plt.subplots(1,2,figsize=(7,3),layout='constrained');names2=['ContrastPenalty','ContrastNoPenalty','SmallCNN','ResNet18','MobileNetV3Small']
    for j,n in enumerate(names2):axs[0].bar(j,m[n]['datasets']['E_test']['raw']['macro_f1'],color=colors[n]);ci=m[n]['datasets']['E_test']['ci95']['macro_f1'];v=m[n]['datasets']['E_test']['raw']['macro_f1'];axs[0].errorbar(j,v,yerr=np.maximum(0,[[v-ci[0]],[ci[1]-v]]),fmt='none',ecolor='black',capsize=2)
    axs[0].set(xticks=range(5),xticklabels=[labels[n] for n in names2],ylabel='Seven-class macro F1');axs[0].tick_params(axis='x',rotation=55);tag(axs[0],'A  DermaMNIST-E, n = 1511')
    phy=np.load(R/'data/ph2_28.npz')['labels']
    for n in names2:
        fpr,tpr,_=roc_curve(phy,ep[n+'_PH2_raw'][:,4]);axs[1].plot(fpr,tpr,label=f"{labels[n]}: {m[n]['datasets']['PH2']['raw']['auc']:.3f}",color=colors[n])
    axs[1].plot([0,1],[0,1],ls='--',color='grey',lw=.8);axs[1].set(xlabel='False positive rate',ylabel='Melanoma sensitivity',xlim=(0,1),ylim=(0,1));axs[1].legend(fontsize=7);tag(axs[1],'B  PH2, 40 MEL / 160 nevi');save(fig,'fig08_external')
    sr=json.loads((R/'sensitivity/sensitivity_results.json').read_text());fig,axs=plt.subplots(2,3,figsize=(7,4.4),layout='constrained')
    for ax,param in zip(axs.flat,CFG['sensitivity_ranges']):
        vv=CFG['sensitivity_ranges'][param];vals=[];sd=[];ds=[]
        for value in vv:
            rows=[r for r in sr if (r['parameter']==param and r['value']==value) or (value==DEFAULT[param] and r['parameter']=='default')];assert len(rows)==3,(param,value,len(rows));sc=[r['tuning_macro_f1'] for r in rows];vals.append(np.mean(sc));sd.append(np.std(sc,ddof=1));ds.append(np.mean([r['graph_spectral_delta_mean'] for r in rows]))
        display={'k':'Neighbors k','K':'Steps K','alpha':'Restart α','sigma_color':'Color scale','sigma_position':'Spatial scale','beta':'Penalty β'};symbol={'k':'k','K':'K','alpha':'α','sigma_color':'σc','sigma_position':'σs','beta':'β'}
        ax.errorbar(vv,vals,yerr=sd,fmt='-o',color=colors['ContrastPenalty'],ms=3,capsize=2);ax.set_xlabel(symbol[param]);ax.set_xticks(vv,[str(value) for value in vv]);ax.set_ylabel('Tuning macro F1');tag(ax,display[param]);other=ax.twinx();other.plot(vv,ds,'--s',color=colors['ContrastNoPenalty'],ms=2,lw=1);other.set_ylabel('Mean spectral ΔS',color=colors['ContrastNoPenalty'],fontsize=7);other.tick_params(labelsize=6)
    save(fig,'fig09_sensitivity')
    fig,axs=plt.subplots(2,4,figsize=(7,3.9),layout='constrained');chosen=[]
    for cl in [0,2,4,6]:
        for correct in [True,False]:
            ids=np.flatnonzero((y==cl)&((pred==y)==correct));idx=int(ids[0]) if len(ids) else int(np.flatnonzero(y==cl)[0]);chosen.append(idx)
    for ax,idx in zip(axs.flat,chosen):ax.imshow(c['test_images'][idx],interpolation='nearest');ax.axis('off');ax.set_title(f"#{idx}: {classes[y[idx]]} → {classes[pred[idx]]}\np(MEL) = {p[idx,4]:.3f}",fontsize=8)
    save(fig,'fig10_examples');(F/'example_selection.json').write_text(json.dumps({'rule':'Lowest native index within each reference category and correctness stratum; fallback to lowest category index if a stratum is absent','indices':chosen},indent=2))
    idx=int(np.flatnonzero(y==4)[0]);clean=c['test_images'][idx:idx+1];dirty=corrupted(clean,'noise',.05,101);fc=features(clean);fd=features(dirty);mc=topology(fc)[0];md=topology(fd)[0];coords=fc[2];fig,axs=plt.subplots(2,3,figsize=(7,4.3),layout='constrained')
    for row,(image,f,mask,title) in enumerate([(clean[0],fc,mc,'Clean'),(dirty[0],fd,md,'Noise σ = 0.05')]):
        axs[row,0].imshow(image,interpolation='nearest');axs[row,0].axis('off');tag(axs[row,0],title+' image')
        ax=axs[row,1]
        for i,j in zip(*np.where(np.triu(mask,1))):ax.plot(coords[[i,j],0],1-coords[[i,j],1],color='#718898',alpha=.35,lw=.5)
        ax.scatter(coords[:,0],1-coords[:,1],c=f[1][0],s=22,edgecolor='black',linewidth=.2);ax.set_aspect('equal');ax.axis('off');tag(ax,'Grid + symmetrized kNN')
        im=axs[row,2].imshow(f[3][0].reshape(7,7),vmin=0,vmax=1,cmap='magma');axs[row,2].axis('off');tag(axs[row,2],'Local variation proxy b')
    fig.colorbar(im,ax=axs[:,2],shrink=.6,label='Proxy (0 to 1)');save(fig,'fig11_topology')
    tr=json.loads((R/'analysis/transfer_reference.json').read_text());fig,axs=plt.subplots(1,2,figsize=(7,2.9),layout='constrained')
    for j,ds in enumerate(['C_test','E_test']):
        v=tr['datasets'][ds]['raw']['macro_f1'];lo,hi=tr['datasets'][ds]['ci95']['macro_f1'];axs[0].bar(j,v,color='#475A79');axs[0].errorbar(j,v,yerr=[[v-lo],[hi-v]],fmt='none',color='black',capsize=3)
    axs[0].set(xticks=[0,1],xticklabels=['C test','E test'],ylabel='Macro F1',ylim=(0,.8));tag(axs[0],'A  Frozen ImageNet features / learned head')
    for n in ['ContrastPenalty','ResNet18']:
        v=[m[n]['datasets']['C_test']['raw']['macro_f1']]+[rob[n][f'noise_{lev}']['conditions']['D']['macro_f1'] for lev in levels];axs[1].plot([0]+levels,v,'-o',ms=3,color=colors[n],label=labels[n])
    vv=[tr['datasets']['C_test']['raw']['macro_f1']]+[tr['corruption_mean'][f'noise_{lev}']['macro_f1'] for lev in levels];ci=np.array([tr['datasets']['C_test']['ci95']['macro_f1']]+[tr['corruption_mean'][f'noise_{lev}']['macro_f1_ci95'] for lev in levels]);axs[1].errorbar([0]+levels,vv,yerr=np.maximum(0,np.stack([np.array(vv)-ci[:,0],ci[:,1]-np.array(vv)])),fmt='-s',label='Frozen ImageNet MobileNetV3',color='#475A79',ms=3,capsize=2);axs[1].set(xlabel='Gaussian noise σ',ylabel='Macro F1');axs[1].legend(fontsize=7);tag(axs[1],'B  Additional reference; fixed weights');save(fig,'fig12_transfer')
    print('Twelve PNG, SVG and PDF figures saved',flush=True)
if __name__=='__main__':main()
