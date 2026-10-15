from optimized_cnn import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
result=json.loads((HERE/'analysis/optimized_cnn.json').read_text());prior=json.loads((R/'analysis/corruption_summary.json').read_text())['models'];summary=json.loads((R/'analysis/summary.json').read_text())['models']
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for rec in result['selections']:
    history=json.loads((HERE/'optimized_cnn/candidates'/str(rec['seed'])/f"lr_{rec['selected_learning_rate']}"/'training.json').read_text())['history'];x=[r['epoch'] for r in history];y=[r['tuning_macro_f1'] for r in history];ax[0].plot(x,y,alpha=.45,lw=.9);ax[0].plot(rec['best_epoch'],rec['tuning_macro_f1'],'o',ms=3)
ax[0].set(xlabel='Epoch',ylabel='Tuning macro F1',title='A  Ten selected candidate histories')
for name,label,color in [('ContrastPenalty','Penalized regional network','#b74528'),('SmallCNN_NoAug','Original unaugmented CNN','#689b7a'),('tuned','Development-tuned CNN','#2d6f9c')]:
    if name=='tuned':y=[result['datasets']['C_test']['raw']['macro_f1']]+[result['corruption_mean'][f'noise_{s}']['macro_f1'] for s in [.02,.05,.1]]
    else:y=[summary[name]['datasets']['C_test']['raw']['macro_f1']]+[prior[name][f'noise_{s}']['conditions']['D']['macro_f1'] for s in [.02,.05,.1]]
    ax[1].plot([0,.02,.05,.1],y,'o-',label=label,color=color,ms=4)
ax[1].set(xlabel='Gaussian noise standard deviation',ylabel='Frozen-ensemble macro F1',title='B  Compact reference and noise response');ax[1].legend(frameon=False,fontsize=8,loc='upper right')
for axis in ax:
    for s in ['top','right']:axis.spines[s].set_visible(False)
fig.tight_layout()
for extension in ['png','svg','pdf']:fig.savefig(HERE/'figures'/('fig_tuned_cnn.'+extension),dpi=400,bbox_inches='tight')
plt.close(fig);print('Compact CNN figure generated',flush=True)
