"""Independent arithmetic, partition, checkpoint, formula and artifact audit."""
from experiment_core import *
from scipy.stats import rankdata
import zipfile,re
from lxml import etree
import pymupdf

def independent(y,p):
    labels=p.argmax(1);cm=np.zeros((7,7),int)
    for actual,pred in zip(y,labels):cm[int(actual),int(pred)]+=1
    tp=np.diag(cm);fn=cm.sum(1)-tp;fp=cm.sum(0)-tp;f1=np.divide(2*tp,2*tp+fn+fp,out=np.zeros(7),where=2*tp+fn+fp>0).mean();ba=(tp/cm.sum(1)).mean();auc=[]
    for cl in range(7):
        pos=y==cl;n1=pos.sum();n0=len(y)-n1;ranks=rankdata(p[:,cl]);auc.append((ranks[pos].sum()-n1*(n1+1)/2)/(n0*n1))
    return dict(macro_f1=float(f1),balanced_accuracy=float(ba),accuracy=float(tp.sum()/len(y)),macro_auc=float(np.mean(auc)))

def main():
    names=CFG['central_models']+CFG['augmentation_controls']+['RGBOnlyGraph'];assert len(names)==15
    d=np.load(OLD/'data/dermamnist_corrected_28.npz');e=np.load(R/'data/dermamnist_extended_28.npz');ys={'C_test':d['test_labels'].ravel(),'E_test':e['test_labels'].ravel()};summary=json.loads((R/'analysis/summary.json').read_text());ensemble=np.load(R/'analysis/ensemble_predictions.npz');dev,dy,groups,_=load_development();raw=torch.tensor(features(dev)[0]);test_im,test_raw,test_f=tensors(d['test_images']);count=0;largest=0;maxlogit=0;checks={}
    for name in names:
        kind=name.replace('_NoAug','');test_q=None if kind in ['NoGraph','RGBOnly','SmallCNN','ResNet18','MobileNetV3Small'] else torch.tensor(graph_matrix(test_f,kind))
        for seed in CFG['main_seeds']:
            p=R/'models'/name/str(seed);training=json.loads((p/'training.json').read_text());assert training['seed']==seed and 1<=training['best_epoch']<=60;assert training['tuning_macro_f1']>=max(v['tuning_macro_f1'] for v in training['history'])-1e-6;assert .1<=training['temperature']<=10
            ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False);assert ck['kind']==name.replace('_NoAug','');assert ck['augment']==(not name.endswith('_NoAug'));assert abs(ck['temperature']-training['temperature'])<1e-10;parts=np.load(p/'development.npz');wanted=partitions(dy,groups,seed)
            for k,field in [('train','train_index'),('tune','tuning_index'),('cal','calibration_index')]:assert np.array_equal(parts[field],wanted[k]);assert len(parts[field])==training['n'][k]
            mu=raw[wanted['train']].mean((0,1));sd=raw[wanted['train']].std((0,1),correction=0).clamp_min(1e-5);assert torch.allclose(ck['mu'],mu,atol=1e-6);assert torch.allclose(ck['sd'],sd,atol=1e-6)
            arrays=np.load(R/'evaluation'/name/str(seed)/'predictions.npz');row=json.loads((R/'evaluation'/name/str(seed)/'metrics.json').read_text())
            model=make_model(kind);model.load_state_dict(ck['state']);logits=predict(model,test_im,test_raw,test_q,ck['mu'],ck['sd'],kind);assert np.max(abs(logits-arrays['C_test_logits']))<1e-5,(name,seed,'Checkpoint prediction disagreement')
            for ds,y in ys.items():
                prob=arrays[ds+'_raw'];assert prob.shape==(len(y),7) and np.isfinite(prob).all();assert np.max(abs(prob.sum(1)-1))<1e-6;v=independent(y,prob)
                for key,value in v.items():err=abs(value-row['datasets'][ds]['raw'][key]);largest=max(largest,err);assert err<2e-7,(name,seed,ds,key,err)
                logits=arrays[ds+'_logits'];exp=np.exp(logits-logits.max(1,keepdims=True));pp=exp/exp.sum(1,keepdims=True);assert np.max(abs(pp-prob))<2e-7
            count+=1
        for ds,y in ys.items():
            v=independent(y,ensemble[name+'_'+ds+'_raw'])
            for key,value in v.items():err=abs(value-summary['models'][name]['datasets'][ds]['raw'][key]);largest=max(largest,err);assert err<2e-7
    assert count==150;sr=json.loads((R/'sensitivity/sensitivity_results.json').read_text());assert len(sr)==42
    assert len({(r['parameter'],r['value'],r['seed']) for r in sr})==42
    for parameter,values in CFG['sensitivity_ranges'].items():
        for value in values:assert len([r for r in sr if (r['parameter']==parameter and r['value']==value) or (value==DEFAULT[parameter] and r['parameter']=='default')])==3
    for row in sr:
        assert row['seed'] in CFG['sensitivity_seeds'] and row['graph_spectral_delta_mean']>=0;prefix={'k':'neighbors','K':'steps'}.get(row['parameter'],row['parameter']);folder=R/'sensitivity'/f"{prefix}_{row['value']}_{row['seed']}";ck=torch.load(folder/'checkpoint.pt',weights_only=False,map_location='cpu');assert ck['actual_graph_configuration']==row['configuration']
    rows=[json.loads(p.read_text()) for p in (R/'corruptions').glob('*/*.json')];assert len(rows)==15*23
    for row in rows:
        arrays=np.load(R/'corruptions'/row['name']/(row['case']+'.npz'))
        for cond,met in row['conditions'].items():
            v=independent(ys['C_test'],arrays[cond])
            for k,value in v.items():assert abs(value-met[k])<2e-7
        assert np.max(abs(arrays['A']-ensemble[row['name']+'_C_test_raw']))<2e-6
        if 'conditional_logit_bound_ratio_max' in row:maxlogit=max(maxlogit,row['conditional_logit_bound_ratio_max']);assert row['conditional_logit_bound_ratio_max']<=1.00001
        if row['name']=='MobileNetV3Small':assert np.all(arrays['D'].argmax(1)==5)
    refs=json.loads((R/'bibliography/references.json').read_text(encoding='utf-8'));audit=json.loads((R/'bibliography/reference_audit_final.json').read_text());cross=json.loads((R/'bibliography/doi_audit.json').read_text(encoding='utf-8'));dc=json.loads((R/'bibliography/datacite_audit.json').read_text(encoding='utf-8'));registered={r['doi'].lower() for r in cross+dc};assert len(refs)==21 and sum(v['doi'] is not None for v in refs)==20
    for ref in refs:
        if ref['doi']:assert ref['doi'].lower() in registered
        assert audit[ref['number']-1]['doi']==ref['doi'];assert '&amp;' not in ref['text'];assert 'et al..' not in ref['text']
    path=R/'final/Skin_Lesion_GNN_Revised_Manuscript.docx';manifest=json.loads((R/'final/document_manifest.json').read_text())
    with zipfile.ZipFile(path) as z:
        root=etree.fromstring(z.read('word/document.xml'));ns={'m':'http://schemas.openxmlformats.org/officeDocument/2006/math','w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'};texts=' '.join(root.xpath('.//w:t/text()',namespaces=ns));assert len(root.xpath('.//m:oMath',namespaces=ns))==manifest['native_math'];assert manifest['equations']>=25;assert manifest['figures']==12
        for marker in ['{{','[TABLE:','[FIG:','[REFERENCES]','[AUTHOR]']:assert marker not in texts
        for required in ['Caglar Uyulan','caglar.uyulan@ikcu.edu.tr','no external financial support','no competing interests']:assert required in texts
        assert '\\frac' not in texts;assert 'et al..' not in texts
        for i in range(1,22):assert '['+str(i)+']' in texts
    pdf=R/'final/Skin_Lesion_GNN_Revised_Manuscript.pdf';doc=pymupdf.open(pdf);pages=[]
    for i,page in enumerate(doc):
        words=page.get_text('words');assert words,(i,'Empty PDF page');outside=[w for w in words if w[0]<28 or w[1]<25 or w[2]>page.rect.width-25 or w[3]>page.rect.height-20];assert not outside,(i,outside[:3]);pages.append({'page':i+1,'words':len(words),'images':len(page.get_images())})
    maps=json.loads((R/'analysis/reporting_maps.json').read_text());assert len(maps['TRIPOD_AI'])==52 and len(maps['CLAIM_2024'])==44
    tr=json.loads((R/'analysis/transfer_reference.json').read_text());tp=np.load(R/'transfer_reference/predictions.npz');tz=np.load(R/'transfer_reference/cache/development.npy');assert tr['head_fits']==10 and tr['learned_parameters']==4039
    for j,seed in enumerate(CFG['main_seeds']):
        ck=torch.load(R/'transfer_reference/models'/str(seed)/'checkpoint.pt',weights_only=False,map_location='cpu');devp=np.load(R/'transfer_reference/models'/str(seed)/'development.npz');assert ck['weight_sha256']==tr['weight_sha256'];wanted=partitions(dy,groups,seed)
        for k,field in [('train','train_index'),('tune','tuning_index'),('cal','calibration_index')]:assert np.array_equal(devp[field],wanted[k])
        z=torch.tensor(tz);assert torch.allclose(ck['mu'],z[wanted['train']].mean(0));assert torch.allclose(ck['sd'],z[wanted['train']].std(0,correction=0).clamp_min(1e-5));head=nn.Linear(576,7);head.load_state_dict(ck['state']);head.eval()
        with torch.no_grad():log=head((torch.tensor(np.load(R/'transfer_reference/cache/C_test.npy'))-ck['mu'])/ck['sd']).numpy()
        assert np.max(abs(log-tp['C_test_seed_logits'][j]))<1e-5
    for ds,y in ys.items():
        v=independent(y,tp[ds+'_raw'])
        for k,value in v.items():assert abs(value-tr['datasets'][ds]['raw'][k])<2e-7
    assert len(tr['corruptions'])==23
    for key,row in tr['corruptions'].items():
        pp=np.load(R/'transfer_reference/corruptions'/(key+'.npz'))['raw'];v=independent(ys['C_test'],pp)
        for k,value in v.items():assert abs(value-row['metrics'][k])<2e-7
    result=dict(status='passed',primary_fits=count,additional_transfer_heads=10,sensitivity_fits=len(sr),corruption_ensembles=len(rows),additional_transfer_corruptions=23,largest_independent_metric_error=largest,max_trained_logit_bound_ratio=maxlogit,equations=manifest['equations'],figures=manifest['figures'],tables=manifest['tables'],reference_count=21,verified_DOI_count=20,no_DOI_exceptions=1,pdf_pages=len(doc),page_audit=pages,docx_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),visual_QA='Requires separate visual inspection of every rendered page; this automated check alone does not establish layout quality.')
    (R/'final/verification_report.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='page_audit'},indent=2),flush=True)
if __name__=='__main__':main()
