from experiment_core import *
def main():
    im,y,g,m=load_development();selected=im[:32];f=features(selected);q=torch.tensor(graph_matrix(f));raw=torch.tensor(f[0]);max_feature_error=0;max_graph_error=0
    for rotation in range(4):
        for flip in [False,True]:
            transformed=np.rot90(selected,rotation,axes=(1,2)).copy()
            if flip:transformed=np.flip(transformed,axis=2).copy()
            target=features(transformed);a,b=augment_region(raw,q,rotation,flip)
            max_feature_error=max(max_feature_error,float(np.abs(a.numpy()-target[0]).max()))
            max_graph_error=max(max_graph_error,float(np.abs(b.numpy()-graph_matrix(target)).max()))
    assert max_feature_error<1e-6
    out=R/'analysis';out.mkdir(exist_ok=True)
    result=dict(n=32,transforms=8,max_descriptor_equivalence_error=max_feature_error,max_graph_reconstruction_error=max_graph_error,scope='Graph augmentation permutes an already constructed graph. kNN distance ties may yield different neighbors if graphs are independently rebuilt; exact raw image reconstruction is asserted for node descriptors, not all tied graph constructions.')
    (out/'augmentation_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
