from pathlib import Path
import sys,json
from experiment_core import *
def main():
    images,y,groups,meta=load_development();f=features(images[:64]);s=graph_matrix(f,return_s=True).astype(np.float64)
    noise=np.random.default_rng(20261014).normal(0,2,images[:64].shape)
    ip=np.clip(images[:64].astype(float)+noise,0,255).astype(np.uint8);fp=features(ip);sp=graph_matrix(fp,return_s=True).astype(np.float64)
    alpha=.2;K=4;a=1-alpha;coefficient=a*(1-a**K)/alpha;records=[];rng=np.random.default_rng(2901)
    for i in range(64):
        z=rng.normal(size=(49,32));zp=z+rng.normal(0,.01,z.shape)
        q=np.eye(49);qp=q.copy()
        for _ in range(K):q=a*s[i]@q+alpha*np.eye(49);qp=a*sp[i]@qp+alpha*np.eye(49)
        norm=np.linalg.norm(s[i]-sp[i],2);bound=coefficient*norm*np.linalg.norm(z)
        error=np.linalg.norm(q@z-qp@z);joint=np.linalg.norm(q@z-qp@zp)
        jointbound=np.linalg.norm(z-zp)+coefficient*norm*np.linalg.norm(zp)
        pool=lambda x:np.r_[x.mean(0),x.max(0)]
        poolratio=np.linalg.norm(pool(q@z)-pool(qp@zp))/(np.sqrt(1+1/49)*joint)
        # Independent finite polynomial, recurrence, finite spectral response and fixed-point checks.
        poly=a**K*np.linalg.matrix_power(s[i],K)+alpha*sum(a**t*np.linalg.matrix_power(s[i],t) for t in range(K))
        lam,U=np.linalg.eigh(np.eye(49)-s[i]);qq=a*(1-lam);g=qq**K+alpha*(1-qq**K)/(1-qq)
        spectral=(U*g)@U.T
        stationary=np.linalg.solve(np.eye(49)-a*s[i],alpha*z);H=z.copy()
        for _ in range(40):H=a*s[i]@H+alpha*z
        records.append(dict(graph=i,finite_graph_bound_ratio=error/bound,joint_bound_ratio=joint/jointbound,pooling_lipschitz_ratio=float(poolratio),operator_norm=float(np.linalg.norm(q,2)),polynomial_error=float(np.abs(q-poly).max()),spectral_error=float(np.abs(q-spectral).max()),fixedpoint_error40=float(np.linalg.norm(H-stationary)),delta_s=float(norm),delta_z=float(np.linalg.norm(z-zp))))
    for key in ['finite_graph_bound_ratio','joint_bound_ratio','pooling_lipschitz_ratio','operator_norm']:assert max(x[key] for x in records)<=1+1e-6
    for key in ['polynomial_error','spectral_error']:assert max(x[key] for x in records)<1e-9
    summary={key:max(x[key] for x in records) for key in records[0] if key!='graph'}
    (R/'analysis').mkdir(exist_ok=True)
    (R/'analysis/finite_theory.json').write_text(json.dumps(dict(alpha=alpha,K=K,coefficient=coefficient,pooling_constant=float(np.sqrt(1+1/49)),n=64,summary=summary,records=records),indent=2))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
