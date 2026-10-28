"""Analytic molecular path derivatives of an odd empirical kernel measure."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.linalg import solve
from .core import persistent_operator,cochains,cochains_jvp,distances,rips
from .shape_geometry import bond_rotation,cochains_jet2
from .ensemble import rms_normalized_jets,odd_norm_jets,odd_statistics


def main(folder):
    root=Path(__file__).resolve().parents[2]
    census=json.loads((root/'data/extension_selection.json').read_text())
    chosen=None
    for selected in census['selected']:
        parent=json.loads((folder/'parents'/(selected['mol_id']+'.json')).read_text())
        if parent.get('status')!='completed':continue
        first=parent['records'][0];x=np.array(first['coordinates']);n=len(x)
        for a,b,order in first['bonds']:
            if order!=1:continue
            try:
                xx,v,acc,moving=bond_rotation(x,first['bonds'],(a,b),a)
                if 3<=len(moving)<=n-3 and np.linalg.norm(v)>1e-6:
                    chosen=(parent,(a,b),a,moving);break
            except ValueError:continue
        if chosen:break
    if chosen is None:raise ValueError('no admissible illustrative bridge torsion')
    parent,edge,moving_vertex,moving=chosen;records=parent['records'];first=records[0]
    x=np.array(first['coordinates']);af=np.array(first['atom_features']);label=first['en_id']
    take=np.array([r['en_id']==label for r in records]);ops=[];sources=[];probes=[]
    for rec in records:
        xx=np.array(rec['coordinates']);op=persistent_operator(xx,2.6,3.2);c,b=cochains(xx,op.small[2],np.array(rec['atom_features']))
        ops.append(op);sources.append(c);probes.append(b)
    original=ops[0];d=distances(x);i,j=np.triu_indices(len(x),1)
    margin=float(min(np.min(np.abs(d[i,j]-2.6)),np.min(np.abs(d[i,j]-3.2))))
    _,v0,a0,_=bond_rotation(x,first['bonds'],edge,moving_vertex)
    vmax=np.max(np.linalg.norm(v0,axis=1));window=min(.015,margin/(20*max(vmax,1e-12)))
    def molecular_jets(theta,tau):
        xx,v,acc,_=bond_rotation(x,first['bonds'],edge,moving_vertex,theta)
        if rips(xx,2.6)!=original.small or rips(xx,3.2)!=original.large:raise AssertionError('filtration chamber changed')
        c,b,dc,db,d2c,d2b=cochains_jet2(xx,original.small[2],af,v)
        ca,ba=cochains_jvp(xx,original.small[2],af,acc)[2:]
        p=[];dp=[];d2p=[];hessian_only=[]
        for k,(op,ck,bk) in enumerate(zip(ops,sources,probes)):
            matrix=op.matrix+tau*np.eye(len(op.matrix));den=max(len(op.small[2]),1)
            if k==0:
                u=solve(matrix,c,assume_a='pos');du=solve(matrix,dc,assume_a='pos')
                dd_h=solve(matrix,d2c,assume_a='pos');dd=solve(matrix,d2c+ca,assume_a='pos')
                p.append((b.T@u/den).ravel());dp.append(((db.T@u+b.T@du)/den).ravel())
                d2p.append((((d2b+ba).T@u+2*db.T@du+b.T@dd)/den).ravel())
                hessian_only.append(((d2b.T@u+2*db.T@du+b.T@dd_h)/den).ravel())
            else:
                u=solve(matrix,ck,assume_a='pos');p.append((bk.T@u/den).ravel());dp.append(np.zeros(18));d2p.append(np.zeros(18));hessian_only.append(np.zeros(18))
        return np.array(p),np.array(dp),np.array(d2p),np.array(hessian_only)
    def kernel_jets(theta,tau=.3,sigma=1.):
        p,v,a,ah=molecular_jets(theta,tau)
        z,dz,ddz=rms_normalized_jets(p,v,a)
        result=odd_norm_jets(z[take],dz[take],ddz[take],sigma)
        _,_,wrong_ddz=rms_normalized_jets(p,v,ah)
        wrong=odd_norm_jets(z[take],dz[take],wrong_ddz[take],sigma)
        return result,wrong
    base,wrong=kernel_jets(0)
    checks=[]
    for h in [window/2,window/6,window/18]:
        plus,_=kernel_jets(h);minus,_=kernel_jets(-h)
        fd=(plus[1]-minus[1])/(2*h)
        checks.append({'step':h,'finite_difference_of_first':fd,'absolute_error':abs(fd-base[2])})
    theta=np.linspace(-window,window,41);curves=np.array([kernel_jets(t)[0] for t in theta])
    taus=np.geomspace(.03,3,25);sigmas=np.linspace(.5,2,23);surface=np.empty((len(sigmas),len(taus)))
    for j,tau in enumerate(taus):
        p,v,a,_=molecular_jets(0,tau);z,dz,ddz=rms_normalized_jets(p,v,a)
        for i,sigma in enumerate(sigmas):surface[i,j]=odd_norm_jets(z[take],dz[take],ddz[take],sigma)[2]
    # A well-conditioned abstract finite-support fixture makes the sharp limit
    # visible without concealing numerical rank loss in molecular samples.
    fixture=np.array([[-3.],[1.],[2.]])
    from .ensemble import odd_kernel
    gram=odd_kernel(fixture);coefficient=1/(np.ones(3)@np.linalg.solve(gram,np.ones(3)))
    lambdas=np.geomspace(1e-7,1,100);scores=np.array([odd_statistics(fixture,regularization=l)['centered_fisher_score'] for l in lambdas])
    payload={'mol_id':parent['mol_id'],'row_key':first['row_key'],'en_id':label,'r':2.6,'s':3.2,'tau':.3,'sigma':1.,
       'axis_edge':edge,'moving_vertex':moving_vertex,'moving_atoms':moving,'coordinates':x.tolist(),'elements':first['elements'],'bonds':first['bonds'],
       'n_parent_conformers':len(records),'n_class_conformers':int(take.sum()),'filtration_margin_angstrom':margin,'theta_window_radians':window,
       'normalization':'RMS recomputed along the path, with analytic first and second derivatives',
       'kernel_jets':base.tolist(),'omitted_path_acceleration_jets':wrong.tolist(),'finite_difference_checks':checks,
       'theta':theta.tolist(),'curve_jets':curves.tolist(),'tau_grid':taus.tolist(),'sigma_grid':sigmas.tolist(),'curvature_surface':surface.tolist(),
       'fixture':{'points':fixture.tolist(),'gram':gram.tolist(),'lambda':lambdas.tolist(),'scores':scores.tolist(),'divergence_coefficient':float(coefficient)}}
    (folder/'ensemble_geometry.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    print(json.dumps({k:payload[k] for k in ['mol_id','n_parent_conformers','n_class_conformers','axis_edge','filtration_margin_angstrom','theta_window_radians','kernel_jets','omitted_path_acceleration_jets','finite_difference_checks']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
