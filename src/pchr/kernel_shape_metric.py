"""Mass-orthonormal molecular sensitivity and the odd-kernel rank bottleneck."""
from pathlib import Path
import argparse,json
import numpy as np
from scipy.linalg import solve
from rdkit import Chem
from .core import persistent_operator,cochains,cochains_jvp,distances,rips
from .ensemble import odd_pullback_metric


def main(folder):
    root=Path(__file__).resolve().parents[2]
    selection=json.loads((root/'data/extension_selection.json').read_text())['selected']
    parent=json.loads((folder/'parents'/(selection[0]['mol_id']+'.json')).read_text());record=parent['records'][0]
    x=np.array(record['coordinates']);f=np.array(record['atom_features']);n=len(x)
    mass=np.array([Chem.GetPeriodicTable().GetAtomicWeight(int(z)) for z in f[:,0]])
    y=x-np.average(x,axis=0,weights=mass);rootmass=np.repeat(np.sqrt(mass),3)
    rigid=[]
    for axis in np.eye(3):rigid.append((np.ones((n,1))*axis).ravel()*rootmass)
    for axis in np.eye(3):rigid.append(np.cross(axis,y).ravel()*rootmass)
    rigid=np.column_stack(rigid);q,_=np.linalg.qr(rigid,mode='complete');horizontal=q[:,6:];lift=horizontal/rootmass[:,None]
    op=persistent_operator(x,2.6,3.2);c,b=cochains(x,op.small[2],f);matrix=op.matrix+.3*np.eye(len(op.matrix))
    u=solve(matrix,c,assume_a='pos');den=max(len(op.small[2]),1);p=(b.T@u/den).ravel();scale=np.linalg.norm(p)
    jacobian=np.empty((18,3*n))
    for j in range(3*n):
        v=np.zeros(3*n);v[j]=1
        _,_,dc,db=cochains_jvp(x,op.small[2],f,v.reshape(n,3));du=solve(matrix,dc,assume_a='pos')
        jacobian[:,j]=((db.T@u+b.T@du)/den).ravel()
    jshape=jacobian@lift/scale;z=p/scale;metric=odd_pullback_metric(z)
    induced=jshape.T@metric@jshape;euclidean=jshape.T@jshape
    left,singular,right=np.linalg.svd(jshape,full_matrices=True);tol=singular[0]*1e-8;rank=int(np.sum(singular>tol))
    c0=(1-2*np.exp(-1.5))/2
    lower=np.linalg.eigvalsh(induced-c0*euclidean);upper=np.linalg.eigvalsh(euclidean-induced)
    if lower.min()<-1e-10 or upper.min()<-1e-10:raise AssertionError('metric sandwich failed')
    directions={'maximum':lift@right[0],'null':lift@right[-1]};checks=[]
    dd=distances(x);i,j=np.triu_indices(n,1);margin=float(min(np.min(np.abs(dd[i,j]-2.6)),np.min(np.abs(dd[i,j]-3.2))))
    for name,flat in directions.items():
        v=flat.reshape(n,3);maxspeed=np.max(np.linalg.norm(v,axis=1));h=min(1e-4,margin/(20*max(maxspeed,1e-12)))
        values=[]
        for sign in [-1,1]:
            xx=x+sign*h*v
            if rips(xx,2.6)!=op.small or rips(xx,3.2)!=op.large:raise AssertionError('shape check crossed chamber')
            cc,bb=cochains(xx,op.small[2],f);uu=solve(matrix,cc,assume_a='pos');values.append((bb.T@uu/den).ravel()/scale)
        numeric=(values[1]-values[0])/(2*h);analytic=jacobian@flat/scale
        checks.append({'direction':name,'step':h,'analytic_derivative_norm':float(np.linalg.norm(analytic)),
            'finite_difference_norm':float(np.linalg.norm(numeric)),'absolute_vector_error':float(np.linalg.norm(numeric-analytic))})
    output={'mol_id':parent['mol_id'],'n_atoms':n,'shape_dimension':3*n-6,'readout_dimension':18,'numerical_rank':rank,
       'rank_relative_tolerance':1e-8,'numerical_nullity':3*n-6-rank,'dimension_lower_bound':max(3*n-24,0),
       'normalization':'fixed baseline response norm for this local metric experiment; RMS-changing derivatives are analysed separately',
       'coordinates':x.tolist(),'elements':record['elements'],'bonds':record['bonds'],'fixed_scale':scale,'feature':z.tolist(),
       'singular_values':singular.tolist(),'induced_metric_eigenvalues':np.linalg.eigvalsh(induced).tolist(),
       'metric_eigenvalues':np.linalg.eigvalsh(metric).tolist(),'sandwich_lower_min_eigenvalue':float(lower.min()),
       'sandwich_upper_min_eigenvalue':float(upper.min()),'direction_checks':checks,'maximum_velocity':directions['maximum'].reshape(n,3).tolist(),
       'null_velocity':directions['null'].reshape(n,3).tolist()}
    (folder/'kernel_shape_metric.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps({k:output[k] for k in ['mol_id','n_atoms','shape_dimension','numerical_rank','numerical_nullity','dimension_lower_bound','direction_checks']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
