"""Intrinsic gradients and a bridge-bond torsion on frozen molecular geometry."""
import hashlib
import json
from pathlib import Path

import numpy as np
from rdkit import Chem
from scipy.linalg import solve

from .core import cochains,cochains_jvp,distances,persistent_operator
from .shape_geometry import bond_rotation,cochains_jet2,horizontal_projection,metric


ROOT=Path(__file__).resolve().parents[2]


def readout_gradient(x,f,op,tau=.3):
    c,b=cochains(x,op.small[2],f);matrix=op.matrix+tau*np.eye(len(op.matrix))
    u=solve(matrix,c,assume_a='pos');gradient=np.zeros_like(x)
    for i in range(len(x)):
        for j in range(3):
            v=np.zeros_like(x);v[i,j]=1
            _,_,dc,db=cochains_jvp(x,op.small[2],f,v)
            gradient[i,j]=(db.T@u+b.T@solve(matrix,dc,assume_a='pos'))[0,0]
    return float((b.T@u)[0,0]),gradient


def run():
    path=ROOT/'data/conformers.json';records=json.loads(path.read_text())['records']
    names=['L-alanine','L-cysteine','L-methionine','(S)-naproxen'];output=[]
    pt=Chem.GetPeriodicTable()
    for name in names:
        rec=next(r for r in records if r['name']==name)
        x=np.asarray(rec['conformers'][0]['coordinates']);f=np.asarray(rec['atom_features'])
        masses=np.array([pt.GetAtomicWeight(int(z)) for z in f[:,0]])
        op=persistent_operator(x,2.6,3.2);p,grad=readout_gradient(x,f,op)
        g=grad/masses[:,None];projection=horizontal_projection(x,g,masses);horizontal=projection['horizontal']
        mirror=x*np.array([-1,1,1]);mop=persistent_operator(mirror,2.6,3.2);pm,gm=readout_gradient(mirror,f,mop)
        mirror_gradient_error=float(np.max(abs(gm+grad*np.array([-1,1,1]))))
        rng=np.random.default_rng(51228);v=horizontal_projection(x,rng.normal(size=x.shape),masses)['horizontal']
        v/=np.sqrt(metric(v,v,masses))
        c,b,dc,db,d2c,d2b=cochains_jet2(x,op.small[2],f,v)
        matrix=op.matrix+.3*np.eye(len(op.matrix));u=solve(matrix,c,assume_a='pos');du=solve(matrix,dc,assume_a='pos')
        hess=float((d2b.T@u+2*db.T@du+b.T@solve(matrix,d2c,assume_a='pos'))[0,0])
        pair=distances(x)[np.triu_indices(len(x),1)]
        margin=min(abs(pair-2.6).min(),abs(pair-3.2).min())
        h=min(1e-3,margin/(8*np.linalg.norm(v,axis=1).max()))
        for coords in (x+h*v,x-h*v):
            check=persistent_operator(coords,2.6,3.2)
            assert check.small==op.small and check.large==op.large
        def first_derivative(coords,velocity):
            cc,bb,dcc,dbb=cochains_jvp(coords,op.small[2],f,velocity)
            uu=solve(matrix,cc,assume_a='pos')
            return float((dbb.T@uu+bb.T@solve(matrix,dcc,assume_a='pos'))[0,0])
        observed=(first_derivative(x+h*v,v)-first_derivative(x-h*v,v))/(2*h)
        item=dict(name=name,readout=p,masses=masses,gradient=grad,
             intrinsic_gradient=horizontal,intrinsic_gradient_norm=np.sqrt(metric(horizontal,horizontal,masses)),
             translation_residual=np.linalg.norm(grad.sum(axis=0)),
             rotation_residual=np.linalg.norm(np.sum(np.cross(projection['centered_coordinates'],grad),axis=0)),
             projection_difference=np.linalg.norm(horizontal-g),mirror_readout_error=abs(pm+p),
             mirror_gradient_error=mirror_gradient_error,horizontal_unit_direction=v,
             intrinsic_hessian_direction=hess,hessian_finite_difference=observed,hessian_error=abs(hess-observed),hessian_step=h)
        if name=='L-methionine':
            _,velocity,acceleration,moving=bond_rotation(x,rec['bonds'],(2,3),0)
            c,b,dc,db,d2c,d2b=cochains_jet2(x,op.small[2],f,velocity)
            _,_,ac,ab=cochains_jvp(x,op.small[2],f,acceleration)
            du=solve(matrix,dc,assume_a='pos')
            straight=float((d2b.T@u+2*db.T@du+b.T@solve(matrix,d2c,assume_a='pos'))[0,0])
            acceleration_term=float((ab.T@u+b.T@solve(matrix,ac,assume_a='pos'))[0,0])
            d=distances(x);pair=d[np.triu_indices(len(x),1)]
            margin=min(abs(pair-2.6).min(),abs(pair-3.2).min())
            max_speed=np.linalg.norm(velocity,axis=1).max()
            step=min(.005,margin/(8*max_speed));convergence=[]
            for hh in (step,step/3,step/9):
                yp,vp,_,_=bond_rotation(x,rec['bonds'],(2,3),0,hh)
                ym,vm,_,_=bond_rotation(x,rec['bonds'],(2,3),0,-hh)
                for coords in (yp,ym):
                    check=persistent_operator(coords,2.6,3.2)
                    assert check.small==op.small and check.large==op.large
                finite=(first_derivative(yp,vp)-first_derivative(ym,vm))/(2*hh)
                convergence.append(dict(step=hh,finite=finite,error=abs(finite-straight-acceleration_term)))
            hvelocity=horizontal_projection(x,velocity,masses)['horizontal']
            bond_error=0.
            for angle in (-.4,.4):
                yy,_,_,_=bond_rotation(x,rec['bonds'],(2,3),0,angle)
                for i,j,*_ in rec['bonds']:
                    bond_error=max(bond_error,abs(np.linalg.norm(yy[i]-yy[j])-np.linalg.norm(x[i]-x[j])))
            item['torsion']=dict(axis_atom_indices=[2,3],moving_atom_indices=moving,
                velocity=velocity,horizontal_velocity=hvelocity,acceleration=acceleration,
                derivative=first_derivative(x,velocity),horizontal_derivative=first_derivative(x,hvelocity),
                coordinate_hessian_term=straight,acceleration_term=acceleration_term,
                total_second_derivative=straight+acceleration_term,convergence=convergence,
                horizontal_squared_speed_fraction=metric(hvelocity,hvelocity,masses)/metric(velocity,velocity,masses),
                minimum_distance_event_margin=margin,max_bond_length_error=bond_error)
        output.append(item)
    out=ROOT/'results/geometry';out.mkdir(parents=True,exist_ok=True)
    payload=dict(coordinates_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
       metric='heavy-atom standard atomic weights; no scale quotient',r=2.6,s=3.2,tau=.3,
       source=0,probe=0,records=output)
    (out/'intrinsic_sensitivity.json').write_text(json.dumps(payload,indent=2,
       default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a)),encoding='utf-8')
    print(json.dumps([{k:r[k] for k in ['name','intrinsic_gradient_norm','projection_difference','mirror_gradient_error','hessian_error']} for r in output],indent=2))
    print(json.dumps(output[2]['torsion'],default=lambda a:a.tolist() if isinstance(a,np.ndarray) else float(a),indent=2))


if __name__=='__main__':run()
