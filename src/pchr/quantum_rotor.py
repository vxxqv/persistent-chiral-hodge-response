"""Exact Fourier Galerkin matrices for explicit bilinear coefficient models.

This is a reduced nuclear-coordinate model with an empirical MMFF potential.
It is not an electronic-structure or receptor-binding calculation.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,json,hashlib
import numpy as np
from scipy.linalg import eigh
from scipy import constants
from .constrained_analysis import ROOT,case_base,write_json
from .torsion_geometry import chart,quotient_metric
from .rotational_kinematics import mechanical_connection,square_holonomy

KAPPA=constants.hbar**2/(2*constants.atomic_mass*1e-20)*constants.Avogadro/4184
PROTOCOL=ROOT/'data/protocols/quantum_kinematics_v1.json'


def modes(cutoff):
    if not isinstance(cutoff,int) or cutoff<0:raise ValueError('nonnegative integer Fourier cutoff required')
    return np.array([(i,j) for i in range(-cutoff,cutoff+1) for j in range(-cutoff,cutoff+1)])


def coefficient_matrix(values,frequencies):
    """Integral <exp(i p.q), c_N exp(i q.q)>/(2pi)^2, exactly.

    c_N is the periodic tensor-product linear interpolant of the vertex array.
    Its kth Fourier coefficient is DFT[k mod N] prod sinc(k_j/N)^2.
    """
    values=np.asarray(values,float);raw_p=np.asarray(frequencies)
    if values.ndim!=2 or min(values.shape)<2 or raw_p.ndim!=2 or raw_p.shape[1]!=2 or not np.isfinite(values).all() or not np.isfinite(raw_p).all() or not np.equal(raw_p,np.rint(raw_p)).all():raise ValueError('finite two-dimensional samples and integer frequency pairs required')
    p=raw_p.astype(int)
    diff=p[:,None,:]-p[None,:,:];shape=np.array(values.shape)
    dft=np.fft.fft2(values)/values.size
    return dft[diff[:,:,0]%shape[0],diff[:,:,1]%shape[1]]*np.prod(np.sinc(diff/shape)**2,axis=2)


def galerkin(energy,rho,tensor,cutoff,kappa=KAPPA):
    p=modes(cutoff);b=coefficient_matrix(rho,p);h=coefficient_matrix(rho*energy,p)
    for i in range(2):
        for j in range(2):h+=kappa*p[:,None,i]*p[None,:,j]*coefficient_matrix(tensor[:,:,i,j],p)
    h=(h+h.conj().T)/2;b=(b+b.conj().T)/2
    return h,b,p


def coefficient_grid(smiles,n):
    rec,axes,full,x,aa,m,ff,baseline=case_base(smiles);q=2*np.pi*np.arange(n)/n
    energy=np.empty((n,n));rho=np.empty((n,n));tensor=np.empty((n,n,2,2));metric=np.empty_like(tensor)
    for i,a in enumerate(q):
        for j,b in enumerate(q):
            y,v=chart(x,aa,[a,b],False);g=quotient_metric(y,v,m)[0];eigen=np.linalg.eigvalsh(g)
            if eigen[0]<=1e-12*eigen[-1]:raise ValueError('singular mass metric')
            energy[i,j]=ff.CalcEnergy(y.ravel().tolist())-baseline;metric[i,j]=g;rho[i,j]=np.sqrt(np.linalg.det(g));tensor[i,j]=rho[i,j]*np.linalg.inv(g)
    return {'angles':q,'energy':energy,'rho':rho,'tensor':tensor,'metric':metric}


def thermal_density(p,c,values,rho,temperature=298.15):
    beta=1/(.00198720425864083*temperature);relative=np.exp(-beta*(values-values[0]));weights=relative/relative.sum()
    nretain=min(len(weights),int(np.searchsorted(np.cumsum(weights),1-1e-12))+1)
    q=2*np.pi*np.arange(rho.shape[0])/rho.shape[0];points=np.array(np.meshgrid(q,q,indexing='ij')).reshape(2,-1).T;density=np.empty(len(points))
    for first in range(0,len(points),256):
        basis=np.exp(1j*points[first:first+256]@p.T);psi=basis@c[:,:nretain]
        density[first:first+256]=(abs(psi)**2)@weights[:nretain]
    mass=density.reshape(rho.shape)*rho;norm=float(mass.mean());probability=mass/mass.sum()
    return probability,{'ground_energy':float(values[0]),'log_partition':float(-beta*values[0]+np.log(relative.sum())),
       'retained_states_for_density':nretain,'discarded_normalized_spectral_weight':float(weights[nretain:].sum()),
       'sampled_density_mass_integral':norm,'mass_quadrature_error':abs(norm-1),'thermal_mean_energy':float(weights@values)}


def run_one(payload):
    side,smiles,n,folder=payload;out=Path(folder);grid=coefficient_grid(smiles,n);spectra={};allarrays={**grid};last=None
    for cutoff in [3,5,7,9,12,15]:
        h,b,p=galerkin(grid['energy'],grid['rho'],grid['tensor'],cutoff)
        values,c=eigh(h,b,check_finite=True,driver='gvd');norm=np.linalg.norm(h@c-b@c*values,axis=0)
        residual=float(np.max(norm/np.maximum(np.linalg.norm(h@c,axis=0)+np.linalg.norm(b@c*values,axis=0),1e-12)))
        spectra[str(cutoff)]={'dimension':len(p),'low_eigenvalues':values[:12].tolist(),'maximum_relative_residual':residual,'minimum_mass_eigenvalue':float(np.linalg.eigvalsh(b)[0])}
        allarrays[f'eigenvalues_K{cutoff}']=values
        if last is not None and np.max(values[:12]-last[:12])>1e-7:raise AssertionError('fixed-operator nested variational monotonicity failed')
        last=values
    probability,thermal=thermal_density(p,c,values,grid['rho']);e=grid['energy'];classical=grid['rho']*np.exp(-(e-e.min())/(.00198720425864083*298.15));classical/=classical.sum()
    allarrays.update(quantum_probability=probability,classical_probability=classical,modes=p,eigenvectors=c)
    np.savez_compressed(out/f'rotor_{side}_n{n}.npz',**allarrays)
    mirror_idx=(-np.arange(n))%n;eg=grid['energy'][np.ix_(mirror_idx,mirror_idx)];rg=grid['rho'][np.ix_(mirror_idx,mirror_idx)];ag=grid['tensor'][mirror_idx][:,mirror_idx]
    hm,bm,pp=galerkin(eg,rg,ag,3);hh,bb,_=galerkin(grid['energy'],grid['rho'],grid['tensor'],3)
    summary={'side':side,'n':n,'kappa_kcal_mol_amu_Angstrom_squared':KAPPA,'spectra':spectra,'thermal':thermal,
       'quantum_vs_classical_position_total_variation':float(.5*np.abs(probability-classical).sum()),
       'mirror_hamiltonian_conjugation_error':float(np.max(abs(hm-hh.conj()))),'mirror_mass_conjugation_error':float(np.max(abs(bm-bb.conj()))),
       'minimum_rho':float(grid['rho'].min()),'maximum_rho':float(grid['rho'].max()),'minimum_coefficient_tensor_eigenvalue':float(np.linalg.eigvalsh(grid['tensor']).min()),
       'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),'scope':'finite Fourier model of explicit bilinear coefficients; not an electronic or biological prediction'}
    write_json(out/f'rotor_{side}_n{n}.json',summary);print(json.dumps({'phase':'rotor','side':side,'n':n,'ground_energies':{k:v['low_eigenvalues'][0] for k,v in spectra.items()},'thermal':thermal}),flush=True)
    return summary


def kinematics_case(smiles):
    rec,axes,full,x,aa,m,ff,baseline=case_base(smiles);y,v,h=chart(x,aa,[0.,0.]);a,inertia,ell,da=mechanical_connection(y,v,m,h)
    curvature=da[0,1]-da[1,0]-np.cross(a[0],a[1]);loops=[square_holonomy(x,aa,m,e) for e in [.2,.1,.05,.025]]
    for loop in loops:
        loop['curvature_vector_error']=float(np.linalg.norm(np.array(loop['rotation_log'])/loop['epsilon']**2+curvature))
    step=1e-5;checks=[]
    for i in range(2):
        delta=np.eye(2)[i]*step;yp,vp=chart(x,aa,delta,False);ym,vm=chart(x,aa,-delta,False)
        fd=(mechanical_connection(yp,vp,m)[0]-mechanical_connection(ym,vm,m)[0])/(2*step)
        checks.append({'axis':i,'connection_derivative_max_error':float(np.max(abs(fd-da[i])))})
    return {'connection':a.tolist(),'connection_derivative':da.tolist(),'curvature':curvature.tolist(),'loops':loops,'derivative_checks':checks,
            'baseline_coordinates':x.tolist(),'masses':m.tolist(),'axes':aa,'scope':'prescribed internal kinematics and torque-free reconstruction, not a force-field dynamics trajectory'}


def main(out,workers=2):
    out.mkdir(parents=True,exist_ok=True);pairs=json.loads((ROOT/'results/biology_cliffs/same_assay_pairs.json').read_text());case=next(p for p in pairs if p['parent_key']=='81dbd5f36b7abada')
    jobs=[(side,case['stereo_'+side],n,str(out)) for side in ('a','b') for n in (33,65,129) if not (out/f'rotor_{side}_n{n}.json').exists()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(run_one,jobs):pass
    write_json(out/'kinematics.json',{s:kinematics_case(case['stereo_'+s]) for s in ('a','b')})
    write_json(out/'run_receipt.json',{'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),'grid_sizes':[33,65,129],'fourier_cutoffs':[3,5,7,9,12,15],
         'kinematics_loop_sizes':[.2,.1,.05,.025],'kappa':KAPPA,'numerical_models':6,'eigenproblems':36,'boundary':'No claim of continuum or physical accuracy without the reported convergence evidence'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args();main(args.out,args.workers)
