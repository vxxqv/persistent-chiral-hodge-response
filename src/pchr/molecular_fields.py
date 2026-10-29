"""Velocity fields, connection curvature and a declared reduced Hamiltonian flow."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
from scipy.integrate import solve_ivp
from scipy import constants
from rdkit.Chem import AllChem
from .constrained_analysis import ROOT,case_base,write_json
from .torsion_geometry import chart,quotient_metric
from .rotational_kinematics import mechanical_connection,hat

TIME_UNIT_FS=np.sqrt(constants.atomic_mass*1e-20/(4184/constants.Avogadro))*1e15


def case_model(smiles):
    rec,axes,full,x,aa,m,fixedff,baseline=case_base(smiles)
    props=AllChem.MMFFGetMoleculeProperties(full,mmffVariant='MMFF94s')
    freeff=AllChem.MMFFGetMoleculeForceField(full,props,nonBondedThresh=100.)
    np.testing.assert_allclose(freeff.CalcEnergy(),baseline,atol=1e-12)
    return x,aa,m,freeff,baseline


def state_values(model,q):
    x,axes,m,ff,baseline=model;y,v,h=chart(x,axes,q);g,vh,dg=quotient_metric(y,v,m,h)
    energy=float(ff.CalcEnergy(y.ravel().tolist())-baseline)
    force_covector=np.array(ff.CalcGrad(y.ravel().tolist())).reshape(y.shape)
    gradient=np.einsum('inj,nj->i',v,force_covector)
    connection,inertia,ell=mechanical_connection(y,v,m)
    return y,v,h,g,vh,dg,energy,gradient,connection,inertia,ell


def trajectory(model,rtol=1e-10,atol=1e-12):
    y,v,h,g,vh,dg,e,gradient,a,inertia,ell=state_values(model,[0.,0.])
    velocity=np.array([np.sqrt(1/g[0,0]),0.]);momentum=g@velocity;initial=np.r_[0.,0.,momentum,np.eye(3).ravel()]
    maxmoment=0.
    def rhs(t,state):
        nonlocal maxmoment
        y,v,h,g,vh,dg,e,gradient,a,inertia,ell=state_values(model,state[:2]);vel=np.linalg.solve(g,state[2:4])
        pdot=.5*np.einsum('i,aij,j->a',vel,dg,vel)-gradient;omega=-vel@a
        maxmoment=max(maxmoment,float(np.linalg.norm(inertia@omega+vel@ell)))
        r=state[4:].reshape(3,3)
        return np.r_[vel,pdot,(r@hat(omega)).ravel()]
    time=np.linspace(0,20,401);solution=solve_ivp(rhs,(0,20),initial,t_eval=time,method='DOP853',rtol=rtol,atol=atol,max_step=.05)
    if not solution.success:raise RuntimeError(solution.message)
    energies=[];kinetic=[];orth=[]
    for state in solution.y.T:
        y,v,h,g,vh,dg,e,*_=state_values(model,state[:2]);kin=.5*state[2:4]@np.linalg.solve(g,state[2:4]);energies.append(e+kin);kinetic.append(kin)
        r=state[4:].reshape(3,3);orth.append(np.linalg.norm(r.T@r-np.eye(3)))
    energies=np.array(energies)
    return {'dimensionless_time':time,'physical_time_fs':time*TIME_UNIT_FS,'state':solution.y.T,'total_energy':energies,'kinetic_energy':np.array(kinetic)}


def run_case(smiles):
    model=case_model(smiles);x,axes,m,ff,baseline=model
    y,v,h,g,vh,dg,e,gradient,a,inertia,ell=state_values(model,[0.,0.]);checks=[]
    for i in range(2):
        delta=np.eye(2)[i]*1e-5;yp=chart(x,axes,delta,False)[0];ym=chart(x,axes,-delta,False)[0]
        fd=(ff.CalcEnergy(yp.ravel().tolist())-ff.CalcEnergy(ym.ravel().tolist()))/2e-5
        checks.append({'axis':i,'analytic_energy_gradient':float(gradient[i]),'finite_difference':float(fd),'absolute_error':abs(float(fd-gradient[i]))})
    q=2*np.pi*np.arange(25)/25;curvature=np.empty((25,25,3));connections=np.empty((25,25,2,3))
    for i,aa in enumerate(q):
        for j,bb in enumerate(q):
            yy,vv,hh=chart(x,axes,[aa,bb]);connection,_,_,derivative=mechanical_connection(yy,vv,m,hh)
            curvature[i,j]=derivative[0,1]-derivative[1,0]-np.cross(connection[0],connection[1]);connections[i,j]=connection
    # Keep both integrator attempts; tolerance comparison is a numerical check.
    fine=trajectory(model,1e-10,1e-12);coarse=trajectory(model,1e-8,1e-10)
    energy_error=float(np.max(abs(fine['total_energy']-fine['total_energy'][0])))
    if energy_error>1e-6:raise AssertionError('reduced-flow energy conservation check failed')
    rotations=fine['state'][:,4:].reshape(-1,3,3)
    report={'energy_gradient_checks':checks,'time_unit_fs':TIME_UNIT_FS,'initial_total_energy_kcal_mol':float(fine['total_energy'][0]),
        'maximum_energy_absolute_drift_kcal_mol':energy_error,'integration_tolerance_max_state_difference':float(np.max(abs(fine['state']-coarse['state']))),
        'maximum_rotation_orthogonality_error':float(np.linalg.norm(rotations.transpose(0,2,1)@rotations-np.eye(3),axis=(1,2)).max()),
        'curvature_norm_range': [float(np.linalg.norm(curvature,axis=2).min()),float(np.linalg.norm(curvature,axis=2).max())],
        'scope':'reduced constrained Hamiltonian model with empirical potential and zero-angular-momentum reconstruction; not full molecular dynamics or an experimental trajectory'}
    arrays={**fine,'coarse_state':coarse['state'],'curvature_angles':q,'curvature_field':curvature,'connection_field':connections,'baseline_coordinates':x,
        'baseline_raw_velocity_fields':v,'baseline_horizontal_velocity_fields':vh,'masses':m}
    return report,arrays


def main(out):
    out.mkdir(parents=True,exist_ok=True);pairs=json.loads((ROOT/'results/biology_cliffs/same_assay_pairs.json').read_text());case=next(p for p in pairs if p['parent_key']=='81dbd5f36b7abada')
    for side in ('a','b'):
        if (out/f'vector_fields_{side}.json').exists():continue
        report,arrays=run_case(case['stereo_'+side]);write_json(out/f'vector_fields_{side}.json',report);np.savez_compressed(out/f'vector_fields_{side}.npz',**arrays)
        print(side,json.dumps(report),flush=True)
    protocol=ROOT/'data/protocols/molecular_vector_fields_v1.json';write_json(out/'run_receipt.json',{'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'cases':2,'time_unit_fs':TIME_UNIT_FS})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args();main(args.out)
