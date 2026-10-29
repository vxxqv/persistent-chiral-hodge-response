"""Cross-check canonical flow against fixed-step RK4 forced geodesics."""
from pathlib import Path
import argparse,json,numpy as np
from pchr.constrained_analysis import ROOT,write_json
from pchr.molecular_fields import case_model,state_values
from pchr.torsion_geometry import christoffel


def main(out=None,reference_dir=None):
    out=ROOT/'results/molecular_fields' if out is None else Path(out)
    reference_dir=ROOT/'results/molecular_fields' if reference_dir is None else Path(reference_dir)
    out.mkdir(parents=True,exist_ok=True)
    pairs=json.loads((ROOT/'results/biology_cliffs/same_assay_pairs.json').read_text());case=next(p for p in pairs if p['parent_key']=='81dbd5f36b7abada');records={}
    for side in ('a','b'):
        model=case_model(case['stereo_'+side]);base=state_values(model,[0.,0.]);g=base[3]
        initial=np.r_[0.,0.,np.sqrt(1/g[0,0]),0.];reference=np.load(reference_dir/f'vector_fields_{side}.npz')['state'][-1,:4];rows=[]
        def rhs(state):
            data=state_values(model,state[:2]);g,dg,gradient=data[3],data[5],data[7];gamma=christoffel(g,dg)
            acceleration=-np.einsum('aij,i,j->a',gamma,state[2:],state[2:])-np.linalg.solve(g,gradient)
            return np.r_[state[2:],acceleration]
        for step in (.2,.1,.05):
            state=initial.copy();energy=[]
            for _ in range(round(20/step)):
                k1=rhs(state);k2=rhs(state+step*k1/2);k3=rhs(state+step*k2/2);k4=rhs(state+step*k3);state+=step*(k1+2*k2+2*k3+k4)/6
                data=state_values(model,state[:2]);energy.append(data[6]+.5*state[2:]@data[3]@state[2:])
            data=state_values(model,state[:2]);canonical=np.r_[state[:2],data[3]@state[2:]]
            rows.append({'step':step,'endpoint_canonical_error':float(np.linalg.norm(canonical-reference)), 'maximum_energy_error':float(np.max(abs(np.array(energy)-.5)))})
        if not(rows[1]['endpoint_canonical_error']<rows[0]['endpoint_canonical_error'] and rows[2]['endpoint_canonical_error']<rows[1]['endpoint_canonical_error']):raise AssertionError('forced-geodesic cross-check did not converge')
        records[side]=rows
    write_json(out/'forced_geodesic_validation.json',{'status':'passed','method':'fixed-step classical RK4 in q and velocity, independently organized from canonical DOP853 q and momentum', 'validation_amendment':'additional numerical diagnostic after the primary trajectories; no physical parameters or reported endpoints changed','cases':records})
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--reference',type=Path);args=parser.parse_args();main(args.output,args.reference)
