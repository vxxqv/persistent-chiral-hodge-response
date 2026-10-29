"""Check numerical source values for figures 16-24 against archived computations."""
from pathlib import Path
import argparse, json, sys
import numpy as np
from pchr.constraints import rigidity_matrix
from pchr.knot_topology import two_strand_jones
import sympy as sp

ROOT=Path(__file__).resolve().parents[1]


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def same(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:same(a[key],b[key])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for aa,bb in zip(a,b):same(aa,bb)
    elif isinstance(a,(float,int)) and not isinstance(a,bool):np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-13)
    else:assert a==b


def main(folder):
    folder=Path(folder)
    if (folder/'source_data').is_dir():folder=folder/'source_data'
    def source(n):return load(next(folder.glob(f'Figure_{n:02}_*.json')))
    a=source(16);x=np.array(a['coordinates']);v=np.array(a['velocity']);r=rigidity_matrix(x,a['pairs']);stress=np.array(a['stress'])
    np.testing.assert_allclose(r@v.ravel(),0,atol=1e-14);np.testing.assert_allclose(r.T@stress,0,atol=1e-14)
    assert np.linalg.matrix_rank(r)==a['rank']==5
    b=np.array([np.linalg.norm(v[i]-v[j])**2 for i,j in a['pairs']]);assert stress@b==a['obstruction']==1
    a=source(17);expected=[load(p) for p in sorted((ROOT/'results/constrained/cohort').glob('*.json'))]
    same(a['records'],expected);assert len(expected)==206
    for name in ['source','same_scale','persistent']:
        assert sum(z['representations'][name]['ranks']['1e-08']==z['torsion_dimension'] for z in expected)==204
    same(a['local_checks'],load(ROOT/'results/constrained/case_local_checks.json'))
    a=source(18)
    for side in ['a','b']:
        with np.load(ROOT/f'results/constrained/case_{side}_n65.npz') as z:
            np.testing.assert_array_equal(a[side]['energy'],z['energy']);np.testing.assert_array_equal(a[side]['angles'],z['angles'])
    a=source(19)
    for key,value in a['case_diagrams'].items():same(value,load(ROOT/f'results/constrained/case_{key[0]}_n{key[2:]}.json'))
    same(a['measures'],load(ROOT/'results/constrained/measure_summary.json'));same(a['grid_comparison'],load(ROOT/'results/constrained/grid_comparison.json'))
    a=source(20);same(a['kinematics'],load(ROOT/'results/quantum_kinematics/kinematics.json'))
    with np.load(ROOT/'results/molecular_fields/vector_fields_a.npz') as z:
        np.testing.assert_array_equal(a['horizontal_velocity_fields'],z['baseline_horizontal_velocity_fields'])
        np.testing.assert_array_equal(a['curvature_grid'],z['curvature_field'])
    a=source(21);same(a['validation'],load(ROOT/'results/molecular_fields/forced_geodesic_validation.json'))
    for side in ['a','b']:
        with np.load(ROOT/f'results/molecular_fields/vector_fields_{side}.npz') as z:
            for key,value in a['trajectories'][side].items():np.testing.assert_array_equal(value,z[key])
    a=source(22)
    for side in ['a','b']:
        for n in [33,65,129]:same(a[f'{side}_{n}'],load(ROOT/f'results/quantum_kinematics/rotor_{side}_n{n}.json'))
        with np.load(ROOT/f'results/quantum_kinematics/rotor_{side}_n129.npz') as z:
            np.testing.assert_array_equal(a['density_'+side]['quantum'],z['quantum_probability'])
            np.testing.assert_allclose(a['density_'+side]['difference_from_classical'],(z['quantum_probability']-z['classical_probability'])*129**2,atol=0,rtol=0)
            np.testing.assert_allclose([z['quantum_probability'].sum(),z['classical_probability'].sum()],[1,1],atol=1e-14)
    a=source(23)
    for pdb in ['1J85','2EFV','1UBQ']:
        same(a[pdb],load(ROOT/f'results/protein_knots/{pdb}.json'));assert sum(a[pdb]['determinant_counts'].values())==64
    a=source(24);same(a['matrix_checks'],load(ROOT/'results/protein_knots/quantum_braid_checks.json'))
    t=sp.Symbol('t');polynomial=sum(int(c)*t**int(k) for c,k in zip(a['jones_coefficients'],a['exponents']))
    assert sp.expand(polynomial-(t+t**3-t**4))==0
    assert a['mirror_coefficients']==a['jones_coefficients'][::-1]
    print('All 9 extension figure source sets passed: constraint stress, 206 torsional ranks, full grids and diagrams, fields, trajectories, quantum models, 192 closures and exact Jones coefficients.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args();main(args.folder)
