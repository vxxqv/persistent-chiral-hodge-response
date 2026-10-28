"""Check figure source values against the frozen pilot and algebraic identities."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap=argparse.ArgumentParser();ap.add_argument('figures',type=Path);args=ap.parse_args()
    root=Path(__file__).resolve().parents[1];folder=args.figures/'source_data'
    def data(stem):return json.loads(next(folder.glob(stem+'*.json')).read_text(encoding='utf-8'))
    first=data('Figure_01');np.testing.assert_allclose(first['signed'],-np.asarray(first['mirror_signed']),atol=1e-13)
    np.testing.assert_allclose(first['eigenvalues'],first['mirror_eigenvalues'],atol=1e-12)
    assert first['distance_error']==0
    third=data('Figure_03');d=np.asarray(third['outside']);z=np.asarray(third['null_vector'])
    np.testing.assert_allclose(d@z,0,atol=1e-12);assert np.linalg.matrix_rank(d)==7
    np.testing.assert_allclose(third['persistent_eigenvalues'],[1,2,2,2,4,4,4,6],atol=1e-12)
    fourth=data('Figure_04');a=np.asarray(fourth['source_spectral_weight']);u=np.asarray(fourth['response_spectral_weight'])
    np.testing.assert_allclose([a.sum(),u.sum()],[1,1],atol=1e-13)
    predicted=a/(np.asarray(fourth['grouped_eigenvalues'])+.3)**2;predicted/=predicted.sum()
    np.testing.assert_allclose(u,predicted,atol=1e-12)
    assert np.all(np.diff(fourth['energy'])<0)
    fifth=data('Figure_05');assert len(fifth['atoms'])==4
    for r in fifth['atoms']:np.testing.assert_allclose(np.sum(r['atom_fraction']),1,atol=1e-13)
    sixth=data('Figure_06');table=pd.read_csv(root/'results/main/molecule_summary.csv')
    assert [r['name'] for r in sixth['molecules']]==list(table.name)
    for col in ['raw_conformer_dispersion','same_conformer_dispersion','pchr_conformer_dispersion']:
        np.testing.assert_array_equal([r[col] for r in sixth['molecules']],table[col])
    assert int((table.pchr_conformer_dispersion<table.raw_conformer_dispersion).sum())==8
    for conf in sixth['ensemble']:
        assert np.isfinite(conf).all()
    eighth=data('Figure_08');np.testing.assert_allclose(eighth['total_derivative'],
        np.asarray(eighth['interior_derivative'])+eighth['boundary_derivative'],atol=1e-15)
    atlas=data('Figure_S01');assert len(atlas['molecules'])==24
    for r in atlas['molecules']:
        np.testing.assert_allclose(sum(r['atom_fractions']),1,atol=1e-13)
        assert len(r['coordinates'])==len(r['elements'])==len(r['atom_fractions'])
    supplement=data('Figure_S02');assert supplement['n_finite']==1176 and supplement['n_total']==1728
    np.testing.assert_allclose(np.median(supplement['response_source_cosine']),.9403973917733,atol=1e-13)
    geometry=data('Figure_09');methionine=next(r for r in geometry['geometry']['records'] if r['name']=='L-methionine')['torsion']
    assert methionine['coordinate_hessian_term']>0 and methionine['total_second_derivative']<0
    assert methionine['convergence'][-1]['error']<1e-10
    np.testing.assert_allclose(methionine['derivative'],methionine['horizontal_derivative'],atol=1e-16)
    assert methionine['max_bond_length_error']<1e-13
    tenth=data('Figure_10');scores=pd.DataFrame(tenth['paired_scores']).set_index('mol_id')
    external=pd.read_csv(root/'results/external_full/separation.csv')
    expected=external[external.radius=='r2.6'].pivot(index='mol_id',columns='representation',values='es_auc')
    assert len(scores)==512
    for name in expected.columns:np.testing.assert_allclose(scores.loc[expected.index,name],expected[name],atol=1e-15)
    for item in tenth['mean_intervals']:np.testing.assert_allclose(item['mean'],expected[item['representation']].mean(),atol=1e-15)
    eleventh=data('Figure_11');parity=pd.read_csv(root/'results/external_full/parity_decomposition.csv')
    assert parity.mol_id.nunique()==337
    assert len(parity[(parity.radius=='r2.6')&(parity.representation=='persistent')])==465
    np.testing.assert_allclose(parity.total_squared,parity.even_squared+parity.odd_squared,atol=1e-14)
    np.testing.assert_allclose(parity.balanced_squared,parity.odd_squared,atol=1e-14)
    for row in eleventh['parent_mean_energies']:
        group=parity[(parity.radius=='r2.6')&(parity.representation==row['representation'])].groupby('mol_id')[['odd_squared','even_squared']].mean()
        for col in group.columns:np.testing.assert_allclose(row[col],group[col].mean(),atol=1e-15)
    twelfth=data('Figure_12');fixture=twelfth['fixture'];g=np.array(fixture['gram']);one=np.ones(len(g))
    np.testing.assert_allclose(fixture['divergence_coefficient'],1/(one@np.linalg.solve(g,one)),atol=1e-15)
    assert np.linalg.eigvalsh(g).min()>0
    thirteenth=data('Figure_13');calculus=json.loads((root/'results/external_full/ensemble_geometry.json').read_text())
    assert thirteenth==calculus and max(r['absolute_error'] for r in calculus['finite_difference_checks'])<2e-10
    fourteenth=data('Figure_14');bio=json.loads((root/'results/biology_cliffs/association_summary.json').read_text())
    assert fourteenth['summary']==bio
    assert (bio['parents'],bio['targets'],bio['analysed_pair_observations'],bio['retained_conformers'])==(103,65,143,1648)
    parent=pd.read_csv(root/'results/biology_cliffs/parent_level_activity.csv').set_index('parent_key')
    assert len(fourteenth['parent_data'])==len(parent)==103
    pairs=pd.read_csv(root/'results/biology_cliffs/descriptor_activity_pairs.csv')
    case=pairs[pairs.parent_key==fourteenth['case_parent']['parent_key']].groupby(['target_chembl_id','target_name']).delta_pki.median()
    for row in fourteenth['case_target_differences']:
        np.testing.assert_allclose(row['delta_pki'],case.loc[(row['target_chembl_id'],row['target_name'])],atol=1e-15)
    fifteenth=data('Figure_15');metric=fifteenth['metric'];c0=(1-2*np.exp(-1.5))/2
    assert (metric['shape_dimension'],metric['numerical_rank'],metric['numerical_nullity'])==(87,18,69)
    assert metric==json.loads((root/'results/external_full/kernel_shape_metric.json').read_text())
    assert min(fifteenth['radial_eigenvalues'])>=c0-1e-14
    for fractions in fifteenth['atom_fractions']:np.testing.assert_allclose(sum(fractions),1,atol=1e-15)
    print('All 17 figure source sets passed: original identities, external paired rows, parent-weighted parity, exact divergence, analytic jets, same-assay cases and shape metric.')


if __name__=='__main__':main()
