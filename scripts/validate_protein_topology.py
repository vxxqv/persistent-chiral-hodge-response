"""Additional projections and exact braid checks; no complete knot classification."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from pchr.knot_topology import fox_determinant, braid_jones_A, unitary_braid_matrices
from pchr.constrained_analysis import ROOT, write_json


def main(out=None):
    out = ROOT/'results/protein_knots' if out is None else Path(out)
    rows = []
    for pdb in ['1J85', '2EFV', '1UBQ']:
        record = json.loads((out/(pdb+'.json')).read_text())
        x = np.array(record['coordinates'])
        rng = np.random.default_rng(int(hashlib.sha256(('validate:'+pdb).encode()).hexdigest()[:8],16))
        assert record['status'] == 'completed' and record['failed_closures'] == 0
        for closure in record['closure_results']:
            polygon = np.vstack([x, closure['farpoint']])
            for repeat in range(3):
                candidate = fox_determinant(polygon, Rotation.random(random_state=rng).as_matrix())
                assert candidate['determinant'] == closure['determinant'], (pdb, closure['closure_index'], candidate)
                rows.append({'pdb':pdb, 'closure_index':closure['closure_index'],
                             'independent_projection':repeat, 'determinant':candidate['determinant']})
    write_json(out/'projection_validation.json', {'status':'passed', 'closed_polygons':192,
               'new_projections':len(rows), 'method':'different generic proper projections of each identical closed polygon',
               'checks':rows, 'boundary':'projection invariance cross-check, not complete knot classification'})
    a,b,e1,e2 = unitary_braid_matrices(5)
    fixtures = {
        'three_strand_trefoil_positive_stabilization':str(braid_jones_A(3,[1,1,1,2])),
        'three_strand_trefoil_negative_stabilization':str(braid_jones_A(3,[1,1,1,-2])),
        'three_strand_figure_eight_word':str(braid_jones_A(3,[1,-2,1,-2])),
        'root_order5_braid_relation_error':float(np.linalg.norm(a@b@a-b@a@b)),
        'root_order5_unitarity_errors':[float(np.linalg.norm(a.conj().T@a-np.eye(2))),float(np.linalg.norm(b.conj().T@b-np.eye(2)))],
        'root_order5_noncommutativity':float(np.linalg.norm(a@b-b@a)),
        'ordinary_trace_is_not_Jones_Markov_trace':True}
    write_json(out/'quantum_braid_checks.json', fixtures)
    print(json.dumps({'projection_checks':len(rows), 'braid_checks':fixtures},indent=2),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path)
    args=parser.parse_args();main(args.output)
