"""Fresh, complete replay of the added quantum, field and protein computations."""
from pathlib import Path
import argparse, hashlib, importlib.metadata, json, os, platform, subprocess, time
import numpy as np
from pchr.constrained_analysis import ROOT, write_json
from pchr.quantum_rotor import main as quantum_main
from pchr.molecular_fields import main as fields_main
from pchr.protein_knots import main as protein_main
from replay_constrained import compare
from validate_molecular_flow import main as validate_flow
from validate_protein_topology import main as validate_topology


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out,workers=2):
    out=Path(out).resolve()
    if out == (ROOT/'results').resolve() or (ROOT/'results').resolve() in out.parents:
        raise ValueError('Replay must use a separate output directory')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty replay directory so no cached run can be skipped')
    out.mkdir(parents=True,exist_ok=True)
    inputs=sorted(set([*ROOT.glob('src/pchr/*.py'),*ROOT.glob('scripts/*.py'),
                       *ROOT.glob('data/protocols/*.json'),*ROOT.glob('data/protein_sources/*'),
                       *ROOT.glob('results/biology_cliffs/ensembles/*.json'),
                       ROOT/'results/biology_cliffs/same_assay_pairs.json']))
    before={str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()}
    started=time.time()
    git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    git_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)
    write_json(out/'launch_record.json',{'status':'running','input_sha256':before,'git_revision':git_revision,
               'git_dirty_state':git_dirty,'thread_settings':{k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']}})
    quantum_main(out/'quantum_kinematics',workers)
    fields_main(out/'molecular_fields')
    validate_flow(out/'molecular_fields',out/'molecular_fields')
    protein_main(out/'protein_knots')
    validate_topology(out/'protein_knots')
    assert before == {str(p.relative_to(ROOT)):sha(p) for p in inputs if p.is_file()}, 'Input changed during replay'
    records=arrays=0;maximum=0.;vector_phase_error=0.
    for folder in ['quantum_kinematics','molecular_fields','protein_knots']:
        reference=ROOT/'results'/folder;fresh=out/folder
        for path in sorted(reference.glob('*.json')):
            if path.name in ['validation_receipt.json','replay_validation.json']:
                continue
            compare(json.loads(path.read_text()),json.loads((fresh/path.name).read_text()),folder+'/'+path.name)
            records+=1
        for path in sorted(reference.glob('*.npz')):
            with np.load(path) as old,np.load(fresh/path.name) as new:
                assert set(old.files)==set(new.files)
                for key in old.files:
                    a,b=old[key],new[key]
                    if key=='eigenvectors':
                        # Eigenvector phases are arbitrary. Equal coefficient
                        # grids and finite spectra/densities are checked below.
                        overlap=np.sum(a.conj()*b,axis=0)
                        phase=np.where(abs(overlap)>0,overlap/np.maximum(abs(overlap),1e-300),1)
                        b=b*phase.conj()[None,:]
                        error=float(np.max(abs(a-b)))
                        vector_phase_error=max(vector_phase_error,error)
                        np.testing.assert_allclose(a,b,rtol=1e-9,atol=1e-10)
                    else:
                        error=float(np.max(abs(a-b))) if a.size else 0.
                        maximum=max(maximum,error)
                        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-11)
                    arrays+=1
    receipt={'status':'passed','run_kind':'fresh computation from archived molecular coordinates and experimental protein files',
             'git_revision':git_revision,'dirty_state_at_launch':git_dirty,'elapsed_seconds':time.time()-started,
             'platform':platform.platform(),'python':platform.python_version(),
             'software':{k:importlib.metadata.version(k) for k in ['numpy','scipy','rdkit','sympy','gudhi','biopython']},
             'input_sha256_before_and_after':before,'json_records_compared':records,'npz_arrays_compared':arrays,
             'maximum_non_eigenvector_array_absolute_error':maximum,'maximum_phase_aligned_eigenvector_error':vector_phase_error,
             'coverage':{'quantum_models':6,'generalized_eigenproblems':36,'molecular_trajectories':4,
                         'curvature_grids':2,'holonomy_loops':8,'protein_closures':192,'additional_projections':576},
             'boundary':'Numerical replay and finite invariant checks; no continuum convergence or independent peer review claim',
             'outputs':{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}}
    write_json(out/'validation_receipt.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['outputs','input_sha256_before_and_after','dirty_state_at_launch']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args();main(args.out,args.workers)
