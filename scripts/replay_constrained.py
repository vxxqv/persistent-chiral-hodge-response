"""Recompute all frozen constrained-conformer results and compare the archive."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from pchr.constrained_analysis import ROOT,PROTOCOL,main


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a,b,location='root'):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),location
        for k in a:
            if k=='output_sha256':continue # arrays are checked directly, not ZIP container metadata
            compare(a[k],b[k],location+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),location
        for i,(aa,bb) in enumerate(zip(a,b)):compare(aa,bb,location+f'[{i}]')
    elif isinstance(a,(float,int)) and not isinstance(a,bool):
        assert np.isclose(a,b,rtol=1e-11,atol=1e-12), (location,a,b)
    else:assert a==b,(location,a,b)


def run(out,workers):
    frozen=ROOT/'results/constrained'
    paths=[PROTOCOL,ROOT/'results/biology_cliffs/same_assay_pairs.json',*sorted((ROOT/'results/biology_cliffs/ensembles').glob('*.json'))]
    before={str(p.relative_to(ROOT)):digest(p) for p in paths}
    expected=json.loads((frozen/'run_receipt.json').read_text())['protocol_sha256']
    assert digest(PROTOCOL)==expected,'protocol differs from the archived computation'
    main(out,workers)
    assert before=={str(p.relative_to(ROOT)):digest(p) for p in paths},'input changed during replay'
    count=0;maxerror=0.
    for path in sorted(frozen.rglob('*.json')):
        if path.name in ('run_receipt.json','validation_receipt.json'):continue
        other=out/path.relative_to(frozen);compare(json.loads(path.read_text()),json.loads(other.read_text()),str(path.relative_to(frozen)));count+=1
    arrays=0
    for path in sorted(frozen.glob('*.npz')):
        with np.load(path) as a,np.load(out/path.name) as b:
            assert a.files==b.files
            for key in a.files:
                err=float(np.max(abs(a[key]-b[key]))) if a[key].size else 0.;maxerror=max(maxerror,err)
                np.testing.assert_allclose(a[key],b[key],rtol=1e-11,atol=1e-12);arrays+=1
    result={'status':'passed','json_records_compared':count,'npz_arrays_compared':arrays,'maximum_array_absolute_error':maxerror,
            'input_files_checked_before_and_after':len(paths),'input_sha256':before,'protocol_sha256':expected,
            'coverage':'all206 first-conformer derivatives and both isomer33/65 grids, all three weight rules, periodic diagrams and local checks',
            'boundary':'deterministic numerical replay is not a proof of universal correctness or independent peer review'}
    (out/'validation_receipt.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='input_sha256'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args();run(args.out,args.workers)
