"""Execute the frozen finite experiment; all results derive from stored geometry."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
from scipy.linalg import eigh

from .core import (atom_energy, cochains, local_signed_map, persistent_operator,
                   response, schur_operator, signed_transport)
from .exact_topology import exact_persistent, octahedral_filling, tetrahedron_subcomplexes


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cos_angle(a,b):
    norm=np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.sum(a*b)/norm) if norm>1e-25 else None


def rayleigh(l,x):
    norm=np.sum(x*x)
    return float(np.sum(x*(l@x))/norm) if norm>1e-25 else 0.


def write_csv(path, rows):
    if not rows: return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def paired_interval(values,seed=90210):
    x=np.asarray(values); rng=np.random.default_rng(seed)
    boot=np.median(rng.choice(x,size=(5000,len(x)),replace=True),axis=1)
    return {"median":float(np.median(x)),"bootstrap_median_95":[float(v) for v in np.quantile(boot,[.025,.975])],"n_molecules":len(x)}


def main(data:Path,out:Path):
    if out.exists(): raise FileExistsError("Choose a new result directory to preserve previous runs")
    out.mkdir(parents=True)
    started=time.time(); before=sha(data)
    panel=json.loads(data.read_text(encoding="utf-8"))
    rng=np.random.default_rng(314159)
    rows=[]; transform=[]; molecules=[]; maps=[]; feature_records=[]
    for record in panel["records"]:
        feats=np.array(record["atom_features"],float); centers=[i for i,_ in record["stereocenters"]]
        primary_raw=[]; primary_pchr=[]; primary_same=[]
        for conf in record["conformers"]:
            x=np.array(conf["coordinates"],float)
            for r in (2.2,2.6,3.0):
                op=persistent_operator(x,r,r+.6)
                chi,b=cochains(x,op.small[2],feats)
                mirror=x*np.array([-1,1,1]); op_m=persistent_operator(mirror,r,r+.6)
                cm,bm=cochains(mirror,op_m.small[2],feats)
                raw=b.T@chi/max(len(op.small[2]),1)
                same=persistent_operator(x,r,r)
                same_out=response(same,chi,b,.3)
                schur_err=float(np.linalg.norm(op.matrix-schur_operator(op)))
                for tau in (.1,.3,1.):
                    value=response(op,chi,b,tau); mirrored=response(op_m,cm,bm,tau)
                    signed=value["signed"]/max(len(op.small[2]),1)
                    errors={"source_parity_error":float(np.max(np.abs(cm+chi),initial=0)),
                            "probe_even_error":float(np.max(np.abs(bm-b),initial=0)),
                            "response_parity_error":float(np.max(np.abs(value['u']+mirrored['u']),initial=0)),
                            "signed_parity_error":float(np.max(np.abs(value['signed']+mirrored['signed']),initial=0)),
                            "energy_even_error":float(np.max(np.abs(value['energy']-mirrored['energy']),initial=0)),
                            "spectrum_even_error":float(np.max(np.abs(value['eigenvalues']-mirrored['eigenvalues']),initial=0))}
                    if max(errors.values())>1e-8: raise AssertionError((record['name'],errors))
                    row={"name":record["name"],"family":record["family"],"conformer_id":conf["conformer_id"],
                         "r":r,"s":r+.6,"tau":tau,"n_atoms":len(x),
                         "n_edges":len(op.small[1]),"n_triangles":len(op.small[2]),"n_tetrahedra_later":len(op.large[3]),
                         "nullity":int(np.sum(value['eigenvalues']<1e-8)),
                         "signed_norm":float(np.linalg.norm(signed)),"raw_signed_norm":float(np.linalg.norm(raw)),
                         "energy":float(np.sum(value['energy'])),"source_norm":float(np.linalg.norm(chi)),
                         "response_source_cosine":cos_angle(value['u'],chi),
                         "raw_rayleigh":rayleigh(op.matrix,chi),"response_rayleigh":rayleigh(op.matrix,value['u']),
                         "solve_residual":value['residual'],"residual_bound":value['residual_bound'],
                         "schur_difference_norm":schur_err,**errors}
                    if row['response_rayleigh']>row['raw_rayleigh']+1e-10: raise AssertionError("Rayleigh inequality")
                    rows.append(row)
                    feature_records.append({"name":record['name'],"conformer_id":conf['conformer_id'],"r":r,"tau":tau,
                                            "signed":signed.ravel().tolist(),"raw":raw.ravel().tolist()})
                    if r==2.6 and tau==.3:
                        primary_raw.append(raw.ravel());primary_pchr.append(signed.ravel())
                        primary_same.append((same_out['signed']/max(len(op.small[2]),1)).ravel())
                        atom=atom_energy(value['u'],op.small[2],len(x))
                        raw_atom=atom_energy(chi,op.small[2],len(x))
                        enrichment=float(np.mean(atom[centers])/np.mean(atom)) if np.mean(atom)>0 else 0
                        raw_enrichment=float(np.mean(raw_atom[centers])/np.mean(raw_atom)) if np.mean(raw_atom)>0 else 0
                        maps.append({"name":record['name'],"conformer_id":conf['conformer_id'],
                                     "enrichment":enrichment,"raw_enrichment":raw_enrichment,
                                     "atom_energy":atom.tolist(),"source_atom_energy":raw_atom.tolist(),
                                     "local_signed":local_signed_map(value['u'],b,op.small[2],len(x)).tolist(),
                                     "triangles":[list(t) for t in op.small[2]],"response":value['u'].tolist(),
                                     "source":chi.tolist(),"probe":b.tolist()})
                if conf['conformer_id']==record['conformers'][0]['conformer_id'] and r==2.6:
                    p=rng.permutation(len(x)); opp=persistent_operator(x[p],r,r+.6)
                    cp,bp=cochains(x[p],opp.small[2],feats[p]); vp=response(opp,cp,bp,.3)
                    original=response(op,chi,b,.3);t=signed_transport(op.small[2],opp.small[2],p)
                    q,_=np.linalg.qr(rng.normal(size=(3,3)))
                    if np.linalg.det(q)<0:q[:,0]*=-1
                    rotated=x@q.T+[11,-4,6];opr=persistent_operator(rotated,r,r+.6)
                    cr,br=cochains(rotated,opr.small[2],feats);vr=response(opr,cr,br,.3)
                    planar=x.copy();planar[:,2]=0
                    cplan,_=cochains(planar,op.small[2],feats)
                    tr={"name":record['name'],"permutation_scalar_error":float(np.max(np.abs(vp['signed']-original['signed']),initial=0)),
                        "permutation_operator_error":float(np.linalg.norm(opp.matrix-t@op.matrix@t.T)),
                        "rotation_translation_error":float(np.max(np.abs(vr['signed']-original['signed']),initial=0)),
                        "planar_source_max":float(np.max(np.abs(cplan),initial=0))}
                    if max(v for k,v in tr.items() if k!='name')>1e-8:raise AssertionError(tr)
                    transform.append(tr)
        def variability(values):
            a=np.asarray(values); center=np.median(a,axis=0)
            return float(np.median(np.linalg.norm(a-center,axis=1))/(np.median(np.linalg.norm(a,axis=1))+1e-12))
        subset=[m for m in maps if m['name']==record['name']]
        molecules.append({"name":record['name'],"family":record['family'],"cid":record['pubchem_cid'],
                          "n_atoms":len(feats),"n_stereocenters":len(centers),"n_conformers":len(primary_raw),
                          "raw_conformer_dispersion":variability(primary_raw),
                          "pchr_conformer_dispersion":variability(primary_pchr),
                          "same_conformer_dispersion":variability(primary_same),
                          "median_stereocenter_enrichment":float(np.median([m['enrichment'] for m in subset])),
                          "median_source_enrichment":float(np.median([m['raw_enrichment'] for m in subset]))})
        print(record['name'],len(rows),flush=True)
    k,l=octahedral_filling(); exact,beta,z,a,d=exact_persistent(k,l)
    summary={"n_molecules":len(molecules),"n_conformers":sum(m['n_conformers'] for m in molecules),
             "n_paired_configurations":len(rows),"n_nonzero_signed":sum(row['signed_norm']>1e-10 for row in rows),
             "max_errors":{key:max(row[key] for row in rows) for key in list(rows[0]) if key.endswith('error') or key in ('solve_residual','residual_bound','schur_difference_norm')},
             "min_response_source_cosine":min(row['response_source_cosine'] for row in rows if row['response_source_cosine'] is not None),
             "median_response_source_cosine":float(np.median([row['response_source_cosine'] for row in rows if row['response_source_cosine'] is not None])),
             "raw_dispersion":paired_interval([m['raw_conformer_dispersion'] for m in molecules]),
             "pchr_dispersion":paired_interval([m['pchr_conformer_dispersion'] for m in molecules]),
             "dispersion_difference":paired_interval([m['pchr_conformer_dispersion']-m['raw_conformer_dispersion'] for m in molecules]),
             "enrichment_difference":paired_interval([m['median_stereocenter_enrichment']-m['median_source_enrichment'] for m in molecules]),
             "n_molecules_dispersion_improved":sum(m['pchr_conformer_dispersion']<m['raw_conformer_dispersion'] for m in molecules),
             "n_molecules_enrichment_improved":sum(m['median_stereocenter_enrichment']>m['median_source_enrichment'] for m in molecules),
             "octahedron":{"small_triangles":len(k[2]),"later_tetrahedra":len(l[3]),"outside_rank":d.rank(),
                           "filling_space_dimension":z.cols,"persistent_beta2":beta,"smallest_eigenvalue_exact":"1"}}
    write_csv(out/'all_results.csv',rows);write_csv(out/'molecule_summary.csv',molecules);write_csv(out/'transform_checks.csv',transform)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+"\n",encoding='utf-8')
    (out/'atom_maps.json').write_text(json.dumps(maps,indent=2)+"\n",encoding='utf-8')
    (out/'features.json').write_text(json.dumps(feature_records,indent=2)+"\n",encoding='utf-8')
    try:rev=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    except Exception:rev=None
    manifest={"status":"completed","assertion_tested":"Frozen molecular paired symmetry and response experiment",
              "git_revision":rev,"input_sha256_before":before,"input_sha256_after":sha(data),
              "python":platform.python_version(),"platform":platform.platform(),
              "software":{p:importlib.metadata.version(p) for p in ('numpy','scipy','rdkit','matplotlib','sympy','gudhi')},
              "elapsed_seconds":time.time()-started,"completed_utc":datetime.now(timezone.utc).isoformat(),
              "outputs":{p.name:sha(p) for p in out.iterdir() if p.is_file()},"seed":314159,"generator":"PCG64"}
    assert manifest['input_sha256_before']==manifest['input_sha256_after']
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+"\n",encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=Path('data/conformers.json'))
    p.add_argument('--output',type=Path,default=Path('results/main'));a=p.parse_args();main(a.data,a.output)
