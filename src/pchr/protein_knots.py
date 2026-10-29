"""Frozen closure-dependent diagnostics of experimental protein backbones."""
from pathlib import Path
from collections import Counter
import argparse,json,hashlib
import numpy as np
import sympy as sp
from Bio.PDB import MMCIFParser
from scipy.spatial.transform import Rotation
from .constrained_analysis import ROOT,write_json
from .knot_topology import fox_determinant,two_strand_jones


def ca_trace(path,chain='A'):
    structure=MMCIFParser(QUIET=True).get_structure(Path(path).stem,path);model=next(iter(structure));residues=[];coords=[]
    for residue in model[chain]:
        if residue.id[0]!=' ' or 'CA' not in residue:continue
        atom=residue['CA']
        if atom.is_disordered():atom=sorted(atom.disordered_get_list(),key=lambda a:(-(a.occupancy or 0),a.altloc))[0]
        residues.append({'number':residue.id[1],'insertion':residue.id[2].strip(),'name':residue.resname,'altloc':atom.altloc.strip(),'occupancy':atom.occupancy})
        coords.append(np.array(atom.coord,dtype=float))
    x=np.array(coords);lengths=np.linalg.norm(np.diff(x,axis=0),axis=1)
    if len(x)<4 or any(r['insertion'] for r in residues) or any(b['number']!=a['number']+1 for a,b in zip(residues[:-1],residues[1:])) or np.any((lengths<2.5)|(lengths>4.5)):
        raise ValueError('missing residues, unsupported insertion codes or noncontinuous CA trace')
    return x,residues,lengths


def main(out):
    out.mkdir(parents=True,exist_ok=True);sources=ROOT/'data/protein_sources';protocol=ROOT/'data/protocols/quantum_kinematics_v1.json';summary={}
    for pdb in ['1J85','2EFV','1UBQ']:
        path=sources/(pdb+'.cif')
        try:
            x,residues,lengths=ca_trace(path);center=x.mean(axis=0);radius=np.linalg.norm(x-center,axis=1).max();rng=np.random.default_rng(29102026+int(hashlib.sha256(pdb.encode()).hexdigest()[:8],16));rows=[]
            for k in range(64):
                zz=1-2*(k+.5)/64;phi=k*np.pi*(3-np.sqrt(5));direction=np.array([np.sqrt(1-zz*zz)*np.cos(phi),np.sqrt(1-zz*zz)*np.sin(phi),zz])
                polygon=np.vstack([x,center+100*radius*direction]);result=None
                for attempt in range(4):
                    q=Rotation.random(random_state=rng).as_matrix()
                    try:
                        result=fox_determinant(polygon,q);s=np.diag([-1.,1.,1.]);t=np.diag([1.,1.,-1.]);mirror=fox_determinant(polygon@s,s@q@t)
                        if result['determinant']!=mirror['determinant'] or result['writhe']!=-mirror['writhe']:raise AssertionError('mirror determinant or crossing-sign check failed')
                        rows.append({'closure_index':k,'direction':direction.tolist(),'farpoint':polygon[-1].tolist(),'projection':q.tolist(),'attempt':attempt+1,**result,'mirror':mirror,'status':'completed'});break
                    except ValueError as e:
                        if attempt==3:rows.append({'closure_index':k,'direction':direction.tolist(),'status':'failed','reason':str(e)})
            counts=Counter(str(r['determinant']) for r in rows if r['status']=='completed');meta=json.loads((sources/(pdb+'.json')).read_text())
            result={'pdb':pdb,'chain':'A','status':'completed','n_residues':len(x),'residues':residues,'coordinates':x.tolist(),'consecutive_CA_distance_range': [float(lengths.min()),float(lengths.max())],
                'resolution_angstrom':meta.get('rcsb_entry_info',{}).get('resolution_combined'),'entry_title':meta.get('struct',{}).get('title'),'primary_citation':meta.get('rcsb_primary_citation',{}),
                'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'closure_results':rows,'determinant_counts':dict(counts),
                'failed_closures':sum(r['status']!='completed' for r in rows),'claim_boundary':'open-chain closure diagnostic; determinant is not a complete classifier and cannot detect mirror handedness'}
        except Exception as e:result={'pdb':pdb,'chain':'A','status':'excluded','reason':repr(e)}
        write_json(out/(pdb+'.json'),result);summary[pdb]={k:result[k] for k in ['status','n_residues','determinant_counts','failed_closures'] if k in result};print(pdb,json.dumps(summary[pdb]),flush=True)
    a=sp.Symbol('A');fixtures={str(m):str(two_strand_jones(m)) for m in range(-9,10,2)}
    write_json(out/'exact_jones_fixtures.json',{'variable':'A; t=A^-4','normalization':'unknot1, positive generator A I+A^-1 e, delta=-A^2-A^-2, writhe factor(-A^3)^-w','two_strand_odd_exponents':fixtures,
       'trefoil_positive':str(two_strand_jones(3)),'mirror_antisymmetric':str(sp.expand(two_strand_jones(3)-two_strand_jones(-3))),'exact_arithmetic':'SymPy Laurent polynomial expansion'})
    write_json(out/'summary.json',{'proteins':summary,'protocol_sha256':hashlib.sha256(protocol.read_bytes()).hexdigest(),'directions':64,'sampling':'deterministic Fibonacci directions, descriptive frequencies, no random-sample confidence interval'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);args=parser.parse_args();main(args.out)
