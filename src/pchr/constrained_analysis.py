"""Frozen realizable-torsion cohort and periodic reduced-energy case study."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,time
import numpy as np
import gudhi
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.linalg import solve
from .constraints import rigidity_matrix,distance_pairs
from .torsion_geometry import bridge_axes,chart,quotient_metric,response_jets,christoffel
from .core import persistent_operator,cochains,distances,rips
from .ensemble import odd_kernel,odd_pullback_metric

ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'data/protocols/constrained_conformers_v1.json'


def write_json(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def rank(s,relative):
    return int(np.sum(s>max(s[0]*relative if len(s) else 0,1e-12)))


def cohort_one(payload):
    path,parent=payload;data=json.loads(Path(path).read_text());rec=data['conformers'][0]
    x=np.array(rec['coordinates']);af=np.array(rec['atom_features']);m=np.array([Chem.GetPeriodicTable().GetAtomicWeight(int(z)) for z in af[:,0]])
    axes=bridge_axes(len(x),rec['bonds']);k=len(axes)
    out={'input_file':Path(path).name,'input_sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'parent_key':parent,'smiles':data['smiles'],
         'conformer_id':rec['conformer_id'],'n_atoms':len(x),'shape_dimension':3*len(x)-6,'torsion_dimension':k,'axes':axes,'status':'completed'}
    if not k:return {**out,'status':'no_eligible_axes','representations':{}}
    y,v=chart(x,axes,np.zeros(k),False);g,vh=quotient_metric(y,v,m)
    ev,u=np.linalg.eigh(g);out['mass_metric_eigenvalues']=ev.tolist()
    if ev[0]<=1e-12*ev[-1]:return {**out,'status':'degenerate_metric','representations':{}}
    lift=u@np.diag(1/np.sqrt(ev))@u.T
    jets=response_jets(x,af,v);representations={}
    for name,j in jets.items():
        scale=np.linalg.norm(j['value']);scaled=j['jacobian']@lift/max(scale,1e-12)
        s=np.linalg.svd(scaled,compute_uv=False);gram=odd_pullback_metric(j['value']/max(scale,1e-12))
        induced=scaled.T@gram@scaled
        representations[name]={'baseline_norm':float(scale),'normalization_at_floor':bool(scale<=1e-12),'singular_values':s.tolist(),
             'ranks':{str(t):rank(s,t) for t in [1e-6,1e-8,1e-10]},'induced_metric_eigenvalues':np.linalg.eigvalsh(induced).tolist()}
    pairs=distance_pairs(len(x),rec['bonds']);r=rigidity_matrix(x,pairs)
    sv=np.linalg.svd(r,compute_uv=False)
    d=distances(x);a,b=np.triu_indices(len(x),1);margin=float(min(np.min(abs(d[a,b]-2.6)),np.min(abs(d[a,b]-3.2))))
    return {**out,'representations':representations,'constraint_rank':rank(sv,1e-8),'constraint_infinitesimal_shape_nullity':3*len(x)-rank(sv,1e-8)-6,
            'constraint_velocity_max_residual':float(np.max(abs(r@v.reshape(k,-1).T))), 'filtration_margin_angstrom':margin}


def run_cohort(out,workers):
    pairs=json.loads((ROOT/'results/biology_cliffs/same_assay_pairs.json').read_text());parent={}
    for p in pairs:
        for side in ('stereo_a','stereo_b'):parent[p[side]]=p['parent_key']
    jobs=[];results=[];start=time.time()
    for path in sorted((ROOT/'results/biology_cliffs/ensembles').glob('*.json')):
        data=json.loads(path.read_text());target=out/'cohort'/path.name
        if target.exists():results.append(json.loads(target.read_text()));continue
        jobs.append((str(path),parent[data['smiles']]))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(cohort_one,j):j for j in jobs}
        for i,future in enumerate(as_completed(futures),1):
            job=futures[future]
            try:result=future.result()
            except Exception as e:result={'input_file':Path(job[0]).name,'parent_key':job[1],'status':'failed','reason':repr(e)}
            write_json(out/'cohort'/result['input_file'],result);results.append(result)
            if i%16==0:print(json.dumps({'phase':'cohort','new':i,'pending':len(jobs)-i,'seconds':round(time.time()-start,1)}),flush=True)
    results.sort(key=lambda r:r['input_file'])
    valid=[r for r in results if r['status']=='completed'];summary={'n_input':len(results),'n_parents':len({r['parent_key'] for r in results}),
        'status_counts':{s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},
        'torsion_dimensions':[r.get('torsion_dimension') for r in results], 'representations':{},'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    for name in ('source','same_scale','persistent'):
        summary['representations'][name]={'full_rank_isomers_by_tolerance':{str(t):sum(r['representations'][name]['ranks'][str(t)]==r['torsion_dimension'] for r in valid) for t in [1e-6,1e-8,1e-10]},
            'rank_histogram':{str(i):sum(r['representations'][name]['ranks']['1e-08']==i for r in valid) for i in range(19)},
            'normalization_floor_count':sum(r['representations'][name]['normalization_at_floor'] for r in valid)}
    write_json(out/'cohort_summary.json',summary)
    return summary


def case_base(smiles):
    path=ROOT/'results/biology_cliffs/ensembles'/(hashlib.sha256(smiles.encode()).hexdigest()[:20]+'.json')
    rec=json.loads(path.read_text())['conformers'][0];x=np.array(rec['coordinates']);axes=bridge_axes(len(x),rec['bonds'])[:2]
    if len(axes)!=2:raise ValueError('frozen case lacks two axes')
    mol=Chem.MolFromSmiles(smiles)
    if [a.GetAtomicNum() for a in mol.GetAtoms()]!=[a[0] for a in rec['atom_features']]:raise ValueError('atomic order mismatch')
    if {tuple(sorted((b.GetBeginAtomIdx(),b.GetEndAtomIdx()))) for b in mol.GetBonds()}!={tuple(sorted(b[:2])) for b in rec['bonds']}:raise ValueError('bond order mismatch')
    conf=Chem.Conformer(len(x));conf.Set3D(True)
    for i,p in enumerate(x):conf.SetAtomPosition(i,p)
    mol.AddConformer(conf);full=Chem.AddHs(mol,addCoords=True)
    props=AllChem.MMFFGetMoleculeProperties(full,mmffVariant='MMFF94s')
    ff=AllChem.MMFFGetMoleculeForceField(full,props,nonBondedThresh=100.)
    for i in range(len(x)):ff.AddFixedPoint(i)
    ff.Initialize();status=ff.Minimize(maxIts=2000,forceTol=1e-6,energyTol=1e-8)
    if status!=0:raise ValueError('hydrogen-only minimization did not converge')
    allx=full.GetConformer().GetPositions();np.testing.assert_allclose(allx[:len(x)],x,atol=1e-12)
    allaxes=[]
    for ax in axes:
        indices=set(ax['moving'])
        indices|={a.GetIdx() for a in full.GetAtoms() if a.GetAtomicNum()==1 and next(iter(a.GetNeighbors())).GetIdx() in indices}
        allaxes.append({**ax,'moving':sorted(indices)})
    masses=np.array([a.GetMass() for a in full.GetAtoms()]);baseline=float(ff.CalcEnergy())
    return rec,axes,full,allx,allaxes,masses,ff,baseline


def value_response(x,af):
    op=persistent_operator(x,2.6,3.2);c,b=cochains(x,op.small[2],af);n=max(len(op.small[2]),1)
    u=solve(op.matrix+.3*np.eye(len(op.matrix)),c,assume_a='pos') if len(c) else c
    return (b.T@u/n).ravel(),(b.T@c/n).ravel(),len(op.small[2])


def persistence(energy):
    complex=gudhi.PeriodicCubicalComplex(vertices=energy,periodic_dimensions=[True,True])
    complex.persistence(homology_coeff_field=2,min_persistence=0)
    if complex.betti_numbers()!=[1,2,1]:raise AssertionError('terminal torus homology failed')
    return {'betti_terminal':complex.betti_numbers(),'intervals':{str(d):[[float(b),None if np.isinf(e) else float(e)] for b,e in complex.persistence_intervals_in_dimension(d)] for d in range(3)}}


def grid_case(payload):
    side,smiles,n,outdir=payload;out=Path(outdir);rec,axes,full,x,aa,m,ff,baseline=case_base(smiles)
    energy=np.empty((n,n));volume=np.empty((n,n));response=np.empty((n,n,18));source=np.empty_like(response);triangles=np.empty((n,n),int)
    angles=2*np.pi*np.arange(n)/n;min_eigen=np.inf;max_preservation=0.;nh=len(rec['coordinates']);af=np.array(rec['atom_features'])
    for i,a in enumerate(angles):
        for j,b in enumerate(angles):
            y,v=chart(x,aa,[a,b],False);g,vh=quotient_metric(y,v,m)
            eigen=np.linalg.eigvalsh(g);min_eigen=min(min_eigen,float(eigen[0]))
            if eigen[0]<=1e-12*eigen[-1]:raise ValueError('singular metric on frozen family')
            energy[i,j]=ff.CalcEnergy(y.ravel().tolist())-baseline;volume[i,j]=np.sqrt(np.linalg.det(g))
            response[i,j],source[i,j],triangles[i,j]=value_response(y[:nh],af)
    target=out/f'case_{side}_n{n}.npz';np.savez_compressed(target,angles=angles,energy=energy,volume=volume,persistent=response,source=source,triangles=triangles,
        baseline_coordinates=x,heavy_coordinates=np.array(rec['coordinates']),masses=m,atom_features=af)
    summary={'side':side,'smiles':smiles,'n':n,'axes':axes,'all_atom_axes':aa,'elements':[a.GetSymbol() for a in full.GetAtoms()],
        'bonds':[[b.GetBeginAtomIdx(),b.GetEndAtomIdx(),b.GetBondTypeAsDouble()] for b in full.GetBonds()],
        'baseline_energy_kcal_mol':baseline,'energy_range':[float(energy.min()),float(energy.max())],
        'metric_volume_range':[float(volume.min()),float(volume.max())],'minimum_metric_eigenvalue':min_eigen,
        'persistence':persistence(energy),'output_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    write_json(out/f'case_{side}_n{n}.json',summary)
    print(json.dumps({'phase':'grid','side':side,'n':n,'energy_range':summary['energy_range']}),flush=True)
    return summary


def case_local_checks(smiles):
    rec,axes,full,x,aa,m,ff,baseline=case_base(smiles);nh=len(rec['coordinates']);af=np.array(rec['atom_features'])
    y,v,h=chart(x,aa,[0.,0.]);g,vh,dg=quotient_metric(y,v,m,h);gamma=christoffel(g,dg)
    jets=response_jets(y[:nh],af,v[:,:nh],h[:,:,:nh])['persistent'];scale=max(np.linalg.norm(jets['value']),1e-12)
    intrinsic=jets['hessian']/scale-np.einsum('ak,kij->aij',jets['jacobian']/scale,gamma)
    d=distances(y[:nh]);i,j=np.triu_indices(nh,1);margin=float(min(np.min(abs(d[i,j]-2.6)),np.min(abs(d[i,j]-3.2))))
    speed=max(np.linalg.norm(v,axis=2).max(),1e-12);step=min(1e-4,margin/(40*speed));checks=[]
    for a in range(2):
        q=np.eye(2)[a]*step;yp,vp=chart(x,aa,q,False);ym,vm=chart(x,aa,-q,False)
        if rips(yp[:nh],2.6)!=rips(y[:nh],2.6) or rips(yp[:nh],3.2)!=rips(y[:nh],3.2) or rips(ym[:nh],2.6)!=rips(y[:nh],2.6) or rips(ym[:nh],3.2)!=rips(y[:nh],3.2):raise AssertionError('local derivative check crossed filtration')
        jp=response_jets(yp[:nh],af,vp[:,:nh])['persistent'];jm=response_jets(ym[:nh],af,vm[:,:nh])['persistent']
        checks.append({'axis':a,'step':step,'relative_jacobian_error':float(np.linalg.norm((jp['value']-jm['value'])/(2*step)-jets['jacobian'][:,a])/max(np.linalg.norm(jets['jacobian'][:,a]),1e-12)),
            'scaled_hessian_error':float(np.linalg.norm((jp['jacobian']-jm['jacobian'])/(2*step)-jets['hessian'][:,:,a])/scale),
            'metric_derivative_error':float(np.max(abs((quotient_metric(yp,vp,m)[0]-quotient_metric(ym,vm,m)[0])/(2*step)-dg[a])))})
    reflection=np.diag([-1.,1.,1.]);mirror=[]
    for q in [[0.,0.],[.21,-.37],[1.8,2.3]]:
        yy,vv=chart(x,aa,q,False);yr,vr=chart(x@reflection,aa,-np.array(q),False)
        p,_,_=value_response(yy[:nh],af);pr,_,_=value_response(yr[:nh],af)
        # MMFF energy evaluated afresh on a reflected molecule as an independent control.
        mirrored=Chem.Mol(full);conf=mirrored.GetConformer()
        for a,point in enumerate(yr):conf.SetAtomPosition(a,point)
        mp=AllChem.MMFFGetMoleculeProperties(mirrored,mmffVariant='MMFF94s');mf=AllChem.MMFFGetMoleculeForceField(mirrored,mp,nonBondedThresh=100.)
        mirror.append({'angles':q,'coordinate_error':float(np.max(abs(yr-yy@reflection))), 'energy_error_kcal_mol':abs(float(mf.CalcEnergy())-float(ff.CalcEnergy(yy.ravel().tolist()))),
            'metric_error':float(np.max(abs(quotient_metric(yr,vr,m)[0]-quotient_metric(yy,vv,m)[0]))),'response_odd_error':float(np.linalg.norm(p+pr))})
    return {'metric':g.tolist(),'metric_derivative':dg.tolist(),'christoffel':gamma.tolist(),'normalized_response':(jets['value']/scale).tolist(),
        'normalized_gradient':(jets['jacobian']/scale).tolist(),'coordinate_hessian':(jets['hessian']/scale).tolist(),'intrinsic_hessian':intrinsic.tolist(),
        'filtration_margin':margin,'finite_difference_checks':checks,'mirror_controls':mirror}


def measures(out):
    fine=[np.load(out/f'case_{s}_n65.npz') for s in ('a','b')]
    scale=float(np.sqrt(np.mean([np.mean(np.sum(z['persistent']**2,axis=2)) for z in fine])))
    if scale<=1e-12:raise ValueError('case RMS below frozen floor')
    result={'shared_fine_uniform_rms':scale,'interpretation':'finite quadrature of three explicitly defined reduced-family measures; no molecular equilibrium claim','grids':{}}
    rt=.00198720425864083*298.15
    for n in (33,65):
        per={};supports=[];weights=[]
        for side in ('a','b'):
            z=np.load(out/f'case_{side}_n{n}.npz');e=z['energy'].ravel();vol=z['volume'].ravel();p=z['persistent'].reshape(-1,18)/scale
            boltz=np.exp(-(e-e.min())/rt);w=np.column_stack([np.ones(len(e)),boltz,boltz*vol]);w/=w.sum(axis=0)
            norm=np.zeros(3)
            for start in range(0,len(e),256):norm+=np.sum(w[start:start+256]*(odd_kernel(p[start:start+256],p)@w),axis=0)
            per[side]={'odd_norm_squared':norm.tolist(),'effective_grid_sizes':(1/(w*w).sum(axis=0)).tolist(),
                'mean_relative_energy':(w.T@e).tolist(),'weights_total_variation_angle_vs_volume':float(.5*np.abs(w[:,1]-w[:,2]).sum())}
            supports.append(p);weights.append(w)
        cross=np.zeros(3)
        for start in range(0,len(supports[0]),256):cross+=np.sum(weights[0][start:start+256]*(odd_kernel(supports[0][start:start+256],supports[1])@weights[1]),axis=0)
        per['odd_measure_distance_squared']=(np.array(per['a']['odd_norm_squared'])+per['b']['odd_norm_squared']-2*cross).tolist()
        result['grids'][str(n)]=per
    for z in fine:z.close()
    write_json(out/'measure_summary.json',result)
    return result


def main(out,workers=4,phase='all'):
    out.mkdir(parents=True,exist_ok=True);start=time.time()
    if phase in ('all','cohort'):run_cohort(out,workers)
    if phase in ('all','case'):
        pairs=json.loads((ROOT/'results/biology_cliffs/same_assay_pairs.json').read_text());case=next(p for p in pairs if p['parent_key']=='81dbd5f36b7abada')
        jobs=[(side,case['stereo_'+side],n,str(out)) for side in ('a','b') for n in (33,65) if not (out/f'case_{side}_n{n}.json').exists()]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(grid_case,jobs):pass
        write_json(out/'case_local_checks.json',{s:case_local_checks(case['stereo_'+s]) for s in ('a','b')})
        measures(out)
        diagrams=[json.loads((out/f'case_{s}_n{n}.json').read_text())['persistence']['intervals'] for s in ('a','b') for n in (33,65)]
        bottleneck={s:{str(d):float(gudhi.bottleneck_distance(np.array([[b,np.inf if e is None else e] for b,e in diagrams[i*2][str(d)]]),
             np.array([[b,np.inf if e is None else e] for b,e in diagrams[i*2+1][str(d)]]),e=0)) for d in range(3)} for i,s in enumerate(('a','b'))}
        write_json(out/'grid_comparison.json',{'bottleneck_distances_kcal_mol':bottleneck,'interpretation':'resolution sensitivity, not a continuum error bound'})
    write_json(out/'run_receipt.json',{'phase':phase,'elapsed_seconds':time.time()-start,'protocol_sha256':hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),'versions':{'numpy':np.__version__,'gudhi':gudhi.__version__,'rdkit':Chem.rdBase.rdkitVersion}})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('out',type=Path);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--phase',choices=['all','cohort','case'],default='all');args=parser.parse_args();main(args.out,args.workers,args.phase)
