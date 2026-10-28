"""Frozen external stereoisomer benchmark; no fitted parameters or outcome tuning."""
from __future__ import annotations
import argparse,collections,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.linalg import solve
from rdkit import Chem,RDLogger
from .core import persistent_operator,cochains,distances
from .ensemble import separation_metrics,odd_statistics
RDLogger.DisableLog('rdApp.warning')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_record(block,expected=None):
    m=Chem.MolFromMolBlock(block,removeHs=False)
    if m is None:raise ValueError('MolBlock parsing failed')
    m=Chem.RemoveHs(m);Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    identity=Chem.MolToSmiles(m,isomericSmiles=True)
    if expected is not None and identity!=expected:raise ValueError('stored stereochemical identity changed')
    spatial=Chem.Mol(m);Chem.AssignStereochemistryFrom3D(spatial,confId=0,replaceExistingTags=True)
    Chem.AssignStereochemistry(spatial,cleanIt=True,force=True)
    if Chem.MolToSmiles(spatial,isomericSmiles=True)!=identity:raise ValueError('3D/stored stereochemistry mismatch')
    x=m.GetConformer().GetPositions()
    if not np.isfinite(x).all() or np.min(distances(x)+np.eye(len(x))*100)<.1:raise ValueError('invalid or colliding geometry')
    return {'coordinates':x.tolist(),'atom_features':[[a.GetAtomicNum(),a.GetDegree(),a.GetTotalNumHs()] for a in m.GetAtoms()],
            'elements':[a.GetSymbol() for a in m.GetAtoms()],
            'bonds':[[b.GetBeginAtomIdx(),b.GetEndAtomIdx(),b.GetBondTypeAsDouble()] for b in m.GetBonds()],
            'stereocentres':Chem.FindMolChiralCenters(m,includeUnassigned=True,useLegacyImplementation=False),
            'stereo_smiles':identity}


def distance_histogram(x,features):
    z=np.asarray(features)[:,0].astype(int)
    element=np.array([0 if v==6 else 1 if v==7 else 2 if v==8 else 3 if v==16 else 4 if v in {9,17,35,53} else 5 for v in z])
    d=distances(x);i,j=np.triu_indices(len(x),1);out=[]
    for a in range(6):
        for b in range(a,6):
            take=((element[i]==a)&(element[j]==b))|((element[i]==b)&(element[j]==a))
            out.extend(np.histogram(d[i[take],j[take]],bins=np.linspace(0,12,17))[0])
    return np.array(out,float)/max(len(i),1)


def descriptor_values(record,r):
    x=np.asarray(record['coordinates']);af=np.asarray(record['atom_features'])
    op=persistent_operator(x,r,r+.6);chi,b=cochains(x,op.small[2],af)
    n=max(len(op.small[2]),1);raw=(b.T@chi/n).ravel()
    matrix=op.matrix+.3*np.eye(len(op.matrix))
    u=solve(matrix,chi,assume_a='pos') if len(matrix) else chi.copy()
    persistent=(b.T@u/n).ravel()
    same_op=op.b2.T@op.b2
    if op.small[3]:
        from .core import boundary
        b3=boundary(op.small[2],op.small[3]);same_op+=b3@b3.T
    same_u=solve(same_op+.3*np.eye(len(same_op)),chi,assume_a='pos') if len(same_op) else chi.copy()
    residual=np.linalg.norm(matrix@u-chi)
    return {'source':raw.tolist(),'same_scale':(b.T@same_u/n).ravel().tolist(),'persistent':persistent.tolist(),
       'triangles':len(op.small[2]),'later_tetrahedra':len(op.large[3]),'nullity':int(np.sum(np.linalg.eigvalsh(op.matrix)<1e-8)),
       'residual':float(residual),'distance_histogram':distance_histogram(x,af).tolist()}


def compute_parent(payload):
    parent,source_rows=payload;records=[]
    try:
        for row in parent['rows']:
            source=source_rows[row['row_index']]
            for k,block in enumerate(source['mol_blocks'][:4]):
                geometry=geometry_record(block,row['stereo_smiles'])
                record={'row_key':source['key'],'en_id':source['en_id'],'conformer_index':k,'embedding_index':source['offset']+k,**geometry}
                record['r2.6']=descriptor_values(geometry,2.6);record['r3.0']=descriptor_values(geometry,3.)
                records.append(record)
        return {'mol_id':parent['mol_id'],'achiral_smiles':parent['rows'][0]['achiral_smiles'],'records':records,'status':'completed'}
    except Exception as e:return {'mol_id':parent['mol_id'],'status':'excluded','reason':str(e)}


def main(source_dir,out,workers=4,limit=None):
    out.mkdir(parents=True,exist_ok=True);parts=out/'parents';parts.mkdir(exist_ok=True)
    root=Path(__file__).resolve().parents[2]
    protocol_path=root/'data/protocols/extension_primary_v1.json';protocol=json.loads(protocol_path.read_text())
    source_file=source_dir/'3dcs-chirality.parquet'
    if sha(source_file)!=protocol['external_benchmark']['parquet_sha256']:raise ValueError('source checksum mismatch')
    source_rows=pq.read_table(source_file).to_pylist()
    running=0
    for row in source_rows:
        assert row['offset']==running and len(row['mol_blocks'])==row['n_conformers'];running+=row['n_conformers']
    assert running==52391
    census=json.loads((source_dir/'eligible_3dcs_structural_census.json').read_text())
    selected=census['eligible'][:512]
    if limit is not None:selected=selected[:limit]
    # Only the selected rows need to cross the Windows process boundary.
    jobs=[];allresults=[];started=time.time()
    for parent in selected:
        target=parts/(parent['mol_id']+'.json')
        if target.exists():allresults.append(json.loads(target.read_text()));continue
        sub={r['row_index']:source_rows[r['row_index']] for r in parent['rows']}
        jobs.append((parent,sub))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(compute_parent,p) for p in jobs]
        for i,f in enumerate(as_completed(futures),1):
            result=f.result();allresults.append(result)
            (parts/(result['mol_id']+'.json')).write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
            if i%8==0:print(json.dumps({'computed_new_parents':i,'pending':len(jobs)-i,'seconds':round(time.time()-started,1),'exclusions':sum(x['status']!='completed' for x in allresults)}),flush=True)
    allresults.sort(key=lambda x:next(p['selection_hash'] for p in selected if p['mol_id']==x['mol_id']))
    embeddings={}
    for model,key in [('gemnet','gemnet'),('mace','arr_0'),('fmg','embeddings')]:
        with np.load(source_dir/f'3dcs_{model}.npz',allow_pickle=False) as z:embeddings[model]=z[key]
        assert embeddings[model].shape[0]==running and np.isfinite(embeddings[model]).all()
    rows=[];ensemble_rows=[]
    for p in allresults:
        if p['status']!='completed':continue
        records=p['records'];labels=np.array([x['en_id'] for x in records]);idx=np.array([x['embedding_index'] for x in records])
        for radius in ['r2.6','r3.0']:
            reps={name:np.array([x[radius][name] for x in records]) for name in ['source','same_scale','persistent','distance_histogram']}
            if radius=='r2.6':reps.update({name:e[idx] for name,e in embeddings.items()})
            for name,values in reps.items():
                scale=max(float(np.sqrt(np.mean(np.sum(values*values,axis=1)))),1e-12)
                normalized=values/scale;metrics=separation_metrics(normalized,labels)
                rows.append({'mol_id':p['mol_id'],'radius':radius,'representation':name,'n_conformers':len(records),'n_stereoisomers':len(set(labels)),
                    'heavy_atoms':len(records[0]['coordinates']),'rms_scale':scale,**metrics})
                if name in ['source','same_scale','persistent']:
                    for label in sorted(set(labels)):
                        point=normalized[labels==label]
                        for bandwidth in [.5,1.,2.]:
                            for lam in [.01,.05,.2]:
                                stats=odd_statistics(point,sigma=bandwidth,regularization=lam)
                                mean=point.mean(axis=0);center=point-mean;cov=center.T@center/len(point)
                                linear=float(mean@solve(cov+lam*np.eye(point.shape[1]),mean,assume_a='pos'))
                                ensemble_rows.append({'mol_id':p['mol_id'],'radius':radius,'representation':name,'en_id':label,
                                     'n_conformers':len(point),'sigma':bandwidth,'lambda':lam,'linear_fisher_score':linear,
                                     **{k:v for k,v in stats.items() if k!='witness_coefficients'}})
    pd.DataFrame(rows).to_csv(out/'separation.csv',index=False);pd.DataFrame(ensemble_rows).to_csv(out/'ensemble_statistics.csv',index=False)
    results=pd.DataFrame(rows);primary=results[(results.radius=='r2.6')].pivot(index='mol_id',columns='representation',values='es_auc')
    diff=(primary.persistent-primary.source).to_numpy();rng=np.random.default_rng(28102026)
    boot=np.array([np.mean(rng.choice(diff,size=len(diff),replace=True)) for _ in range(5000)])
    summary={'selected_parents':len(selected),'completed_parents':sum(x['status']=='completed' for x in allresults),
       'exclusions':[x for x in allresults if x['status']!='completed'],
       'conformers':sum(len(x.get('records',[])) for x in allresults),
       'mean_metrics':results.groupby(['radius','representation'])[['es_auc','nn_tie_aware','silhouette']].mean().reset_index().to_dict('records'),
       'primary_paired_auc_difference':{'mean':float(diff.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'n_parents':len(diff)},
       'protocol_sha256':sha(protocol_path),'source_sha256':sha(source_file),'seconds':time.time()-started,
       'max_solve_residual':max(x[r]['residual'] for p in allresults for x in p.get('records',[]) for r in ['r2.6','r3.0'])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--limit',type=int)
    a=p.parse_args();main(a.source_dir,a.output,a.workers,a.limit)
