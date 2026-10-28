"""Conformer ensembles and parent-level activity associations for curated pairs."""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from rdkit import Chem,RDLogger,DataStructs
from rdkit.Chem import AllChem,rdFingerprintGenerator
from .external_benchmark import descriptor_values,geometry_record
from .ensemble import mmd_squared,parity_mmd_decomposition
RDLogger.DisableLog('rdApp.warning')


def generate_molecule(smiles):
    seed=int(hashlib.sha256(('pchr-activity-20261028:'+smiles).encode()).hexdigest()[:8],16)%(2**31-1)
    molecule=Chem.MolFromSmiles(smiles);full=Chem.AddHs(molecule)
    p=AllChem.ETKDGv3();p.randomSeed=seed;p.numThreads=1;p.pruneRmsThresh=-1
    ids=list(AllChem.EmbedMultipleConfs(full,numConfs=8,params=p));failures=[];records=[]
    if not AllChem.MMFFHasAllMoleculeParams(full):return {'smiles':smiles,'seed':seed,'status':'excluded','reason':'MMFF_parameters_unavailable','conformers':[]}
    opt=AllChem.MMFFOptimizeMoleculeConfs(full,numThreads=1,maxIters=2000,mmffVariant='MMFF94s')
    for cid,(status,energy) in zip(ids,opt):
        if status!=0:failures.append({'conformer_id':cid,'reason':'MMFF_not_converged','status':status});continue
        try:
            heavy=Chem.RemoveHs(full);block=Chem.MolToMolBlock(heavy,confId=cid)
            geometry=geometry_record(block,smiles)
            record={'conformer_id':cid,'energy_kcal_mol':energy,**geometry}
            for r in [2.6,3.]:record[f'r{r:.1f}']=descriptor_values(geometry,r)
            records.append(record)
        except Exception as e:failures.append({'conformer_id':cid,'reason':str(e)})
    return {'smiles':smiles,'seed':seed,'status':'completed' if len(records)>=4 else 'excluded','conformers':records,'failures':failures,'requested':8}


def target_independent_limits(pairs):
    if not len(pairs):return []
    rows=[]
    for (parent,a,b),g in pairs.groupby(['parent_key','stereo_a','stereo_b']):
        per=g.groupby('target_chembl_id').delta_pki.median()
        reliable=per[np.abs(per)>=.3-1e-12];positive=int((reliable>0).sum());negative=int((reliable<0).sum())
        rows.append({'parent_key':parent,'stereo_a':a,'stereo_b':b,'n_targets':len(per),'target_variance_lower_bound':float(np.var(per)),
            'n_thresholded_targets':len(reliable),'positive_targets':positive,'negative_targets':negative,
            'rank_reversal':positive>0 and negative>0,'target_blind_classification_error_lower_bound':min(positive,negative)/len(reliable) if len(reliable) else None,
            'target_signed_differences':per.to_dict()})
    return rows


def main(folder,workers=4):
    pairs=json.loads((folder/'same_assay_pairs.json').read_text());start=time.time()
    cache=folder/'ensembles';cache.mkdir(exist_ok=True)
    smiles=sorted({p[s] for p in pairs for s in ['stereo_a','stereo_b']});molecules={};jobs=[]
    for smi in smiles:
        path=cache/(hashlib.sha256(smi.encode()).hexdigest()[:20]+'.json')
        if path.exists():molecules[smi]=json.loads(path.read_text())
        else:jobs.append(smi)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(generate_molecule,s):s for s in jobs}
        for i,future in enumerate(as_completed(futures),1):
            smi=futures[future]
            try:result=future.result()
            except Exception as e:result={'smiles':smi,'status':'excluded','reason':str(e),'conformers':[]}
            molecules[smi]=result
            (cache/(hashlib.sha256(smi.encode()).hexdigest()[:20]+'.json')).write_text(json.dumps(result,separators=(',',':')),encoding='utf-8')
            if i%16==0:print(json.dumps({'generated_molecules':i,'pending':len(jobs)-i,'seconds':round(time.time()-start,1)}),flush=True)
    rows=[];fail=[]
    for p in pairs:
        a,b=molecules[p['stereo_a']],molecules[p['stereo_b']]
        if a['status']!='completed' or b['status']!='completed':fail.append({'parent_key':p['parent_key'],'assay':p['assay_chembl_id'],'reason':'insufficient_converged_valid_conformers'});continue
        row={**p,'n_conformers_a':len(a['conformers']),'n_conformers_b':len(b['conformers'])}
        for radius in ['r2.6','r3.0']:
            for name in ['source','same_scale','persistent','distance_histogram']:
                x=np.array([c[radius][name] for c in a['conformers']]);y=np.array([c[radius][name] for c in b['conformers']])
                scale=max(np.sqrt(np.mean(np.sum(np.r_[x,y]**2,axis=1))),1e-12);x/=scale;y/=scale
                row[f'{name}_{radius}_mmd']=np.sqrt(mmd_squared(x,y))
                if name!='distance_histogram' and p['relation']=='enantiomers':
                    decomp=parity_mmd_decomposition(x,y)
                    row[f'{name}_{radius}_balanced_mmd']=np.sqrt(decomp['odd_squared'])
        for include in [False,True]:
            generator=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=include)
            fa=generator.GetFingerprint(Chem.MolFromSmiles(p['stereo_a']));fb=generator.GetFingerprint(Chem.MolFromSmiles(p['stereo_b']))
            row['ecfp4_chiral' if include else 'ecfp4_achiral']=1-DataStructs.TanimotoSimilarity(fa,fb)
        rows.append(row)
    frame=pd.DataFrame(rows);frame.to_csv(folder/'descriptor_activity_pairs.csv',index=False)
    # Equal parent weights. First average repeated assays within each pair and
    # target, then average pair/target rows within the constitutional parent.
    numeric=['absolute_delta_pki']+[c for c in frame.columns if c.endswith('_mmd') or c.startswith('ecfp4_')]
    pairtarget=frame.groupby(['parent_key','stereo_a','stereo_b','target_chembl_id'])[numeric].mean().reset_index()
    parents=pairtarget.groupby('parent_key')[numeric].mean();parents.to_csv(folder/'parent_level_activity.csv')
    rng=np.random.default_rng(28102026);indices=rng.integers(0,len(parents),size=(5000,len(parents)))
    correlations=[]
    for col in numeric[1:]:
        x=parents[col].to_numpy();y=parents.absolute_delta_pki.to_numpy()
        finite=np.isfinite(x)&np.isfinite(y)
        if not finite.all():
            x,y=x[finite],y[finite]
            local_rng=np.random.default_rng(28102026+int(hashlib.sha256(col.encode()).hexdigest()[:6],16))
            boot_indices=local_rng.integers(0,len(x),size=(5000,len(x)))
        else:boot_indices=indices
        if np.std(x)==0 or np.std(y)==0:
            correlations.append({'representation':col,'spearman':None,'bootstrap95':None,'n_parents':len(x),'reason':'constant predictor or response'});continue
        estimate=float(spearmanr(x,y).statistic)
        boot=[]
        for idx in boot_indices:
            if np.ptp(x[idx])>0 and np.ptp(y[idx])>0:boot.append(float(spearmanr(x[idx],y[idx]).statistic))
        correlations.append({'representation':col,'spearman':estimate,'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),'n_parents':len(x),'valid_bootstrap_replicates':len(boot)})
    limits=target_independent_limits(frame)
    (folder/'target_conditioning_limits.json').write_text(json.dumps(limits,indent=2),encoding='utf-8')
    summary={'curated_same_assay_pairs':len(pairs),'analysed_pair_observations':len(frame),'parents':len(parents),'targets':int(frame.target_chembl_id.nunique()),
       'generated_stereoisomers':len(molecules),'retained_stereoisomers':sum(v['status']=='completed' for v in molecules.values()),
       'retained_conformers':sum(len(v['conformers']) for v in molecules.values() if v['status']=='completed'),
       'failed_pair_observations':fail,'generation_failures':[{k:v for k,v in m.items() if k!='conformers'} for m in molecules.values() if m['status']!='completed'],
       'activity_associations':correlations,'pairs_with_multiple_targets':sum(x['n_targets']>=2 for x in limits),
       'pairs_with_recorded_rank_reversals':sum(x['rank_reversal'] for x in limits),'elapsed_seconds':time.time()-start,
       'inference_boundary':'Exploratory association across parents; no target-conditioned prediction model, causal inference or equilibrium weighting.'}
    (folder/'association_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8');print(json.dumps(summary,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);p.add_argument('--workers',type=int,default=4);a=p.parse_args();main(a.folder,a.workers)
