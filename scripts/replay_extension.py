"""Recompute external descriptors and separation from archived selected inputs.

No full source download, learned-model fitting or manuscript file is required.
The output directory must differ from the archived results directory.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,json,hashlib
import numpy as np
import pandas as pd
from pchr.external_benchmark import descriptor_values
from pchr.ensemble import separation_metrics

ROOT=Path(__file__).resolve().parents[1]

def recompute(path):
    parent=json.loads(path.read_text());maximum=0.
    for record in parent['records']:
        for radius in ['r2.6','r3.0']:
            actual=descriptor_values(record,float(radius[1:]))
            for name in ['source','same_scale','persistent','distance_histogram']:
                error=float(np.max(np.abs(np.asarray(actual[name])-record[radius][name]),initial=0))
                maximum=max(maximum,error)
            if actual['nullity']!=record[radius]['nullity']:raise AssertionError('Betti mismatch')
            record[radius]=actual
    return parent,maximum

def main(out,recompute_descriptors=False,workers=4):
    source=ROOT/'results/external_full';out=out.resolve()
    if out==source.resolve():raise ValueError('Choose a separate replay output directory')
    out.mkdir(parents=True,exist_ok=True)
    paths=sorted((source/'parents').glob('*.json'));parents=[];maximum=0.
    if recompute_descriptors:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i,f in enumerate(as_completed([pool.submit(recompute,p) for p in paths]),1):
                parent,error=f.result();parents.append(parent);maximum=max(maximum,error)
                if i%64==0:print(json.dumps({'recomputed_parents':i,'maximum_component_error':maximum}),flush=True)
    else:parents=[json.loads(p.read_text()) for p in paths]
    with np.load(ROOT/'data/extension_sources/published_embeddings_selected.npz',allow_pickle=False) as arrays:
        offsets={int(source):i for i,source in enumerate(arrays['source_indices'])}
        models={name:arrays[name].copy() for name in ['gemnet','mace','fmg']}
    rows=[]
    for parent in sorted(parents,key=lambda p:p['mol_id']):
        records=parent['records'];labels=[r['en_id'] for r in records]
        indices=[offsets[r['embedding_index']] for r in records]
        for radius in ['r2.6','r3.0']:
            reps={name:np.array([r[radius][name] for r in records]) for name in ['source','same_scale','persistent','distance_histogram']}
            if radius=='r2.6':reps.update({name:a[indices] for name,a in models.items()})
            for name,values in reps.items():
                scale=max(float(np.sqrt(np.mean(np.sum(values**2,axis=1)))),1e-12)
                rows.append({'mol_id':parent['mol_id'],'radius':radius,'representation':name,**separation_metrics(values/scale,labels)})
    table=pd.DataFrame(rows).sort_values(['mol_id','radius','representation']).reset_index(drop=True)
    expected=pd.read_csv(source/'separation.csv').sort_values(['mol_id','radius','representation']).reset_index(drop=True)
    assert table[['mol_id','radius','representation']].equals(expected[['mol_id','radius','representation']])
    np.testing.assert_allclose(table[['es_auc','nn_tie_aware','silhouette']],expected[['es_auc','nn_tie_aware','silhouette']],atol=1e-12,rtol=1e-12)
    primary=table[table.radius=='r2.6'].pivot(index='mol_id',columns='representation',values='es_auc')
    delta=(primary.persistent-primary.source).to_numpy();rng=np.random.default_rng(28102026)
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    baseline=json.loads((source/'summary.json').read_text())
    np.testing.assert_allclose([delta.mean(),*np.quantile(boot,[.025,.975])],
        [baseline['primary_paired_auc_difference']['mean'],*baseline['primary_paired_auc_difference']['bootstrap95']],atol=1e-13)
    if maximum>1e-10:raise AssertionError('descriptor replay error exceeds tolerance')
    table.to_csv(out/'separation_replayed.csv',index=False)
    result={'parents':len(parents),'conformers':sum(len(p['records']) for p in parents),
        'recomputed_descriptors':recompute_descriptors,'maximum_component_error':maximum,
        'primary_difference':float(delta.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist(),
        'all_separation_metrics_match':True,'selected_embeddings_sha256':hashlib.sha256((ROOT/'data/extension_sources/published_embeddings_selected.npz').read_bytes()).hexdigest()}
    (out/'replay_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--recompute-descriptors',action='store_true');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();main(a.output,a.recompute_descriptors,a.workers)
