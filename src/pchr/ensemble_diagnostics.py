"""Labelled post-primary parity diagnostic; preserve the primary benchmark."""
from pathlib import Path
from itertools import combinations
import json,argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from .ensemble import parity_mmd_decomposition,mmd_squared,odd_statistics


def mirror_smiles(smi):
    m=Chem.MolFromSmiles(smi)
    for atom in m.GetAtoms():
        if atom.GetChiralTag() in {Chem.ChiralType.CHI_TETRAHEDRAL_CW,Chem.ChiralType.CHI_TETRAHEDRAL_CCW}:atom.InvertChirality()
    Chem.AssignStereochemistry(m,cleanIt=True,force=True)
    return Chem.MolToSmiles(m,isomericSmiles=True)


def main(folder):
    rows=[];types=[];checks=[]
    for path in sorted((folder/'parents').glob('*.json')):
        parent=json.loads(path.read_text());records=parent.get('records',[])
        if not records:continue
        labels=sorted({r['en_id'] for r in records});smiles={l:next(r['stereo_smiles'] for r in records if r['en_id']==l) for l in labels}
        for la,lb in combinations(labels,2):
            relation='enantiomers' if mirror_smiles(smiles[la])==smiles[lb] else 'other_stereoisomers'
            types.append({'mol_id':parent['mol_id'],'en_a':la,'en_b':lb,'relation':relation})
            if relation!='enantiomers':continue
            for radius in ['r2.6','r3.0']:
                for rep in ['source','same_scale','persistent']:
                    values=np.array([r[radius][rep] for r in records]);scale=max(np.sqrt(np.mean(np.sum(values*values,axis=1))),1e-12);values/=scale
                    a=values[[r['en_id']==la for r in records]];b=values[[r['en_id']==lb for r in records]]
                    d=parity_mmd_decomposition(a,b)
                    rows.append({'mol_id':parent['mol_id'],'en_a':la,'en_b':lb,'radius':radius,'representation':rep,'n_a':len(a),'n_b':len(b),**d})
                    checks.append(d['projection_identity_error'])
                # Coordinate reflection leaves distance histograms fixed. Mix
                # empirical laws with equal class weights, even for unequal n.
                values=np.array([r[radius]['distance_histogram'] for r in records]);scale=max(np.sqrt(np.mean(np.sum(values*values,axis=1))),1e-12);values/=scale
                a=values[[r['en_id']==la for r in records]];b=values[[r['en_id']==lb for r in records]]
                mix_a=np.r_[a,b];mix_b=np.r_[b,a]
                wa=np.r_[np.full(len(a),.5/len(a)),np.full(len(b),.5/len(b))]
                wb=np.r_[np.full(len(b),.5/len(b)),np.full(len(a),.5/len(a))]
                balanced=mmd_squared(mix_a,mix_b,weights_x=wa,weights_y=wb)
                rows.append({'mol_id':parent['mol_id'],'en_a':la,'en_b':lb,'radius':radius,'representation':'distance_histogram','n_a':len(a),'n_b':len(b),
                    'total_squared':mmd_squared(a,b),'odd_squared':0.,'even_squared':mmd_squared(a,b),'balanced_squared':balanced,'projection_identity_error':balanced})
    table=pd.DataFrame(rows);table.to_csv(folder/'parity_decomposition.csv',index=False)
    pd.DataFrame(types).to_csv(folder/'stereoisomer_pair_types.csv',index=False)
    counts=pd.DataFrame(types).relation.value_counts().to_dict()
    summary={'status':'post-primary diagnostic; not a frozen confirmatory outcome','pair_relations':counts,
        'enantiomer_parents':int(table.mol_id.nunique()),'projection_identity_max_error':max(checks,default=0),
        'mean_energies':table.groupby(['radius','representation'])[['total_squared','odd_squared','even_squared','balanced_squared']].mean().reset_index().to_dict('records')}
    if checks and max(checks)>1e-12:raise AssertionError('parity projection identity')
    (folder/'parity_diagnostic_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);a=p.parse_args();main(a.folder)
