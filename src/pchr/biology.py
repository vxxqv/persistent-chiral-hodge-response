"""Same-assay stereoisomer curation from raw ChEMBL records."""
from pathlib import Path
from itertools import combinations
import argparse,json,collections,hashlib
import numpy as np
import pandas as pd
from rdkit import Chem
from .ensemble_diagnostics import mirror_smiles


def curate(raw,out):
    out.mkdir(parents=True,exist_ok=True)
    molecules={};assays={};targets={};activities={}
    for p in raw.glob('*molecules*.json'):
        for row in json.loads(p.read_text())['molecules']:molecules[row['molecule_chembl_id']]=row
    for p in raw.glob('assay_metadata*.json'):
        for row in json.loads(p.read_text())['assays']:assays[row['assay_chembl_id']]=row
    for p in raw.glob('target_metadata*.json'):
        for row in json.loads(p.read_text())['targets']:targets[row['target_chembl_id']]=row
    for p in raw.glob('*activities*.json'):
        for row in json.loads(p.read_text())['activities']:activities[row['activity_id']]=row
    identities={};counts=collections.Counter();audit=[];accepted=[]
    for mid,record in molecules.items():
        try:
            smi=record['molecule_structures']['canonical_smiles'];m=Chem.MolFromSmiles(smi)
            if m is None or not 6<=m.GetNumHeavyAtoms()<=32 or len(Chem.GetMolFrags(m))!=1:continue
            if any(a.GetAtomicNum() not in {6,7,8,9,15,16,17,35,53} for a in m.GetAtoms()):continue
            centers=Chem.FindMolChiralCenters(m,includeUnassigned=True,useLegacyImplementation=False)
            if not centers or any(s=='?' for _,s in centers):continue
            graph=Chem.MolToSmiles(m,isomericSmiles=False);iso=Chem.MolToSmiles(m,isomericSmiles=True)
            identities[mid]={'graph':graph,'stereo_smiles':iso,'heavy_atoms':m.GetNumHeavyAtoms()}
        except Exception:continue
    for aid,a in activities.items():
        reason=None
        mid=a['molecule_chembl_id'];assay=assays.get(a['assay_chembl_id']);target=targets.get(a['target_chembl_id'])
        if mid not in identities:reason='invalid_or_underdefined_identity'
        elif assay is None or target is None:reason='metadata_unavailable'
        elif assay['confidence_score']!=9:reason='assay_confidence_not_9'
        elif target['target_type']!='SINGLE PROTEIN' or target['organism']!='Homo sapiens':reason='target_not_human_single_protein'
        elif assay['target_chembl_id']!=a['target_chembl_id']:reason='target_identity_disagreement'
        elif (a['standard_type'],a['standard_relation'],a['standard_units'],a['assay_type'])!=('Ki','=','nM','B'):reason='endpoint_not_exact_binding_Ki'
        elif a.get('data_validity_comment') is not None:reason='validity_flag'
        elif str(a.get('potential_duplicate') or 0)!='0':reason='potential_duplicate'
        try:
            value=float(a['standard_value']);pk=float(a['pchembl_value'])
            if not np.isfinite([value,pk]).all() or value<=0 or abs(9-np.log10(value)-pk)>.011:reason=reason or 'pKi_reconstruction_failed'
        except (TypeError,ValueError):reason=reason or 'missing_numeric_pKi'
        if reason:counts[reason]+=1;audit.append({'activity_id':aid,'status':'excluded','reason':reason});continue
        row={'activity_id':aid,'molecule_chembl_id':mid,'assay_chembl_id':a['assay_chembl_id'],'target_chembl_id':a['target_chembl_id'],
            'target_name':target['pref_name'],'document_chembl_id':a['document_chembl_id'],'pki':pk,'standard_value_nm':value,**identities[mid]}
        accepted.append(row);audit.append({'activity_id':aid,'status':'accepted','reason':None})
    pd.DataFrame(audit).to_csv(out/'activity_filter_audit.csv',index=False)
    pd.DataFrame(accepted).to_csv(out/'accepted_activity_records.csv',index=False)
    aggregates=[];pairs=[]
    if accepted:
        frame=pd.DataFrame(accepted)
        for (graph,stereo,assay_id,target_id),g in frame.groupby(['graph','stereo_smiles','assay_chembl_id','target_chembl_id']):
            if g.pki.max()-g.pki.min()>1:
                counts['replicate_range_above_1']+=1;continue
            aggregates.append({'graph':graph,'stereo_smiles':stereo,'assay_chembl_id':assay_id,'target_chembl_id':target_id,
                'target_name':g.target_name.iloc[0],'pki':float(g.pki.median()),'n_records':len(g),
                'molecule_ids':sorted(set(g.molecule_chembl_id)),'activity_ids':sorted(g.activity_id.tolist()),'document_ids':sorted(set(g.document_chembl_id))})
        if aggregates:
            aframe=pd.DataFrame(aggregates)
            for (graph,assay_id,target_id),g in aframe.groupby(['graph','assay_chembl_id','target_chembl_id']):
                for (_,a),(_,b) in combinations(list(g.sort_values('stereo_smiles').iterrows()),2):
                    delta=float(a.pki-b.pki)
                    pairs.append({'parent_key':hashlib.sha256(graph.encode()).hexdigest()[:16],'graph':graph,'stereo_a':a.stereo_smiles,'stereo_b':b.stereo_smiles,
                        'assay_chembl_id':assay_id,'target_chembl_id':target_id,'target_name':a.target_name,
                        'pki_a':a.pki,'pki_b':b.pki,'delta_pki':delta,'absolute_delta_pki':abs(delta),'cliff_ge1':abs(delta)>=1,
                        'relation':'enantiomers' if mirror_smiles(a.stereo_smiles)==b.stereo_smiles else 'other_stereoisomers',
                        'molecule_ids_a':a.molecule_ids,'molecule_ids_b':b.molecule_ids,'activity_ids_a':a.activity_ids,'activity_ids_b':b.activity_ids,
                        'document_ids_a':a.document_ids,'document_ids_b':b.document_ids})
    (out/'compound_assay_aggregates.json').write_text(json.dumps(aggregates,indent=2),encoding='utf-8')
    (out/'same_assay_pairs.json').write_text(json.dumps(pairs,indent=2),encoding='utf-8')
    if pairs:pd.DataFrame(pairs).to_csv(out/'same_assay_pairs.csv',index=False)
    summary={'raw_unique_activities':len(activities),'source_molecules':len(molecules),'valid_stereochemical_molecules':len(identities),
        'accepted_records':len(accepted),'compound_assay_aggregates':len(aggregates),'same_assay_pair_observations':len(pairs),
        'paired_parents':len(set(p['parent_key'] for p in pairs)),'paired_targets':len(set(p['target_chembl_id'] for p in pairs)),
        'filter_counts':dict(counts),'same_assay_pair_documentation':'Each row retains original compound, assay, target, document and activity identifiers. No cross-assay potency pooling enters a pair.'}
    (out/'curation_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
    return pairs


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();curate(a.raw,a.output)
