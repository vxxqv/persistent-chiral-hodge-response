import json
import numpy as np
import pandas as pd
from pchr.biology import curate
from pchr.biology_ensembles import target_independent_limits


def test_same_assay_pairing_and_fail_closed_flags(tmp_path):
    raw=tmp_path/'raw';raw.mkdir();out=tmp_path/'out'
    molecules=[{'molecule_chembl_id':mid,'molecule_structures':{'canonical_smiles':smi}}
        for mid,smi in [('M1','C[C@H](O)CCNC'),('M2','C[C@@H](O)CCNC')]]
    assays=[{'assay_chembl_id':a,'confidence_score':9,'target_chembl_id':t}
        for a,t in [('A1','T1'),('A2','T2')]]
    targets=[{'target_chembl_id':t,'pref_name':t,'target_type':'SINGLE PROTEIN','organism':'Homo sapiens'} for t in ['T1','T2']]
    def record(i,mid,assay,target,pki,**kw):
        row=dict(activity_id=i,molecule_chembl_id=mid,assay_chembl_id=assay,target_chembl_id=target,
            standard_type='Ki',standard_relation='=',standard_units='nM',assay_type='B',
            standard_value=str(10**(9-pki)),pchembl_value=str(pki),document_chembl_id='D1',
            data_validity_comment=None,potential_duplicate=0)
        row.update(kw)
        return row
    activities=[record(1,'M1','A1','T1',8),record(2,'M2','A1','T1',7),
        record(3,'M1','A2','T2',6), # no M2 in this assay: must not cross-pool
        record(4,'M2','A2','T2',7,potential_duplicate=1),
        record(5,'M2','A2','T2',7,data_validity_comment='Outside typical range'),
        record(6,'M2','A2','T1',7), # activity/assay target disagreement
        record(7,'M2','A2','T2',7,pchembl_value='1.00')]
    for name,key,value in [('source_molecules','molecules',molecules),('assay_metadata','assays',assays),
        ('target_metadata','targets',targets),('source_activities','activities',activities)]:
        (raw/(name+'.json')).write_text(json.dumps({key:value}))
    pairs=curate(raw,out)
    assert len(pairs)==1 and pairs[0]['assay_chembl_id']=='A1'
    assert abs(pairs[0]['delta_pki'])==1 and pairs[0]['relation']=='enantiomers'
    assert pairs[0]['activity_ids_a']+pairs[0]['activity_ids_b'] in [[1,2],[2,1]]
    summary=json.loads((out/'curation_summary.json').read_text())
    assert summary['accepted_records']==3
    assert summary['filter_counts']=={'potential_duplicate':1,'validity_flag':1,
        'target_identity_disagreement':1,'pKi_reconstruction_failed':1}


def test_target_limits_use_target_medians_and_recorded_threshold():
    frame=pd.DataFrame([{'parent_key':'P','stereo_a':'A','stereo_b':'B','target_chembl_id':t,'delta_pki':y}
        for t,y in [('T1',-.3),('T1',-.3),('T2',.8),('T3',.1)]])
    result=target_independent_limits(frame)[0]
    assert result['n_targets']==3 and result['n_thresholded_targets']==2
    assert result['rank_reversal'] and result['target_blind_classification_error_lower_bound']==.5
    np.testing.assert_allclose(result['target_variance_lower_bound'],np.var([-.3,.8,.1]))
