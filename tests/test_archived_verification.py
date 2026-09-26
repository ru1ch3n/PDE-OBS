import copy
import json

import h5py
import numpy as np
import pytest

from pdeobs import archived_verification as av
from pdeobs.evaluation import EvaluationConfig
from pdeobs.strict_inference import target_frame_metadata
from pdeobs.strict_score import CONTRACT_VERSION, score_prediction_file


def fixture(task):
    frames = [[0, 2, 4, 6], [0, 5, 10, 15]] if task == 'rollout' else [[0], [0]]
    rows = [{'sample_id':str(i), 'T':len(v), 'stored_frame_indices':v,
             **({'stored_time_values':[float(x)/100 for x in v]} if task=='rollout' else {})}
            for i,v in enumerate(frames)]
    ec = EvaluationConfig(task=task, horizon=3, history_steps=1,
                          rollout_target_offset=0, data_layout='channels_last', device='cpu')
    shape = [3, 4, 4, 1] if task=='rollout' else [4, 4, 1]
    target = np.ones((2,*shape),dtype=np.float32)
    inputs = np.ones((2,1,4,4,1) if task=='rollout' else (2,4,4,1),dtype=np.float32)
    batch={'input':inputs,'mask':np.ones_like(inputs),'target':target,'metadata':rows}
    contract={'schema_version':CONTRACT_VERSION,'task':task,'split':'demo',
              'expected_ids':['0','1'],'expected_shape':shape,
              'target_time_indices':[1,2,3] if task=='rollout' else [0],
              'time_index_semantics':'stored_frame_positions; per_identity_source_mapping_bound',
              'frame_mapping_by_identity':{r['sample_id']:target_frame_metadata(r,ec) for r in rows},
              'checkpoint_id':'synthetic-test-checkpoint','observation_id':'synthetic', 'projection':False}
    return ec,batch,contract


@pytest.mark.parametrize('task',['recovery','rollout'])
def test_retains_arrays_and_recomputes_every_identity(task,tmp_path,monkeypatch):
    ec,b,c=fixture(task)
    monkeypatch.setattr(av,'predict_batch',lambda model,batch,config:(batch['target']*1.25,batch['target']))
    path=tmp_path/'predictions.h5'
    report=av.export_bound_predictions(None,[b],config=ec,contract=c,prediction_path=path)
    assert report['status']=='valid'
    assert report['scored_identity_count']==2
    assert report['summary']['rel_l2_joint_mean']==pytest.approx(.25)
    assert score_prediction_file(path,c)==report
    with h5py.File(path) as h:
        saved=[json.loads(s) for s in h['metadata_json'].asstr()[:]]
        assert [r['target_source_frame_indices'] for r in saved]==([ [2,4,6],[5,10,15] ] if task=='rollout' else [[0],[0]])
    with pytest.raises(FileExistsError):
        av.export_bound_predictions(None,[b],config=ec,contract=c,prediction_path=path)


def test_nonfinite_prediction_is_retained_but_invalid(tmp_path,monkeypatch):
    ec,b,c=fixture('rollout')
    pred=b['target'].copy();pred[0,0,0,0,0]=np.nan
    monkeypatch.setattr(av,'predict_batch',lambda *args:(pred,b['target']))
    path=tmp_path/'predictions.h5'
    r=av.export_bound_predictions(None,[b],config=ec,contract=c,prediction_path=path)
    assert r['status']=='invalid' and r['scored_identity_count']==0
    with h5py.File(path) as h: assert np.isnan(h['prediction'][0,0,0,0,0])


@pytest.mark.parametrize('fault',['duplicate','missing','time'])
def test_wrong_identity_or_frame_fails_without_final_file(tmp_path,monkeypatch,fault):
    ec,b,c=fixture('rollout')
    monkeypatch.setattr(av,'predict_batch',lambda *args:(b['target'],b['target']))
    if fault=='duplicate': b['metadata'][1]['sample_id']='0'
    if fault=='missing': c['expected_ids'].append('2')
    if fault=='time': b['metadata'][0]['stored_frame_indices']=[0,3,6,9]
    path=tmp_path/'predictions.h5'
    with pytest.raises(ValueError):
        av.export_bound_predictions(None,[b],config=ec,contract=c,prediction_path=path)
    assert not path.exists()


def test_hash_and_split_mismatch_are_fail_closed(tmp_path):
    p=tmp_path/'checkpoint';p.write_bytes(b'fixture')
    with pytest.raises(ValueError): av.require_hash(p,'0'*64)
    with pytest.raises(ValueError): av.assert_split({'seed':1},{'seed':2})


def test_poor_finite_prediction_not_rejected_for_quality():
    r={'status':'valid','summary':{'rel_l2_joint_mean':12.0}}
    c=av.compare_score(r,{'relative_l2':12.0})
    assert c['within_reporting_tolerance'] and c['strict_relative_l2']==12.0
