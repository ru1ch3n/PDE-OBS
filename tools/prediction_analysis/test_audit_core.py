import copy
import itertools
import unittest
from audit_core import *

def fixture(task='recovery'):
    ids=['synthetic-%03d'%i for i in range(200)]
    times=[0] if task=='recovery' else [1,2,3]
    frames={i:{'target_stored_positions':times,'target_source_frame_indices':times,
               'target_physical_time_values':None if task=='recovery' else [.1,.2,.3]} for i in ids}
    row={'identity':'synthetic/model/random_50pct','task':task,'training_cohort':'synthetic-test-only','actual_epochs':200,
         'checkpoint_id':'original','split':{'test_sample_ids_sha256':ids_hash(ids)},'blocks':{v:{'relative_l2':1.0} for v in VIEWS}}
    release={'checkpoint_anonymized_sha256':'released'}
    prov={'dataset_manifest':{},'split':row['split']}
    mask={'protocol':'synthetic'}
    contract={'identity':row['identity'],'observation_id':VIEWS[0],'task':task,'split':'paper-test',
     'checkpoint_id':'released','original_checkpoint_id':'original','dataset_manifest_sha256':canonical({}),
     'historical_split_sha256':canonical(row['split']),'training_cohort':row['training_cohort'],'actual_epochs':200,
     'mask_config':mask,'projection':False,'schema_version':'pdeobs-strict-contract-v1',
     'time_index_semantics':'stored_frame_positions; per_identity_source_mapping_bound',
     'verification_adapter':'pdeobs-archived-prediction-verification-v1','expected_ids':ids,
     'frame_mapping_by_identity':frames,'target_time_indices':times,
     'expected_shape':[128,128,1] if task=='recovery' else [3,128,128,1]}
    values=[{'identity':i,'rel_l2_joint':.9+.2*(n%2)} for n,i in enumerate(ids)]
    if task=='rollout':
        for r in values:r['rel_l2_by_horizon']=[{'horizon':h,'target_time_index':h,'rel_l2':float(h)} for h in (1,2,3)]
    score={'status':'valid','errors':[],'scoring_version':'pdeobs-strict-v1','epsilon':1e-12,
      'raw_arrays_checked':True,'projection_applied':False,'artifact_contract_binding':'pdeobs-strict-inference-v1_sha256_bound',
      'config':copy.deepcopy(contract),'config_sha256':canonical(contract),'identity_set_sha256':canonical(sorted(ids)),
      'expected_identity_order':ids,'per_identity':values,'summary':{'identity_count':200,'rel_l2_joint_mean':1.},
      'checkpoint_id':'released','observation_id':VIEWS[0],'task':task}
    for k in ('expected_identity_count','actual_prediction_identity_count','actual_target_identity_count','scored_identity_count'):score[k]=200
    if task=='rollout':score['summary']['rel_l2_by_horizon_mean']=[{'horizon':h,'target_time_index':h,'rel_l2_mean':float(h)} for h in (1,2,3)]
    comparison={'archived_relative_l2':1.,'strict_relative_l2':1.,'relative_difference':0.,'absolute_difference':0.,
      'strict_status':'valid','within_reporting_tolerance':True,'tolerance':{'atol':1e-7,'rtol':1e-4}}
    done={'test_view':VIEWS[0],'status':'valid','samples':200,'predictions_sha256':'p','score_sha256':'s','comparison':copy.deepcopy(comparison),'seconds':1.}
    return [row,release,prov,contract,score,comparison,done,{'score.json':'s','predictions.h5':'p'},mask,frames]

class AuditTests(unittest.TestCase):
    def test_static(self):
        result=audit_block(*fixture());self.assertEqual(result['joint']['mean'],1.);self.assertGreater(result['joint']['std'],.1)
    def test_rollout_joint_not_horizon_average(self):
        result=audit_block(*fixture('rollout'));self.assertEqual(result['joint']['mean'],1.);self.assertEqual([x['mean'] for x in result['horizons']],[1.,2.,3.])
    def reject(self,mutate,task='recovery'):
        f=fixture(task);mutate(f)
        with self.assertRaises((ValueError,KeyError)):audit_block(*f)
    def test_invalid(self):self.reject(lambda f:f[4].update(status='invalid'))
    def test_count(self):self.reject(lambda f:f[4].update(scored_identity_count=199))
    def test_nan(self):self.reject(lambda f:f[4]['per_identity'][0].update(rel_l2_joint=float('nan')))
    def test_negative(self):self.reject(lambda f:f[4]['per_identity'][0].update(rel_l2_joint=-1.))
    def test_duplicate(self):self.reject(lambda f:f[4]['per_identity'][0].update(identity='synthetic-001'))
    def test_missing(self):self.reject(lambda f:f[4]['per_identity'].pop())
    def test_prediction_hash(self):self.reject(lambda f:f[7].update({'predictions.h5':'tampered'}))
    def test_score_hash(self):self.reject(lambda f:f[7].update({'score.json':'tampered'}))
    def test_checkpoint(self):self.reject(lambda f:f[3].update(checkpoint_id='wrong'))
    def test_mask(self):self.reject(lambda f:f[3].update(mask_config={'protocol':'wrong'}))
    def test_shape(self):self.reject(lambda f:f[3].update(expected_shape=[64,64,1]))
    def test_dataset(self):self.reject(lambda f:f[3].update(dataset_manifest_sha256='wrong'))
    def test_source_frames(self):self.reject(lambda f:f[3]['frame_mapping_by_identity'].update({'synthetic-000':{'target_stored_positions':[1,2,3],'target_source_frame_indices':[2,3,4],'target_physical_time_values':[.1,.2,.3]}}),'rollout')
    def test_physical_time(self):self.reject(lambda f:f[3]['frame_mapping_by_identity']['synthetic-000'].update(target_physical_time_values=None),'rollout')
    def test_horizon(self):self.reject(lambda f:f[4]['per_identity'][0]['rel_l2_by_horizon'][0].update(target_time_index=2),'rollout')
    def test_mean(self):self.reject(lambda f:f[4]['summary'].update(rel_l2_joint_mean=1.1))
    def test_unbound(self):self.reject(lambda f:f[4].update(artifact_contract_binding='unbound'))
    def test_tolerance(self):self.reject(lambda f:f[5]['tolerance'].update(rtol=.01))
    def test_old_score_not_substituted(self):self.reject(lambda f:f[5].update(strict_relative_l2=.9))
    def test_no_projection(self):self.reject(lambda f:f[3].update(projection=True))
    def test_grid(self):validate_grid('/'.join(i) for i in itertools.product(PDES,METHODS,VIEWS))
    def test_grid_missing(self):
        with self.assertRaises(ValueError):validate_grid([])
    def test_grid_duplicate(self):
        items=['/'.join(i) for i in itertools.product(PDES,METHODS,VIEWS)];items[-1]=items[0]
        with self.assertRaises(ValueError):validate_grid(items)

if __name__=='__main__':unittest.main()
