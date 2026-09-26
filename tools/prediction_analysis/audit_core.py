"""Terminal metadata audit, independent of the frozen GPU verifier.

No model loading, inference, training or scientific gate changes. File I/O is
implemented separately; these functions are pure and tested with metadata only.
"""
import hashlib
import itertools
import json
import math
import statistics

PDES = ['darcy','poisson','helmholtz','heat','reaction_diffusion','burgers','navier_stokes']
METHODS = ['ufno_2d','fno','cno','deeponet','gnot','transolver','pino']
VIEWS = ['random_50pct','horizontal_lines_50pct','vertical_lines_50pct','clustered_50pct','block_observed_50pct','line_sensors_50pct','boundary_band_50pct','random_65pct','random_80pct']
PLAN_SHA = 'd138e5289e503f0b2ecf389ee6cf16e6d69695596b768f9d0013374db93104c4'
VERSION = 'pdeobs-terminal-prediction-audit/20260925-v1'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def ids_hash(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()

def real(value):
    require(type(value) in (int,float) and math.isfinite(value), 'nonfinite or non-real number')
    return float(value)

def close(a,b):
    return math.isclose(real(a),real(b),rel_tol=1e-12,abs_tol=1e-15)

def summary(values):
    values=[real(x) for x in values]
    require(len(values)==200 and all(x>=0 for x in values), 'incomplete/negative score population')
    return {'mean':statistics.fmean(values),'std':statistics.stdev(values),'n':200,'ddof':1}

def validate_grid(identities):
    values=list(identities)
    expected={'/'.join(x) for x in itertools.product(PDES,METHODS,VIEWS)}
    require(len(values)==441 and len(set(values))==441 and set(values)==expected,'incomplete or duplicate 441 identity grid')

def validate_frames(mapping,ids,task):
    require(set(mapping)==set(ids),'frame identity set')
    positions=[0] if task=='recovery' else [1,2,3]
    for identity in ids:
        f=mapping[identity]
        require(f['target_stored_positions']==positions,'stored frame positions')
        frames=f['target_source_frame_indices']
        require(len(frames)==len(positions) and all(type(x) is int and x>=0 for x in frames),'source frames')
        require(all(b>a for a,b in zip(frames,frames[1:])),'source frame order')
        times=f['target_physical_time_values']
        if task=='rollout':
            require(isinstance(times,list) and len(times)==3,'missing physical time')
            times=[real(x) for x in times]
            require(all(b>a for a,b in zip(times,times[1:])),'physical time order')
        elif times is not None:
            require(len(times)==1,'static time shape');real(times[0])

def released_checkpoint(released):
    """Published checkpoint digest of a release row (manifest v3 `checkpoint_sha256`, v2 `checkpoint_anonymized_sha256`)."""
    return released['checkpoint_sha256'] if 'checkpoint_sha256' in released else released['checkpoint_anonymized_sha256']

def audit_block(row,released,prov,contract,score,comparison,complete,file_hashes,mask,source_frames):
    identity=row['identity'];view=complete['test_view'];task=row['task']
    require(task in ('recovery','rollout') and view in VIEWS,'task/view')
    require(complete['status']=='valid' and complete['samples']==200,'block incomplete')
    require(complete['score_sha256']==file_hashes['score.json'],'score file hash mismatch')
    require(complete['predictions_sha256']==file_hashes['predictions.h5'],'prediction file hash mismatch')
    require(complete['comparison']==comparison,'completion comparison mismatch')
    bindings={'identity':identity,'observation_id':view,'task':task,'split':'paper-test',
      'checkpoint_id':released_checkpoint(released),'original_checkpoint_id':row['checkpoint_id'],
      'dataset_manifest_sha256':canonical(prov['dataset_manifest']),'historical_split_sha256':canonical(prov['split']),
      'training_cohort':row['training_cohort'],'actual_epochs':row['actual_epochs'],'mask_config':mask,
      'projection':False,'schema_version':'pdeobs-strict-contract-v1',
      'time_index_semantics':'stored_frame_positions; per_identity_source_mapping_bound',
      'verification_adapter':'pdeobs-archived-prediction-verification-v1'}
    for k,v in bindings.items():require(contract.get(k)==v,'contract binding: '+k)
    ids=contract['expected_ids']
    require(len(ids)==200 and len(set(ids))==200 and all(isinstance(i,str) and i for i in ids),'200 unique identities')
    require(ids_hash(ids)==row['split']['test_sample_ids_sha256'],'historical test identity set mismatch')
    require(set(ids)==set(source_frames),'test identities differ from pinned source split')
    validate_frames(contract['frame_mapping_by_identity'],ids,task)
    require(contract['frame_mapping_by_identity']==source_frames,'source physical time mapping mismatch')
    require(contract['target_time_indices']==([0] if task=='recovery' else [1,2,3]),'target time indices')
    require(contract['expected_shape']==([128,128,1] if task=='recovery' else [3,128,128,1]),'shape')
    require(score.get('status')=='valid' and score.get('errors')==[],'invalid strict score')
    for k in ('expected_identity_count','actual_prediction_identity_count','actual_target_identity_count','scored_identity_count'):
        require(score.get(k)==200,'score count: '+k)
    require(score.get('raw_arrays_checked') is True and score.get('projection_applied') is False,'raw/projection state')
    require(score.get('artifact_contract_binding')=='pdeobs-strict-inference-v1_sha256_bound','unbound artifact')
    require(score.get('scoring_version')=='pdeobs-strict-v1' and score.get('epsilon')==1e-12,'scorer version/epsilon')
    require(score['config']==contract and score['config_sha256']==canonical(contract),'score contract hash')
    require(score['identity_set_sha256']==canonical(sorted(ids)),'score identity set hash')
    require(score['expected_identity_order']==ids,'score identity order')
    for k in ('checkpoint_id','observation_id','task'):require(score[k]==contract[k],'score identity: '+k)
    values=score['per_identity']
    require(len(values)==200 and [r['identity'] for r in values]==ids,'per-record identity order')
    require(score['summary']['identity_count']==200,'summary count')
    result=summary([r['rel_l2_joint'] for r in values])
    require(close(result['mean'],score['summary']['rel_l2_joint_mean']),'per-record joint mean mismatch')
    horizons=[]
    if task=='rollout':
        recorded=score['summary']['rel_l2_by_horizon_mean']
        require(len(recorded)==3,'horizon summary coverage')
        for h in range(3):
            errors=[]
            for r in values:
                require(len(r['rel_l2_by_horizon'])==3,'per-record horizon coverage')
                point=r['rel_l2_by_horizon'][h]
                require(point['horizon']==h+1 and point['target_time_index']==h+1,'per-record horizon mapping')
                errors.append(point['rel_l2'])
            s=summary(errors)
            require(recorded[h]['horizon']==h+1 and recorded[h]['target_time_index']==h+1,'summary horizon mapping')
            require(close(s['mean'],recorded[h]['rel_l2_mean']),'per-record horizon mean mismatch')
            horizons.append({'horizon':h+1,**s})
    old=real(row['blocks'][view]['relative_l2']);new=real(score['summary']['rel_l2_joint_mean'])
    result['mean']=new  # Keep the authoritative frozen-scorer float after redundant arithmetic validation.
    require(comparison['strict_status']=='valid' and close(comparison['archived_relative_l2'],old) and close(comparison['strict_relative_l2'],new),'comparison scores')
    require(comparison['tolerance']=={'atol':1e-7,'rtol':1e-4},'historical tolerance changed')
    relative=abs(new-old)/max(abs(old),1e-12)
    require(close(relative,comparison['relative_difference']) and close(abs(new-old),comparison['absolute_difference']),'comparison arithmetic')
    strict=abs(new-old)<=1e-7+1e-4*abs(old)
    require(comparison['within_reporting_tolerance']==strict,'strict comparison flag')
    category='old_strict' if strict else 'minor_le_1pct' if relative<=.01 else 'difference_gt_1pct'
    return {'test_view':view,'joint':result,'horizons':horizons,'per_identity':values,
            'comparison':comparison,'comparison_category':category,'file_hashes':file_hashes,
            'contract_sha256':canonical(contract),'seconds':complete['seconds']}

def audit_row_metadata(row,released,prov,completion,plan_dataset):
    require(prov['archived_row']==row and prov['released_file']==released,'provenance differs from frozen plan')
    require(prov['dataset_manifest']==plan_dataset,'dataset manifest mismatch')
    require(row['checkpoint_id']==prov['original_checkpoint_id']==released.get('checkpoint_original_sha256',row['checkpoint_id']),'original checkpoint mapping')
    require(released_checkpoint(released)==prov['actual_checkpoint_id'],'released checkpoint mapping')
    for k,v in row['split'].items():
        if k!='schema_version':require(prov['split'].get(k)==v,'split field mismatch: '+k)
    require(prov['mask_seed']==row['split']['seed'],'mask seed')
    require(completion['identity']==row['identity'] and completion['status']=='complete','row not complete')
    require(completion.get('optimizer_updates')==0 and completion.get('historical_completion_credit_delta')==0,'runtime scope')
    blocks=completion['blocks']
    require(len(blocks)==9 and {b['test_view'] for b in blocks}==set(VIEWS),'row nine unique views')
    return {b['test_view']:b for b in blocks}
