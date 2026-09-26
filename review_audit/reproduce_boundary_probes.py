import json, tempfile, numpy as np
from pathlib import Path
from pdeobs import api
from pdeobs.api.infer import PredictionBundle
rows=[]
def record(name, fn):
    try: rows.append({'case':name,'result':fn()})
    except Exception as e: rows.append({'case':name,'exception':type(e).__name__,'message':str(e)})
target=np.stack([np.ones((4,4,1)),np.ones((4,4,1))*2]).astype(np.float32)
mask=np.ones_like(target)
inputs=api.InferenceInput(target.copy(),mask.copy(),sample_ids=['a','b'])
bundle=PredictionBundle(target.copy(),['a','b'],[0],'recovery',inputs=inputs)
record('canonical_target_order',lambda:api.evaluate(bundle,target,target_ids=['a','b'])['status'])
record('reordered_targets_correct_ids',lambda:api.evaluate(bundle,target[::-1],target_ids=['b','a']))
record('numpy_target_ids',lambda:api.evaluate(bundle,target,target_ids=np.array(['a','b']))['status'])
record('float_target_time_indices',lambda:api.evaluate(bundle,target,target_ids=['a','b'],target_time_indices=[0.9])['status'])
for value in [float('nan'),float('inf'),-1.0,0.5,2.0]:
    def probe(v=value):
        raw=mask.copy();raw.flat[0]=v
        inp=api.InferenceInput(target.copy(),raw,sample_ids=['a','b'])
        b=PredictionBundle(target.copy(),['a','b'],[0],'recovery',inputs=inp)
        rep=api.evaluate(b,target,target_ids=['a','b'])
        return {'converted_mask':float(inp.mask.flat[0]),'score_status':rep['status']}
    record('raw_mask_'+str(value),probe)
with tempfile.TemporaryDirectory(prefix='pdeobs-probe-') as d:
    cfg={'out':d,'stages':['trian']}
    record('unknown_stage_validates',lambda:api.validate_pipeline(cfg))
    record('unknown_stage_runs',lambda:api.run(cfg))
print(json.dumps(rows,indent=2,default=str))
