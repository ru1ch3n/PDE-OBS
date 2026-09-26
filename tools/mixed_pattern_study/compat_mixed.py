"""Data-free frozen-shape/optimizer check for one mixed-pattern study row; no dataset access, no credit."""
import hashlib, json, os, sys, time
from pathlib import Path
import torch, yaml
import mixed_builder  # noqa: F401  (accepts the study view in the frozen builder)
from pdeobs.one_setting import load_campaign, build_experiment_config
from pdeobs.runner import _method, _training_config
from pdeobs.training import Trainer

pde, method, view, expected, receipt = sys.argv[1:]
HERE = Path(__file__).resolve().parent
repo = Path(os.environ['PDEOBS_REPOSITORY'])
assert os.environ.get('PDEOBS_GPU_NODE_AUTHORIZED') == 'authorized-gpu-node'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1
device = torch.cuda.get_device_name(0)
assert expected.upper() in device.upper(), device
assert expected.upper() == 'V100'
pins = {'configs/campaign/all_pde_one_setting_10method_9x9_v7.yaml':'5fb5aae1a1f098537c286db020759b2578f124fd70df2f7520103c02e403080b', 'configs/method/all_pde_source_faithful_registry_v7.yaml':'7d56a79c9ac7f08252a15e8bf0707c7dc3c630e1a7c9448034ba1b50d1f738da', 'src/pdeobs/training.py':'9d7758ddafa973dddafbeab7c03fdd32b9ca94da771cf6e2f78189340d17cb59', 'src/pdeobs/runner.py':'a393ed8ca3c145af3752bf56127dbedbef9e74fb6fadf6b27dbb5f2aadedf578'}
for name, sha in pins.items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == sha
row = json.loads((HERE/'plan.json').read_text())['rows'][0]
assert row['identity'] == '/'.join((pde, method, view)) and mixed_builder.accepts(view)
campaign = load_campaign(repo/'configs/campaign/all_pde_one_setting_10method_9x9_v7.yaml')
registry = yaml.safe_load((repo/'configs/method/all_pde_source_faithful_registry_v7.yaml').read_text())
cfg = build_experiment_config(campaign, registry, dataset_root=os.environ['PDEOBS_DATA'], pde=pde, method=method, view=view, epochs=500)
assert mixed_builder.KIND != 'mixture' or cfg['data']['mask'] == {'protocol': mixed_builder.SPEC}
if row['protocol'] == 'pdeobs.max200-min120-then-train-data-patience10/20260920':
    from budget200 import derive_config
    cfg = derive_config(cfg)
else:
    assert row['protocol'] == 'pdeobs.fixed500-last-checkpoint/v7'
batch_size = cfg['training']['batch_size']
torch.manual_seed(7319)
model = _method(cfg)
trainer = Trainer(model, _training_config(cfg, Path(receipt).parent/'diagnostic-only-no-study-credit', None))
dynamic = trainer.config.task == 'rollout'
shape = (batch_size, trainer.config.horizon + trainer.config.rollout_target_offset, 1, 64, 64) if dynamic else (batch_size,1,64,64)
target = torch.randn(shape)*0.1
mask = (torch.rand((batch_size,1,64,64)) > 0.5).float()
observed = (target[:,0] if dynamic else target)*mask
metadata = {'pde':pde, 'boundary':'periodic' if dynamic else 'dirichlet', 'stored_time_values':[0.0,0.1,0.2,0.3], 'parameters':{'domain_id':'unit_square_cell_centered_v1','forcing_id':'unit_square_sine_mix_v1','forcing_amplitude':0.1,'diffusivity':0.01,'reaction_rate':1.0,'viscosity':0.01,'wavenumber':1.0}}
batch = {'observations':observed,'mask':mask,'target':target,'geometry':torch.zeros((batch_size,1,64,64)),'metadata':[metadata.copy() for _ in range(batch_size)],'pde_condition':torch.ones((batch_size,1,64,64))}
started=time.time()
losses=[trainer.run_epoch([batch], training=True) for _ in range(5)]
torch.cuda.synchronize()
assert all(torch.isfinite(torch.tensor(losses))) and not trainer.health_events
assert all(torch.isfinite(parameter).all().item() for parameter in model.parameters())
record={'schema':'pdeobs.synthetic-first-family-compatibility/v1+mixed-study','identity':'/'.join((pde,method,view)),'protocol':row['protocol'],'device':device,'torch':str(torch.__version__),'frozen_batch':batch_size,'synthetic_updates':5,'target_shape':list(shape),'horizon':trainer.config.horizon,'teacher_forcing':trainer.config.teacher_forcing_ratio,'physics_loss':trainer.config.physics_loss,'losses':losses,'seconds':time.time()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'events':trainer.health_events,'dataset_records_touched':0,'test_records_touched':0,'training_credit':0,'evaluation_credit':0,'passed':True,'scope':'shape/optimizer/runtime only, not performance or acceptance'}
with open(receipt,'x') as output:
    json.dump(record,output,sort_keys=True);output.write('\n')
print(json.dumps(record),flush=True)
