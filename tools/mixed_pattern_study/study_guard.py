"""Execution gate for one study row (mixture or seed subset): pins, overlay manifest, authorization, single ownership."""
import fcntl,hashlib,json,os,pathlib,subprocess
HERE=pathlib.Path(__file__).resolve().parent
def H(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(pathlib.Path(p).read_text())
def study_view_ok(a,view):
 if a.get('kind','mixture')=='mixture':return view==a['training_view']
 sv=(a.get('seed_views') or {}).get(view);return bool(sv) and view==sv['base_view']+'@seed'+str(int(sv['training_seed'])) and sv['base_view'] in a['paper_views']
def verify(p):
 auth=HERE/'study-authorization.json';assert H(auth)==p['authorization_sha256']
 a=J(auth);assert a['study_id']==p['study_id']==p['wave'] and a['host']==p['host']=='<gpu-node>' and a['other_clusters_forbidden'] is True
 for n,d in p['bundle_pins'].items():assert H(HERE/n)==d,n
 for n,d in p['source_pins'].items():assert H(n)==d,n
 assert H(p['overlay_manifest'])==p['overlay_manifest_sha256']
 man=J(p['overlay_manifest'])
 assert man['production_repository']==p['repository'] and man['commit']==p['repository_commit'] and man['overlay_src']==p['overlay_src']
 for rel,item in man['files'].items():assert H(pathlib.Path(p['overlay_src'])/'pdeobs'/rel)==item['sha256'],rel
 assert set(man['changed'])=={'dataset.py','mask_specs.py'}
 assert man['files']['dataset.py']['sha256']==a['release_dataset_sha256'] and man['files']['mask_specs.py']['sha256']==a['release_mask_specs_sha256']
 src='production-'+os.environ.get('PDEOBS_PRODUCTION_COMMIT_LABEL','<COMMIT-1>')  # the production commit id is not published; set the label on the production host
 assert man['files']['masks.py']['source']==src and man['files']['evaluation.py']['source']==src
 assert subprocess.check_output(['git','-C',p['repository'],'rev-parse','HEAD'],universal_newlines=True).strip()==p['repository_commit']
 assert not subprocess.check_output(['git','-C',p['repository'],'status','--porcelain','--untracked-files=all'],universal_newlines=True).strip()
 assert len(p['rows'])==1
 for r in p['rows']:
  pde,method,view=r['identity'].split('/')
  assert study_view_ok(a,view) and view not in a['paper_views'] and r['mode']=='new'
  assert r['protocol'] in a['protocols'] and r['epochs']==a['protocols'][r['protocol']]['epochs']
  assert r['training_gate'].startswith(p['training_gate_root']+'/') and r['output'].startswith(p['training_root']+'/')
  assert r['identity_sha256']==hashlib.sha256(r['identity'].replace('/','|').encode()).hexdigest()
  assert any(x['identity']==r['identity'] and x['protocol']==r['protocol'] for x in a['rows'])
def free(p):
 return all(not pathlib.Path(r['training_gate']).exists() and not any(pathlib.Path(r['output']+s).exists() for s in ('','.lock','.authorized-owner')) for r in p['rows'])
def precheck(p):verify(p);return free(p)
def claim_all(p,job):
 verify(p);root=pathlib.Path(p['training_gate_root']);root.mkdir(parents=True,exist_ok=True)
 with (root/'.cohort-claim.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if not free(p):return dict(acquired=False,reason='owned_or_completed_same_filesystem_alias',dataset_records_touched=0,cuda_access=False)
  made=[]
  try:
   for r in sorted(p['rows'],key=lambda r:r['identity_sha256']):
    gate=pathlib.Path(r['training_gate']);gate.parent.mkdir(parents=True,exist_ok=True);gate.mkdir()
    token=r['training_owner']+'|'+job
    with (gate/'.pack-owner').open('x') as f:f.write(token)
    made.append((gate,token))
  except FileExistsError:
   for gate,token in made:
    assert (gate/'.pack-owner').read_text()==token and not (gate/'.started').exists()
    (gate/'.pack-owner').unlink();gate.rmdir()
   return dict(acquired=False,reason='alias_race',dataset_records_touched=0,cuda_access=False)
  return dict(acquired=True,identities=[r['identity'] for r in p['rows']],claims_persist_after_start=True)
