"""Mixed-pattern study worker on the GPU node: one row, one GPU (UUID flock), same-job nine-view EVA, no preemption."""
import fcntl,hashlib,importlib.util,json,os,pathlib,socket,subprocess,sys,time,traceback
HERE=pathlib.Path(__file__).resolve().parent;BASE=pathlib.Path('/path/to/pdeobs')
def H(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def J(p):return json.loads(pathlib.Path(p).read_text())
def rec(name,d):
 p=HERE/'receipts'/str(os.getpid())/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**d),f,indent=2)
def load(name):
 s=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def old_eva_waiting():
 for f in (BASE/'_staging').glob('*/rows/*/evaluation-device-wait.json'):
  d=J(f);pid=d.get('pid')
  if pid and pathlib.Path('/proc/'+str(pid)).exists():return True
 return False
def active_new_eva():
 for f in (BASE/'_runtime-gates/budget200-single-eva-waiters').glob('*.json'):
  if pathlib.Path('/proc/'+str(J(f)['pid'])).exists():return True
 return False
def acquire_gpu(p,eva=False):
 root=BASE/'_runtime-gates/authorized-the GPU node-evaluation-gpus';root.mkdir(parents=True,exist_ok=True)
 deadline=time.monotonic()+72*3600
 while True:
  if time.monotonic()>deadline:raise RuntimeError('bounded resource wait exhausted; no automatic retry')
  if old_eva_waiting() or (not eva and active_new_eva()):time.sleep(10);continue
  for index,uuid in p['gpus'].items():
   lock=(root/(uuid+'.lock')).open('a+')
   try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:lock.close();continue
   apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True)
   used=int(subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
   avail=int(next(x for x in pathlib.Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])
   if uuid in apps or used or avail<32*1024*1024 or old_eva_waiting() or (not eva and active_new_eva()):
    lock.close();continue
   return lock,int(index),uuid
  time.sleep(10)
def env_for(p):
 return dict(os.environ,PDEOBS_GPU_NODE_AUTHORIZED='authorized-gpu-node',PDEOBS_REPOSITORY=p['repository'],PDEOBS_DATA=p['dataset'],
  PYTHONPATH=p['overlay_src']+':'+str(HERE)+':'+p['repository']+'/src:'+p['site_packages'],
  OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',OPENBLAS_NUM_THREADS='6',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',
  CUBLAS_WORKSPACE_CONFIG=':4096:8',HDF5_USE_FILE_LOCKING='FALSE',PYTHONUNBUFFERED='1')
def main(p):
 assert socket.gethostname() in ('<gpu-node>',) and not os.environ.get('SLURM_JOB_ID')
 guard=load('study_guard');guard.verify(p)
 if not guard.precheck(p):rec('skip.json',dict(reason='already_claimed',cuda_access=False));return 0
 r=p['rows'][0]
 rec('waiting.json',dict(identity=r['identity'],protocol=r['protocol'],pid=os.getpid(),no_preemption=True))
 gpu,index,uuid=acquire_gpu(p)
 try:
  job=str(os.getpid());claim=guard.claim_all(p,job);rec('claim.json',claim)
  if not claim['acquired']:return 0
  gate=pathlib.Path(r['training_gate']);token=r['training_owner']+'|'+job
  with (gate/'.started').open('x') as f:f.write(token)
  # Byte-level pins do not open held-out sample records.
  for item in J(HERE/'required-data.json'):
   f=pathlib.Path(p['dataset'])/item['path'];assert f.stat().st_size==item['bytes'] and H(f)==item['sha256']
  env=env_for(p);env['CUDA_VISIBLE_DEVICES']=uuid
  rec('device.json',dict(gpu=index,uuid=uuid,pid=os.getpid(),identity=r['identity'],protocol=r['protocol']))
  comp=HERE/'receipts'/job/'compatibility.json'
  rc=subprocess.call([p['python'],'-B',str(HERE/'compat_mixed.py'),*r['identity'].split('/'),'V100',str(comp)],env=env)
  if rc:rec('compatibility-failure.json',dict(returncode=rc));return rc
  out=pathlib.Path(r['output']);out.parent.mkdir(parents=True,exist_ok=True)
  with pathlib.Path(str(out)+'.authorized-owner').open('x') as f:f.write(token)
  with pathlib.Path(str(out)+'.lock').open('x') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   log=(HERE/'logs'/('training-'+job+'.log')).open('xb')
   proc=subprocess.Popen([p['python'],'-B',str(HERE/'mixed_train.py'),'run','0'],env=env,stdout=log,stderr=subprocess.STDOUT)
   rec('training-start.json',dict(child_pid=proc.pid,identity=r['identity'],gpu=index,uuid=uuid,log=log.name))
   rc=proc.wait();log.close()
  rec('training-exit.json',dict(returncode=rc,completion_exists=(out/'completion.json').exists()))
 finally:gpu.close()
 if rc:return rc
 # Pre-test eligibility must not reserve a GPU for rejected training.
 try:load('mixed_eva').metadata(p,r,HERE/'receipts'/job/'pre-eva-eligibility.json')
 except Exception as e:
  rec('evaluation-pending-or-rejected.json',dict(error=repr(e),traceback=traceback.format_exc(),training_success=True,test_records_touched=0));return 0
 waitroot=BASE/'_runtime-gates/budget200-single-eva-waiters';waitroot.mkdir(parents=True,exist_ok=True)
 with (waitroot/(job+'.json')).open('x') as f:json.dump(dict(pid=os.getpid(),identity=r['identity']),f)
 with (BASE/'_runtime-gates/budget200-single-eva.lock').open('a+') as evlock:
  fcntl.flock(evlock,fcntl.LOCK_EX)
  gp,index,uuid=acquire_gpu(p,eva=True)
  try:
   env['CUDA_VISIBLE_DEVICES']=uuid;env['PDEOBS_EVA_PARENT']=job;env['PDEOBS_EVA_GPU_UUID']=uuid
   log=(HERE/'logs'/('evaluation-'+job+'.log')).open('xb')
   proc=subprocess.Popen([p['python'],'-B',str(HERE/'study_worker.py'),'eval'],env=env,stdout=log,stderr=subprocess.STDOUT)
   rec('evaluation-start.json',dict(child_pid=proc.pid,gpu=index,uuid=uuid,log=log.name));rc=proc.wait();log.close()
   rec('evaluation-exit.json',dict(returncode=rc))
   return 0  # EVA status is separately recorded; do not erase training success.
  finally:gp.close()
def evaluate(p):
 assert os.environ['PDEOBS_GPU_NODE_AUTHORIZED']=='authorized-gpu-node'
 parent=int(os.environ['PDEOBS_EVA_PARENT']);assert os.getppid()==parent
 uuid=os.environ['PDEOBS_EVA_GPU_UUID'];assert uuid in p['gpus'].values() and os.environ['CUDA_VISIBLE_DEVICES']==uuid
 eva=load('mixed_eva')
 os.environ['PDEOBS_RECEIPT_JOB']='the GPU node-'+str(parent)
 eva.allocation=lambda _:dict(type='authorized_the GPU node',node=socket.gethostname(),gpu_uuid=uuid,parent_pid=parent)
 return eva.evaluate(p,0)
if __name__=='__main__':
 try:raise SystemExit(evaluate(J(HERE/'plan.json')) if len(sys.argv)>1 and sys.argv[1]=='eval' else main(J(HERE/'plan.json')))
 except Exception as e:rec('failure.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
