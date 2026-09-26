"""Independent public-cut arithmetic and anonymity gate; CPU/file metadata only."""
import argparse,gzip,hashlib,json,math,re,statistics
from collections import defaultdict
from pathlib import Path
from audit_core import PDES,VIEWS,validate_grid
from analyze_new import analyze

def check(root):
    manifest=json.loads((root/'manifest.json').read_text());index=json.loads((root/'index.json').read_text());records=index['records'];validate_grid(records)
    if {p.name for p in root.iterdir()}!=set(manifest['files'])|{'manifest.json'}:raise ValueError('unmanifested or missing result-cut file')
    for name,pin in manifest['files'].items():
        path=root/name
        if path.stat().st_size!=pin['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=pin['sha256']:raise ValueError('public manifest mismatch: '+name)
    suspicious=re.compile(r'(?:[A-Z]:[\\/]|/home/|/gpfs/|(?:ghp_|hf_)[A-Za-z0-9]{12,}|PRIVATE KEY)',re.I)
    for path in root.iterdir():
        if path.suffix in ('.json','.md','.py') and suspicious.search(path.read_text(encoding='utf-8')):raise ValueError('private path/identity/credential-like marker: '+path.name)
    n=0
    for pde in PDES:
        groups=defaultdict(list)
        with gzip.open(root/f'per_identity_{pde}.jsonl.gz','rt') as stream:
            for line in stream:
                if suspicious.search(line):raise ValueError('private marker in per-identity export')
                r=json.loads(line);groups[r['model'],r['test_view']].append(r);n+=1
        if len(groups)!=567:raise ValueError('per-PDE group count')
        for (identity,view),group in groups.items():
            if len(group)!=200 or len({r['identity'] for r in group})!=200:raise ValueError('test identity count')
            b=records[identity]['blocks'][view]
            values=[r['rel_l2_joint'] for r in group]
            if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):raise ValueError('invalid per-record score')
            if not math.isclose(statistics.fmean(values),b['joint']['mean'],rel_tol=1e-12,abs_tol=1e-15):raise ValueError('exported joint mean')
            if not math.isclose(statistics.stdev(values),b['joint']['std'],rel_tol=1e-12,abs_tol=1e-15):raise ValueError('exported joint SD')
            if records[identity]['task']=='rollout':
                for h in range(3):
                    values=[r['rel_l2_by_horizon'][h] for r in group]
                    if not math.isclose(statistics.fmean(values),b['horizons'][h]['mean'],rel_tol=1e-12,abs_tol=1e-15):raise ValueError('exported horizon mean')
                    if not math.isclose(statistics.stdev(values),b['horizons'][h]['std'],rel_tol=1e-12,abs_tol=1e-15):raise ValueError('exported horizon SD')
    if n!=3969*200:raise ValueError('full per-identity coverage')
    saved=json.loads((root/'statistics.json').read_text());fresh=analyze(records)
    if saved!=fresh:raise ValueError('aggregate recomputation mismatch')
    return {'status':'passed','models':441,'blocks':3969,'record_cell_errors':n,'full_array_files_freshly_hashed':index['audit']['fresh_prediction_hashes'],
      'new_inference':0,'old_scores_used_for_current_statistics':False,'anonymity_pattern_gate':'passed; not a guarantee about external deposits',
      'aggregate_recomputation':'exact JSON equality','mean_sd_recomputation':'all cells and all dynamic horizons'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('cut',type=Path);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();result=check(a.cut)
    with a.receipt.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
