#!/usr/bin/env bash
# Fresh node: venv + install v0.2.0, essentials test, then the 14-tier coverage sweep (nohup-friendly).
set -u
W=/workspace/pdeobs/sweep-run
mkdir -p $W/{logs,work,qa_full,dist}; cd $W
LOG=logs/coverage.log; : > $LOG
echo "=== coverage start $(date -u)" | tee -a $LOG
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | tee -a $LOG
python -m venv --system-site-packages envs/v020 >>$LOG 2>&1
V=envs/v020/bin/python
$V -m pip install -q --upgrade pip >>$LOG 2>&1
$V -m pip install -q "numpy>=1.24,<2.3" "scipy>=1.12" "h5py>=3.8" "PyYAML>=6.0" "pytest>=7.4" pytest-cov >>$LOG 2>&1
rm -rf work/exact && mkdir -p work/exact
$V -c "import zipfile;zipfile.ZipFile('dist/PDE_OBS_v020_code.zip').extractall('work/exact')"
$V -m pip install -q work/exact/PDE_OBS_code >>$LOG 2>&1
$V -c "import pdeobs,pdeobs.api as a,torch;print('pdeobs',pdeobs.__version__,'torch',torch.__version__,'cuda',torch.cuda.is_available())" | tee -a $LOG
sha256sum dist/PDE_OBS_v020_code.zip | tee -a $LOG
export PATH="$W/envs/v020/bin:$PATH"
echo "=== [E] benchmark essentials" | tee -a $LOG
$V -m pytest -q -p no:cacheprovider work/exact/PDE_OBS_code/tests/test_benchmark_essentials.py --basetemp work/ess-temp > qa_full/essentials.txt 2>&1
echo "essentials_rc=$?  $(tail -1 qa_full/essentials.txt)" | tee -a $LOG
echo "=== [C] 14-tier coverage sweep" | tee -a $LOG
$V remote_full_coverage.py $W >>$LOG 2>&1
echo "=== coverage DONE $(date -u)" | tee -a $LOG
touch qa_full/SWEEP_DONE
