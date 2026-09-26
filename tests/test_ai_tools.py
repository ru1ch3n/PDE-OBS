"""Bounded helper behavior; never launches the optional training smoke from tests."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / 'tools' / 'ai_assist.py'


def test_ai_doctor_readonly_protected_sources():
    run = subprocess.run([sys.executable, str(TOOL), 'doctor', '--json'], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0, run.stderr
    report = json.loads(run.stdout)
    assert report['changed_protected_files'] == []
    assert report['missing_indexed_files'] == []
    assert report['protected_scientific_files_checked'] > 100


def test_context_is_bounded_and_reports_truncation():
    run = subprocess.run([sys.executable, str(TOOL), 'context', '--task', 'scoring', '--max-chars', '1000'], capture_output=True, text=True, timeout=15)
    assert run.returncode == 0
    assert len(run.stdout) < 1300
    assert 'TRUNCATED' in run.stdout and 'sha256=' in run.stdout


def test_smoke_refuses_existing_output_before_loading_model(tmp_path):
    marker = tmp_path / 'keep.txt'; marker.write_text('do not delete')
    run = subprocess.run([sys.executable, str(TOOL), 'smoke', '--out', str(tmp_path)], capture_output=True, text=True, timeout=15)
    assert run.returncode == 2
    assert marker.read_text() == 'do not delete'
    assert not (tmp_path / 'ai_smoke_receipt.json').exists()


def test_smoke_refuses_writing_into_source_tree():
    output = ROOT / '_forbidden_ai_smoke_output'
    assert not output.exists()
    run = subprocess.run([sys.executable, str(TOOL), 'smoke', '--out', str(output)], capture_output=True, text=True, timeout=15)
    assert run.returncode == 2
    assert not output.exists()
