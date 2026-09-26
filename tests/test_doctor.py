from __future__ import annotations

from pdeobs.doctor import checks_succeeded, format_checks, run_doctor
import pytest


def test_local_doctor_returns_readable_checks() -> None:
    checks = run_doctor(cluster="local")
    rendered = format_checks(checks)
    assert "Python >= 3.10" in rendered
    assert isinstance(checks_succeeded(checks), bool)


def test_generic_slurm_doctor_uses_declared_roots_and_allocation(monkeypatch, tmp_path):
    monkeypatch.setenv('PDEOBS_DATA', str(tmp_path / 'data'))
    monkeypatch.setenv('PDEOBS_RUNS', str(tmp_path / 'runs'))
    monkeypatch.setenv('SLURM_JOB_ID', 'synthetic-allocation')
    monkeypatch.setattr('pdeobs.doctor.shutil.which', lambda name: '/scheduler/' + name)
    checks = run_doctor(cluster='slurm')
    assert checks_succeeded(checks)
    assert any(c.name == 'not on login node' and c.ok for c in checks)
    # User-provided roots are allowed on any filesystem. Assert that the
    # implementation uses those roots, not an assumed workstation/site path.
    by_name = {check.name: check for check in checks}
    for label, directory in (('data root', 'data'), ('run root', 'runs')):
        assert by_name[label].detail == str((tmp_path / directory).resolve())
        assert by_name[label].ok and by_name[label].required
        assert (tmp_path / directory).is_dir()
    assert not list(tmp_path.rglob('.pdeobs-write-test'))


def test_doctor_rejects_undeclared_cluster():
    with pytest.raises(ValueError, match='local or slurm'):
        run_doctor(cluster='undeclared')
