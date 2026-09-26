from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pdeobs.download import DownloadError, download_release


def test_missing_manifest_never_contacts_an_author_endpoint(monkeypatch, capsys, tmp_path):
    from pdeobs.cli import main
    from pdeobs.download import DEFAULT_RELEASE_MANIFEST_URL, load_release_manifest
    import urllib.request

    def forbidden(*args, **kwargs):
        pytest.fail("missing manifest attempted network access")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    assert DEFAULT_RELEASE_MANIFEST_URL is None
    with pytest.raises(DownloadError, match="explicit"):
        load_release_manifest(None)
    destination = tmp_path / "must-not-be-created"
    status = main(["download", "--tier", "tiny", "--output", str(destination)])
    assert status == 2
    assert "--manifest" in capsys.readouterr().err
    assert not destination.exists()


def test_local_manifest_download_is_verified_and_resumable(tmp_path: Path) -> None:
    source = tmp_path / "release"
    source.mkdir()
    payload = b"small deterministic fixture\n"
    (source / "fixture.bin").write_bytes(payload)
    manifest = {
        "schema_version": 1,
        "name": "fixture-release",
        "tiers": ["tiny"],
        "files": [
            {
                "path": "fixture.bin",
                "tier": "tiny",
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size": len(payload),
            }
        ],
    }
    manifest_path = source / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output = tmp_path / "downloaded"

    first = download_release(manifest_path, output, "tiny")
    second = download_release(manifest_path, output, "tiny")

    assert first == second == [output.resolve() / "fixture.bin"]
    assert first[0].read_bytes() == payload


def test_release_manifest_schema_is_enforced(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.json"
    manifest.write_text(json.dumps({"tiers": ["tiny"], "files": []}), encoding="utf-8")
    with pytest.raises(DownloadError, match="schema_version"):
        download_release(manifest, tmp_path / "output", "tiny")


def test_gated_release_manifest_fails_closed(tmp_path: Path) -> None:
    manifest = tmp_path / "gated.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "gated-release",
                "status": "gated",
                "gating_reason": "archive URL and checksum are not published",
                "tiers": ["tiny"],
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    output = tmp_path / "output"
    with pytest.raises(DownloadError, match="not published"):
        download_release(manifest, output, "tiny")
    assert not output.exists()


def test_published_release_manifest_cannot_be_empty(tmp_path: Path) -> None:
    manifest = tmp_path / "empty-published.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "empty-release",
                "status": "published",
                "tiers": ["tiny"],
                "files": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(DownloadError, match="files must be non-empty"):
        download_release(manifest, tmp_path / "output", "tiny")
