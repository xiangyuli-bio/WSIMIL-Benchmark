"""Actual Git verification remains strict without refreshing the source index."""
import os
from pathlib import Path
import subprocess

import pytest

from milplus import upstream


def repository(tmp_path, monkeypatch):
    root = tmp_path / 'sources' / 'test_source'
    root.mkdir(parents=True)
    def git(*args):
        return subprocess.check_output(['git', *args], cwd=root, text=True).strip()
    git('init', '-q')
    git('config', 'user.name', 'Read-only source verifier test')
    git('config', 'user.email', 'test@example.invalid')
    (root / 'model.py').write_text('VALUE = 1\n')
    git('add', 'model.py')
    git('commit', '-q', '-m', 'Fixed source test')
    commit = git('rev-parse', 'HEAD')
    monkeypatch.setenv('MILPLUS_SOURCE_ROOT', str(root.parent))
    monkeypatch.setattr(upstream, 'resource', lambda name: {'test_source': {'commit': commit}})
    return root, commit, git


def test_clean_timestamp_drift_and_existing_lock_do_not_modify_any_index_bytes(tmp_path, monkeypatch):
    root, commit, git = repository(tmp_path, monkeypatch)
    index = root / '.git/index'
    before = index.read_bytes(), index.stat().st_mtime_ns
    target = root / 'model.py'
    stat = target.stat()
    os.utime(target, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2000000000))
    assert upstream.verified_source('test_source') == (root, commit)
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before
    lock = root / '.git/index.lock'
    lock.write_bytes(b'existing_unrelated_lock_must_remain_unchanged')
    assert upstream.verified_source('test_source') == (root, commit)
    assert lock.read_bytes() == b'existing_unrelated_lock_must_remain_unchanged'
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before


@pytest.mark.parametrize('staged', [False, True])
def test_real_tracked_modifications_still_rejected_without_writing_the_index(tmp_path, monkeypatch, staged):
    root, commit, git = repository(tmp_path, monkeypatch)
    (root / 'model.py').write_text('VALUE = 2\n')
    if staged:
        git('add', 'model.py')
    index = root / '.git/index'
    before = index.read_bytes(), index.stat().st_mtime_ns
    with pytest.raises(ValueError, match='tracked local modifications'):
        upstream.verified_source('test_source')
    assert (index.read_bytes(), index.stat().st_mtime_ns) == before


def test_different_HEAD_still_rejected(tmp_path, monkeypatch):
    root, commit, git = repository(tmp_path, monkeypatch)
    (root / 'extra.txt').write_text('new commit\n')
    git('add', 'extra.txt')
    git('commit', '-q', '-m', 'Different commit')
    with pytest.raises(ValueError, match='version mismatch'):
        upstream.verified_source('test_source')
