"""Preserve numerical test receipts before pytest reclaims successful fixtures."""
import hashlib
import os
import shutil
from pathlib import Path

import pytest


def pytest_configure(config):
    destination = os.environ.get('MILPLUS_TEST_RECEIPTS')
    config._milplus_receipts = None
    if destination:
        path = Path(destination).resolve()
        path.mkdir(parents=True, exist_ok=False)
        config._milplus_receipts = path


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    root = getattr(item.config, '_milplus_receipts', None)
    source = item.funcargs.get('tmp_path')
    if root is None or report.when != 'call' or not report.passed or source is None:
        return
    source = Path(source)
    target = root/hashlib.sha256(item.nodeid.encode()).hexdigest()[:24]
    for path in source.rglob('*'):
        if path.is_symlink() or not path.is_file() or path.suffix not in {'.json', '.yaml', '.yml', '.csv', '.xml', '.log'}:
            continue
        destination = target/path.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(destination.read_bytes()).digest():
            raise ValueError('Numerical receipt copy differs before temporary fixture cleanup')
