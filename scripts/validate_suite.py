"""Run the complete CPU suite in isolated processes and retain every worker's evidence."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from milplus.io import digest_file, digest_object, write_json
from milplus.provenance import software_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--junit-output', default='docs/validation/pytest_cpu.xml')
    parser.add_argument('--timings-junit', action='append', default=[], help='Additional historical timings; never used as current pass evidence')
    parser.add_argument('--keep-passing-tmp', action='store_true', help='Keep large passing fixtures as well as receipts')
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError('workers must be positive')
    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    files = sorted((root/'tests').glob('test_*.py'))
    if not files:
        raise ValueError('No test modules found')
    def suite_identity():
        return digest_object({str(p.relative_to(root)): digest_file(p) for p in sorted((root/'tests').rglob('*.py'))})
    identity, tests_sha = software_identity(), suite_identity()
    def runner_identity():
        return digest_object({name: digest_file(root/'scripts'/name)
                              for name in ['validate_suite.py', 'pytest_receipts.py']})
    runner_sha = runner_identity()
    weights = {}
    previous = root/args.junit_output
    timing_inputs = [previous, *[Path(p).resolve() for p in args.timings_junit]]
    timing_hashes = {}
    for timing in timing_inputs:
        if timing.is_file():
            observed = {}
            for case in ET.parse(timing).getroot().iter('testcase'):
                key = '/'.join(case.attrib['classname'].split('.')[:2])+'.py'
                observed[key] = observed.get(key, 0.)+float(case.attrib.get('time', 0))
            weights.update(observed)
            timing_hashes[str(timing)] = digest_file(timing)
    default = max(weights.values(), default=10.)
    groups = [[] for _ in range(min(args.workers, len(files)))]; costs = [0.]*len(groups)
    for path in sorted(files, key=lambda p: weights.get(str(p.relative_to(root)), default), reverse=True):
        index = min(range(len(groups)), key=costs.__getitem__)
        groups[index].append(path)
        costs[index] += weights.get(str(path.relative_to(root)), default)
    summary = {'state': 'running', 'software': identity, 'test_source_sha256': tests_sha,
               'validation_runner_sha256': runner_sha,
               'historical_timing_inputs': timing_hashes,
               'passing_fixture_retention': 'all' if args.keep_passing_tmp else 'metadata_receipts',
               'test_files': [str(p.relative_to(root)) for p in files], 'workers': []}
    started = time.monotonic(); jobs, handles = [], []
    def interrupted(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    try:
        for index, group in enumerate(groups):
            directory = output/f'worker_{index}'; directory.mkdir()
            command = [sys.executable, '-m', 'pytest', *map(str, sorted(group)),
                       '--basetemp='+str(directory/'tmp'), '--junitxml='+str(directory/'pytest.xml')]
            environment = dict(os.environ)
            if not args.keep_passing_tmp:
                command += ['-p', 'scripts.pytest_receipts', '-o', 'tmp_path_retention_policy=failed']
                environment['MILPLUS_TEST_RECEIPTS'] = str(directory/'receipts')
            log = (directory/'pytest.log').open('w'); handles.append(log)
            job = subprocess.Popen(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            jobs.append(job)
            summary['workers'].append({'pid': job.pid, 'files': [str(p.relative_to(root)) for p in sorted(group)],
                                       'directory': str(directory), 'returncode': None})
        write_json(output/'summary.json', summary)
        for job, record in zip(jobs, summary['workers']):
            record['returncode'] = job.wait()
        suites = ET.Element('testsuites'); counters = {k: 0 for k in ['tests', 'errors', 'failures', 'skipped']}
        for record in summary['workers']:
            for suite in ET.parse(Path(record['directory'])/'pytest.xml').getroot().iter('testsuite'):
                suites.append(suite)
                for key in counters:
                    counters[key] += int(suite.attrib.get(key, 0))
        summary.update(counters, wall_seconds=time.monotonic()-started)
        if software_identity() != identity or suite_identity() != tests_sha or runner_identity() != runner_sha:
            raise ValueError('Runtime or test source changed while verification was running')
        summary['state'] = 'passed' if all(j.returncode == 0 for j in jobs) and not any(counters[k] for k in ['errors', 'failures', 'skipped']) else 'failed'
        ET.ElementTree(suites).write(output/'combined.xml', encoding='utf-8', xml_declaration=True)
        if summary['state'] == 'passed':
            previous.parent.mkdir(parents=True, exist_ok=True)
            previous.write_bytes((output/'combined.xml').read_bytes())
        write_json(output/'summary.json', summary)
        print(json.dumps({k: v for k, v in summary.items() if k != 'workers'}, indent=2))
        if summary['state'] != 'passed':
            raise SystemExit(1)
    except BaseException as error:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
        for job in jobs:
            try:
                job.wait(timeout=10)
            except subprocess.TimeoutExpired:
                job.kill(); job.wait()
        summary.update(state='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=repr(error),
                       wall_seconds=time.monotonic()-started)
        write_json(output/'summary.json', summary)
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == '__main__':
    main()
