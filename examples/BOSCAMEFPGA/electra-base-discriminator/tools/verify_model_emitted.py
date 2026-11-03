#!/usr/bin/env python3
"""Strict local comparison of downloaded full-model outputs to emitted-SSA execution."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from build_model_host import sha, ROOT

# Declared independently of observed hardware results. No widening after failure.
ATOL, RTOL = 1e-5, 1e-5


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--reference', type=Path, default=ROOT / 'build/model-host-emitted')
    a = p.parse_args()
    manifest = json.loads((a.run / 'artifacts/manifest.json').read_text())
    task = json.loads((a.run / 'task.json').read_text())
    build = json.loads((a.run / 'build-manifest.json').read_text())
    reference = json.loads((a.reference / 'manifest.json').read_text())
    params = json.loads((a.run / 'params-manifest.json').read_text())
    if manifest['id'] != task['id'] or reference['params_sha256'] != params['params_sha256']:
        raise ValueError('Task or checkpoint mismatch')
    built_inputs = {x['path']: x['sha256'] for x in build['inputs']}
    records = []
    for path, digest in reference['source_hashes'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Host source changed after reference execution: ' + path)
        if path in built_inputs and digest != built_inputs[path]:
            raise ValueError('Uploaded graph/adapter and local graph/adapter source differs: ' + path)
        if path.endswith('/kernel.ll'):
            parent = Path(path).parent
            result = json.loads((parent / 'build-result.json').read_text())['artifacts']
            if result['kernel.ll'] != digest or result['kernel.o'] != built_inputs[str(parent / 'kernel.o')]:
                raise ValueError('Local scalar LLVM and uploaded raw kernel provenance differs')
    expected = {x['name']: x for x in reference['files']}
    if set(expected) != {x['name'] for x in manifest['artifacts']}:
        raise ValueError('Missing or extra full-model output')
    for item in manifest['artifacts']:
        name = item['name']
        actual_file, reference_file = a.run / 'artifacts' / (name + '.bin'), a.reference / (name + '.bin')
        if (sha(actual_file) != item['sha256'] or sha(reference_file) != expected[name]['sha256']
                or actual_file.stat().st_size != item['nbytes'] or item['nbytes'] != expected[name]['bytes']):
            raise ValueError('Output/reference file hash or length mismatch')
        got, want = np.fromfile(actual_file, dtype='<f4'), np.fromfile(reference_file, dtype='<f4')
        diff = np.abs(got.astype('f8') - want.astype('f8'))
        finite = bool(np.isfinite(got).all() and np.isfinite(want).all())
        record = dict(name=name, elements=len(got), finite=finite,
                      max_abs=float(diff.max()), rmse=float(np.sqrt(np.mean(diff*diff))),
                      errors=int(np.count_nonzero(diff > ATOL + RTOL * np.abs(want))),
                      bit_exact=bool(np.array_equal(got.view('u4'), want.view('u4'))),
                      actual_sha256=item['sha256'], expected_sha256=expected[name]['sha256'])
        records.append(record)
        print(json.dumps(record), flush=True)
    report = dict(status='PASS' if all(x['finite'] and x['errors'] == 0 for x in records) else 'FAIL',
                  task_id=task['id'], atol=ATOL, rtol=RTOL, tensors=records,
                  elements=sum(x['elements'] for x in records), verifier_sha256=sha(Path(__file__)),
                  reference_manifest_sha256=sha(a.reference / 'manifest.json'),
                  scope=reference['scope'])
    (a.run / 'local-emitted-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
