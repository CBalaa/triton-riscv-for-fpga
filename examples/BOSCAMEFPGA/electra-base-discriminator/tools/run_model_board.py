#!/usr/bin/env python3
"""Run one immutable full ELECTRA image, download every tensor, then validate locally."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'tools'))
from run_returned_outputs import (BASE, ENDPOINT, TOKEN, cli, now, save, sha, wait_ready,
                                  validate_artifact_metadata, verify_uart_declarations)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, default=ROOT / 'build/model-board')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1800)
    local_client = ROOT.parents[2].parent / 'fpga-scheduler/.venv/bin/fpga-scheduler'
    default_client = str(local_client) if local_client.is_file() else 'fpga-scheduler'
    p.add_argument('--client', default=os.environ.get('FPGA_SCHEDULER_CLIENT', default_client),
                   help='scheduler CLI executable; default: FPGA_SCHEDULER_CLIENT or sibling checkout/PATH')
    p.add_argument('--endpoint', default=os.environ.get('FPGA_SCHEDULER_URL', ENDPOINT),
                   help='fpga112 scheduler URL; default: FPGA_SCHEDULER_URL or local port 18762')
    p.add_argument('--token-file', type=Path,
                   default=Path(os.environ.get('FPGA_SCHEDULER_TOKEN_FILE', str(TOKEN))),
                   help='fpga112 token file; default: FPGA_SCHEDULER_TOKEN_FILE or repository secret path')
    a = p.parse_args()
    # The imported helpers use this same list; keep preflight/run/download on
    # one explicitly configured endpoint without changing shared module files.
    BASE[:] = [a.client, '--endpoint', a.endpoint, '--token-file', str(a.token_file), '--json']
    a.output.mkdir(parents=True, exist_ok=False)
    wait_ready(a.output)
    build_manifest = json.loads((a.build / 'build-manifest.json').read_text())
    for name in ('electra-model.bin', 'electra-model.elf', 'build-manifest.json',
                 'params-manifest.json', 'elf-audit.json'):
        shutil.copyfile(a.build / name, a.output / name)
    frozen = a.output / 'sources'
    for path in [*ROOT.glob('model_runtime/*'), ROOT / 'board_model.ld',
                 Path(__file__), ROOT / 'tools/model_reference.py',
                 ROOT / 'tools/verify_model_board.py', ROOT / 'tools/prepare_model_board.py',
                 ROOT / 'tools/build_model_bin.py', ROOT / 'tools/build_model_host.py',
                 ROOT / 'tools/verify_model_emitted.py', ROOT / 'tools/check_model_weights.py']:
        if path.is_file():
            dest = frozen / path.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    ident = uuid.uuid4().hex
    record = dict(model=ROOT.name, server='fpga112', task_id=ident, started_utc=now(),
                  image_sha256=sha(a.output / 'electra-model.bin'), timeout=a.timeout,
                  status='SUBMITTING', source_sha256={str(x.relative_to(frozen)): sha(x)
                                                    for x in frozen.rglob('*') if x.is_file()})
    save(a.output / 'result.json', record)
    print('RUN', ident, flush=True)
    command = BASE + ['run', str(a.output / 'electra-model.bin'), '--id', ident,
                      '--name', ROOT.name + '/full-model', '--timeout', str(a.timeout),
                      '--queue-timeout', '3600']
    with (a.output / 'client.stdout').open('w') as out, (a.output / 'client.stderr').open('w') as err:
        process = subprocess.run(command, stdout=out, stderr=err)
    try:
        task = json.loads((a.output / 'client.stdout').read_text())
    except json.JSONDecodeError:
        task = cli(['job', ident], a.output, 'recover-same-task')
        save(a.output / 'task.json', task)
        raise RuntimeError('CLI disconnected; stop new submissions and inspect original task ' + ident)
    save(a.output / 'task.json', task)
    result = task.get('result') or {}
    details = result.get('details') or {}
    uart = result.get('uart_output') or result.get('output') or ''
    (a.output / 'uart.log').write_text(uart)
    record.update(state=task.get('state'), fpga=task.get('fpga'), execution_status=result.get('execution_status'),
                  exit_code=result.get('exit_code'), app_exit_code=result.get('app_exit_code'),
                  client_returncode=process.returncode, details=details, finished_utc=now())
    for name in ('adapter_cycles', 'model_cycles', 'snapshot_cycles', 'elapsed_cycles'):
        found = re.findall(name + r'=0x([0-9a-fA-F]+)', uart)
        record[name] = int(found[-1], 16) if found else None
    save(a.output / 'result.json', record)
    if (task.get('id') != ident or task.get('state') != 'completed' or result.get('execution_status') != 'exited'
            or result.get('exit_code') != 0 or result.get('app_exit_code') != 0
            or details.get('image_sha256') != record['image_sha256']
            or details.get('application_started') is not True or details.get('reset_confirmed') is not True
            or details.get('ddr_readback_matches') is not True or details.get('artifact_error')):
        record['status'] = 'EXECUTION_FAILED'
        save(a.output / 'result.json', record)
        raise RuntimeError('Full-model execution not successful: ' + json.dumps(record))
    expected = {('embedding' if i == 0 else f'layer_{i-1:02d}' if i < 13 else 'logits'):
                dict(dtype='<f4', shape=[1,128,768] if i < 13 else [128],
                     nbytes=128*768*4 if i < 13 else 128*4) for i in range(14)}
    listed = cli(['artifacts', ident], a.output, 'artifacts-list')
    if listed.get('id') != ident:
        raise ValueError('artifact job ID mismatch')
    actual = validate_artifact_metadata(listed.get('artifacts'), expected)
    if validate_artifact_metadata(result.get('artifacts'), expected) != actual:
        raise ValueError('task/listed artifact metadata mismatch')
    verify_uart_declarations(uart, expected, actual)
    artifacts = a.output / 'artifacts'
    artifacts.mkdir()
    save(artifacts / 'manifest.json', dict(id=ident, artifacts=list(actual.values())))
    for name, item in actual.items():
        path = artifacts / (name + '.bin')
        cli(['download', ident, name, str(path)], a.output, 'download-' + name)
        if path.stat().st_size != item['nbytes'] or sha(path) != item['sha256']:
            raise ValueError('Downloaded tensor failed SHA/size: ' + name)
    record.update(status='DOWNLOADED', artifacts=list(actual.values()))
    save(a.output / 'result.json', record)
    independent = subprocess.run([sys.executable, ROOT / 'tools/verify_model_board.py',
                                  '--run', a.output, '--build', a.build])
    host = a.output / 'local-emitted-oracle'
    subprocess.run([sys.executable, ROOT / 'tools/build_model_host.py',
                    '--build', a.build, '--output', host,
                    '--kernels', build_manifest.get('kernels', str(ROOT / 'build/kernels'))], check=True)
    implementation = subprocess.run([sys.executable, ROOT / 'tools/verify_model_emitted.py',
                                     '--run', a.output, '--reference', host])
    passed = independent.returncode == 0 and implementation.returncode == 0
    record.update(status='PASS' if passed else 'LOCAL_VALIDATION_FAILED',
                  implementation_status='PASS' if implementation.returncode == 0 else 'FAIL',
                  independent_torch_w8a8_status='PASS' if independent.returncode == 0 else 'FAIL',
                  implementation_reference='actual model graph/emitted scalar LLVM; exact independent C integer dot',
                  local_emitted_validation_sha256=sha(a.output / 'local-emitted-validation.json'))
    save(a.output / 'result.json', record)
    print(json.dumps(record, indent=2), flush=True)
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
