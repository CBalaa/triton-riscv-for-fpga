#!/usr/bin/env python3
"""Render saved fpga112 operator evidence without making scheduler requests."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, default=ROOT / 'build/board-runs/fpga112-60s-20261103')
    parser.add_argument('--output', type=Path, default=ROOT / 'BOARD_RESULTS.md')
    args = parser.parse_args()
    records = []
    for case in inventory():
        path = args.runs / case['name'] / 'result.json'
        record = json.loads(path.read_text()) if path.is_file() else {'case': case['name'], 'status': 'NOT_RUN'}
        records.append(record)
    counts = Counter(record['status'] for record in records)
    pending = any(record['status'] in ('NOT_RUN', 'SUBMITTING', 'CONNECTION_OR_SUBMISSION_ERROR') for record in records)
    lines = [
        '# E5-small-v2 board results — fpga112, 2026-11-03', '',
        'Target: `fpga112`; scheduler endpoint: `http://127.0.0.1:18762`.',
        'Each task uses a 60-second application execution deadline, excluding queue wait and DDR loading; queue timeout is 3,600 seconds.', '',
        'Coverage: ' + ', '.join(f'**{status}: {count}**' for status, count in sorted(counts.items())) + '.', '',
    ]
    if not pending:
        after_return = sum(record.get('phase') == 'kernel_returned_before_timeout' for record in records)
        before_return = sum(record.get('phase') == 'no_completed_kernel_cycle' for record in records)
        lines += [f"Completed kernel cycle records: **{sum(bool(record.get('kernel_cycles')) for record in records)}/{len(records)}**. Timeout after adapter return: **{after_return}**; timeout without observed adapter return: **{before_return}**; TRAP: **{counts['TRAP']}**; normal exit with validation failure: **{counts['EXITED_FAIL']}**.", '']
    if (args.runs / 'connection-interrupted-attempts').is_dir():
        lines += ['An initial API connection failure was recovered by querying the original task ID and confirming `Task not found` before a new submission. That unexecuted attempt is archived under `connection-interrupted-attempts/` and excluded from the execution counts.', '']
    if pending:
        lines += ['The suite is incomplete. Connection/submission errors have no confirmed execution result; they are not classified as application hangs. Resume only after querying any existing task ID.', '']
    lines += [
        '| Operator | Status | Timeout phase | Kernel cycles | Job ID | FPGA port |',
        '| --- | --- | --- | ---: | --- | --- |',
    ]
    for record in records:
        cycles = ', '.join(str(value) for value in record.get('kernel_cycles', [])) or '—'
        task = record.get('task_id', '—')
        port = str(record.get('fpga')) if record.get('fpga') is not None else '—'
        phase = record.get('phase', '—')
        lines.append(f"| `{record['case']}` | {record['status']} | {phase} | {cycles} | `{task}` | {port} |")
    lines += ['',
        '`PASS` requires scheduler `exited`, application exit code zero, complete adapter returns and cycles, validation execution, the expected checked count and zero errors. `HANG_60S` means the application execution deadline expired; `TRAP` is reported separately. `EXITED_FAIL` means normal application termination failed the validation contract.', '',
        'For a timeout, `kernel_returned_before_timeout` means the typed adapter completed and a cycle count was captured; whole-launch completion and numerical validation remain unconfirmed. `no_completed_kernel_cycle` means no adapter completion was observed. Neither connection failures nor queue/loading time are counted as application hangs.', '',
        'Kernel cycles measure the complete typed adapter, including program-grid dispatch, DMA and final fence. They exclude input initialization, scalar validation and UART. A task with no completed adapter has no measured kernel cycle count.', '',
        'Per-case evidence is under `' + str(args.runs.relative_to(ROOT)) + '/<operator>/`: immutable image, SHA256 in `result.json`, ELF audit, board configuration, scheduler `task.json`, client stdout/stderr and `uart.log`. Missing final task/UART files indicate an unconfirmed or unexecuted submission.', '',
    ]
    args.output.write_text('\n'.join(lines))
    print(json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
