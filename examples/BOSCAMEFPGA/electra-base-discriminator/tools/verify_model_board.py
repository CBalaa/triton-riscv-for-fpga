#!/usr/bin/env python3
"""Compare actual downloaded full-model tensors to the matching independent W8A8 oracle."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ElectraForPreTraining
from model_reference import forward

ROOT = Path(__file__).resolve().parents[1]
# Frozen before the first board submission. Quantized full-chain rounding is
# measured separately from FP32 checkpoint accuracy; every returned element is checked.
TOLERANCE = {'embedding': (1e-4, 1e-4), 'hidden': (.02, .01), 'logits': (.02, .01)}


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def metrics(got, want, atol, rtol):
    difference = np.abs(got.astype(np.float64) - want.astype(np.float64))
    finite = bool(np.isfinite(got).all() and np.isfinite(want).all())
    return dict(elements=got.size, finite=finite, max_abs=float(difference.max()),
                rmse=float(np.sqrt(np.mean(difference**2))),
                errors=int(np.count_nonzero(difference > atol + rtol * np.abs(want))),
                atol=atol, rtol=rtol)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--build', type=Path, default=ROOT / 'build/model-board')
    p.add_argument('--assets', type=Path, default=ROOT / 'assets')
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_grad_enabled(False)
    params = json.loads((a.run / 'params-manifest.json').read_text())
    build = json.loads((a.run / 'build-manifest.json').read_text())
    if sha(a.build / 'params.bin') != params['params_sha256']:
        raise ValueError('Reference checkpoint blob mismatch')
    artifact_manifest = json.loads((a.run / 'artifacts/manifest.json').read_text())
    task = json.loads((a.run / 'task.json').read_text())
    if artifact_manifest['id'] != task['id']:
        raise ValueError('Artifact/task provenance mismatch')
    model = ElectraForPreTraining.from_pretrained(a.assets, local_files_only=True,
                                                 attn_implementation='eager').eval()
    inputs = {name: torch.tensor(item['values'], dtype=torch.long)
              for name, item in params['inputs'].items()}
    ids, types, mask = (inputs[name] for name in ('input_ids', 'token_type_ids', 'attention_mask'))
    snapshots = {}
    print('Computing matching checkpoint W8A8 oracle locally', flush=True)
    expected = forward(model.state_dict(), ids, types, mask, 'w8a8', [], snapshots)
    with torch.inference_mode():
        fp32 = model(input_ids=ids[None], token_type_ids=types[None],
                     attention_mask=mask[None]).logits[0].numpy()
    np.savez(a.run / 'local-w8a8-reference.npz', **{k: v.numpy() for k, v in snapshots.items()}, fp32_logits=fp32)
    records = []
    for item in artifact_manifest['artifacts']:
        path = a.run / 'artifacts' / (item['name'] + '.bin')
        if path.stat().st_size != item['nbytes'] or sha(path) != item['sha256']:
            raise ValueError('Invalid downloaded tensor: ' + item['name'])
        got = np.fromfile(path, dtype='<f4').reshape(-1)
        want = snapshots[item['name']].numpy().reshape(-1)
        key = item['name'] if item['name'] in ('embedding', 'logits') else 'hidden'
        record = dict(name=item['name'], sha256=item['sha256'], **metrics(got, want, *TOLERANCE[key]))
        records.append(record)
        print(json.dumps(record), flush=True)
    actual = np.fromfile(a.run / 'artifacts/logits.bin', dtype='<f4')
    report = dict(status='PASS' if all(x['finite'] and x['errors'] == 0 for x in records) else 'FAIL',
                  task_id=task['id'], checkpoint_sha256=params['checkpoint'],
                  image_sha256=build['binary_sha256'], elements=sum(x['elements'] for x in records),
                  tensors=records, expected_w8a8_vs_fp32=metrics(expected.numpy(), fp32, 0, 0),
                  actual_vs_fp32=metrics(actual, fp32, 0, 0),
                  scope='one real tokenizer sentence pair; all 12 layers and raw discriminator logits, including padding')
    (a.run / 'local-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
