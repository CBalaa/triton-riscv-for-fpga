#!/usr/bin/env python3
"""Execute the real E5 NH/E6 model; download outputs and validate locally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parents[2]
SIBLING_CLIENT=REPO.parent/'fpga-scheduler/.venv/bin/fpga-scheduler'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')


def cli(base,arguments,directory,label):
    result=subprocess.run(base+arguments,capture_output=True,text=True)
    (directory/(label+'.stdout')).write_text(result.stdout)
    (directory/(label+'.stderr')).write_text(result.stderr)
    if result.returncode:raise RuntimeError(f'scheduler {label} failed: '+result.stderr[-1200:])
    return json.loads(result.stdout)


def local_validate(directory,record):
    import numpy as np
    for name,expected_hash in record.get('frozen_files',{}).items():
        if sha(directory/name)!=expected_hash:raise ValueError('frozen evidence SHA mismatch '+name)
    for name,expected_hash in record.get('emitted_reference_files',{}).items():
        if sha(directory/name)!=expected_hash:raise ValueError('emitted reference SHA mismatch '+name)
    build=json.loads((directory/'build.json').read_text())
    manifest=json.loads((directory/'artifacts/manifest.json').read_text())
    if manifest['id']!=record['task_id']:raise ValueError('download manifest job mismatch')
    actual={item['name']:item for item in manifest['artifacts']}
    expected={item['name']:item for item in build['outputs']}
    if set(actual)!=set(expected):raise ValueError('incomplete model outputs')
    comparisons=[];arrays={}
    references=['w8a8','fp32']
    if record.get('emitted_reference_files'):references.append('emitted')
    for name,contract in expected.items():
        metadata=actual[name]
        for key in ('shape','dtype','nbytes'):
            if metadata[key]!=contract[key]:raise ValueError('artifact contract mismatch '+name+'/'+key)
        path=directory/'artifacts'/(name+'.bin')
        if path.stat().st_size!=contract['nbytes'] or sha(path)!=metadata['sha256']:
            raise ValueError('downloaded BIN length/SHA mismatch '+name)
        got=np.fromfile(path,dtype='<f4').reshape(contract['shape'])
        if not np.isfinite(got).all():raise ValueError('nonfinite model output '+name)
        arrays[name]=got
        for reference in references:
            reference_path=directory/(reference+'_'+name+'.bin')
            want=np.fromfile(reference_path,dtype='<f4').reshape(contract['shape'])
            difference=got.astype(np.float64)-want.astype(np.float64)
            a,b=got.ravel().astype(np.float64),want.ravel().astype(np.float64)
            comparison={'output':name,'reference':reference,'elements':got.size,'max_abs_error':float(np.abs(difference).max()),
                'rmse':float(np.sqrt(np.mean(difference*difference))),
                'cosine':float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),
                'download_sha256':sha(path),'reference_sha256':sha(reference_path)}
            if reference=='emitted':comparison['bitwise_equal']=sha(path)==sha(reference_path)
            if name=='layer_hidden':
                comparison['per_layer_max_abs_error']=[float(np.abs(d).max()) for d in difference]
                comparison['per_layer_rmse']=[float(np.sqrt(np.mean(d*d))) for d in difference]
            comparisons.append(comparison)
    # Quantized-runtime agreement is separate from quantization error to FP32.
    sentence=next(c for c in comparisons if c['output']=='sentence_embedding' and c['reference']=='w8a8')
    hidden=next(c for c in comparisons if c['output']=='last_hidden' and c['reference']=='w8a8')
    norm=float(np.linalg.norm(arrays['sentence_embedding'].astype(np.float64)))
    # These checks use only downloaded target tensors and the frozen input.
    # They independently establish that exports and the pooling path agree.
    checkpoint_error=float(np.abs(arrays['layer_hidden'][-1].astype(np.float64)-
                                  arrays['last_hidden'].astype(np.float64)).max())
    inputs=json.loads((directory/'input_reference.json').read_text())
    mask=np.asarray(inputs['attention_mask'],dtype=np.float64)
    pooled=(arrays['last_hidden'].astype(np.float64)*mask[:,None]).sum(axis=0)/mask.sum()
    pooled/=np.linalg.norm(pooled)
    pooling_error=float(np.abs(pooled-arrays['sentence_embedding'].astype(np.float64)).max())
    consistency={'final_layer_to_last_hidden_max_abs_error':checkpoint_error,
                 'masked_mean_l2_to_sentence_max_abs_error':pooling_error,
                 'pooling_reference':'FP64 NumPy recomputation from downloaded last_hidden and frozen input mask',
                 'final_layer_required_exact':True,'pooling_max_abs_error':1e-5}
    emitted_agreement=(all(c['bitwise_equal'] for c in comparisons if c['reference']=='emitted')
                       if 'emitted' in references else None)
    passed=(sentence['max_abs_error']<=0.01 and sentence['cosine']>=0.999 and
            hidden['max_abs_error']<=0.2 and hidden['rmse']<=0.03 and abs(norm-1.0)<=1e-4 and
            checkpoint_error==0.0 and pooling_error<=1e-5 and emitted_agreement is not False)
    report={'status':'MODEL_NUMERICAL_PASS' if passed else 'MODEL_NUMERICAL_FAIL','task_id':record['task_id'],
        'validation':'all complete downloaded model outputs compared locally with real-checkpoint W8A8 and independent HF FP32 references',
        'board_math_mocked':False,'sentence_l2_norm':norm,'comparisons':comparisons,'output_consistency':consistency,
        'emitted_implementation_agreement':emitted_agreement,
        'acceptance':{'reference':'host W8A8 contracts; target FP32 reduction/math differences allowed',
                      'sentence_max_abs_error':0.01,'sentence_cosine_min':0.999,
                      'last_hidden_max_abs_error':0.2,'last_hidden_rmse':0.03,'sentence_norm_abs_error':1e-4,
                      'emitted_reference_when_provided':'all three complete output BINs must match byte for byte'},
        'fp32_comparison_role':'reports quantization plus implementation error; FP32 equality is not required'}
    save(directory/'local-validation.json',report)
    print(report['status'],json.dumps(comparisons[:2]),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client',default=os.environ.get('FPGA_SCHEDULER_CLIENT',str(SIBLING_CLIENT) if SIBLING_CLIENT.is_file() else 'fpga-scheduler'),
                        help='scheduler executable (FPGA_SCHEDULER_CLIENT; sibling virtualenv or PATH by default)')
    parser.add_argument('--endpoint',default=os.environ.get('FPGA_SCHEDULER_URL','http://127.0.0.1:18762'),
                        help='selected fpga112 scheduler URL (FPGA_SCHEDULER_URL)')
    parser.add_argument('--token-file',type=Path,default=os.environ.get('FPGA_SCHEDULER_TOKEN_FILE',str(REPO/'.agent/secrets/fpga112.token')),
                        help='local scheduler credential file (FPGA_SCHEDULER_TOKEN_FILE); never copied into evidence')
    parser.add_argument('--build',type=Path,default=ROOT/'build/model-board')
    parser.add_argument('--output',type=Path,help='evidence directory; defaults to a new UTC-dated run, required for --validate-only')
    parser.add_argument('--timeout',type=int,default=1800)
    parser.add_argument('--validate-only',action='store_true',help='only recheck already downloaded model files locally; no scheduler calls')
    parser.add_argument('--emitted-reference',type=Path,help='freeze build_model_host.py outputs and require byte equality (with --validate-only)')
    args=parser.parse_args()
    if args.validate_only and args.output is None:parser.error('--validate-only requires --output PATH to an existing evidence directory')
    if args.output is None:args.output=ROOT/'build/model-board-runs'/time.strftime('run-%Y%m%d-%H%M%S',time.gmtime())
    args.build=args.build.resolve();args.output=args.output.resolve()
    base=[args.client,'--endpoint',args.endpoint,'--token-file',str(args.token_file.expanduser().resolve()),'--json']
    if args.emitted_reference and not args.validate_only:parser.error('--emitted-reference requires --validate-only')
    args.output.mkdir(parents=True,exist_ok=True)
    result_path=args.output/'result.json'
    if args.validate_only:
        record=json.loads(result_path.read_text())
        original=args.output/'local-validation-initial.json'
        if not original.exists() and (args.output/'local-validation.json').exists():
            shutil.copyfile(args.output/'local-validation.json',original)
        if args.emitted_reference:
            reference=args.emitted_reference.resolve()
            metadata=json.loads((reference/'manifest.json').read_text())
            params=json.loads((args.output/'params.json').read_text())
            if metadata['status']!='LOCAL_GRAPH_EXECUTED' or metadata.get('fpga_executed') is not False:
                raise ValueError('not a local emitted-implementation reference')
            if metadata['params_sha256']!=params['sha256'] or metadata['inputs_sha256']!=sha(args.output/'inputs.bin'):
                raise ValueError('emitted reference input/checkpoint mismatch')
            files={}
            for item in metadata['files']:
                source=reference/(item['name']+'.bin')
                if sha(source)!=item['sha256'] or source.stat().st_size!=item['bytes']:
                    raise ValueError('emitted reference BIN mismatch')
                target=args.output/('emitted_'+item['name']+'.bin')
                shutil.copyfile(source,target);files[target.name]=sha(target)
            if set(files)!={'emitted_'+name+'.bin' for name in ('sentence_embedding','last_hidden','layer_hidden')}:
                raise ValueError('incomplete emitted reference')
            shutil.copyfile(reference/'manifest.json',args.output/'emitted_manifest.json')
            files['emitted_manifest.json']=sha(args.output/'emitted_manifest.json')
            for source,expected_hash in metadata['source_hashes'].items():
                source=Path(source)
                if sha(source)!=expected_hash:raise ValueError('emitted source changed after reference execution')
                target=args.output/'emitted-sources'/source.relative_to(REPO)
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(source,target)
                files[str(target.relative_to(args.output))]=expected_hash
            record['emitted_reference_files']=files
        report=local_validate(args.output,record)
        shutil.copyfile(Path(__file__),args.output/'local-validator.py')
        record.update(local_validation_sha256=sha(args.output/'local-validation.json'),
                      local_validator_sha256=sha(args.output/'local-validator.py'),
                      local_recheck_status=report['status'])
        save(result_path,record)
        return report['status']!='MODEL_NUMERICAL_PASS'
    if result_path.exists():
        record=json.loads(result_path.read_text())
        task=cli(base,['job',record['task_id']],args.output,'recovery-query')
        if task['state'] not in ('completed','failed','cancelled'):
            raise RuntimeError('original task is nonterminal; stop without resubmission '+record['task_id'])
        save(args.output/'task.json',task)
        if record.get('status')=='PASS':
            print('CACHED PASS',record['task_id']);return 0
    else:
        health=cli(base,['health'],args.output,'health-before')
        devices=cli(base,['devices'],args.output,'devices-before')
        if health.get('stopping') or health.get('scheduler_error') or not health.get('dispatcher_alive'):
            raise RuntimeError('scheduler is unhealthy; no new submission')
        if not any(d.get('available') and d.get('state')=='READY' for d in devices['devices']):
            raise RuntimeError('no READY FPGA; no submission')
        build=json.loads((args.build/'build.json').read_text())
        if sha(args.build/'e5_model.bin')!=build['image_sha256']:raise ValueError('stale model binary')
        audit=json.loads((args.build/'elf-audit.json').read_text())
        if audit['status']!='PASS' or audit['elf_sha256']!=sha(args.build/'e5_model.elf'):raise ValueError('stale ELF audit')
        names=['e5_model.bin','e5_model.elf','elf-audit.json','build.json','input_reference.json','params.json',
               'inputs.bin','operator_trace.json']+[p.name for p in args.build.glob('w8a8_*.bin')]+[p.name for p in args.build.glob('fp32_*.bin')]
        for name in names:shutil.copyfile(args.build/name,args.output/name)
        for relative,expected in build['sources'].items():
            if sha(REPO/relative)!=expected:raise ValueError('source changed after build '+relative)
            target=args.output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/relative,target)
        record={'status':'SUBMITTING','server':'fpga112','task_id':uuid.uuid4().hex,'image_sha256':build['image_sha256'],
                'image_bytes':build['image_bytes'],'timeout':args.timeout,'frozen_files':{name:sha(args.output/name) for name in names},
                'reference_input':json.loads((args.output/'input_reference.json').read_text())}
        save(result_path,record)
        command=base+['run',str(args.output/'e5_model.bin'),'--id',record['task_id'],'--name','e5-small-v2/full-model-real-checkpoint',
                      '--timeout',str(args.timeout),'--queue-timeout','3600']
        record['command']=command;save(result_path,record)
        print('RUN',record['task_id'],'bytes',record['image_bytes'],flush=True)
        # run remains foreground and owns its lease through task completion.
        with (args.output/'client.stdout').open('w') as out,(args.output/'client.stderr').open('w') as err:
            process=subprocess.Popen(command,stdout=out,stderr=err)
            started=time.monotonic();last=-60
            while process.poll() is None:
                elapsed=time.monotonic()-started
                if elapsed-last>=60:print('MODEL_CLIENT_ALIVE',record['task_id'],'elapsed_seconds',int(elapsed),flush=True);last=elapsed
                time.sleep(2)
        record['client_returncode']=process.returncode;save(result_path,record)
        try:task=json.loads((args.output/'client.stdout').read_text())
        except json.JSONDecodeError:
            record['status']='CONNECTION_OR_CLIENT_ERROR';save(result_path,record)
            task=cli(base,['job',record['task_id']],args.output,'same-id-recovery')
            save(args.output/'task.json',task)
            raise RuntimeError('client failed; original task queried, no resubmission')
        save(args.output/'task.json',task)
    result=task.get('result') or {};details=result.get('details') or {}
    record.update(state=task['state'],fpga=task.get('fpga'),execution_status=result.get('execution_status'),
                  app_exit_code=result.get('app_exit_code'),exit_code=result.get('exit_code'),details=details,scheduler_seconds=result.get('seconds'))
    uart=result.get('uart_output') or result.get('output') or '';(args.output/'uart.log').write_text(uart)
    cycles=re.findall(r'model_cycles=0x([0-9a-fA-F]+)',uart)
    record['model_cycles']=[int(c,16) for c in cycles]
    save(result_path,record)
    if (task['id']!=record['task_id'] or task['state']!='completed' or result.get('execution_status')!='exited' or
            result.get('exit_code')!=0 or result.get('app_exit_code')!=0 or details.get('ddr_readback_matches') is not True or
            details.get('image_sha256')!=record['image_sha256'] or details.get('artifact_error') or
            not re.search(r'kernel_calls=1649 checkpoints=13',uart)):
        record['status']='APPLICATION_OR_PLATFORM_FAIL';save(result_path,record)
        print(record['status'],record['task_id'],uart[-4000:],flush=True);return 1
    artifacts=cli(base,['artifacts',record['task_id']],args.output,'artifacts-list')
    if artifacts['id']!=record['task_id']:raise ValueError('artifact list task mismatch')
    target=args.output/'artifacts';target.mkdir(exist_ok=True);save(target/'manifest.json',artifacts)
    for item in artifacts['artifacts']:
        path=target/(item['name']+'.bin')
        cli(base,['download',record['task_id'],item['name'],str(path)],args.output,'download-'+item['name'])
        if path.stat().st_size!=item['nbytes'] or sha(path)!=item['sha256']:raise ValueError('download SHA mismatch')
    record['artifacts']=artifacts['artifacts'];record['status']='DOWNLOADED';save(result_path,record)
    report=local_validate(args.output,record)
    record['status']='PASS' if report['status']=='MODEL_NUMERICAL_PASS' else 'NUMERICAL_FAIL'
    record['local_validation_sha256']=sha(args.output/'local-validation.json');save(result_path,record)
    print(record['status'],record['task_id'],'fpga',record['fpga'],'model_cycles',record['model_cycles'],flush=True)
    return record['status']!='PASS'


if __name__=='__main__':raise SystemExit(main())
