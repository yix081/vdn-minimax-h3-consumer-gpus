"""Versioned SGLang release entry point with explicit hardware recipes and archived configuration."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXTENSIONS=HERE/'extensions'
SOURCE_REVISION='d72e59508b7554045cb51827f9b8d0f08c7a3abc'
ADALN_PATCH_ID='exact-loaded-projection-adaln-cache'
STREAM_PATCH_ID='streamed-quant-loader'
LIFETIME_PATCH_ID='linear-lifetime'
SP_STREAM_PATCH_ID='sp-main-stream'
RANKLOCAL_PATCH_ID='ranklocal-loader'
ADALN_EXTENSION_SHA256='1781b71b299f8dd7c5664ccb18a18249ffccffd62d1933e497f20c9ae068d029'
MODEL_MANIFEST_SHA256='116bf53898646148133048b02e5c703fa7df2515d0775005bbf4642c2d837594'
SOURCE_FILES={
    ADALN_PATCH_ID:(Path('multimodal_gen/runtime/models/dits/minimax_h3.py'),
                    '2a61d5c8b0418eed72fd3443cc47b6a06ed8512f6c834b608aa4e3e55e7489fa',
                    'df1d9caa615ee5f563ac2bd29e5ba6c433cfaaf1cca01b2c0ea1d693b5764f09'),
    STREAM_PATCH_ID:(Path('multimodal_gen/runtime/loader/fsdp_load.py'),
                     '9f01ae268ebd7d0e3abc40b295e25a6d8b24818a8bd110cb1f1e15ab4298c828',
                     '5d726735b4eeb634c36a6d3d3bf19f47dd54f3ea00e3f46cf9ff721d8a676205'),
    LIFETIME_PATCH_ID:(Path('multimodal_gen/runtime/models/dits/minimax_h3_vdn.py'),
                       'ac3b7aaff8b77bb722498ae903b8412511959a6606fe3e3863c424639040b89f',
                       '9661344060b5ddaea6958819e0d99bffaad2fe4166dd6c31a805ba8792e294b2'),
    SP_STREAM_PATCH_ID:(Path('multimodal_gen/runtime/models/dits/minimax_h3_vdn_attention.py'),
                        'ff71881cec73e3656a4ed412a10847119bbd85d591cc0fee0f5f6a59096128b8',
                        '9753309a3808dbf5081360787facb71039154ad1ee0d687fc97a41f7f2a35ae2'),
    # three files: the loader, the text-encoder loader and a new helper (original None = must be absent)
    RANKLOCAL_PATCH_ID:[(Path('multimodal_gen/runtime/loader/fsdp_load.py'),
                         '9f01ae268ebd7d0e3abc40b295e25a6d8b24818a8bd110cb1f1e15ab4298c828',
                         '310b143cbd78b8db6496fa81569821aa4f6dd03a912ffc1607462c44b811ce94'),
                        (Path('multimodal_gen/runtime/loader/component_loaders/text_encoder_loader.py'),
                         'd38ceb91ad8171cd6fda4a14ae9ccc758852bcfdc5b6d79131b8f2161c6b7c58',
                         '3089d143758036f67513b083fc8e5575be72945674d05e2c92df8c4f63f52ea2'),
                        (Path('multimodal_gen/runtime/loader/affordable_ranklocal_guard.py'),
                         None,
                         'b0a8d62ca3b89ee01d81f7827dd8d94b7f3e5338df173d2640601ee0fc7013d7')],
}
# (patches, adaln cache on, streamed loader on, linear lifetime on, sequence-parallel main stream on, rank-local loader on)
SOURCE_PROFILES={
    'pristine-upstream':(frozenset(),False,False,False,False,False),
    'adaln-patched-cache-off':(frozenset({ADALN_PATCH_ID}),False,False,False,False,False),
    'adaln-patched-cache-on':(frozenset({ADALN_PATCH_ID}),True,False,False,False,False),
    'streamed-quant-loader':(frozenset({STREAM_PATCH_ID}),False,True,False,False,False),
    'streamed-quant-loader-lifetime-off':(frozenset({STREAM_PATCH_ID,LIFETIME_PATCH_ID}),False,True,False,False,False),
    'streamed-quant-loader-lifetime-on':(frozenset({STREAM_PATCH_ID,LIFETIME_PATCH_ID}),False,True,True,False,False),
    'sp-main-stream':(frozenset({SP_STREAM_PATCH_ID}),False,False,False,True,False),
    'ranklocal-loader-sp-main-stream':(frozenset({RANKLOCAL_PATCH_ID,SP_STREAM_PATCH_ID}),False,False,False,True,True),
}
ALLOWED_ENV={'LEANVDN_SGLANG_ADALN_CACHE','LEANVDN_ADALN_VERIFY','LEANVDN_STREAM_QUANT_LOAD',
             'LEANVDN_LINEAR_LIFETIME','LEANVDN_SP_MAIN_STREAM','LEANVDN_H3_MEMORY_LIFETIME','NCCL_NVLS_ENABLE',
             'SGLANG_DIFFUSION_STAGE_LOGGING','SGLANG_DIFFUSION_SYNC_STAGE_PROFILING',
             'PYTORCH_ALLOC_CONF','PYTORCH_CUDA_ALLOC_CONF','SGLANG_CACHE_DIT_ENABLED',
             'AFFORDABLE_H3_RANKLOCAL_STREAM','AFFORDABLE_H3_LOAD_LOCK_TIMEOUT_S'}
# set by this runner, never by a recipe: the rank-local loader's lock file and receipt directory live in the run directory
RUNNER_ENV={'AFFORDABLE_H3_LOAD_LOCK','AFFORDABLE_RANK_RECEIPTS'}


def sha256_file(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()

def atomic(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2,default=str)+'\n');tmp.replace(path)

def probe(path):
    result=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-show_format','-of','json',str(path)],timeout=180))
    streams=[{k:s.get(k) for k in ['codec_type','codec_name','width','height','r_frame_rate','nb_read_frames','duration','sample_rate','channels']} for s in result['streams']]
    return {'streams':streams,'bytes':path.stat().st_size,'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest()}

def full_decode(path):
    """Decode both streams completely; return SHA-256 of the raw video frames and of the PCM audio."""
    hashes={}
    for name,args in (('video',['-map','0:v:0','-c:v','rawvideo']),('audio',['-map','0:a:0','-c:a','pcm_s16le'])):
        out=subprocess.run(['ffmpeg','-v','error','-xerror','-threads','1','-i',str(path)]+args+['-f','hash','-hash','sha256','-'],
                           check=True,timeout=300,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True).stdout
        digest_line=next((line for line in out.splitlines() if line.startswith('SHA256=')),None)
        if digest_line is None:raise RuntimeError(f'ffmpeg produced no {name} hash for {path}')
        hashes[name]=digest_line.split('=',1)[1].strip()
    return hashes


def check_resolved(resolved,expected):
    if not isinstance(resolved,dict):raise ValueError('Resolved server arguments unavailable')
    for key,value in expected.items():
        if key not in resolved or resolved[key]!=value:
            raise ValueError(f'Resolved configuration mismatch: {key}: expected {value!r}, got {resolved.get(key)!r}')


def validate_model_cache():
    if os.environ.get('HF_HUB_OFFLINE')!='1' or os.environ.get('TRANSFORMERS_OFFLINE')!='1':
        raise RuntimeError('Export HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1 for the pinned model snapshot')
    raw_home=os.environ.get('HF_HOME')
    if not raw_home:
        raise RuntimeError('HF_HOME must point to the cache created by tools/stage_model.py')
    hf_home=Path(raw_home).expanduser().resolve()
    receipt_path=hf_home/'staging-receipt.json'
    if not receipt_path.is_file():
        raise RuntimeError(f'Model staging receipt is missing: {receipt_path}')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('status')!='verified' or receipt.get('manifest_sha256')!=MODEL_MANIFEST_SHA256:
        raise RuntimeError('Model staging receipt does not match the release manifest')
    if Path(receipt.get('hf_home','')).resolve()!=hf_home:
        raise RuntimeError('Model staging receipt belongs to a different HF_HOME')
    manifest_path=HERE/'model-manifest.json'
    if sha256_file(manifest_path)!=MODEL_MANIFEST_SHA256:
        raise RuntimeError('Model manifest hash mismatch')
    manifest=json.loads(manifest_path.read_text())
    refs={}
    for spec in manifest['models'].values():
        ref=hf_home/'hub'/('models--'+spec['repo'].replace('/','--'))/'refs/main'
        if not ref.is_file() or ref.read_text().strip()!=spec['revision']:
            raise RuntimeError(f'Model cache is not pinned to {spec["repo"]}@{spec["revision"]}')
        refs[spec['repo']]=spec['revision']
    return {'hf_home':str(hf_home),'receipt':str(receipt_path),
            'manifest_sha256':MODEL_MANIFEST_SHA256,'revisions':refs}


def validate_recipe(recipe):
    if recipe.get('source_revision')!=SOURCE_REVISION:
        raise ValueError('Unreviewed source revision')
    patches=recipe.get('patches',[])
    if (not isinstance(patches,list) or len(patches)!=len(set(patches)) or
            any(p not in SOURCE_FILES for p in patches)):
        raise ValueError(f'Unexpected recipe patches: {patches!r}')
    environment=recipe.get('environment',{})
    if not isinstance(environment,dict):
        raise ValueError('Recipe environment must be an object')
    for key,value in environment.items():
        if key not in ALLOWED_ENV or not isinstance(value,str):
            raise ValueError(f'Unexpected recipe environment entry: {key!r}')
    state=(frozenset(patches),
           environment.get('LEANVDN_SGLANG_ADALN_CACHE')=='1',
           environment.get('LEANVDN_STREAM_QUANT_LOAD')=='1',
           environment.get('LEANVDN_LINEAR_LIFETIME')=='1',
           environment.get('LEANVDN_SP_MAIN_STREAM')=='1',
           environment.get('AFFORDABLE_H3_RANKLOCAL_STREAM')=='1')
    source_profile=recipe.get('source_profile')
    if source_profile not in SOURCE_PROFILES:
        raise ValueError(f'Unknown or missing source_profile: {source_profile!r}')
    expected_state=SOURCE_PROFILES[source_profile]
    if state!=expected_state:
        raise ValueError(
            f'Source profile {source_profile!r} requires patch/cache state {expected_state}, '
            f'got {state}'
        )
    server_args=recipe.get('server_args')
    if not isinstance(server_args,dict):
        raise ValueError('Recipe server_args must be an object')
    if server_args.get('num_gpus') not in (1,2,4,8):
        raise ValueError('Recipe num_gpus must be 1, 2, 4, or 8')
    if environment.get('AFFORDABLE_H3_RANKLOCAL_STREAM')=='1':
        # the loader's own guard repeats these checks at load time; fail early with a clear message
        if server_args['num_gpus'] not in (2,4,8):
            raise ValueError('The rank-local loader is for 2, 4 or 8 GPUs')
        try:lock_timeout=float(environment['AFFORDABLE_H3_LOAD_LOCK_TIMEOUT_S'])
        except (KeyError,ValueError):raise ValueError('Rank-local recipes need AFFORDABLE_H3_LOAD_LOCK_TIMEOUT_S') from None
        if not 0<lock_timeout<float(server_args.get('dist_timeout',0)):
            raise ValueError('AFFORDABLE_H3_LOAD_LOCK_TIMEOUT_S must be positive and below server_args.dist_timeout')
        if server_args.get('encoder_parallel')!='fold' or server_args.get('tp_size',1)!=1 or server_args.get('ulysses_degree')!=server_args['num_gpus']:
            raise ValueError('The rank-local loader needs encoder_parallel=fold, tp_size=1 and ulysses_degree=num_gpus')
    protocol=recipe.get('performance_protocol',{'feasibility':1,'warmups':2,'timings':3})
    if (not isinstance(protocol,dict) or set(protocol)!={'feasibility','warmups','timings'} or
            any(not isinstance(protocol[k],int) or protocol[k]<0 for k in protocol)):
        raise ValueError('Invalid performance_protocol')
    if protocol['timings']<1:
        raise ValueError('performance_protocol needs at least one formal timing')
    if recipe.get('case_profile') not in (None,'short','long'):
        raise ValueError('case_profile must be short or long')
    guard=recipe.get('hardware_guard')
    if guard is not None:
        if (not isinstance(guard,dict) or set(guard)!={'name_contains','minimum_memory_mib','compute_capability'} or
                not isinstance(guard['name_contains'],str) or not guard['name_contains'] or
                not isinstance(guard['minimum_memory_mib'],int) or guard['minimum_memory_mib']<1 or
                not isinstance(guard['compute_capability'],list) or len(guard['compute_capability'])!=2 or
                any(not isinstance(value,int) for value in guard['compute_capability'])):
            raise ValueError('Invalid hardware_guard')


def validate_runtime_source(sglang_package,recipe):
    package_root=Path(sglang_package.__file__).resolve().parent
    source_root=package_root.parents[1]
    try:
        revision=subprocess.check_output(
            ['git','-C',str(source_root),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError,FileNotFoundError):
        raise RuntimeError(f'SGLang must be installed from the pinned Git checkout: {source_root}') from None
    if revision!=SOURCE_REVISION:
        raise RuntimeError(f'SGLang revision mismatch: expected {SOURCE_REVISION}, got {revision}')
    declared=set(recipe.get('patches',[]))
    result={'package_root':str(package_root),'source_root':str(source_root),'revision':revision,'source_files':{}}
    # one expectation per file: a declared patch's hash, else the pinned original, else absence
    expectations={}
    for patch_id,spec in SOURCE_FILES.items():
        for relative,original_sha,patched_sha in (spec if isinstance(spec,list) else [spec]):
            current=expectations.setdefault(relative,{'original':original_sha,'patch':None,'expected':original_sha})
            if patch_id in declared:
                if current['patch'] is not None:
                    raise RuntimeError(f'Patches {current["patch"]!r} and {patch_id!r} both change {relative}; a recipe may declare only one of them')
                current.update(patch=patch_id,expected=patched_sha)
    for relative,item in expectations.items():
        path=package_root/relative
        if item['expected'] is None:
            if path.exists():
                raise RuntimeError(f'{relative} is added by a patch this recipe does not declare; reset the checkout')
            continue
        if not path.is_file():
            raise RuntimeError(f'Pinned SGLang source is missing: {path}')
        actual=sha256_file(path)
        if actual!=item['expected']:
            raise RuntimeError(
                f'SGLang source hash mismatch for {relative}: expected {item["expected"]}, got {actual}. '
                'Use the pinned source and apply only the recipe-declared patches.'
            )
        result['source_files'][str(relative)]={'sha256':actual,'patch':item['patch']}
    if ADALN_PATCH_ID in declared:
        extension=EXTENSIONS/'sglang_adaln_extension.py'
        extension_sha=sha256_file(extension) if extension.is_file() else None
        if extension_sha!=ADALN_EXTENSION_SHA256:
            raise RuntimeError(f'AdaLN extension hash mismatch: expected {ADALN_EXTENSION_SHA256}, got {extension_sha}')
        if str(EXTENSIONS) not in sys.path:
            sys.path.insert(0,str(EXTENSIONS))
        result.update(adaln_extension=str(extension),adaln_extension_sha256=extension_sha)
    return result


def schedule_cases(cases,suite,performance_protocol=None):
    if not cases or any(c['expected_frames'] not in (124,345) for c in cases):
        raise ValueError('Only frozen 124/345-frame cases are allowed')
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('Duplicate case IDs')
    schedule=[]
    warmed=set()
    for case in cases:
        shape=case['expected_frames']
        if suite=='performance':
            protocol=performance_protocol or {'feasibility':1,'warmups':2,'timings':3}
            schedule.extend((case,'feasibility',i) for i in range(protocol['feasibility']))
            schedule.extend((case,'warmup',i) for i in range(protocol['warmups']))
            schedule.extend((case,'timing',i) for i in range(protocol['timings']))
        elif suite=='quality':
            if shape not in warmed:
                schedule.extend((case,'warmup',i) for i in range(2))
                warmed.add(shape)
            schedule.append((case,'quality',0))
        else:
            raise ValueError('Unknown suite')
    return schedule


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--recipe',type=Path,required=True)
    p.add_argument('--suite',choices=['performance','quality'],default='performance')
    a=p.parse_args()
    a.output=a.output.expanduser().resolve()
    a.recipe=a.recipe.expanduser().resolve()
    a.cases=a.cases.expanduser().resolve()
    recipe=json.loads(a.recipe.read_text())
    validate_recipe(recipe)
    model_cache=validate_model_cache()
    for key in ALLOWED_ENV|RUNNER_ENV:
        os.environ.pop(key,None)
    for key,value in recipe.get('environment',{}).items():
        os.environ[key]=value
    a.diagnostic=False
    a.output.mkdir(parents=True,exist_ok=False)
    if os.environ.get('AFFORDABLE_H3_RANKLOCAL_STREAM')=='1':
        os.environ['AFFORDABLE_H3_LOAD_LOCK']=str(a.output/'dit-load.lock')
        os.environ['AFFORDABLE_RANK_RECEIPTS']=str(a.output/'rank-receipts')
    if a.diagnostic:
        os.environ['SGLANG_DIFFUSION_STAGE_LOGGING']='1'
        os.environ['SGLANG_DIFFUSION_SYNC_STAGE_PROFILING']='1'
    config=recipe['server_args'];cases=json.loads(a.cases.read_text())
    case_profile=recipe.get('case_profile')
    expected_frames={'short':124,'long':345}.get(case_profile)
    if expected_frames is not None and any(case.get('expected_frames')!=expected_frames for case in cases):
        raise ValueError(f'Recipe requires the {case_profile} case file ({expected_frames} frames)')
    protocol=recipe.get('performance_protocol',{'feasibility':1,'warmups':2,'timings':3})
    planned=schedule_cases(cases,a.suite,protocol)
    os.environ['LEANVDN_ADALN_REPORT_DIR']=str(a.output/'cache-diagnostics')
    receipt={'recipe':recipe,'recipe_path':str(a.recipe),'cases_path':str(a.cases),'output_path':str(a.output),'recipe_sha256':hashlib.sha256(a.recipe.read_bytes()).hexdigest(),'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases_sha256':hashlib.sha256(a.cases.read_bytes()).hexdigest(),'model_cache':model_cache,'suite':a.suite,'request_count':len(planned)}
    atomic(a.output/'release-receipt.json',receipt)
    os.environ['SGLANG_PERF_LOG_DIR']=str(a.output/'native-metrics')
    os.environ['SGLANG_TORCH_PROFILER_DIR']=str(a.output/'profile')
    report={'status':'loading','server_args':config,'cases':cases,'runs':[],'source_revision':SOURCE_REVISION,'synchronized_diagnostic':a.diagnostic,'timing_boundary':'Host call generate() through finalized local audio/video file; media validation and tensor profiling excluded from timing cohorts.','created_at':time.time(),'performance_protocol':protocol,'suite':a.suite,'release_receipt':receipt}
    atomic(a.output/'report.json',report)
    generator=None
    try:
        import sglang
        report['runtime_source']=validate_runtime_source(sglang,recipe)
        atomic(a.output/'report.json',report)
        import torch
        visible_gpu_count=torch.cuda.device_count()
        if visible_gpu_count!=config['num_gpus']:
            raise RuntimeError(
                f'Recipe requires {config["num_gpus"]} visible GPUs, but PyTorch sees {visible_gpu_count}. '
                'Set CUDA_VISIBLE_DEVICES to exactly the recipe GPU count.'
            )
        guard=recipe.get('hardware_guard')
        if guard:
            expected_capability=tuple(guard['compute_capability'])
            for device_index in range(visible_gpu_count):
                properties=torch.cuda.get_device_properties(device_index)
                memory_mib=properties.total_memory//(1024*1024)
                actual_capability=torch.cuda.get_device_capability(device_index)
                if guard['name_contains'] not in properties.name:
                    raise RuntimeError(f'GPU {device_index} is {properties.name!r}; recipe requires {guard["name_contains"]!r}')
                if memory_mib<guard['minimum_memory_mib']:
                    raise RuntimeError(f'GPU {device_index} has {memory_mib} MiB; recipe requires at least {guard["minimum_memory_mib"]} MiB')
                if actual_capability!=expected_capability:
                    raise RuntimeError(f'GPU {device_index} capability is {actual_capability}; recipe requires {expected_capability}')
        from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
        report['torch']=torch.__version__
        report['visible_gpu_count']=visible_gpu_count
        report['hardware']=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version','--format=csv'],text=True)
        start=time.perf_counter();generator=DiffGenerator.from_pretrained(**config);report['load_seconds']=time.perf_counter()-start
        resolved=generator.server_args if hasattr(generator,'server_args') else None
        resolved_values=vars(resolved) if hasattr(resolved,'__dict__') else None
        atomic(a.output/'resolved-server-args.json',resolved_values)
        check_resolved(resolved_values,recipe.get('expected_resolved',{}))
        report['status']='running';atomic(a.output/'report.json',report)
        schedule=planned
        for index,(case,kind,repeat) in enumerate(schedule):
            params={'prompt':case['prompt'],'seed':case['seed'],'task':'t2va','conditions':[],'target':{'short_edge':768,'aspect_ratio':'16:9','duration_seconds':case['duration_seconds']},'num_inference_steps':9,'flow_shift':12.0,'audio_flow_shift':3.0,'num_outputs_per_prompt':1,'save_output':True,'return_file_paths_only':True,'output_path':str(a.output),'output_file_name':f'{index:03d}-{case["id"]}-{kind}.mp4'}
            if kind=='profile':params.update(profile=True,profile_all_stages=True,num_profiled_timesteps=8)
            row={'case_id':case['id'],'kind':kind,'repeat':repeat,'params':params,'started_at':time.time()}
            report['active_request']=row;atomic(a.output/'report.json',report)
            start=time.perf_counter();r=generator.generate(sampling_params_kwargs=params);row['complete_request_seconds']=time.perf_counter()-start
            if isinstance(r,list):
                if len(r)!=1:raise RuntimeError(f'Expected one generation result, got {len(r)}')
                r=r[0]
            if r is None:raise RuntimeError('SGLang returned no generation result')
            row.update(size=r.size,peak_memory_mb=r.peak_memory_mb,metrics=r.metrics,output_file_path=r.output_file_path)
            row['media']=probe(Path(r.output_file_path))
            video=next(s for s in row['media']['streams'] if s['codec_type']=='video');audio=next(s for s in row['media']['streams'] if s['codec_type']=='audio')
            if (video['width'],video['height'])!=(1344,768):raise RuntimeError(f'Unexpected video dimensions: {video}')
            if int(video['nb_read_frames'])!=case['expected_frames']:raise RuntimeError(f'Unexpected frame count: {video}')
            if int(audio['channels'])!=2 or int(audio['sample_rate'])!=32000:raise RuntimeError(f'Unexpected audio format: {audio}')
            if video['r_frame_rate']!='24/1':raise RuntimeError(f'Unexpected frame rate: {video}')
            decoded=full_decode(Path(r.output_file_path))
            row['full_audio_video_decode_verified']=True
            row['decoded_video_sha256']=decoded['video'];row['decoded_audio_sha256']=decoded['audio']
            steps=row['metrics'].get('steps',[])
            row['native_per_step_metrics_available']=bool(steps)
            row['configured_dit_forwards']=8
            if len(steps) not in (0,8):raise RuntimeError(f'Unexpected measured DiT forwards: {len(steps)}')
            report['runs'].append(row);report.pop('active_request',None);atomic(a.output/'report.json',report)
            print(json.dumps({'case_id':row['case_id'],'kind':kind,'seconds':row['complete_request_seconds'],'stages_ms':row['metrics'].get('stages'),'gpu_peak_mb':row['peak_memory_mb']}),flush=True)
        report['status']='completed'
    except BaseException:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        report['ended_at']=time.time();atomic(a.output/'report.json',report)
        if generator is not None:generator.shutdown()

if __name__=='__main__':main()
