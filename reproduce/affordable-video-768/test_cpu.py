"""CPU-only integrity checks for the public reproduction package."""
import hashlib
import importlib.util
import json
import os
import copy
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner=load_module('affordable_video_generate',ROOT/'generate.py')
patcher=load_module('affordable_video_b200_patch',ROOT/'tools/apply_b200_adaln_patch.py')
stream_patcher=load_module('affordable_video_stream_patch',ROOT/'tools/apply_stream_quant_patch.py')
lifetime_patcher=load_module('affordable_video_lifetime_patch',ROOT/'tools/apply_linear_lifetime_patch.py')
sp_patcher=load_module('affordable_video_sp_stream_patch',ROOT/'tools/apply_sp_main_stream_patch.py')
ranklocal_patcher=load_module('affordable_video_ranklocal_patch',ROOT/'tools/apply_ranklocal_loader_patch.py')


def file_sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


class PackageTests(unittest.TestCase):
    def test_all_recipes_pass_static_validation(self):
        recipes=sorted((ROOT/'recipes').glob('*.json'))
        self.assertEqual(len(recipes),29)
        for path in recipes:
            with self.subTest(path=path.name):
                runner.validate_recipe(json.loads(path.read_text()))

    def test_source_profiles_are_explicit(self):
        recipes=[json.loads(path.read_text()) for path in sorted((ROOT/'recipes').glob('*.json'))]
        profiles=[recipe['source_profile'] for recipe in recipes]
        self.assertEqual(profiles.count('pristine-upstream'),13)
        self.assertEqual(profiles.count('adaln-patched-cache-off'),1)
        self.assertEqual(profiles.count('adaln-patched-cache-on'),4)
        self.assertEqual(profiles.count('streamed-quant-loader'),2)
        self.assertEqual(profiles.count('streamed-quant-loader-lifetime-off'),1)
        self.assertEqual(profiles.count('streamed-quant-loader-lifetime-on'),2)
        self.assertEqual(profiles.count('ranklocal-loader-sp-main-stream'),6)

    def test_single_gpu_recipes_have_hardware_and_case_guards(self):
        names=['rtx5090-short','rtx5090-long','rtx-pro6000-short','rtx-pro6000-long',
               'l40s-short','l40s-long','rtx-pro5000-short','rtx-pro5000-long','rtx4090-short','rtx4090-native-short']
        for name in names:
            with self.subTest(name=name):
                recipe=json.loads((ROOT/'recipes'/f'{name}.json').read_text())
                self.assertIn('hardware_guard',recipe)
                self.assertIn(recipe['case_profile'],name)

    def test_source_profile_mismatch_is_rejected(self):
        recipe=json.loads((ROOT/'recipes/b200-1-native-ablation.json').read_text())
        invalid=copy.deepcopy(recipe)
        invalid['patches']=[]
        with self.assertRaisesRegex(ValueError,'patch/cache state'):
            runner.validate_recipe(invalid)

    def test_native_b200_server_args_match_recorded_hashes(self):
        manifest=json.loads((ROOT/'source-manifest.json').read_text())
        for gpu_count in (1,2,4,8):
            with self.subTest(gpu_count=gpu_count):
                public=json.loads((ROOT/f'recipes/b200-{gpu_count}-native.json').read_text())
                serialized=(json.dumps(public['server_args'],indent=2)+'\n').encode()
                self.assertEqual(hashlib.sha256(serialized).hexdigest(),
                                 manifest['archived_native_b200_server_args_sha256'][str(gpu_count)])

    def test_cases_have_expected_schedule(self):
        manifest=json.loads((ROOT/'source-manifest.json').read_text())
        for name,frames in [('short.json',124),('long.json',345)]:
            cases=json.loads((ROOT/'cases'/name).read_text())
            self.assertTrue(cases)
            self.assertTrue(all(case['expected_frames']==frames for case in cases))
            self.assertEqual(file_sha(ROOT/'cases'/name),manifest['cases_sha256'][Path(name).stem])
            self.assertEqual(len(runner.schedule_cases(cases,'performance')),6*len(cases))
            self.assertEqual(len(runner.schedule_cases(cases,'performance',
                {'feasibility':0,'warmups':2,'timings':3})),5*len(cases))

    def test_single_gpu_server_args_match_recorded_hashes(self):
        manifest=json.loads((ROOT/'source-manifest.json').read_text())
        for name in manifest['archived_single_gpu_server_args_canonical_sha256']:
            with self.subTest(name=name):
                public=json.loads((ROOT/'recipes'/f'{name}.json').read_text())['server_args']
                canonical=json.dumps(public,sort_keys=True,separators=(',',':')).encode()
                self.assertEqual(hashlib.sha256(canonical).hexdigest(),
                    manifest['archived_single_gpu_server_args_canonical_sha256'][name])

    def test_manifest_hashes(self):
        manifest=json.loads((ROOT/'source-manifest.json').read_text())
        self.assertEqual(file_sha(ROOT/'extensions/sglang_adaln_extension.py'),
                         manifest['b200_adaln_extension_sha256'])
        self.assertEqual(file_sha(ROOT/'requirements-frozen.txt'),
                         manifest['requirements_frozen_sha256'])
        self.assertEqual(file_sha(ROOT/'model-manifest.json'),
                         manifest['model_manifest_sha256'])
        self.assertEqual(runner.MODEL_MANIFEST_SHA256,
                         manifest['model_manifest_sha256'])
        self.assertEqual(patcher.ORIGINAL_SHA256,
                         manifest['sglang']['original_model_source_sha256'])
        self.assertEqual(patcher.PATCHED_SHA256,
                         manifest['sglang']['b200_patched_model_source_sha256'])
        sources=manifest['sglang']['source_files']
        for tool in (stream_patcher,lifetime_patcher,sp_patcher):
            entry=sources[str(tool.REL)]
            self.assertEqual(tool.ORIGINAL_SHA256,entry['original_sha256'])
            self.assertEqual(tool.PATCHED_SHA256,entry['patched_sha256'])
        ranklocal=manifest['ranklocal_loader']['files']
        self.assertEqual(ranklocal[str(ranklocal_patcher.REL)],{'original_sha256':ranklocal_patcher.ORIGINAL_SHA256,'patched_sha256':ranklocal_patcher.PATCHED_SHA256})
        self.assertEqual(ranklocal[str(ranklocal_patcher.ENCODER_REL)],{'original_sha256':ranklocal_patcher.ENCODER_ORIGINAL_SHA256,'patched_sha256':ranklocal_patcher.ENCODER_PATCHED_SHA256})
        self.assertEqual(ranklocal[str(ranklocal_patcher.HELPER_REL)],{'original_sha256':None,'patched_sha256':ranklocal_patcher.HELPER_SHA256})
        self.assertEqual(file_sha(ROOT/'tools/ranklocal_guard.py'),ranklocal_patcher.HELPER_SHA256)
        runner_files={str(Path('python/sglang')/rel):(orig,patched) for spec in runner.SOURCE_FILES.values() for rel,orig,patched in (spec if isinstance(spec,list) else [spec])}
        for rel,entry in ranklocal.items():
            self.assertEqual(runner_files[rel],(entry['original_sha256'],entry['patched_sha256']))

    def test_ranklocal_recipes_are_guarded(self):
        for n in (2,4,8):
            for length in ('short','long'):
                recipe=json.loads((ROOT/'recipes'/f'rtx5090-{n}-{length}.json').read_text())
                self.assertEqual(recipe['patches'],['ranklocal-loader','sp-main-stream'])
                self.assertEqual(recipe['server_args']['num_gpus'],n)
                self.assertEqual(recipe['server_args']['ulysses_degree'],n)
                self.assertEqual(recipe['expected_resolved']['vae_cpu_offload'],length=='long')
                runner.validate_recipe(recipe)
        recipe=json.loads((ROOT/'recipes/rtx5090-2-short.json').read_text())
        single=copy.deepcopy(recipe);single['server_args']['num_gpus']=1;single['server_args']['ulysses_degree']=1
        with self.assertRaisesRegex(ValueError,'2, 4 or 8'):
            runner.validate_recipe(single)
        slow=copy.deepcopy(recipe);slow['environment']['AFFORDABLE_H3_LOAD_LOCK_TIMEOUT_S']='3600'
        with self.assertRaisesRegex(ValueError,'dist_timeout'):
            runner.validate_recipe(slow)

    def test_readme_uses_one_controller_process(self):
        readme=(ROOT/'README.md').read_text()
        self.assertNotIn('\ntorchrun ',readme)
        self.assertIn('one ordinary Python process',readme)

    def test_patch_against_checkout_when_supplied(self):
        source=os.environ.get('SGLANG_SOURCE')
        if not source:
            self.skipTest('Set SGLANG_SOURCE to run the pinned-source patch check')
        for tool in (patcher,stream_patcher,lifetime_patcher,sp_patcher,ranklocal_patcher):
            with self.subTest(tool=tool.__name__):
                path=Path(source).expanduser().resolve()/tool.REL
                text=path.read_text()
                if tool.digest(text)==tool.PATCHED_SHA256:
                    self.assertEqual(tool.digest(text),tool.PATCHED_SHA256)
                else:
                    self.assertEqual(tool.digest(text),tool.ORIGINAL_SHA256)
                    self.assertEqual(tool.digest(tool.modify(text)),tool.PATCHED_SHA256)

    def test_runtime_source_gate_against_pinned_checkout(self):
        source=os.environ.get('SGLANG_SOURCE')
        if not source:
            self.skipTest('Set SGLANG_SOURCE to run the pinned-source profile check')
        source=Path(source).expanduser().resolve()
        with tempfile.TemporaryDirectory() as temporary:
            package=Path(temporary)/'sglang'; package.mkdir()
            (package/'__init__.py').write_text('')
            for spec in runner.SOURCE_FILES.values():
                for relative,original,_ in (spec if isinstance(spec,list) else [spec]):
                    if original is None:continue  # added by a patch; must be absent on the pristine tree
                    target=package/relative; target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_text((source/'python/sglang'/relative).read_text())
            fake=types.SimpleNamespace(__file__=str(package/'__init__.py'))
            pristine=json.loads((ROOT/'recipes/l40s-short.json').read_text())
            revision=runner.SOURCE_REVISION+'\n'
            with mock.patch.object(runner.subprocess,'check_output',return_value=revision):
                runner.validate_runtime_source(fake,pristine)
            for tool in (stream_patcher,lifetime_patcher):
                relative=Path(tool.REL).relative_to('python/sglang')
                path=package/relative; path.write_text(tool.modify(path.read_text()))
            patched=json.loads((ROOT/'recipes/rtx-pro5000-short.json').read_text())
            with mock.patch.object(runner.subprocess,'check_output',return_value=revision):
                runner.validate_runtime_source(fake,patched)
                with self.assertRaisesRegex(RuntimeError,'source hash mismatch'):
                    runner.validate_runtime_source(fake,pristine)


if __name__=='__main__':
    unittest.main()
