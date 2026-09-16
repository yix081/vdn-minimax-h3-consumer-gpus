"""Small offline fixtures exercise model staging without downloading weights."""
import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('stage_model', ROOT/'tools/stage_model.py')
staging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(staging)


class StagingTests(unittest.TestCase):
    def fixture(self, root):
        data = b'model-weight-fixture'
        manifest = {'schema_version': 1, 'models': {'test': {
            'repo': 'example/model', 'revision': '1'*40,
            'files': [{'file': 'weights/model.bin', 'bytes': len(data),
                       'sha256': hashlib.sha256(data).hexdigest()}]}}}
        path = root/'manifest.json'
        path.write_text(json.dumps(manifest))
        def download(**kwargs):
            self.assertEqual(kwargs['revision'], '1'*40)
            self.assertFalse(kwargs['token'])
            target = Path(kwargs['cache_dir'])/'models--example--model'/'snapshots'/('1'*40)
            (target/'weights').mkdir(parents=True, exist_ok=True)
            (target/'weights/model.bin').write_bytes(data)
            return str(target)
        return path, manifest, download

    def test_staging_is_pinned_and_verify_only_does_not_download_or_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,download=self.fixture(root);home=root/'cache'
            receipt=staging.stage(home,manifest_path=path,downloader=download)
            receipt_path=home/'staging-receipt.json';before=receipt_path.read_bytes()
            self.assertEqual(receipt['status'],'verified')
            self.assertEqual(staging.stage(home,manifest_path=path,verify_only=True),receipt)
            self.assertEqual(receipt_path.read_bytes(),before)

    def test_only_file_by_file_repositories_are_filtered(self):
        """SGLang resolves the base model with snapshot_download and the hub's offline mode
        rejects a partial snapshot, so those repositories download whole; MiniMax-H3 is read
        file by file and downloads only the manifest files."""
        seen={}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest,download=self.fixture(root)
            def recording(**kwargs):
                seen[kwargs['repo_id']]=kwargs.get('allow_patterns');return download(**kwargs)
            staging.stage(root/'a',manifest_path=path,downloader=recording)
            self.assertEqual(seen['example/model'],['weights/model.bin'])
            manifest['models']['test']['repo']='OpenVDN/vdn-minimax-h3'
            path.write_text(json.dumps(manifest))
            def whole(**kwargs):
                seen[kwargs['repo_id']]=kwargs.get('allow_patterns')
                target=Path(kwargs['cache_dir'])/'models--OpenVDN--vdn-minimax-h3'/'snapshots'/('1'*40)
                (target/'weights').mkdir(parents=True,exist_ok=True);(target/'weights/model.bin').write_bytes(b'model-weight-fixture');return str(target)
            staging.stage(root/'b',manifest_path=path,downloader=whole)
            self.assertIsNone(seen['OpenVDN/vdn-minimax-h3'])
        self.assertIn('MiniMaxAI/MiniMax-H3',{m['repo'] for m in staging.load_manifest()['models'].values()})
        self.assertNotIn('MiniMaxAI/MiniMax-H3',staging.COMPLETE_SNAPSHOT_REPOS)

    def test_corruption_is_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,download=self.fixture(root);home=root/'cache'
            staging.stage(home,manifest_path=path,downloader=download)
            weight=next(home.rglob('model.bin'));weight.write_bytes(b'x'*weight.stat().st_size)
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                staging.stage(home,manifest_path=path,verify_only=True)

    def test_existing_different_default_revision_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,_,download=self.fixture(root);home=root/'cache'
            ref=home/'hub/models--example--model/refs/main';ref.parent.mkdir(parents=True)
            ref.write_text('2'*40)
            with self.assertRaisesRegex(RuntimeError,'another revision'):
                staging.stage(home,manifest_path=path,downloader=download)
            self.assertEqual(ref.read_text(),'2'*40)

    def test_no_refs_are_published_after_failed_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest,download=self.fixture(root);home=root/'cache'
            manifest['models']['test']['files'][0]['sha256']='0'*64;path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError,'checksum mismatch'):
                staging.stage(home,manifest_path=path,downloader=download)
            self.assertFalse(list(home.rglob('main')))
            self.assertFalse((home/'staging-receipt.json').exists())

    def test_unsafe_manifest_paths_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,manifest,_=self.fixture(root)
            for name in ['../outside','/outside','weights/../outside','weights\\outside']:
                modified=copy.deepcopy(manifest);modified['models']['test']['files'][0]['file']=name
                path.write_text(json.dumps(modified))
                with self.subTest(name=name),self.assertRaises(ValueError):staging.load_manifest(path)

    def test_recorded_model_revisions_and_files(self):
        manifest=staging.load_manifest()
        self.assertEqual(len(manifest['models']),3)
        self.assertEqual(sum(len(m['files']) for m in manifest['models'].values()),78)
        self.assertEqual(manifest['models']['overlay']['revision'],'7de18275dddfe59da36a234e222bcdd274963bc3')


if __name__ == '__main__':
    unittest.main()
