"""Resource/config contracts, including a source-independent wheel install."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import patch

from realistic_carplate_generation import common
from pipeline_fixture import require_test_fonts

REPO = Path(__file__).resolve().parents[1]
LOCAL_FONTS = {'assets_dir': str(REPO / 'assets'),
               'fonts': {'han': 'fonts/preview.ttc', 'latin': 'fonts/plate_latin.ttf'}}


def install_test_wheel(test, tmp):
    build = tmp / 'build_source'
    build.mkdir()
    shutil.copy(REPO / 'pyproject.toml', build)
    for name in ('README.md', 'LICENSE', 'NOTICE'):
        if (REPO / name).is_file():
            shutil.copy(REPO / name, build)
    shutil.copytree(REPO / 'src', build / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.egg-info'))
    result = subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-index', '--no-deps', '--no-build-isolation',
                             '--wheel-dir', str(tmp / 'wheels'), str(build)], capture_output=True, text=True)
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    wheel = next((tmp / 'wheels').glob('*.whl'))
    installed = tmp / 'installed'
    result = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', '--no-index',
                             '--target', str(installed), str(wheel)], capture_output=True, text=True)
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    return installed


class PackagingResourceTests(unittest.TestCase):
    def test_wheel_includes_project_license_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            install_test_wheel(self, tmp)
            with zipfile.ZipFile(next((tmp / 'wheels').glob('*.whl'))) as archive:
                metadata = next(p for p in archive.namelist() if p.endswith('.dist-info/METADATA'))
                self.assertIn('License-Expression: Apache-2.0', archive.read(metadata).decode())
                for name in ('LICENSE', 'NOTICE'):
                    self.assertTrue(any(p.endswith('.dist-info/licenses/' + name) for p in archive.namelist()))

    def test_published_models_have_licenses_integrity_and_no_private_paths(self):
        entries = common.read_json(REPO / 'assets/models/provenance.json')['models']
        manifest = {row['path']: row for row in common.read_json(REPO / 'assets/manifest.json')['files']}
        self.assertEqual(len(entries), 4)
        for row in entries:
            with self.subTest(model=row['path']):
                self.assertTrue(row['distributed'])
                self.assertIn(row['license_spdx_id'], ('Apache-2.0', 'AGPL-3.0'))
                self.assertTrue((REPO / row['licence_material']).is_file())
                data = (REPO / row['path']).read_bytes()
                self.assertEqual(len(data), row['bytes'])
                self.assertEqual(common.sha256(REPO / row['path']), row['sha256'])
                self.assertEqual(manifest[row['path']]['sha256'], row['sha256'])
                # Keep official HyperLPR binaries byte-identical, including their
                # already-public upstream build metadata. Local exports must be clean.
                if not row['path'].startswith('assets/models/hyperlpr3/'):
                    self.assertIsNone(re.search(rb'/(?:home|Users)/[A-Za-z0-9_.-]+/', data),
                                      'Local model contains a private home path: ' + row['path'])
        self.assertTrue((REPO / 'assets/models/e3/verification.json').is_file())

    def test_fonts_are_local_resources_not_public_assets(self):
        manifest = common.read_json(REPO / 'assets/manifest.json')
        self.assertFalse(any(row['path'].startswith('assets/fonts/') for row in manifest['files']))
        self.assertIn('/assets/fonts/', (REPO / '.gitignore').read_text().splitlines())
        self.assertFalse((REPO / 'THIRD_PARTY_NOTICES.md').exists())

    def test_defaults_are_package_data_not_project_files(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, RCG_HOME=tmp):
            cfg = common.load_config()
            self.assertEqual(cfg['fonts']['han'], 'fonts/plate_han.ttf')
            self.assertEqual(cfg['models'], {'yolo_hints': False, 'e3_ocr': False})
            self.assertEqual(cfg['roots'], {})
            self.assertIsNone(cfg['heldout_registry'])
            self.assertIsNone(cfg['heldout_fingerprints'])
            self.assertFalse(cfg['generate_report'])
            cfg['fonts']['han'] = 'changed'
            self.assertEqual(common.load_config()['fonts']['han'], 'fonts/plate_han.ttf')
            self.assertEqual(common.project_home(), Path(tmp))

    def test_config_nested_overrides_do_not_erase_font_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'override.json'
            path.write_text(json.dumps({'fonts': {'han': 'fonts/preview.ttc'}, 'roots': {'a': tmp}}))
            cfg = common.load_config(path, roots=['b=' + tmp])
            self.assertEqual(cfg['fonts']['latin'], 'fonts/plate_latin.ttf')
            self.assertEqual(set(cfg['roots']), {'a', 'b'})

    def test_wheel_defaults_without_any_user_fonts(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            installed = install_test_wheel(self, tmp)
            probe = '''
import importlib.metadata as md
from pathlib import Path
from realistic_carplate_generation.common import load_config, asset_check
from realistic_carplate_generation.rendering import renderer
import realistic_carplate_generation.common as c
assert Path(c.__file__).is_relative_to(Path.cwd() / 'installed'), c.__file__
cfg = load_config()
assert cfg['fonts']['han'] == 'fonts/plate_han.ttf'
assert asset_check([])['files'] == []
assert all('extra == "detection"' in r for r in md.requires('realistic-carplate-generation') if r.startswith('onnxruntime'))
try:
    renderer(cfg)
except ValueError as exc:
    assert 'fonts/plate_han.ttf' in str(exc), str(exc)
else:
    raise AssertionError('missing user fonts must not silently select system fonts')
print('wheel_without_fonts_passed')
'''
            env = dict(os.environ, PYTHONPATH=str(installed), RCG_HOME=str(tmp / 'outside'))
            env.pop('RCG_ASSETS_DIR', None)
            result = subprocess.run([sys.executable, '-B', '-c', probe], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('wheel_without_fonts_passed', result.stdout)

    def test_isolated_wheel_has_canonical_defaults_and_optional_detection(self):
        require_test_fonts(LOCAL_FONTS)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            installed = install_test_wheel(self, tmp)
            external = tmp / 'user_assets/fonts'
            external.mkdir(parents=True)
            for name in ('preview.ttc', 'plate_latin.ttf'):
                shutil.copy(REPO / 'assets/fonts' / name, external / name)
            probe = '''
import importlib.metadata as md, json
from pathlib import Path
import realistic_carplate_generation.common as c
assert Path(c.__file__).is_relative_to(Path.cwd() / 'installed'), c.__file__
cfg = c.load_config()
assert cfg['fonts']['han'] == 'fonts/plate_han.ttf'
assert cfg['models'] == {'yolo_hints': False, 'e3_ocr': False}
assert c.asset_check(required=[])['files'] == []
req = md.requires('realistic-carplate-generation')
assert all('extra == "detection"' in r for r in req if r.startswith('onnxruntime'))
try:
    from realistic_carplate_generation.rendering import renderer
    renderer(cfg)
except ValueError as exc:
    assert 'fonts/plate_han.ttf' in str(exc), str(exc)
else:
    raise AssertionError('wheel must not silently use system fonts')
cfg['assets_dir'] = str(Path.cwd() / 'user_assets')
cfg['fonts']['han'] = 'fonts/preview.ttc'
from realistic_carplate_generation.rendering import render_plate, render_asset_check
identity = render_asset_check(cfg)
c.write_json(Path(cfg['assets_dir']) / 'manifest.json', identity)
assert render_plate('京A12345', 'blue', cfg=cfg).shape == (280, 880, 3)
with (Path(cfg['assets_dir']) / 'fonts/preview.ttc').open('ab') as stream:
    stream.write(b'tamper')
try:
    renderer(cfg)
except ValueError as exc:
    assert '哈希改变' in str(exc), str(exc)
else:
    raise AssertionError('cached renderer accepted tampered wheel assets')
print(json.dumps({'wheel_probe': 'passed', 'config': cfg}, ensure_ascii=False))
'''
            env = dict(os.environ, PYTHONPATH=str(installed), RCG_HOME=str(tmp / 'outside'))
            env.pop('RCG_ASSETS_DIR', None)
            (tmp / 'outside').mkdir()
            result = subprocess.run([sys.executable, '-c', probe], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('"wheel_probe": "passed"', result.stdout)


class SelectiveResourceTests(unittest.TestCase):
    def test_asset_directory_precedence_legacy_prefix_and_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {'assets_dir': str(root / 'explicit')}
            with patch.dict(os.environ, RCG_ASSETS_DIR=str(root / 'env')):
                self.assertEqual(common.asset_path('assets/fonts/a.ttf', cfg), root / 'explicit/fonts/a.ttf')
                self.assertEqual(common.asset_path('fonts/a.ttf'), root / 'env/fonts/a.ttf')
            with self.assertRaisesRegex(ValueError, '越界'):
                common.asset_path('../escape', cfg)
            (root / 'explicit').mkdir()
            (root / 'explicit/link').symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, '越界'):
                common.asset_path('link/escape', cfg)
            with self.assertRaisesRegex(ValueError, '越过'):
                common.source_path({'root': 'data', 'path': '../escape'}, {'data': str(root / 'data')})

    def test_selective_manifest_tamper_missing_and_unregistered_user_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'fonts').mkdir()
            font = root / 'fonts/a.ttf'
            font.write_bytes(b'font')
            cfg = {'assets_dir': tmp}
            self.assertEqual(common.asset_check([], cfg)['files'], [])
            identity = common.asset_check(['fonts/a.ttf'], cfg)
            self.assertEqual(identity['files'][0]['sha256'], common.sha256(font))
            manifest = {'schema_version': 1, 'files': identity['files'] + [
                {'path': 'assets/models/disabled.onnx', 'bytes': 4, 'sha256': '0' * 64}]}
            common.write_json(root / 'manifest.json', manifest)
            self.assertEqual(common.asset_check(['assets/fonts/a.ttf'], cfg), identity)
            font.write_bytes(b'evil')
            with self.assertRaisesRegex(ValueError, '哈希改变'):
                common.asset_check(['fonts/a.ttf'], cfg)
            font.unlink()
            with self.assertRaisesRegex(ValueError, 'fonts/a.ttf'):
                common.asset_check(['fonts/a.ttf'], cfg)
            with self.assertRaises(ValueError):
                common.asset_check(None, cfg)

    def test_renderer_explicit_fonts_identity_relocation_and_cached_tamper(self):
        from realistic_carplate_generation import rendering
        require_test_fonts(LOCAL_FONTS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / 'assets'
            (assets / 'fonts').mkdir(parents=True)
            for name in ('preview.ttc', 'plate_latin.ttf'):
                shutil.copy(REPO / 'assets/fonts' / name, assets / 'fonts' / name)
            cfg = common.load_config()
            cfg.update(assets_dir=str(assets))
            cfg['fonts']['han'] = 'fonts/preview.ttc'  # Explicit test font, not a production fallback.
            identity = rendering.render_asset_check(cfg)
            common.write_json(assets / 'manifest.json', identity)
            self.assertEqual(len(identity['files']), 2)
            image = rendering.render_plate('京A12345', 'blue', cfg=cfg)
            self.assertEqual(image.shape, (280, 880, 3))
            reporting_cfg = dict(cfg, generate_report=True,
                                 fonts=dict(cfg['fonts'], preview='fonts/missing-preview.ttc'))
            self.assertEqual(rendering.render_asset_check(reporting_cfg), identity)
            # Exercise the same explicit fonts through a real full replacement.
            import cv2
            import numpy as np
            scene = np.full((120, 300, 3), 100, dtype=np.uint8)
            source = root / 'scene.png'
            cv2.imwrite(str(source), scene)
            record = {'root': 'data', 'path': source.name, 'sha256': common.sha256(source),
                      'width': 300, 'height': 120}
            target = {'colour': 'blue', 'plate_type': 'blue', 'text': '京A12345', 'seed': 1,
                      'quad': [[240, 80], [40, 80], [40, 30], [240, 30]], 'target_id': 'test'}
            job_cfg = dict(cfg, render_mode='full', augment=False, format='png', export_crops=False)
            encoded, crops, audit = rendering.render_job(record, {'plates': [target]}, {'data': str(root)}, job_cfg)
            output = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(output.shape, scene.shape)
            self.assertFalse(crops)
            self.assertEqual(audit['png_saved_outside_changed_pixels'], 0)
            moved = root / 'moved'
            shutil.copytree(assets, moved)
            relocated = dict(cfg, assets_dir=str(moved))
            self.assertEqual(rendering.render_asset_check(relocated), identity)
            with (assets / 'fonts/preview.ttc').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, '哈希改变'):
                rendering.renderer(cfg)
            (moved / 'fonts/preview.ttc').unlink()
            with self.assertRaisesRegex(ValueError, 'fonts/preview.ttc'):
                rendering.renderer(relocated)

    def test_render_job_validates_quad_before_rendering(self):
        import cv2
        import numpy as np
        from realistic_carplate_generation import rendering
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'scene.png'
            cv2.imwrite(str(path), np.zeros((80, 200, 3), dtype=np.uint8))
            record = {'root': 'data', 'path': path.name, 'sha256': common.sha256(path), 'width': 200, 'height': 80}
            target = {'colour': 'blue', 'plate_type': 'blue', 'text': '京A12345', 'seed': 1,
                      'quad': [[220, 60], [20, 60], [20, 10], [220, 10]]}
            cfg = common.load_config()
            cfg['assets_dir'] = str(Path(tmp) / 'missing_fonts')
            with self.assertRaisesRegex(ValueError, '四角越界'):
                rendering.render_job(record, {'plates': [target]}, {'data': tmp}, cfg)


if __name__ == '__main__':
    unittest.main()
