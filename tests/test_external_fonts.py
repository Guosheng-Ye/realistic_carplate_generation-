"""External font paths retain integrity without leaking deployment locations."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from realistic_carplate_generation import common, rendering, reporting

REPO = Path(__file__).resolve().parents[1]


class ExternalFontTests(unittest.TestCase):
    def test_font_path_absolute_tilde_and_legacy_boundary(self):
        self.assertTrue(callable(getattr(common, 'font_path', None)), 'font_path is required')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = {'assets_dir': str(root / 'assets')}
            self.assertEqual(common.font_path(root / 'outside.ttf', cfg), root / 'outside.ttf')
            with patch.dict('os.environ', HOME=tmp):
                self.assertEqual(common.font_path('~/outside.ttf', cfg), root / 'outside.ttf')
            self.assertEqual(common.font_path('assets/fonts/a.ttf', cfg), root / 'assets/fonts/a.ttf')
            with self.assertRaisesRegex(ValueError, '越界'):
                common.font_path('../outside.ttf', cfg)
            for path in ('../outside.ttf', str(root / 'outside.ttf')):
                with self.assertRaisesRegex(ValueError, '越界'):
                    common.asset_path(path, cfg)
            with self.assertRaisesRegex(ValueError, '越过'):
                common.source_path({'root': 'data', 'path': '../outside.ttf'}, {'data': str(root / 'data')})
    def external_config(self, root):
        root.mkdir(parents=True)
        for name in ('preview.ttc', 'plate_latin.ttf'):
            source = REPO / 'assets/fonts' / name
            if not source.is_file():
                self.skipTest('Local test font unavailable: ' + name)
            shutil.copyfile(source, root / name)
        return {'assets_dir': str(root.parent / 'unrelated_assets'), 'fonts': {
            'han': {'path': str(root / 'preview.ttc'), 'index': 0},
            'latin': str(root / 'plate_latin.ttf')}}

    def test_external_render_identity_is_content_based_and_relocation_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.external_config(root / 'first')
            image = rendering.render_plate('京A12345', 'blue', cfg=cfg)
            self.assertEqual(image.shape, (280, 880, 3))
            identity = rendering.render_asset_check(cfg)
            self.assertNotIn(tmp, json.dumps(identity))
            self.assertEqual(identity['fonts']['han']['index'], 0)
            moved = self.external_config(root / 'second')
            self.assertEqual(rendering.render_asset_check(moved), identity)
            before = rendering.renderer(cfg)
            with (root / 'first/plate_latin.ttf').open('ab') as stream:
                stream.write(b'changed-content')
            self.assertNotEqual(rendering.render_asset_check(cfg), identity)
            self.assertIsNot(rendering.renderer(cfg), before)
            (root / 'first/preview.ttc').unlink()
            with self.assertRaises(ValueError):
                rendering.renderer(cfg)
    def test_absolute_registered_font_cannot_bypass_manifest_even_when_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.external_config(root / 'assets/fonts')
            cfg['assets_dir'] = str(root / 'assets')
            identity = common.asset_check(['fonts/preview.ttc', 'fonts/plate_latin.ttf'], cfg)
            common.write_json(root / 'assets/manifest.json', identity)
            before = rendering.render_asset_check(cfg)
            self.assertEqual(before['files'], identity['files'])
            rendering.renderer(cfg)
            with (root / 'assets/fonts/preview.ttc').open('ab') as stream:
                stream.write(b'tamper')
            with self.assertRaisesRegex(ValueError, '哈希改变'):
                rendering.renderer(cfg)
    def test_report_loads_external_preview_and_missing_preview_keeps_ascii_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = self.external_config(root / 'fonts')
            cfg['fonts']['preview'] = {'path': str(root / 'fonts/preview.ttc'), 'index': 0}
            original_fonts = json.loads(json.dumps(cfg['fonts']))
            common.write_json(root / 'index.json', {'roots': {}, 'records': []})
            common.write_json(root / 'plan.json', {'jobs': []})
            common.write_jsonl(root / 'annotations.jsonl', [])
            common.write_json(root / 'summary.json', {})
            result = reporting.report(root, cfg)
            self.assertEqual(result['preview_font_status'], 'configured')
            self.assertEqual(cfg['fonts'], original_fonts)
            cfg['fonts']['preview'] = str(root / 'missing-preview.ttc')
            self.assertEqual(reporting.report(root, cfg)['preview_font_status'], 'ascii_fallback')
            self.assertEqual(cfg['fonts']['han'], original_fonts['han'])
            self.assertEqual(cfg['fonts']['latin'], original_fonts['latin'])


if __name__ == '__main__':
    unittest.main()
