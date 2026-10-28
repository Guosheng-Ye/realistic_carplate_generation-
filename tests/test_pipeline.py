"""Engineering regressions on procedural fixtures, not photo/OCR accuracy."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from unittest.mock import patch

from realistic_carplate_generation.common import load_config, read_json, read_jsonl, write_json, write_jsonl
from realistic_carplate_generation.geometry import polygon_mask
from realistic_carplate_generation.planner import build_plan
from realistic_carplate_generation.pipeline import generate
from realistic_carplate_generation.reporting import report
from realistic_carplate_generation.validation import validate
from pipeline_fixture import make_fixture, require_test_fonts


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rcg-pipeline-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cfg = load_config()
        self.cfg.update(format='png', reuse_blue='auto', heldout_registry=None,
                        heldout_fingerprints=None, generate_report=False)
        # OFL preview glyphs are explicit test assets, never a production fallback.
        self.cfg['fonts'] = dict(han='fonts/preview.ttc', latin='fonts/plate_latin.ttf',
                                 preview='fonts/preview.ttc')
        self.index = make_fixture(self.root / 'sources', self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def assert_png_outside_unchanged(self, output, index, plan):
        records = {r['id']: r for r in index['records']}
        for job in plan['jobs']:
            row = records[job['source_id']]
            original = cv2.imread(str(Path(index['roots'][row['root']]) / row['path']))
            saved = cv2.imread(str(output / job['image']))
            mask = np.zeros(original.shape[:2], np.uint8)
            for plate in job['plates']:
                mask |= polygon_mask(original.shape, plate['quad'])
            np.testing.assert_array_equal(saved[mask == 0], original[mask == 0])
            self.assertTrue(np.any(saved[mask > 0] != original[mask > 0]))

    def test_atomic_resume_and_output_tampering(self):
        plan = build_plan(self.index, 4, self.cfg)
        first = copy.deepcopy(self.cfg)
        first['_stop_after'] = 2
        with self.assertRaises(InterruptedError):
            generate(self.index, plan, self.root / 'run', first, workers=2)
        self.assertFalse((self.root / 'run/labels.jsonl').exists())
        self.assertFalse((self.root / 'run/annotations.jsonl').exists())
        self.assertEqual(read_json(self.root / 'run/progress.json')['completed'], 2)
        bad = copy.deepcopy(self.cfg)
        bad['seed'] += 1
        with self.assertRaisesRegex(ValueError, '恢复配置'):
            generate(self.index, plan, self.root / 'run', bad, workers=2, resume=True)
        stats = generate(self.index, plan, self.root / 'run', self.cfg, workers=2, resume=True)
        self.assertEqual((stats['image_count'], stats['crop_count']), (4, 4))
        self.assertEqual(stats['png_saved_outside_changed_pixels'], 0)
        self.assertEqual(read_json(self.root / 'run/progress.json')['status'], 'complete')
        self.assertFalse((self.root / 'run/review').exists())
        self.assertEqual(stats['ocr_accuracy'], 'not_measured_without_independent_human_GT')
        self.assert_png_outside_unchanged(self.root / 'run', self.index, plan)
        labels = read_jsonl(self.root / 'run/labels.jsonl')
        with (self.root / 'run' / labels[0]['image']).open('ab') as stream:
            stream.write(b'tamper')
        with self.assertRaisesRegex(ValueError, '写盘图片哈希'):
            validate(self.root / 'run', self.cfg)

    def test_data_root_relocation_and_source_tampering(self):
        plan = build_plan(self.index, 4, self.cfg)
        generate(self.index, plan, self.root / 'run', self.cfg, workers=2)
        moved = self.root / 'moved_sources'
        shutil.copytree(self.root / 'sources', moved)
        shutil.rmtree(self.root / 'sources')
        cfg = copy.deepcopy(self.cfg)
        cfg['roots']['fixture'] = str(moved)
        self.assertEqual(validate(self.root / 'run', cfg)['image_count'], 4)
        used = read_json(self.root / 'run/plan.json')['jobs'][0]['source_id']
        record = next(r for r in self.index['records'] if r['id'] == used)
        with (moved / record['path']).open('ab') as stream:
            stream.write(b'changed')
        with self.assertRaisesRegex(ValueError, '来源哈希'):
            validate(self.root / 'run', cfg)

    def test_full_plate_mode_and_multi_output(self):
        index = copy.deepcopy(self.index)
        two = next(r for r in index['records'] if len(r['plates']) == 2)
        green = next(r for r in index['records'] if len(r['plates']) == 1 and r['plates'][0]['colour'] == 'green')
        index['records'] = [two, green]
        cfg = copy.deepcopy(self.cfg)
        cfg['render_mode'] = 'full'
        plan = build_plan(index, 4, cfg, 'all')
        summary = generate(index, plan, self.root / 'multi', cfg, workers=2)
        self.assertEqual((summary['image_count'], summary['plate_count'], summary['crop_count']), (4, 5, 5))
        self.assertFalse((self.root / 'multi/labels.jsonl').exists())
        annotations = read_jsonl(self.root / 'multi/annotations.jsonl')
        self.assertEqual(len(annotations), 4)
        self.assertEqual(sum(len(row['plates']) for row in annotations), 5)
        self.assertEqual(len(list((self.root / 'multi/images').glob('*.png'))), 4)
        self.assertEqual(len(list((self.root / 'multi/crops/images').glob('*.png'))), 5)
        self.assertEqual(len(read_jsonl(self.root / 'multi/crops/labels.jsonl')), 5)
        self.assertEqual(summary['png_saved_outside_changed_pixels'], 0)
        self.assert_png_outside_unchanged(self.root / 'multi', index, plan)


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='rcg-report-')
        self.root = Path(self.temp.name)
        source = self.root / 'sources'
        source.mkdir()
        Image.new('RGB', (600, 220), '#285c99').save(source / 'source.png')
        (self.root / 'images').mkdir()
        image = 'images/output&sample.png'
        Image.new('RGB', (600, 220), '#326090').save(self.root / image)
        quad = [[499, 179], [60, 179], [60, 40], [499, 40]]
        self.text = '京A<&"123'
        write_json(self.root / 'index.json', dict(roots={'fixture': str(source)},
            records=[dict(id='procedural', root='fixture', path='source.png')]))
        write_json(self.root / 'plan.json', dict(jobs=[dict(source_id='procedural')]))
        write_jsonl(self.root / 'annotations.jsonl', [dict(image=image,
            plates=[dict(text=self.text, plate_type='blue', quad=quad)])])
        write_json(self.root / 'summary.json', dict(scope='procedural only <not real-photo accuracy>'))
        self.cfg = load_config()
        write_json(self.root / 'config.json', self.cfg)

    def tearDown(self):
        self.temp.cleanup()

    def test_missing_preview_font_cli_retains_unicode_and_escapes_html(self):
        self.cfg['fonts'] = dict(preview='fonts/not-installed.ttf')
        config = self.root / 'report-config.json'
        write_json(config, self.cfg)
        proc = subprocess.run([sys.executable, '-B', '-m', 'realistic_carplate_generation',
            'report', '--dataset', str(self.root), '--config', str(config)],
            capture_output=True, text=True, env=dict(os.environ), timeout=30)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        result = json.loads(proc.stdout)
        self.assertEqual(result.get('preview_font_status'), 'ascii_fallback')
        page = (self.root / 'review/index.html').read_text(encoding='utf-8')
        self.assertIn('京A&lt;&amp;&quot;123', page)
        self.assertIn('output&amp;sample.png', page)
        self.assertIn('&lt;not real-photo accuracy&gt;', page)
        self.assertIn('thumbnail labels use ASCII identifiers', page)
        self.assertNotIn(self.text, page)
        self.assertEqual(read_json(self.root / 'review/samples.json')[0]['text'], self.text)
        with Image.open(self.root / 'review/000001.jpg') as preview:
            self.assertEqual(preview.size, (1000, 174))
        self.assertTrue((self.root / 'review/overview.jpg').is_file())

    def test_preview_font_is_configurable_without_changing_plate_fonts(self):
        self.cfg['fonts'] = dict(preview='fonts/plate_latin.ttf',
                                 han='fonts/missing-production.ttf', latin='fonts/missing-production.ttf')
        require_test_fonts(self.cfg, roles=('preview',))
        before = copy.deepcopy(self.cfg)
        with patch.object(ImageFont, 'truetype', wraps=ImageFont.truetype) as load_font:
            result = report(self.root, self.cfg)
        self.assertTrue(str(load_font.call_args.args[0]).endswith('/fonts/plate_latin.ttf'))
        self.assertEqual(load_font.call_args.kwargs.get('index'), 0)
        self.assertEqual(result.get('preview_font_status'), 'configured')
        self.assertEqual(self.cfg, before)
        self.assertTrue((self.root / 'review/index.html').is_file())

    def test_preview_font_accepts_ttc_face_configuration(self):
        self.cfg['fonts'] = dict(preview=dict(path='fonts/preview.ttc', index=2))
        require_test_fonts(self.cfg, roles=('preview',))
        with patch.object(ImageFont, 'truetype', wraps=ImageFont.truetype) as load_font:
            result = report(self.root, self.cfg)
        self.assertEqual(load_font.call_args.kwargs.get('index'), 2)
        self.assertEqual(result.get('preview_font_status'), 'configured')

    def test_missing_preview_font_draws_only_ascii_thumbnail_labels(self):
        self.cfg['fonts'] = dict(preview='fonts/not-installed.ttf')
        labels = []
        original = ImageDraw.ImageDraw.text

        def capture_label(draw, xy, text, *args, **kwargs):
            labels.append(text)
            return original(draw, xy, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, 'text', capture_label):
            result = report(self.root, self.cfg)
        self.assertEqual(result.get('preview_font_status'), 'ascii_fallback')
        self.assertEqual(labels, ['Source', 'Sample 000001 / blue'])
        self.assertTrue(all(label.isascii() for label in labels))
