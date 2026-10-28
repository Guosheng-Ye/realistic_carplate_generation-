import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from realistic_carplate_generation import sources


class SourcesAssetsTests(unittest.TestCase):
    def test_empty_detection_can_be_imported(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'detect.jsonl'
            p.write_text(json.dumps(dict(background=str(Path(tmp) / 'a.jpg'), plates=[])) + '\n')
            roots, tasks = sources.import_tasks(p, {}, 'auto')
            self.assertEqual(tasks[0][1], [])

    def test_content_path_and_live_physical_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'a.jpg'; p.write_bytes(b'source')
            digest = hashlib.sha256(b'source').hexdigest()
            row = dict(root='data', path=p.name, sha256=digest); roots = {'data': tmp}; stat = p.stat()
            for frozen, locations in [({digest}, (set(), set(), 1)), (set(), ({str(p)}, set(), 1)),
                                      (set(), (set(), {(stat.st_dev, stat.st_ino)}, 1))]:
                with self.assertRaises(ValueError): sources.check_source_separation(row, roots, frozen, locations)
            sources.check_source_separation(row, roots, set(), (set(), set(), 0))

    def test_unconfigured_holdout_is_reported_not_verified_end_to_end(self):
        from pipeline_fixture import make_fixture
        from realistic_carplate_generation.common import load_config
        from realistic_carplate_generation.planner import build_plan
        from realistic_carplate_generation.pipeline import generate
        cfg = load_config(); cfg.update(format='png', reuse_blue='auto', heldout_registry=None,
            heldout_fingerprints=None, generate_report=False)
        cfg['fonts'] = {'han': 'fonts/preview.ttc', 'latin': 'fonts/plate_latin.ttf'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); make_fixture(root / 'sources', cfg)
            index = sources.prepare(root / 'sources/index.json', None, cfg, 'manifest')
            self.assertIn('heldout_verification', index)
            self.assertEqual(index['heldout_verification'], 'not_configured')
            self.assertEqual(index['heldout_sha256'], [])
            # A stale index field must not enable a policy which was not configured.
            index['heldout_sha256'] = [index['records'][0]['sha256']]
            summary = generate(index, build_plan(index, 4, cfg), root / 'output', cfg, workers=1)
            self.assertEqual(summary['heldout_verification'], 'not_configured')
            self.assertIsNone(summary['heldout_content_intersection'])
            self.assertIsNone(summary['heldout_path_intersection'])
            self.assertIsNone(summary['heldout_physical_intersection'])

    def test_heldout_is_explicit_optional_and_strict(self):
        self.assertTrue(hasattr(sources, 'heldout_hashes'), 'optional heldout loader missing')
        self.assertEqual(sources.heldout_hashes({}), set())
        self.assertEqual(sources.heldout_locations({}), (set(), set(), 0))
        self.assertEqual(sources.separation_status({}, (set(), set(), 0))['heldout_verification'], 'not_configured')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'registry.json'
            with self.assertRaises(OSError): sources.heldout_hashes({'heldout_registry': str(p)})
            p.write_text('{bad')
            with self.assertRaises(ValueError): sources.heldout_hashes({'heldout_registry': str(p)})
            p.write_text('{}')
            with self.assertRaises(ValueError): sources.heldout_hashes({'heldout_registry': str(p)})
            p.write_text('["' + 'a' * 64 + '"]')
            self.assertEqual(sources.heldout_hashes({'heldout_registry': str(p)}), {'a' * 64})
            with self.assertRaises(ValueError): sources.heldout_locations({'heldout_fingerprints': str(p)})
            p.write_text('{bad\n')
            with self.assertRaises(ValueError): sources.heldout_locations({'heldout_fingerprints': str(p)})


if __name__ == '__main__': unittest.main()
