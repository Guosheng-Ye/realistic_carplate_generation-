"""The repository entry point: edit one config, run one command."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]


def runner():
    spec = importlib.util.spec_from_file_location('rcg_user_run', REPO / 'run.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunTests(unittest.TestCase):
    def test_default_config_prefers_ignored_local_file_without_changing_template(self):
        module = runner()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / 'config.json'
            template.write_text('{"count": 100}')
            with patch.object(module, 'ROOT', root):
                self.assertEqual(module.default_config(), template)
                local = root / 'config.local.json'
                local.write_text('{"count": 3}')
                self.assertEqual(module.default_config(), local)
                self.assertEqual(template.read_text(), '{"count": 100}')

    def test_single_plate_type_generation_does_not_require_mixed_quotas(self):
        import inspect
        from realistic_carplate_generation.rules import make_samples, valid_text, summarize
        self.assertIn('plate_type', inspect.signature(make_samples).parameters)
        for kind in ('blue', 'nev_small', 'nev_large'):
            with self.subTest(kind=kind):
                rows = make_samples(37, seed=7, plate_type=kind)
                self.assertEqual(len(rows), 37)
                self.assertTrue(all(row['plate_type'] == kind and valid_text(row['text'], kind) for row in rows))
                self.assertEqual(summarize(rows)['plate_type'][kind], 37)

    def test_single_type_replacement_uses_only_matching_source_colours(self):
        from realistic_carplate_generation.common import load_config
        from realistic_carplate_generation.planner import build_plan
        from test_planner import record
        for kind in ('blue', 'nev_small', 'nev_large'):
            with self.subTest(kind=kind):
                cfg = load_config(); cfg.update(plate_type=kind, reuse_blue='auto')
                source = record('mixed', ['blue', 'green'])
                source['plates'][1]['native_height'] = 80
                plan = build_plan({'records': [source]}, 3, cfg)
                self.assertEqual(plan['plate_count'], 3)
                self.assertTrue(all(p['plate_type'] == kind for job in plan['jobs'] for p in job['plates']))
                expected = 'blue' if kind == 'blue' else 'green'
                self.assertTrue(all(p['colour'] == expected for job in plan['jobs'] for p in job['plates']))

    def test_bad_settings_fail_before_creating_output(self):
        cases = [{'count': 0}, {'workers': 0}, {'mode': 'unknown'}, {'format': 'jpeg'},
                 {'resume': 'yes'}, {'output': ''}, {'fonts': {'han': 123}}, {'cout': 3}]
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            for case in cases:
                with self.subTest(case=case):
                    config.write_text(json.dumps(case))
                    with self.assertRaises(ValueError):
                        runner().load_settings(config)
            self.assertFalse((Path(tmp) / 'outputs').exists())

    def test_render_writes_images_labels_and_refuses_overwrite(self):
        import numpy as np
        from realistic_carplate_generation.common import read_jsonl, sha256
        module = runner()
        self.assertTrue(hasattr(module, 'run_render'), 'config-driven rendering is missing')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps({'count': 3, 'output': 'result', 'plate_type': 'blue'}))
            cfg = module.load_settings(config)
            face = np.full((140, 440, 3), 80, dtype=np.uint8)
            with patch('realistic_carplate_generation.rendering.renderer'), \
                    patch('realistic_carplate_generation.rendering.render_plate', return_value=face):
                result = module.run_render(cfg)
                self.assertEqual(result['image_count'], 3)
                output = Path(cfg['output'])
                labels = read_jsonl(output / 'labels.jsonl')
                self.assertEqual(len(labels), 3)
                self.assertTrue(all((output / row['image']).is_file() for row in labels))
                before = sha256(output / labels[0]['image'])
                with self.assertRaisesRegex(ValueError, '输出目录'):
                    module.run_render(cfg)
                self.assertEqual(sha256(output / labels[0]['image']), before)

    def test_replace_automatically_prepares_plans_and_resumes_without_redetection(self):
        module = runner()
        self.assertTrue(hasattr(module, 'run_replace'), 'one-step photo replacement is missing')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base / 'photos').mkdir()
            config = base / 'config.json'
            config.write_text(json.dumps({'mode': 'replace', 'input': 'photos', 'output': 'result', 'count': 2}))
            cfg = module.load_settings(config)
            index = {'records': [], 'roots': {}}
            plan = {'image_count': 2, 'targets': 'single', 'jobs': []}
            with patch('realistic_carplate_generation.rendering.renderer'), \
                    patch('realistic_carplate_generation.sources.prepare', return_value=index) as prepare, \
                    patch('realistic_carplate_generation.planner.build_plan', return_value=plan) as planner, \
                    patch('realistic_carplate_generation.pipeline.generate', return_value={'image_count': 2}) as generate:
                module.run_replace(cfg)
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(planner.call_count, 1)
                passed = generate.call_args.args[3]
                self.assertNotIn('resume', passed)
                self.assertNotIn('output', passed)
                self.assertEqual(passed['plate_type'], 'blue')
                output = Path(cfg['output']); output.mkdir()
                for name, value in [('index.json', index), ('plan.json', plan), ('config.json', passed)]:
                    (output / name).write_text(json.dumps(value))
                cfg['resume'] = True
                module.run_replace(cfg)
                self.assertEqual(prepare.call_count, 1)
                self.assertEqual(planner.call_count, 1)
                self.assertTrue(generate.call_args.kwargs['resume'])
                cfg['count'] = 3
                with self.assertRaisesRegex(ValueError, 'count'):
                    module.run_replace(cfg)

    def test_main_uses_config_without_a_subcommand(self):
        import contextlib
        import io
        module = runner()
        self.assertTrue(hasattr(module, 'main'), 'one-command CLI is missing')
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            config.write_text(json.dumps({'count': 1}))
            with patch.object(module, 'run_render', return_value={'image_count': 1}), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(module.main(['--config', str(config)]), 0)
            self.assertIn('完成', output.getvalue())

    def test_real_single_type_replace_validates_and_resumes(self):
        from pipeline_fixture import make_fixture
        from realistic_carplate_generation.common import load_config, read_json, sha256
        cfg = load_config()
        cfg['fonts'] = {'han': 'fonts/preview.ttc', 'latin': 'fonts/plate_latin.ttf'}
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            make_fixture(base / 'input', cfg)
            config = base / 'config.json'
            config.write_text(json.dumps({'mode': 'replace', 'input': 'input/index.json',
                'output': 'result', 'count': 3, 'plate_type': 'blue', 'workers': 1,
                'format': 'png', 'augment': False,
                'fonts': {'han': str(REPO / 'assets/fonts/preview.ttc'),
                          'latin': str(REPO / 'assets/fonts/plate_latin.ttf')}}))
            import contextlib
            import io
            module = runner(); settings = module.load_settings(config)
            with contextlib.redirect_stdout(io.StringIO()) as console:
                result = module.run_replace(settings)
            self.assertNotIn('"stage"', console.getvalue())
            self.assertIn('3/3', console.getvalue())
            self.assertEqual(result['image_count'], 3)
            self.assertEqual(result['crop_count'], 3)
            self.assertEqual(result['png_saved_outside_changed_pixels'], 0)
            image = base / 'result/images/000001.png'
            before = sha256(image)
            settings['resume'] = True
            self.assertEqual(module.run_replace(settings)['image_count'], 3)
            self.assertEqual(sha256(image), before)
            self.assertEqual(read_json(base / 'result/progress.json')['status'], 'complete')

    def test_paths_are_relative_to_config_not_working_directory(self):
        self.assertTrue((REPO / 'run.py').is_file(), 'one-command entry point is missing')
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            config = base / 'config.json'
            config.write_text(json.dumps({'input': 'photos', 'output': 'result', 'count': 3,
                                         'fonts': {'han': 'fonts/han.ttf', 'latin': 'fonts/latin.ttf'}}))
            with patch('os.getcwd', return_value='/'):
                cfg = runner().load_settings(config)
            self.assertEqual(cfg['input'], str(base / 'photos'))
            self.assertEqual(cfg['output'], str(base / 'result'))
            self.assertEqual(cfg['fonts']['han'], str(base / 'fonts/han.ttf'))
            self.assertEqual(cfg['fonts']['latin'], str(base / 'fonts/latin.ttf'))
            self.assertEqual(cfg['count'], 3)
            self.assertEqual(cfg['mode'], 'render')


if __name__ == '__main__':
    unittest.main()
