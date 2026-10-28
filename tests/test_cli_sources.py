import contextlib
import inspect
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from realistic_carplate_generation import api, cli, sources


class PublicFlowTests(unittest.TestCase):
    def test_core_doctor_accepts_missing_optional_runtime_metadata(self):
        original = cli.importlib.metadata.version
        def installed_version(name):
            if name == 'onnxruntime':
                raise cli.importlib.metadata.PackageNotFoundError(name)
            return original(name)
        with patch.object(cli.importlib.metadata, 'version', side_effect=installed_version), \
                patch('realistic_carplate_generation.inference.Engine') as engine:
            result = cli.doctor(models=False, cfg={'roots': {}, 'required_assets': []})
        self.assertEqual(result['status'], 'passed')
        self.assertNotIn('onnxruntime', result['missing_dependencies'])
        engine.assert_not_called()

    def test_doctor_is_portable_models_opt_in_and_checks_roots(self):
        self.assertIs(inspect.signature(cli.doctor).parameters['models'].default, False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); image = root / 'input.jpg'; image.write_bytes(b'probe')
            cfg = {'roots': {'data': tmp}, 'required_assets': []}
            index = {'roots': {}, 'records': [{'root': 'data', 'path': image.name}]}
            result = cli.doctor(cfg=cfg, index=index)
            self.assertEqual(result['input_images_checked'], 1)
            self.assertNotIn('wheels_verified', result)
            self.assertNotIn('model_contracts', result)
            with patch('realistic_carplate_generation.inference.Engine') as engine:
                cli.doctor(models=True, cfg=cfg, index=index)
                engine.assert_called_once_with(cfg)
            image.unlink()
            with self.assertRaisesRegex(ValueError, '来源文件不存在'): cli.doctor(cfg=cfg, index=index)
            cfg['roots']['data'] = str(root / 'missing')
            with self.assertRaisesRegex(ValueError, '数据根目录不存在'): cli.doctor(cfg=cfg)

    def test_configuration_reaches_engine_and_renderer(self):
        self.assertIn('cfg', inspect.signature(api.render).parameters)
        self.assertIn('cfg', inspect.signature(api.detect).parameters)
        cfg = {'roots': {}, 'marker': 'explicit'}
        with patch('realistic_carplate_generation.inference.Engine') as engine:
            api.detect('image', cfg=cfg)
            engine.assert_called_once_with(cfg)
        with patch.object(api, '_render', return_value='rendered') as render:
            self.assertEqual(api.render('京A12345', 'blue', cfg=cfg), 'rendered')
            render.assert_called_once_with('京A12345', 'blue', 20261023, cfg=cfg)
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp, patch('realistic_carplate_generation.inference.Engine') as engine:
            p = Path(tmp) / 'input.png'; cv2.imwrite(str(p), np.zeros((100, 200, 3), np.uint8))
            engine.return_value.detect.return_value = []
            with self.assertRaisesRegex(ValueError, '没有合格底图'): sources.prepare(p, None, cfg)
            engine.assert_called_once_with(cfg)

    def test_signature_tracks_only_render_assets_with_config(self):
        from realistic_carplate_generation import pipeline
        self.assertTrue(hasattr(pipeline, 'render_asset_check'), 'render-specific identity missing')
        cfg = {'roots': {}}; index = {'records': []}; plan = {'jobs': []}
        with patch.object(pipeline, 'render_asset_check', return_value={'fonts': 'one'}) as assets:
            first = pipeline.signature(cfg, plan, index)
            assets.assert_called_once_with(cfg)
        with patch.object(pipeline, 'render_asset_check', return_value={'fonts': 'two'}):
            self.assertNotEqual(first, pipeline.signature(cfg, plan, index))

    def test_validation_is_mandatory_and_reporting_opt_in(self):
        from realistic_carplate_generation import pipeline
        from realistic_carplate_generation.common import read_json
        index = {'roots': {}, 'records': [], 'heldout_sha256': []}
        plan = {'jobs': [], 'image_count': 0, 'targets': 'single'}
        for enabled in (False, True):
            with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'render_asset_check', return_value={'files': []}), \
                    patch('realistic_carplate_generation.validation.validate', return_value={'status': 'checked'}) as validate, \
                    patch('realistic_carplate_generation.reporting.report') as report:
                cfg = {'generate_report': enabled}
                pipeline.generate(index, plan, tmp, cfg, workers=1)
                validate.assert_called_once_with(Path(tmp).resolve(), cfg)
                self.assertEqual(report.call_count, int(enabled))
                self.assertEqual(read_json(Path(tmp) / 'progress.json')['status'], 'complete')
        with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline, 'render_asset_check', return_value={'files': []}), \
                patch('realistic_carplate_generation.validation.validate', side_effect=ValueError('invalid output')), \
                patch('realistic_carplate_generation.reporting.report') as report:
            with self.assertRaisesRegex(ValueError, 'invalid output'): pipeline.generate(index, plan, tmp, {}, workers=1)
            report.assert_not_called()
            self.assertFalse((Path(tmp) / 'progress.json').exists())

    def test_invalid_imported_quad_never_reaches_rendering(self):
        from realistic_carplate_generation import pipeline
        record = {'width': 400, 'height': 200}
        malformed = [ [[450, 100], [20, 100], [20, 20], [450, 20]],
                      [[200, 100], [20, 20], [20, 100], [200, 20]],
                      [[200, float('nan')], [20, 100], [20, 20], [200, 20]] ]
        for quad in malformed:
            with patch.object(pipeline, 'render_job', return_value='must not render') as render:
                with self.assertRaises(ValueError):
                    pipeline.worker((record, {'plates': [{'quad': quad}]}, {}, {'min_plate_height': 16}))
                render.assert_not_called()

    def test_legacy_commands_and_arguments_are_removed(self):
        self.assertFalse(hasattr(cli, 'replay_baseline'))
        self.assertFalse(hasattr(cli, 'export_bundle'))
        for function in (sources.prepare, sources.import_tasks, api.prepare):
            self.assertNotIn('legacy_proposals', inspect.signature(function).parameters)
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit): cli.main(['--help'])
        self.assertNotIn('replay', output.getvalue())
        self.assertNotIn('export-bundle', output.getvalue())


if __name__ == '__main__': unittest.main()
