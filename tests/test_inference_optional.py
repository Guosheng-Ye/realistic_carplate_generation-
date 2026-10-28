"""Model-free regression tests for optional inference dependencies/resources."""
import os
from pathlib import Path
import subprocess
import sys
import unittest
import hashlib
import json
import tempfile
import numpy as np
from unittest.mock import patch, Mock

from realistic_carplate_generation import inference


class LazyTeacherTests(unittest.TestCase):
    def test_recognition_files_and_session_are_lazy_and_verified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            det, rec, tokens = b'det-fixture', b'rec-fixture', b'["", "A"]'
            names = ['y5fu_640x_sim.onnx', 'rpv3_mdict_160_r3.onnx']
            (root / names[0]).write_bytes(det)
            (root / 'provenance.json').write_text(json.dumps({
                'models': [dict(name=n, sha256=hashlib.sha256(b).hexdigest()) for n, b in zip(names, (det, rec))],
                'tokens_sha256': hashlib.sha256(tokens).hexdigest()}))
            runtime = Mock()
            session = runtime.InferenceSession.return_value
            session.get_inputs.return_value = [Mock(name='input', shape=[1,3,640,640])]
            session.run.return_value = [np.array([[[0., 1.]]], np.float32)]
            with patch.object(inference, '_runtime', return_value=runtime):
                teacher = inference.PlateTeacher(root)
                self.assertIsNone(teacher.rec)
                self.assertEqual(runtime.InferenceSession.call_count, 1)
                (root / names[1]).write_bytes(b'tampered')
                (root / 'tokens.json').write_bytes(tokens)
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    teacher.recognize(np.zeros((10,20,3), np.uint8))
                self.assertIsNone(teacher.rec)
                (root / names[1]).write_bytes(rec)
                result = teacher.recognize(np.zeros((10,20,3), np.uint8))
                self.assertEqual(result['text'], 'A')
                self.assertEqual(runtime.InferenceSession.call_count, 2)
                teacher.recognize(np.zeros((10,20,3), np.uint8))
                self.assertEqual(runtime.InferenceSession.call_count, 2)


class OptionalEngineTests(unittest.TestCase):
    def test_detection_restores_crop_offsets_without_optional_models(self):
        teacher = Mock(rec=None)
        teacher.detect.side_effect = lambda image: [dict(bbox=[10,10,20,10],
            corners=[[10,10],[30,10],[30,20],[10,20]], score=.9,
            corners_source='model_predicted', teacher='HyperLPR3', layer=0)]
        with patch.object(inference, 'asset_check'), \
             patch.object(inference, 'asset_path', side_effect=lambda relative, cfg=None: Path(relative)), \
             patch.object(inference, 'PlateTeacher', return_value=teacher):
            engine = inference.Engine()
            image = np.zeros((200,400,3), np.uint8)
            rows = engine.detect(image, ocr=False)
            teacher.recognize.assert_not_called()
            self.assertTrue(any(r['bbox'][:2] == [187,98] for r in rows), rows)
            offset = next(r for r in rows if r['bbox'][:2] == [187,98])
            self.assertEqual(offset['quad'], [[207,108],[187,108],[187,98],[207,98]])
            with patch.object(engine, 'e3_recognize') as e3:
                rows = engine.detect(image, ocr=True)
                e3.assert_not_called()
                self.assertIn('hyperlpr', rows[0]['ocr'])
                self.assertNotIn('e3', rows[0]['ocr'])

    def test_e3_requires_opt_in_and_checks_only_its_resources_when_used(self):
        with patch.object(inference, 'asset_check') as check, \
             patch.object(inference, 'asset_path', side_effect=lambda relative, cfg=None: Path(relative)), \
             patch.object(inference, 'PlateTeacher'), patch.object(inference, 'Detector') as yolo:
            engine = inference.Engine()
            with self.assertRaisesRegex(ValueError, 'e3_ocr'):
                engine.e3_recognize(np.zeros((32,112,3), np.uint8))
            engine = inference.Engine({'models': {'yolo_hints': True, 'e3_ocr': True}})
            yolo.assert_called_once()
            self.assertIsNone(engine.e3)
            check.reset_mock()
            runtime = Mock()
            runtime.InferenceSession.return_value.get_inputs.return_value = [Mock(name='input')]
            runtime.InferenceSession.return_value.run.return_value = [None, None]
            with patch.object(inference, '_runtime', return_value=runtime), \
                 patch.object(inference, 'read_json', return_value={'characters': [], 'provinces': []}), \
                 patch.object(inference, 'decode', return_value={'text': 'fixture'}):
                self.assertEqual(engine.e3_recognize(np.zeros((32,112,3), np.uint8)), {'text': 'fixture'})
            self.assertEqual(set(check.call_args.kwargs['required']),
                {'assets/models/e3/model.onnx', 'assets/models/e3/verification.json'})

    def test_default_engine_checks_detection_only_and_contracts_are_non_loading(self):
        teacher = Mock()
        teacher.rec = None
        teacher.det.get_inputs.return_value = []
        teacher.det.get_outputs.return_value = []
        with patch.object(inference, 'asset_check') as check, \
             patch.object(inference, 'asset_path', side_effect=lambda relative, cfg=None: Path(relative)), \
             patch.object(inference, 'PlateTeacher', return_value=teacher), \
             patch.object(inference, 'Detector') as yolo:
            engine = inference.Engine(cfg={})
            self.assertIsNone(engine.yolo)
            yolo.assert_not_called()
            required = check.call_args.kwargs['required']
            self.assertIn('assets/models/hyperlpr3/y5fu_640x_sim.onnx', required)
            self.assertNotIn('assets/models/hyperlpr3/rpv3_mdict_160_r3.onnx', required)
            self.assertFalse(any('/e3/' in p or '/yolo/' in p for p in required))
            self.assertEqual(set(engine.contracts()), {'hyper_detection'})
            teacher.recognize.assert_not_called()


class OptionalRuntimeTests(unittest.TestCase):
    def test_import_and_geometry_work_without_onnxruntime(self):
        script = r'''
import sys
class BlockORT:
    def find_spec(self, fullname, *args):
        if fullname == 'onnxruntime' or fullname.startswith('onnxruntime.'):
            raise ModuleNotFoundError('blocked optional runtime')
sys.meta_path.insert(0, BlockORT())
import numpy as np
from realistic_carplate_generation import api
from realistic_carplate_generation.inference import decode_hyper, Detector
from realistic_carplate_generation.rules import make_samples
assert decode_hyper(np.zeros((1, 0, 15)), 100, 80, 1, 0, 0) == []
assert len(make_samples(2, 42)) == 2
try:
    Detector('unused.onnx')
except ImportError as exc:
    assert '[detection]' in str(exc), str(exc)
else:
    raise AssertionError('model use must explain missing detection dependency')
'''
        env = dict(os.environ)
        env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1] / 'src')
        result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
