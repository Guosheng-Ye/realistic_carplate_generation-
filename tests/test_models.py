import importlib.util
import unittest
import numpy as np

from realistic_carplate_generation.common import asset_path, asset_check
from realistic_carplate_generation.inference import Engine, Detector, PlateTeacher, decode_hyper, decode


class ModelsTests(unittest.TestCase):
    """Pure decoding fixtures: no runtime, model files or engine construction."""
    def test_corner_mapping_is_not_bbox_fallback(self):
        raw=np.asarray([[[20,20,10,4,.9,16,19,24,18,24,22,16,23,.8,.1]]],np.float32)
        result=decode_hyper(raw,100,80,1,0,0,.2)
        self.assertEqual(result[0]['corners_source'],'model_predicted')
        self.assertEqual(result[0]['corners'][0],[16,19])

    def test_letterbox_coordinates_are_restored(self):
        raw=np.asarray([[[60,60,20,8,.9,52,58,68,56,68,64,52,66,.8,.1]]],np.float32)
        result=decode_hyper(raw,100,80,2,20,20,.2)[0]
        self.assertEqual(result['corners'],[[16,19],[24,18],[24,22],[16,23]])
        self.assertEqual(result['bbox'],[15,18,10,5])

    def test_unmapped_hyper_token_is_explicit(self):
        teacher=PlateTeacher.__new__(PlateTeacher)
        teacher.tokens=['']+[str(i) for i in range(1,77)]
        teacher.rec_name='fixture'
        class Fake:
            def run(self,*args):
                out=np.zeros((1,20,78),np.float32);out[:,:,0]=1;out[0,3]=0;out[0,3,77]=1
                return [out]
        teacher.rec=Fake()
        result=teacher.recognize(np.zeros((32,100,3),np.uint8))
        self.assertEqual(result['unsupported_token_ids'],[77])
        self.assertEqual(result['text'],'\ufffd')
        self.assertEqual(result['score'],1.)

    def test_e3_other_province_is_explicitly_unsupported(self):
        ctc=np.zeros((56,73));province=np.zeros(32);province[31]=1
        result=decode(ctc,province,list('A'*72),list('省'*31))
        self.assertEqual(result['text'],'')
        self.assertEqual(result['province'],'<other>')
        self.assertEqual(result['decode_status'],'unsupported')
        self.assertEqual(result['decode_reason'],'province_other')

    def test_malformed_or_nonfinite_tensors_are_rejected(self):
        for raw in (np.zeros((1,2,14)), np.full((1,2,15),np.nan)):
            with self.assertRaises(ValueError):decode_hyper(raw,100,80,1,0,0)


class ModelIntegrationTests(unittest.TestCase):
    """Run installed local CPU models; skip only genuinely missing prerequisites."""
    def resources(self, *paths):
        if importlib.util.find_spec('onnxruntime') is None:
            self.skipTest('optional inference runtime missing; install [detection]')
        missing=[p for p in paths if not asset_path(p).is_file()]
        if missing:self.skipTest('optional local model resources missing: '+', '.join(missing))
        asset_check(required=list(paths))  # Hash mismatches must fail, never skip.

    def hyper_resources(self, ocr=False):
        paths=['assets/models/hyperlpr3/provenance.json','assets/models/hyperlpr3/y5fu_640x_sim.onnx']
        if ocr:paths+=['assets/models/hyperlpr3/rpv3_mdict_160_r3.onnx','assets/models/hyperlpr3/tokens.json']
        self.resources(*paths)

    def test_real_default_detection_contract_and_blank_input(self):
        self.hyper_resources()
        engine=Engine()
        contracts=engine.contracts()
        self.assertEqual(set(contracts),{'hyper_detection'})
        self.assertEqual(contracts['hyper_detection']['outputs'][0]['shape'],[1,25200,15])
        self.assertEqual(engine.detect(np.zeros((200,400,3),np.uint8)),[])
        self.assertIsNone(engine.teacher.rec)
        self.assertIsNone(engine.yolo)
        self.assertIsNone(engine.e3)

    def test_real_hyper_ocr_lazy_session(self):
        self.hyper_resources(ocr=True)
        teacher=PlateTeacher(asset_path('assets/models/hyperlpr3'))
        self.assertIsNone(teacher.rec)
        result=teacher.recognize(np.zeros((48,160,3),np.uint8))
        self.assertIsNotNone(teacher.rec)
        self.assertEqual(teacher.rec.get_inputs()[0].shape,[1,3,48,160])
        self.assertIsInstance(result['text'],str)
        self.assertTrue(np.isfinite(result['score']))
        self.assertGreaterEqual(result['score'],0)
        self.assertLessEqual(result['score'],1.00001)
        self.assertIsInstance(result['unsupported_token_ids'],list)

    def test_real_yolo_contract_and_blank_input(self):
        self.resources('assets/models/yolo/detector.onnx')
        detector=Detector(asset_path('assets/models/yolo/detector.onnx'),confidence=.22)
        self.assertEqual(detector.session.get_outputs()[0].shape,[1,300,6])
        self.assertEqual(detector.run(np.zeros((200,400,3),np.uint8)),([],[]))

    def test_real_e3_opt_in_diagnostic_contract(self):
        self.hyper_resources()
        self.resources('assets/models/e3/model.onnx','assets/models/e3/verification.json')
        engine=Engine({'models':{'e3_ocr':True}})
        self.assertIsNone(engine.e3)
        result=engine.e3_recognize(np.zeros((32,112,3),np.uint8))
        self.assertEqual(engine.contracts()['e3']['inputs'][0]['shape'],[1,3,64,224])
        self.assertEqual(result['decode_contract'],'e3_dual_v2_other_reject')
        self.assertIn(result['decode_status'],('decoded','unsupported'))
        self.assertIsInstance(result['ctc_text'],str)
        self.assertIsNone(engine.teacher.rec)
        self.assertIsNone(engine.yolo)
