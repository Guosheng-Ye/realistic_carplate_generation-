import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
from PIL import Image
from realistic_carplate_generation.cli import main
from realistic_carplate_generation.common import read_jsonl


class ExifTests(unittest.TestCase):
    def test_detector_uses_encoded_pixel_coordinates_for_oriented_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);image=root/'oriented.jpg';exif=Image.Exif();exif[274]=6
            pixels=Image.new('RGB',(200,100),(255,0,0))
            pixels.paste((0,0,255),(100,0,200,100));pixels.save(image,exif=exif)
            self.assertEqual(cv2.imread(str(image)).shape[:2],(200,100))
            expected=dict(bbox=[150,10,30,20],quad=[[180,30],[150,30],[150,10],[180,10]],
                          quad_origin='model_predicted',teacher='fixture')
            test=self
            class StubEngine:
                def __init__(self, cfg=None):pass
                def detect(self, decoded, ocr=False):
                    test.assertEqual(decoded.shape[:2],(100,200))
                    test.assertGreater(int(decoded[10,10,2]),240)  # encoded red left
                    test.assertGreater(int(decoded[10,180,0]),240)  # encoded blue right
                    test.assertFalse(ocr)
                    return [expected]
            with patch('realistic_carplate_generation.inference.Engine',StubEngine), \
                 contextlib.redirect_stdout(io.StringIO()):
                result=main(['detect','--input',str(image),'--out',str(root/'candidates.jsonl')])
            self.assertEqual(result,0)
            row=read_jsonl(root/'candidates.jsonl')[0]
            self.assertEqual((row['width'],row['height']),(200,100))
            self.assertEqual(row['plates'],[expected])
