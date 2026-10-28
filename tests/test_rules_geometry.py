import random
import unittest
from collections import Counter

import cv2
import numpy as np

from realistic_carplate_generation.rules import make_samples, summarize, valid_text, PROVINCES, LETTERS, SMALL_THIRD
from realistic_carplate_generation.geometry import validate_quad, rectify, polygon_mask, prune_overlaps
from realistic_carplate_generation.augment import augment_plate


class RulesTests(unittest.TestCase):
    def test_arbitrary_counts_and_exact_62000(self):
        for n in (1, 2, 31, 619, 620, 621, 62000):
            samples = make_samples(n, 20261023)
            stats = summarize(samples)
            self.assertEqual(stats['total'], n)
            self.assertTrue(all(valid_text(s['text'], s['plate_type']) for s in samples))
            self.assertEqual(len({s['text'] for s in samples}), n)
            if n == 62000:
                self.assertEqual(stats['plate_type'], {'blue':24800,'nev_small':27900,'nev_large':9300})
                self.assertEqual(set(stats['province'].values()), {2000})
                self.assertEqual(set(stats['nev_small_third'].values()), {2790})
                for p in PROVINCES:
                    self.assertEqual(stats['province_plate_type'][p], {'blue':800,'nev_small':900,'nev_large':300})
                    city=Counter(s['text'][1] for s in samples if s['text'][0]==p)
                    self.assertLessEqual(max(city.values())-min(city.values()),1)
                    third=Counter(s['text'][2] for s in samples if s['text'][0]==p and s['plate_type']=='nev_small')
                    self.assertEqual({third[c] for c in SMALL_THIRD},{90})

    def test_seed_and_negative_text_cases(self):
        self.assertEqual(make_samples(620, 1), make_samples(620, 1))
        self.assertNotEqual(make_samples(620, 1), make_samples(620, 2))
        for text,kind in [('京I12345','blue'),('京O12345','blue'),('京AABC12','blue'),('京AI12345','nev_small'),('京AL12345','nev_small'),('京A12345A','nev_large')]:
            self.assertFalse(valid_text(text,kind))


class GeometryTests(unittest.TestCase):
    def test_roundtrip_subpixel_quad_and_native_crop(self):
        image=np.full((300,600,3),95,np.uint8)
        quad=[[541.5,200.5],[40.5,220.5],[60.5,50.5],[520.5,45.5]]
        face,crop,meta=rectify(image,quad,'green')
        self.assertEqual(face.shape,(140,480,3))
        self.assertGreater(crop.size,0)
        pts=np.float32(quad)[None]
        out=cv2.perspectiveTransform(cv2.perspectiveTransform(pts,np.asarray(meta['to_face'])),np.asarray(meta['to_scene']))
        self.assertLess(np.max(np.abs(out-pts)),.01)

    def test_rejects_clipping_and_overlaps(self):
        with self.assertRaises(ValueError):validate_quad([[610,200],[0,200],[0,0],[610,0]],600,300)
        q=[[500,200],[40,200],[40,50],[500,50]]
        kept,rejected=prune_overlaps([dict(quad=q,native_height=150),dict(quad=q,native_height=149)],(300,600))
        self.assertEqual((len(kept),len(rejected)),(1,1))

    def test_every_augmentation_family_is_confined_to_same_subpixel_mask(self):
        image=np.full((300,600,3),(160,90,25),np.uint8)
        q=np.float32([[541.5,200.5],[40.5,220.5],[60.5,50.5],[520.5,45.5]])
        mask=polygon_mask(image.shape,q);families=set();numbers=set()
        for seed in range(32):
            result,audit=augment_plate(image,q,seed,mask)
            np.testing.assert_array_equal(result[mask==0],image[mask==0])
            families.update(audit['operations']);numbers.add(len(audit['operations']))
            self.assertLessEqual(len(audit['operations']),3)
        self.assertEqual(numbers,{0,1,2,3})
        self.assertEqual(families,{'blur','exposure','shadow','noise','jpeg','weather'})
