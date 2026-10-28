"""Private procedural pipeline input; never a detector/OCR benchmark."""
from pathlib import Path
import unittest

import cv2
import numpy as np

from realistic_carplate_generation.common import asset_path, sha256, write_json
from realistic_carplate_generation.rendering import encode, render_plate


def require_test_fonts(cfg, roles=('han', 'latin')):
    for role in roles:
        spec = cfg['fonts'][role]
        path = spec if isinstance(spec, str) else spec['path']
        if not asset_path(path, cfg).is_file():
            raise unittest.SkipTest('Local test font missing: ' + path + '; see README font setup')


def make_fixture(root, cfg):
    require_test_fonts(cfg)
    root = Path(root)
    root.mkdir(parents=True)
    records = []
    for number, kinds in enumerate((('blue',), ('nev_small',), ('blue', 'blue'))):
        image = np.zeros((380, 1100, 3), np.uint8)
        image[:] = (45 + number * 15, 65, 80)
        cv2.rectangle(image, (16, 16), (1083, 363), (90, 100, 110), 2)
        plates = []
        for target_id, kind in enumerate(kinds):
            text = '京A12345' if kind == 'blue' else '京AD12345'
            face = render_plate(text, kind, seed=20261023 + number + target_id, cfg=cfg)
            face = cv2.resize(face, (face.shape[1] // 2, face.shape[0] // 2), interpolation=cv2.INTER_AREA)
            h, w = face.shape[:2]
            x, y = 40 + target_id * 530, 120
            image[y:y + h, x:x + w] = face
            plates.append(dict(id=str(target_id), colour='blue' if kind == 'blue' else 'green',
                quad=[[x + w - 1, y + h - 1], [x, y + h - 1], [x, y], [x + w - 1, y]],
                native_height=h - 1, confidence=1, quad_origin='procedural_fixture_not_human_GT'))
        path = root / f'procedural-{number}.png'
        path.write_bytes(encode(image, 'png'))
        digest = sha256(path)
        records.append(dict(id=digest, root='fixture', path=path.name, sha256=digest,
                            width=image.shape[1], height=image.shape[0], plates=plates))
    index = dict(schema_version=1, roots={'fixture': str(root.resolve())}, records=records,
                 heldout_sha256=[], detection_completeness='procedural_only_not_real_photo_accuracy')
    write_json(root / 'index.json', index)
    return index
