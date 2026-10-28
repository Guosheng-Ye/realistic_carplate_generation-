import random
import cv2
import numpy as np
from . import _render_base as base

def augment_plate(image, quad, seed, plate_mask=None):
    """0-3 requested families, confined to the photographed plate before JPG export."""
    rng = random.Random(seed)
    q = np.asarray(quad, np.float32)
    x0, y0 = np.maximum(0, np.floor(q.min(0)).astype(int) - 2)
    x1, y1 = np.minimum((image.shape[1], image.shape[0]), np.ceil(q.max(0)).astype(int) + 3)
    # Rasterize in the same full-image coordinates as the material edit. A
    # second local rasterization can differ at fractional/scanline edge pixels.
    if plate_mask is None:
        plate_mask = np.zeros(image.shape[:2], np.uint8)
        cv2.fillConvexPoly(plate_mask, np.rint(q).astype(np.int32), 255)
    base.require(plate_mask.shape == image.shape[:2], '车牌增强掩膜尺寸不匹配')
    mask = plate_mask[y0:y1, x0:x1]
    alpha = mask.astype(np.float32) / 255
    roi = image[y0:y1, x0:x1]
    height = min(np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2]))
    augmented, quality, operations = base.augment_region(roi, alpha, rng, height)
    if "jpeg" in operations:
        ok, data = cv2.imencode(".jpg", augmented, [cv2.IMWRITE_JPEG_QUALITY, quality])
        base.require(ok, "局部 JPEG 增强编码失败")
        decoded = cv2.imdecode(data, cv2.IMREAD_COLOR)
        augmented[mask > 0] = decoded[mask > 0]
    result = image.copy()
    result[y0:y1, x0:x1][mask > 0] = augmented[mask > 0]
    return result, {"operations": operations, "local_jpeg_quality": quality if "jpeg" in operations else None,
                    "outside_plate_before_export_changed_pixels": 0}

