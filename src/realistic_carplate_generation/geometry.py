"""Coordinates are always BR, BL, TL, TR in original decoded-image pixels."""
import cv2
import numpy as np

from .common import require


def validate_quad(quad, width, height, min_height=0):
    q = np.asarray(quad, np.float32)
    require(q.shape == (4, 2) and np.isfinite(q).all(), '四角必须为有限的 4×2 数组')
    require(cv2.isContourConvex(q) and cv2.contourArea(q, oriented=True) > 4, '四角非凸或顺序错误')
    require(bool((q.min(0) >= 0).all() and (q.max(0) <= (width - 1, height - 1)).all()), '四角越界')
    a, b = np.linalg.norm(q[0] - q[3]), np.linalg.norm(q[1] - q[2])
    c, d = np.linalg.norm(q[0] - q[1]), np.linalg.norm(q[2] - q[3])
    require(min(a, b) >= min_height, '车牌过小')
    require(1.8 <= (c + d) / max(1e-6, a + b) <= 6.5, '车牌长宽比异常')
    require(min(a, b) / max(a, b) > .45 and min(c, d) / max(c, d) > .45, '透视过于极端')
    return q


def native_height(q):
    q = np.asarray(q, np.float32)
    return float((np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2)


def polygon_mask(shape, quad):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, np.rint(quad).astype(np.int32), 255)
    return mask


def rectify(image, quad, colour):
    q = validate_quad(quad, image.shape[1], image.shape[0])
    width, height = (440 if colour == 'blue' else 480), 140
    corners = np.float32([[width - 1, height - 1], [0, height - 1], [0, 0], [width - 1, 0]])
    forward = cv2.getPerspectiveTransform(q, corners)
    inverse = cv2.getPerspectiveTransform(corners, q)
    face = cv2.warpPerspective(image, forward, (width, height), flags=cv2.INTER_LINEAR)
    x, y = np.floor(q.min(0)).astype(int)
    r, b = np.ceil(q.max(0)).astype(int) + 1
    return face, image[y:b, x:r].copy(), {'quad': q.tolist(), 'to_face': forward.tolist(),
        'to_scene': inverse.tolist(), 'crop_xyxy': [int(x), int(y), int(r), int(b)],
        'width': width, 'height': height}


def prune_overlaps(plates, shape):
    """Prefer the largest reliable carrier; never edit overlapping supports twice."""
    kept, rejects = [], []
    occupied = np.zeros(shape[:2], np.uint8)
    for plate in sorted(plates, key=lambda p: (-p['native_height'], -p.get('confidence', 1), str(p['quad']))):
        mask = polygon_mask(shape, plate['quad'])
        if np.any((occupied > 0) & (mask > 0)):
            rejects.append(dict(reason='overlapping_plate_candidate', plate=plate))
        else:
            kept.append(plate)
            occupied |= mask
    return kept, rejects
