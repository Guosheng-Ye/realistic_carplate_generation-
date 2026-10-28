import math
import cv2
import numpy as np
from . import _render_base as base

def face_geometry(quad, colour):
    width, height = (440 if colour == "blue" else 480), 140
    target = np.array(((width - 1, height - 1), (0, height - 1), (0, 0), (width - 1, 0)), np.float32)
    quad = np.asarray(quad, np.float32)
    return width, height, cv2.getPerspectiveTransform(quad, target), cv2.getPerspectiveTransform(target, quad)

def erase_old_letters(face, colour):
    """Extract ink by local contrast; inpaint ink only, retaining the carrier."""
    height, width = face.shape[:2]
    if colour == "blue":
        # White lettering loses saturation, including wide strokes that a local
        # top-hat filter would mistake for carrier material.
        contrast = np.rint(face.min(2).astype(np.float32) / np.maximum(face.max(2), 1) * 255).astype(np.uint8)
    else:
        signal = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        carrier = cv2.morphologyEx(signal, cv2.MORPH_CLOSE, np.ones((23, 25), np.uint8))
        contrast = cv2.subtract(carrier, signal)
    band = np.zeros((height, width), np.uint8)
    band[18:height - 17, :] = 255
    threshold, _ = cv2.threshold(contrast[band > 0], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    threshold = max(12, threshold)
    core = ((contrast > threshold) & (band > 0)).astype(np.uint8)
    # Ignore bolts, isolated rain specks, and the original separator dot.
    count, components, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    large = np.zeros_like(core)
    protected = np.zeros_like(core)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if (x <= 1 or x + w >= width - 1) and area >= 20:
            # CCPD quads can include a 10-22 px rectified metal rim. Its
            # nearly full-band trace is material, rather than a clipped glyph.
            if w <= 22 and h >= 94 or h < 35:
                protected[components == index] = 255
                continue
            raise ValueError("原字符被四角裁切，排除此底图以免旧字残留")
        if 90 <= x + w / 2 <= 155 and 53 <= y + h / 2 <= 88 and w <= 12 and h <= 12:
            protected[components == index] = 255
        elif h >= 2 and area >= 6:
            large[components == index] = 255
    core = large
    base.require(0.025 <= np.count_nonzero(core) / np.count_nonzero(band) <= 0.48,
                 "原字形对比不足或强反光，无法可靠清字")
    neighbourhood = cv2.dilate(core, np.ones((9, 9), np.uint8))
    mask = ((neighbourhood > 0) & (contrast > threshold * 0.18) & (band > 0)).astype(np.uint8) * 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    mask[band == 0] = 0
    mask[cv2.dilate(protected, np.ones((3, 3), np.uint8)) > 0] = 0
    clean = cv2.inpaint(face, mask, 5, cv2.INPAINT_NS)
    return clean, mask, core

def lettering_mask(sample, renderer, old_core):
    height, width = old_core.shape
    rows = np.flatnonzero(np.count_nonzero(old_core, axis=1) >= max(5, width * 0.018))
    base.require(len(rows) >= 35, "无法确定原字符高度")
    top, bottom = int(rows[0]), int(rows[-1]) + 1
    char_height = min(98, max(64, bottom - top))
    top = max(19, min(122 - char_height, round((top + bottom - char_height) / 2)))
    xs = (18, 76, 157, 210, 263, 316, 369) if sample.plate_type == "blue" else (
        20, 76, 151, 204, 257, 310, 363, 416)
    slot_width = (45 if sample.plate_type == "blue" else 44)
    mask = np.zeros((height, width), np.uint8)
    for x, char in zip(xs, sample.text):
        glyph = np.asarray(renderer.glyph(char, slot_width, char_height))
        mask[top:top + char_height, x:x + slot_width] = np.maximum(
            mask[top:top + char_height, x:x + slot_width], glyph)
    return mask.astype(np.float32) / 255

def estimate_ink_blur(face, clean, core, colour):
    """Estimate source edge softness from ink-to-carrier contrast, in face pixels."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float32)
    carrier = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    difference = np.maximum(0, gray - carrier if colour == "blue" else carrier - gray)
    amplitude = max(float(np.percentile(difference[core > 0], 85)), 10)
    coverage = np.clip(difference / amplitude, 0, 1)
    dx = cv2.Sobel(coverage, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(coverage, cv2.CV_32F, 0, 1, ksize=3) / 8
    gradient = np.hypot(dx, dy)
    edges = gradient[(coverage > 0.2) & (coverage < 0.8) & (gradient > 0.025)]
    if len(edges) < 20:
        return 1.0
    # For a Gaussian step edge, maximum normalized slope is 1/(sqrt(2*pi)*sigma).
    return float(np.clip(0.39894228 / np.percentile(edges, 65), 0.45, 4.0))

def recolour_large_nev(clean):
    """Yellow left panel with measured illumination; preserve rim, bolts and marker."""
    height, width = clean.shape[:2]
    paint = np.zeros((height, width), np.float32)
    paint[3:height - 3, 3:144] = 1
    # Keep the photographed registration marker and the actual mounting caps.
    # Protect the small photographed marker itself, with a circular support.
    # A rectangular exclusion would leave a visible green notch in yellow paint.
    cv2.circle(paint, (135, 72), 7, 0, -1)
    gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    local = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 6)
    candidates = ((local - gray) > 16).astype(np.uint8)
    candidates[25:height - 23] = 0
    count, regions, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    protected = np.zeros_like(candidates)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if 3 <= w <= 25 and 3 <= h <= 22 and 8 <= area <= 330:
            protected[regions == index] = 255
    protected = cv2.dilate(protected, np.ones((5, 5), np.uint8))
    paint[protected > 0] = 0
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32)
    target = cv2.cvtColor(np.uint8([[[57, 205, 246]]]), cv2.COLOR_BGR2LAB)[0, 0].astype(np.float32)
    # Strong specular whites remain pale, rather than becoming a flat yellow patch.
    chroma = np.clip((255 - lab[:, :, 0]) / 65, 0.15, 1)
    changed = lab.copy()
    changed[:, :, 1:] = 128 + (target[1:] - 128) * chroma[:, :, None]
    yellow = cv2.cvtColor(np.clip(changed, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    result = np.rint(clean * (1 - paint[:, :, None]) + yellow * paint[:, :, None]).astype(np.uint8)
    return result, paint

def edit_plate(original, quad, sample, renderer, seed):
    colour = "blue" if sample.plate_type == "blue" else "green"
    base.require(sample.plate_type in base.PLATE_TYPES, "不支持的车牌类型")
    base.require(base.valid_text(sample.text, sample.plate_type), "非法号码")
    width, height, to_face, to_scene = face_geometry(quad, colour)
    face = cv2.warpPerspective(original, to_face, (width, height), flags=cv2.INTER_LINEAR)
    clean, old_mask, old_core = erase_old_letters(face, colour)
    alpha = lettering_mask(sample, renderer, old_core)
    q = np.asarray(quad, np.float32)
    native_width = (np.linalg.norm(q[0] - q[1]) + np.linalg.norm(q[2] - q[3])) / 2
    native_height = (np.linalg.norm(q[0] - q[3]) + np.linalg.norm(q[1] - q[2])) / 2
    scale = math.sqrt(width * height / max(1, native_width * native_height))
    source_sigma = estimate_ink_blur(face, clean, old_core, colour)
    size_sigma = min(1.3, max(0.45, scale * 0.48))
    sigma = min(3.0, max(size_sigma, math.sqrt(max(0.2, source_sigma ** 2 - 0.55 ** 2))))
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)
    # Measured photographed ink colour, with the photographed illumination field.
    values = face[old_core > 0].astype(np.float32)
    luminance = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(luminance, (0, 0), 10)
    reference = float(np.median(smooth[20:122, 12:width - 12]))
    light = np.clip(smooth / max(reference, 15), 0.72, 1.28)
    level = np.percentile(values, 72 if colour == "blue" else 28, axis=0)
    if colour == "blue":
        level = np.clip(level, 105, 245)
    else:
        level = np.clip(level, 5, 65)
    ink = np.broadcast_to(level, face.shape).copy() * (0.65 + 0.35 * light[:, :, None])
    # Small directional edge response suggests embossed lettering, not a shadow decal.
    dx = cv2.Sobel(alpha, cv2.CV_32F, 1, 0, ksize=3) / 8
    dy = cv2.Sobel(alpha, cv2.CV_32F, 0, 1, ksize=3) / 8
    relief = np.clip(-dx * 0.5 - dy * 0.8, -1, 1) * 4
    grain = clean.astype(np.float32) - cv2.GaussianBlur(clean.astype(np.float32), (0, 0), 1.0)
    ink += relief[:, :, None] + grain * 0.22
    rng = np.random.default_rng(seed)
    ink += rng.normal(0, 0.4, (height, width, 1)).astype(np.float32)
    material, paint_mask = (recolour_large_nev(clean) if sample.plate_type == "nev_large" else
                            (clean, np.zeros(clean.shape[:2], np.float32)))
    edited = material.astype(np.float32) * (1 - alpha[:, :, None]) + ink * alpha[:, :, None]
    edited = np.clip(edited, 0, 255).astype(np.uint8)
    support = np.maximum(old_mask.astype(np.float32) / 255, (alpha > 0.002).astype(np.float32))
    support = cv2.GaussianBlur(support, (0, 0), 0.45)
    support[:18] = 0
    support[height - 17:] = 0
    support[:, :2] = 0
    support[:, width - 2:] = 0
    support = np.maximum(support, paint_mask)
    scene_h, scene_w = original.shape[:2]
    # Work only in the original plate's bounding rectangle.
    x0, y0 = np.floor(q.min(0)).astype(int)
    x1, y1 = np.ceil(q.max(0)).astype(int) + 1
    offset = np.array(((1, 0, -x0), (0, 1, -y0), (0, 0, 1)), np.float64)
    to_roi = offset @ to_scene
    warped = cv2.warpPerspective(edited, to_roi, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    weights = cv2.warpPerspective(support, to_roi, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
    quad_mask = np.zeros((scene_h, scene_w), np.uint8)
    cv2.fillConvexPoly(quad_mask, np.rint(q).astype(np.int32), 255)
    weights *= quad_mask[y0:y1, x0:x1].astype(np.float32) / 255
    roi = original[y0:y1, x0:x1]
    result = original.copy()
    result[y0:y1, x0:x1] = np.rint(roi * (1 - weights[:, :, None]) + warped * weights[:, :, None]).astype(np.uint8)
    changed = np.any(result != original, axis=2)
    base.require(not np.any(changed & (quad_mask == 0)), "车牌之外出现像素变化")
    audit = {"outside_plate_changed_pixels": 0, "changed_pixels": int(changed.sum()),
             "plate_pixels": int(np.count_nonzero(quad_mask)), "quad_unchanged": True,
             "ink_blur_sigma_face": round(sigma, 4), "source_ink_sigma_face": round(source_sigma, 4),
             "source_native_plate_height": round(float(native_height), 2),
             "yellow_green_left_panel": sample.plate_type == "nev_large",
             "original_border_bolts_carrier_retained": True}
    return result, quad_mask, audit
