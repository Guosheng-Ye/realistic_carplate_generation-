"""Shared rendering implementation for the API, preview and dataset workers."""
import io
import random
from pathlib import Path
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image

from . import _render_base as base, material
from .augment import augment_plate
from .common import font_path, asset_check, load_config, require, decode_source, source_path, _asset_relative, sha256, _asset_root
from .geometry import polygon_mask, native_height, rectify, validate_quad


def _font_settings(cfg=None):
    defaults = load_config()['fonts']
    defaults.update((cfg or {}).get('fonts', {}))
    return {role: (dict(path=value, index=0) if isinstance(value, str) else dict(value))
            for role, value in defaults.items()}


def render_asset_check(cfg=None):
    """Fingerprint the two plate fonts; report previews and models are independent."""
    fonts = _font_settings(cfg)
    roles = ['han', 'latin']
    relative, external, logical = [], [], {}
    for role in roles:
        value = fonts[role]['path']
        if Path(value).expanduser().is_absolute():
            path = font_path(value, cfg)
            if path.is_relative_to(_asset_root(cfg)):
                name = path.relative_to(_asset_root(cfg)).as_posix()
                relative.append(name)
                logical[role] = 'assets/' + name
                continue
            require(path.is_file(), f'字体缺失: {role}')
            digest = sha256(path)
            name = 'fonts/sha256/' + digest
            external.append({'path': name, 'bytes': path.stat().st_size, 'sha256': digest})
            logical[role] = name
        else:
            relative.append(value)
            logical[role] = 'assets/' + _asset_relative(value)
    identity = asset_check(relative, cfg)
    rows = {row['path']: row for row in identity['files'] + external}
    identity['files'] = [rows[key] for key in sorted(rows)]
    identity['fonts'] = {role: {'path': logical[role], 'index': fonts[role].get('index', 0)}
                         for role in roles}
    return identity


@lru_cache(maxsize=8)
def _renderer(han, latin, content_identity):
    base.validate_font(han, base.PROVINCES); base.validate_font(latin, base.LETTERS + base.DIGITS)
    return base.PlateRenderer(han, latin)


def renderer(cfg=None):
    # Check before the cache lookup: a cached renderer must not conceal changed files.
    identity = render_asset_check(cfg)
    fonts = _font_settings(cfg)
    specs = [base.FontSpec(str(font_path(fonts[role]['path'], cfg)), int(fonts[role].get('index', 0)), True)
             for role in ('han', 'latin')]
    try:
        return _renderer(*specs, tuple(row['sha256'] for row in identity['files']))
    except OSError as exc:
        raise ValueError(f'字体无法加载，请检查 fonts 配置: {exc}') from exc


def render_plate(text, plate_type, seed=20261023, cfg=None):
    return renderer(cfg).render(base.Sample(text, plate_type), random.Random(seed))


def full_replace(image, quad, sample, seed, cfg=None):
    face = renderer(cfg).render(sample, random.Random(seed))
    q = np.asarray(quad, np.float32)
    x, y = np.floor(q.min(0)).astype(int)
    r, b = np.ceil(q.max(0)).astype(int) + 1
    h, w = face.shape[:2]
    source = np.float32([[w - 1, h - 1], [0, h - 1], [0, 0], [w - 1, 0]])
    transform = cv2.getPerspectiveTransform(source, (q - (x, y)).astype(np.float32))
    warped = cv2.warpPerspective(face, transform, (r - x, b - y), borderMode=cv2.BORDER_REPLICATE)
    mask = polygon_mask(image.shape, q)
    local = mask[y:b, x:r].astype(np.float32) / 255
    weights = cv2.GaussianBlur(local, (0, 0), .6) * local
    roi = image[y:b, x:r]
    warped = base.match_lighting(warped, roi, local, native_height(q))
    result = image.copy()
    result[y:b, x:r] = np.rint(roi * (1 - weights[:, :, None]) + warped * weights[:, :, None]).astype(np.uint8)
    return result, mask, dict(render_mode='full', original_border_bolts_carrier_retained=False)


def encode(image, fmt='jpg', quality=96):
    stream = io.BytesIO()
    if fmt == 'png':
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(stream, 'PNG', compress_level=3)
    else:
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(stream, 'JPEG', quality=quality, subsampling=0)
    return stream.getvalue()


def render_job(record, job, roots, cfg):
    original = decode_source(record, roots)
    result = original.copy(); union = np.zeros(original.shape[:2], np.uint8); audits = []
    for target in job['plates']:
        require(target['colour'] == ('blue' if target['plate_type'] == 'blue' else 'green'), '号码牌型与原牌颜色不兼容')
        sample = base.Sample(target['text'], target['plate_type'])
        require(base.valid_text(sample.text, sample.plate_type), '非法目标号码')
        q, seed = target['quad'], target['seed']
        validate_quad(q, result.shape[1], result.shape[0], cfg.get('min_plate_height', 0))
        mask = polygon_mask(result.shape, q)
        require(not np.any((mask > 0) & (union > 0)), '替换目标重叠')
        if cfg['render_mode'] == 'material':result, mask, audit = material.edit_plate(result, q, sample, renderer(cfg), seed)
        else:result, mask, audit = full_replace(result, q, sample, seed, cfg)
        if cfg['augment']:
            result, aug = augment_plate(result, q, seed + 17, mask)
        else:aug = dict(operations=[], local_jpeg_quality=None, outside_plate_before_export_changed_pixels=0)
        union |= mask
        audits.append(dict(**audit, **aug, target_id=target['target_id'], text=target['text'],
                           plate_type=target['plate_type'], quad=q, seed=seed))
    outside = np.any(result != original, axis=2) & (union == 0)
    require(not np.any(outside), '合成或增强改变了所有目标掩膜之外的像素')
    encoded = encode(result, cfg['format'], cfg['jpeg_quality'])
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    require(decoded is not None and decoded.shape == original.shape, '输出无法解码或尺寸改变')
    if cfg['format'] == 'png':require(np.array_equal(decoded[union == 0], original[union == 0]), 'PNG 保存后牌外像素改变')
    crops = []
    if cfg['export_crops']:
        for target in job['plates']:
            face, _, meta = rectify(decoded, target['quad'], target['colour'])
            w, h = meta['width'], meta['height']
            crop_image = f'images/{target["plate_index"]:06d}.png'
            label = dict(image=crop_image, text=target['text'], plate_type=target['plate_type'],
                         quad=[[w - 1, h - 1], [0, h - 1], [0, 0], [w - 1, 0]],
                         background=str(source_path(record, roots)))
            crops.append(dict(path='crops/' + crop_image, data=encode(face, 'png'), label=label,
                              source_quad=target['quad'], source_image=job['image'], target_id=target['target_id']))
    return encoded, crops, dict(plates=audits, outside_plate_before_encoding_changed_pixels=0,
        png_saved_outside_changed_pixels=0 if cfg['format'] == 'png' else None,
        width=original.shape[1], height=original.shape[0], render_mode=cfg['render_mode'])
