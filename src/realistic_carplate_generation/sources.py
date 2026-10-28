"""Adapters for CCPD, path lists and native JSON/JSONL proposals."""
import hashlib
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import read_json, read_jsonl, write_json, source_path, require, decode_source
from .geometry import validate_quad, native_height, rectify, prune_overlaps
from . import material


def infer_colour(face):
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    blue = np.mean((hsv[:, :, 0] > 90) & (hsv[:, :, 0] < 140) & (hsv[:, :, 1] > 70))
    green = np.mean((hsv[:, :, 0] > 30) & (hsv[:, :, 0] < 90) & (hsv[:, :, 1] > 30))
    require(blue > .25 or green > .20, '不支持的牌面颜色')
    return 'blue' if blue > .25 else 'green'


def qualify(image, proposal, cfg):
    require(proposal.get('quad') is not None, '只有检测框，缺少可靠四角')
    require(proposal.get('confidence', 1) >= cfg['confidence'], '四角预测置信度不足')
    q = validate_quad(proposal['quad'], image.shape[1], image.shape[0], cfg['min_plate_height'])
    require(bool((q.min(0) >= 2).all() and (q.max(0) <= (image.shape[1] - 3, image.shape[0] - 3)).all()), '牌面过于贴边')
    face, _, _ = rectify(image, q, 'green')
    colour = proposal.get('colour') or infer_colour(face)
    require(colour in ('blue', 'green'), '无效颜色类别')
    face, _, _ = rectify(image, q, colour)
    clean, _, core = material.erase_old_letters(face, colour)
    rows = np.flatnonzero(np.count_nonzero(core, axis=1) >= max(5, face.shape[1] * .018))
    require(len(rows) >= 35, '无法确定旧字符高度')
    return dict(quad=q.tolist(), colour=colour, native_height=native_height(q),
                confidence=proposal.get('confidence', 1), quad_origin=proposal.get('quad_origin', 'imported_not_verified'),
                source_ink_sigma=material.estimate_ink_blur(face, clean, core, colour),
                **({'ocr': proposal['ocr']} if 'ocr' in proposal else {}))


def parse_ccpd(path):
    fields = Path(path).stem.split('-')
    require(len(fields) == 7, '不是 CCPD 文件名')
    points = [list(map(int, item.split('&'))) for item in fields[3].split('_')]
    require(len(points) == 4 and all(len(point) == 2 for point in points), 'CCPD 四角格式错误')
    return dict(quad=points, quad_origin='CCPD_filename', confidence=1,
                **({'colour': 'green'} if 'ccpd_green' in str(path) else {'colour': 'blue'} if 'ccpd_base' in str(path) else {}))


def import_tasks(input_path, cfg, source_type):
    path = Path(input_path).expanduser().resolve()
    if path.suffix == '.json':
        doc = read_json(path)
        if 'records' in doc:
            roots = dict(doc['roots']); roots.update(cfg.get('roots', {}))
            return roots, [(source_path(row, roots), row['plates'], row['root'], row['path'], row.get('sha256')) for row in doc['records']]
    if path.suffix == '.jsonl':
        raw_rows = read_jsonl(path); roots = dict(cfg.get('roots', {})); tasks = []
        for row in raw_rows:
            if 'root' in row:
                p = source_path(row, roots); alias, relative = row['root'], row['path']
            else:
                p = Path(row['background']).resolve(); alias = 'import_' + hashlib.sha256(str(p.parent).encode()).hexdigest()[:12]
                roots[alias] = str(p.parent); relative = p.name
            proposals = row['plates'] if 'plates' in row else [dict(quad=row['quad'], colour=row.get('colour'),
                confidence=row.get('confidence', 1), quad_origin=row.get('quad_origin', 'imported_not_verified'))]
            tasks.append((p, proposals, alias, relative, row.get('sha256')))
        return roots, tasks
    if path.is_dir():
        paths = sorted(p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        roots = dict(cfg.get('roots', {})); roots['input'] = str(path)
        relative = lambda p:p.relative_to(path).as_posix()
    elif path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
        paths = [path]; roots = dict(cfg.get('roots', {})); roots['input'] = str(path.parent)
        relative = lambda p:p.name
    else:
        paths = [Path(line.strip()).expanduser().resolve() for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
        roots = dict(cfg.get('roots', {}))
        # Different path-list parents become independently remappable aliases.
        tasks = []
        for p in paths:
            alias = 'list_' + hashlib.sha256(str(p.parent).encode()).hexdigest()[:12]
            roots[alias] = str(p.parent)
            proposals = []
            if source_type in ('auto', 'ccpd'):
                try:proposals = [parse_ccpd(p)]
                except (ValueError, IndexError):
                    if source_type == 'ccpd':raise
            tasks.append((p, proposals, alias, p.name, None))
        return roots, tasks
    tasks = []
    for p in paths:
        proposals, expected = [], None
        if source_type in ('auto', 'ccpd'):
            try:proposals = [parse_ccpd(p)]
            except (ValueError, IndexError):
                if source_type == 'ccpd':raise
        tasks.append((p, proposals, 'input', relative(p), expected))
    return roots, tasks


def prepare(input_path, output, cfg, source_type='auto', ocr=False):
    roots, tasks = import_tasks(input_path, cfg, source_type)
    frozen = heldout_hashes(cfg)
    locations = heldout_locations(cfg)
    engine, records, rejects, grouped, hashes = None, [], [], {}, set()
    for path, proposals, alias, relative, expected in tasks:
        try:
            raw = path.read_bytes(); digest = hashlib.sha256(raw).hexdigest()
            require(not expected or expected == digest, '导入预标与原图哈希不一致')
            check_source_separation(dict(root=alias,path=relative,sha256=digest), roots, frozen, locations)
            image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            require(image is not None, '图像无法解码')
            if not proposals:
                if engine is None:
                    from .inference import Engine
                    engine = Engine(cfg)
                proposals = engine.detect(image, ocr=ocr)
            plates = []
            for proposal in proposals:
                try:plates.append(qualify(image, proposal, cfg))
                except (ValueError, cv2.error) as exc:rejects.append(dict(path=str(path), reason=str(exc), proposal=proposal))
            if digest in grouped:
                grouped[digest]['plates'].extend(plates)
                continue
            require(bool(plates), '没有合格车牌')
            row = dict(id=digest, root=alias, path=relative, sha256=digest, width=image.shape[1], height=image.shape[0], plates=plates)
            grouped[digest] = row; records.append(row); hashes.add(digest)
        except (ValueError, OSError, cv2.error) as exc:rejects.append(dict(path=str(path), reason=str(exc)))
    for row in records:
        row['plates'], overlap = prune_overlaps(row['plates'], (row['height'], row['width']))
        rejects.extend(dict(path=row['path'], **r) for r in overlap)
        for i, plate in enumerate(row['plates']):plate['id'] = str(i)
    doc = dict(schema_version=1, roots=roots, records=records, rejection_count=len(rejects),
               rejection_reasons=dict(Counter(r['reason'] for r in rejects)), rejections=rejects,
               source_image_count=len(tasks), qualified_image_count=len(records),
               heldout_sha256=sorted(frozen), **separation_status(cfg, locations), detection_completeness='all_qualified_detected_targets_only_not_recall_or_human_GT')
    require(bool(records), f'没有合格底图: {doc["rejection_reasons"]}')
    if output:write_json(output, doc)
    return doc


def resolved_roots(index, cfg):
    roots = dict(index['roots']); roots.update(cfg.get('roots', {}))
    return roots


def heldout_hashes(cfg):
    """Only explicitly configured registries participate in separation checks."""
    raw = cfg.get('heldout_registry')
    if raw is None:return set()
    rows = read_json(Path(raw).expanduser().resolve())
    require(isinstance(rows, list) and all(isinstance(v, str) and len(v) == 64 and
            all(c in '0123456789abcdef' for c in v) for v in rows), '留出哈希注册表必须为 SHA256 数组')
    return set(rows)


def separation_status(cfg, locations):
    content = cfg.get('heldout_registry') is not None
    paths = cfg.get('heldout_fingerprints') is not None
    return dict(heldout_verification='configured_checks_passed' if content or paths else 'not_configured',
                heldout_content_intersection=0 if content else None,
                heldout_path_intersection=0 if paths else None,
                heldout_physical_intersection=0 if paths and locations[1] else None,
                heldout_historical_paths=len(locations[0]), heldout_physical_files_available=len(locations[1]))


def heldout_locations(cfg):
    """Optional explicit JSONL paths; inode checks use existing files only."""
    raw = cfg.get('heldout_fingerprints')
    if raw is None:return set(), set(), 0
    path = Path(raw).expanduser().resolve()
    require(path.is_file(), '缺少显式留出路径记录')
    rows = read_jsonl(path); paths, physical = set(), set()
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get('path'), str) and bool(row['path']), '留出路径记录格式错误')
        p = Path(row['path']).expanduser().resolve(); paths.add(str(p))
        try:
            stat = p.stat(); physical.add((stat.st_dev, stat.st_ino))
        except FileNotFoundError:pass
    return paths, physical, len(rows)


def check_source_separation(record, roots, frozen, locations):
    paths, physical, _ = locations
    path = source_path(record, roots); stat = path.stat()
    require(record['sha256'] not in frozen, '来源与历史留出集内容重叠')
    require(str(path) not in paths, '来源与历史留出集路径重叠')
    require((stat.st_dev, stat.st_ino) not in physical, '来源与历史留出集当前物理文件重叠')
