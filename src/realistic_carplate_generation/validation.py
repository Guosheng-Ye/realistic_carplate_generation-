"""Validate actual image files, coordinates, journals and integer distributions."""
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .common import read_json, read_jsonl, write_json, sha256, require, decode_source, source_path, dumps
from .geometry import polygon_mask, validate_quad
from .rules import summarize, apportion, PLATE_TYPES, WEIGHTS
from .sources import resolved_roots, heldout_hashes, heldout_locations, check_source_separation, separation_status


def validate(root, cfg):
    root = Path(root).resolve()
    index, plan = read_json(root / 'index.json'), read_json(root / 'plan.json')
    from .pipeline import signature
    require(signature(cfg, plan, index) == read_json(root / 'run_identity.json')['signature'], '运行身份、资源、配置或代码改变')
    roots = resolved_roots(index, cfg); sources = {r['id']: r for r in index['records']}
    frozen = heldout_hashes(cfg); locations = heldout_locations(cfg)
    targets_by_source = {j['source_id']: j['plates'] for j in plan['jobs']}
    annotations, audits = read_jsonl(root / 'annotations.jsonl'), read_jsonl(root / 'audit.jsonl')
    require(len(annotations) == len(audits) == plan['image_count'], '图片标注或审计数量不正确')
    require(len(list((root / 'images').iterdir())) == plan['image_count'], '图像目录文件数量不正确')
    if plan['targets'] == 'single':
        labels = read_jsonl(root / 'labels.jsonl')
        require(len(labels) == plan['image_count'] and all(set(r) == {'image', 'text', 'plate_type', 'quad', 'background'} for r in labels), '单牌五字段标签错误')
    samples, uses, checked_sources, crop_count, operations, counts = [], Counter(), set(), 0, Counter(), Counter()
    for i, (ann, audit, job) in enumerate(zip(annotations, audits, plan['jobs'])):
        record = sources[job['source_id']]
        require(ann['image'] == job['image'] == audit['output']['path'], '图片路径与计划不一致')
        expected = [{k:p[k] for k in ('text', 'plate_type', 'quad')} for p in job['plates']]
        require(ann['plates'] == expected, '号码、类型或四角与计划不同')
        require(ann['background'] == audit['labels'][0]['background'], '来源路径与原始生成审计不一致')
        require(audit['source_sha256'] == record['sha256'], '来源身份错误')
        require(audit['audit']['outside_plate_before_encoding_changed_pixels'] == 0, '编码前牌外像素变化')
        require(sha256(root / ann['image']) == audit['output']['sha256'], '写盘图片哈希错误')
        image = cv2.imread(str(root / ann['image']))
        require(image is not None and image.shape[:2] == (record['height'], record['width']), '输出尺寸或解码错误')
        source = decode_source(record, roots) if cfg['format'] == 'png' or record['id'] not in checked_sources else None
        union = np.zeros(image.shape[:2], np.uint8)
        require(len(job['plates']) == len(audit['audit']['plates']), '逐牌审计数量不正确')
        for target, entry in zip(job['plates'], audit['audit']['plates']):
            require(entry['text'] == target['text'] and entry['seed'] == target['seed'], '逐牌审计与计划不一致')
            q = validate_quad(target['quad'], record['width'], record['height'], cfg['min_plate_height'])
            require(any(p['quad'] == target['quad'] and p['colour'] == target['colour'] for p in record['plates']), '四角不是合格来源四角')
            mask = polygon_mask(image.shape, q)
            require(not np.any((union > 0) & (mask > 0)), '目标掩膜相交')
            union |= mask; samples.append(target)
            require(len(entry['operations']) <= 3 and len(set(entry['operations'])) == len(entry['operations']), '增强操作数错误')
            operations.update(entry['operations']); counts[len(entry['operations'])] += 1
        if cfg['format'] == 'png':require(np.array_equal(image[union == 0], source[union == 0]), 'PNG 保存后牌外像素不同')
        if plan['targets'] == 'single':require(labels[i] == audit['labels'][0], '五字段标签与事务日志不一致')
        if record['id'] not in checked_sources:
            check_source_separation(record, roots, frozen, locations)
            checked_sources.add(record['id'])
        uses[record['id']] += 1
        for crop in audit['crops']:
            require(sha256(root / crop['path']) == crop['sha256'], '裁剪文件哈希错误')
            img = cv2.imread(str(root / crop['path']))
            require(img is not None, '裁剪解码失败')
            h, w = img.shape[:2]
            require(crop['label']['quad'] == [[w - 1, h - 1], [0, h - 1], [0, 0], [w - 1, 0]], '裁剪四角坐标系错误')
            require(crop['source_image'] == ann['image'] and any(p['quad'] == crop['source_quad'] for p in job['plates']), '裁剪来源映射错误')
            crop_count += 1
    stats = summarize(samples)
    require(stats == plan['target_counts'], '实际号码分布与计划不同')
    kind = cfg.get('plate_type', 'all')
    weights = WEIGHTS if kind == 'all' else tuple(int(k == kind) for k in PLATE_TYPES)
    require(list(stats['plate_type'].values()) == apportion(len(samples), weights), '全局牌型配额错误')
    require(len(samples) == plan['plate_count'], '实际替换车牌数错误')
    if cfg['export_crops']:
        crop_labels = read_jsonl(root / 'crops/labels.jsonl')
        require(crop_count == len(crop_labels) == len(samples), 'OCR 裁剪或标签数错误')
        expected_crop_labels = [c['label'] for a in audits for c in a['crops']]
        require(crop_labels == expected_crop_labels, '裁剪标签与已核验裁剪映射不一致')
    for ident, count in uses.items():
        targets = targets_by_source[ident]
        maximum = cfg['reuse_blue'] if any(t['colour'] == 'blue' for t in targets) else cfg['reuse_green']
        require(maximum == 'auto' or count <= int(maximum), '底图复用超出限制')
    summary = dict(status='complete_engineering_validated', image_count=len(annotations), plate_count=len(samples),
        crop_count=crop_count, distributions=stats, unique_backgrounds=len(uses),
        background_reuse_histogram=dict(Counter(uses.values())), augmentation_families=dict(operations),
        augmentation_count_histogram=dict(counts), outside_before_encoding_changed_pixels=0,
        png_saved_outside_changed_pixels=0 if cfg['format'] == 'png' else None,
        final_jpeg_pixel_identity=False if cfg['format'] == 'jpg' else None,
        original_dimensions_and_quads_preserved=True, source_hashes_unchanged=True,
        output_hashes_and_decoding_checked=True, **separation_status(cfg, locations),
        annotations_sha256=sha256(root / 'annotations.jsonl'), audit_sha256=sha256(root / 'audit.jsonl'),
        visual_acceptance='requires_human_review', ocr_accuracy='not_measured_without_independent_human_GT')
    write_json(root / 'summary.json', summary)
    return summary
