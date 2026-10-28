"""Self-contained static HTML review with real source/output/crop comparisons."""
import html
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from .common import read_json, read_jsonl, write_json, atomic_bytes, font_path, source_path
from .geometry import rectify
from .sources import resolved_roots


def report(root, cfg):
    root = Path(root).resolve()
    index, plan = read_json(root / 'index.json'), read_json(root / 'plan.json')
    annotations = read_jsonl(root / 'annotations.jsonl'); records = {r['id']:r for r in index['records']}
    roots = resolved_roots(index, cfg)
    out = root / 'review'; out.mkdir(exist_ok=True)
    try:
        preview = cfg.get('fonts', {}).get('preview', 'fonts/preview.ttc')
        spec = dict(path=preview, index=0) if isinstance(preview, str) else preview
        font = ImageFont.truetype(str(font_path(spec['path'], cfg)), 18, index=spec.get('index', 0))
        font_status = 'configured'
    except (OSError, ValueError):
        # Review-only fallback: never alter the production han/latin font settings.
        font = ImageFont.load_default()
        font_status = 'ascii_fallback'
    chosen, seen = [], set()
    for ann, job in zip(annotations, plan['jobs']):
        for plate in ann['plates']:
            key = (plate['text'][0], plate['plate_type'])
            if key not in seen:
                chosen.append((ann, job, plate)); seen.add(key)
    pieces, gallery = [], []
    for i, (ann, job, plate) in enumerate(chosen):
        record = records[job['source_id']]
        source = cv2.imread(str(source_path(record, roots)), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        image = cv2.imread(str(root / ann['image']))
        colour = 'blue' if plate['plate_type'] == 'blue' else 'green'
        a, _, _ = rectify(source, plate['quad'], colour); b, _, _ = rectify(image, plate['quad'], colour)
        canvas = Image.new('RGB', (1000, 174), '#eeeeee'); draw = ImageDraw.Draw(canvas)
        canvas.paste(Image.fromarray(cv2.cvtColor(a, cv2.COLOR_BGR2RGB)), (6, 30))
        canvas.paste(Image.fromarray(cv2.cvtColor(b, cv2.COLOR_BGR2RGB)), (510, 30))
        source_label = 'Source' if font_status == 'ascii_fallback' else '原图'
        target_label = (f'Sample {i + 1:06d} / ' + plate['plate_type'].encode('ascii', 'replace').decode('ascii')
                        if font_status == 'ascii_fallback' else plate['text'] + ' / ' + plate['plate_type'])
        draw.text((6, 4), source_label, font=font, fill='black')
        draw.text((510, 4), target_label, font=font, fill='black')
        path = out / f'{i + 1:06d}.jpg'; canvas.save(path, quality=96, subsampling=0)
        pieces.append(f'<figure><figcaption>{html.escape(plate["text"])} — {html.escape(ann["image"])}</figcaption><a href="../{html.escape(ann["image"],quote=True)}"><img src="{path.name}" loading="lazy"></a></figure>')
        gallery.append(dict(image=ann['image'], text=plate['text'], plate_type=plate['plate_type'], preview=path.name))
    if chosen:
        sheet = Image.new('RGB', (1000, 174 * min(6, len(chosen))), '#eeeeee')
        for i in range(min(6, len(chosen))):
            with Image.open(out / f'{i + 1:06d}.jpg') as im:sheet.paste(im, (0, i * 174))
        sheet.save(out / 'overview.jpg', quality=96, subsampling=0)
    summary = read_json(root / 'summary.json')
    page = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>车牌替换校验与审图</title><style>body{max-width:1100px;margin:24px auto;font-family:sans-serif;background:#f3f4f6}pre{white-space:pre-wrap}img{max-width:100%}figure{margin:20px 0;background:white;padding:12px}</style><h1>车牌替换校验与审图</h1><p>左侧原牌、右侧合成牌。点击对照图可打开整张合成图。自动检测四角和 OCR 回读均不是人工真值。</p><p>工程校验与视觉接受分开记录；JPG 的牌外像素可能存在编码差异。</p><pre>' + html.escape(__import__('json').dumps(summary, ensure_ascii=False, indent=2)) + '</pre>' + ''.join(pieces) + '</html>'
    if font_status == 'ascii_fallback':
        page = page.replace('</h1>', '</h1><p>Preview font unavailable; thumbnail labels use ASCII identifiers. Unicode labels remain below.</p>', 1)
    atomic_bytes(out / 'index.html', page.encode('utf-8'))
    write_json(out / 'samples.json', gallery)
    return dict(review_html=str(out / 'index.html'), reviewed_sample_count=len(chosen),
                human_review_status='pending', preview_font_status=font_status)
