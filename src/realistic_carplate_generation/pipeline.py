"""Transactional generation journal; manifest publication follows validation."""
import fcntl
import hashlib
import multiprocessing
import platform
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import PIL

from .common import (sha256, read_json, dumps, write_json, write_jsonl, require,
                     atomic_bytes, source_path)
from .rendering import render_job, render_asset_check
from .sources import resolved_roots, heldout_hashes, heldout_locations, check_source_separation


def code_identity():
    return {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def signature(cfg, plan, index):
    clean_cfg = {key: value for key, value in cfg.items() if key not in ('roots', '_stop_after')}
    return hashlib.sha256(dumps(dict(config=clean_cfg, plan=plan,
        sources=index['records'], frozen=index.get('heldout_sha256', []), assets=render_asset_check(cfg), code=code_identity())).encode()).hexdigest()


def worker_init():
    cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)


def worker(task):
    record, job, roots, cfg = task
    from .geometry import validate_quad
    # Validate every target before any rendering/back-paste, including full mode.
    for target in job['plates']:
        validate_quad(target['quad'], record['width'], record['height'], cfg['min_plate_height'])
    return render_job(record, job, roots, cfg)


def generate(index, plan, out_root, cfg, workers=4, resume=False):
    require(1 <= workers <= 12, 'workers 必须为 1–12')
    root = Path(out_root).resolve()
    if root.exists() and not resume:require(not any(root.iterdir()), '输出目录非空，禁止覆盖；本任务恢复需 --resume')
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / '.run.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    db = None
    try:
        roots = resolved_roots(index, cfg); records = {row['id']: row for row in index['records']}
        sig = signature(cfg, plan, index)
        db = sqlite3.connect(root / 'journal.sqlite3')
        db.execute('PRAGMA journal_mode=WAL'); db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, image TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)')
        prior = db.execute("SELECT value FROM metadata WHERE key='signature'").fetchone()
        if prior:require(resume and prior[0] == sig, '恢复配置、代码、模型、字体、计划或来源身份改变')
        else:
            require(not resume, '没有可恢复的事务日志')
            db.execute('INSERT INTO metadata VALUES (?,?)', ('signature', sig)); db.commit()
            write_json(root / 'index.json', index); write_json(root / 'plan.json', plan)
            write_json(root / 'config.json', cfg)
            write_json(root / 'run_identity.json', dict(signature=sig, code=code_identity(), assets=render_asset_check(cfg),
                versions=dict(python=platform.python_version(), numpy=np.__version__, pillow=PIL.__version__, opencv=cv2.__version__)))
        completed = db.execute('SELECT id,payload FROM images ORDER BY id').fetchall()
        for expected, (ident, raw) in enumerate(completed):
            require(ident == expected, '事务日志不是连续提交前缀')
            import json
            payload = json.loads(raw)
            for item in [payload['output'], *payload['crops']]:
                require(sha256(root / item['path']) == item['sha256'], '恢复前已提交图片或裁剪哈希错误')
        # Check every source before resuming, including those not yet rendered.
        frozen = heldout_hashes(cfg); locations = heldout_locations(cfg)
        for row in records.values():
            check_source_separation(row, roots, frozen, locations)
            require(sha256(source_path(row, roots)) == row['sha256'], '底图哈希改变')
        start = len(completed); started = time.monotonic()
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=worker_init) as pool:
            for offset in range(start, len(plan['jobs']), workers * 2):
                jobs = plan['jobs'][offset:offset + workers * 2]
                tasks = [(records[j['source_id']], j, roots, cfg) for j in jobs]
                for local, (encoded, crops, audit) in enumerate(pool.map(worker, tasks)):
                    position = offset + local; job = jobs[local]; record = records[job['source_id']]
                    digest = hashlib.sha256(encoded).hexdigest()
                    outputs = [(job['image'], encoded, digest)] + [(c['path'], c['data'], hashlib.sha256(c['data']).hexdigest()) for c in crops]
                    for relative, data, expected_sha in outputs:
                        path = root / relative
                        if path.exists():require(sha256(path) == expected_sha, '未提交的同名文件与确定性重跑不一致，保留现场停止')
                        else:atomic_bytes(path, data)
                    background = str(source_path(record, roots))
                    labels = [dict(image=job['image'], text=p['text'], plate_type=p['plate_type'], quad=p['quad'], background=background)
                              for p in job['plates']]
                    crop_rows = [dict(path=c['path'], sha256=hashlib.sha256(c['data']).hexdigest(), label=c['label'],
                                      source_quad=c['source_quad'], source_image=c['source_image'], target_id=c['target_id']) for c in crops]
                    payload = dict(output=dict(path=job['image'], sha256=digest), source_id=record['id'],
                                   source_sha256=record['sha256'], labels=labels, crops=crop_rows, audit=audit)
                    with db:db.execute('INSERT INTO images VALUES (?,?,?)', (position, job['image'], dumps(payload)))
                    done = position + 1
                    write_json(root / 'progress.json', dict(status='generating', completed=done, total=plan['image_count'],
                        elapsed_seconds=round(time.monotonic() - started, 2)))
                    if cfg.get('_stop_after') and done >= cfg['_stop_after']:raise InterruptedError('test_requested_interruption')
                print(f"已生成 {offset + len(jobs)}/{plan['image_count']} 张图片", flush=True)
        publish(root, db, plan)
        from .validation import validate
        summary = validate(root, cfg)
        if cfg.get('generate_report', False):
            from .reporting import report
            report(root, cfg)
        write_json(root / 'progress.json', dict(status='complete', completed=plan['image_count'], total=plan['image_count']))
        return summary
    finally:
        if db is not None:db.close()
        lock.close()


def publish(root, db, plan):
    import json
    payloads = [json.loads(raw) for (raw,) in db.execute('SELECT payload FROM images ORDER BY id')]
    require(len(payloads) == plan['image_count'], '任务未完成，不发布标签')
    annotations = []
    for item in payloads:
        labels = item['labels']
        annotations.append(dict(image=item['output']['path'], background=labels[0]['background'],
            plates=[{k: row[k] for k in ('text', 'plate_type', 'quad')} for row in labels]))
    write_jsonl(root / 'annotations.jsonl', annotations)
    if plan['targets'] == 'single':write_jsonl(root / 'labels.jsonl', [p['labels'][0] for p in payloads])
    write_jsonl(root / 'audit.jsonl', payloads)
    if any(p['crops'] for p in payloads):write_jsonl(root / 'crops/labels.jsonl', [c['label'] for p in payloads for c in p['crops']])
