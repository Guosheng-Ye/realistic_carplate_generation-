"""The rcg CLI. No implicit downloads or dependencies on the parent checkout."""
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import cv2

from .common import (project_home, asset_check, load_config, read_json, write_json,
                     write_jsonl, require, sha256, dumps, atomic_bytes)


def doctor(models=False, cfg=None, index=None):
    cfg = load_config() if cfg is None else cfg
    versions, missing = {}, []
    required = ('numpy', 'Pillow', 'opencv-python', 'scipy')
    if models:
        required += ('onnxruntime',)
    for name in required:
        try:versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None; missing.append(name)
    manifest = asset_check(required=cfg.get('required_assets', []), cfg=cfg)
    result = dict(status='missing_dependencies' if missing else 'passed', platform=platform.platform(),
                  python=platform.python_version(), versions=versions, missing_dependencies=missing,
                  assets_verified=len(manifest['files']), runtime='CPU only')
    if cfg.get('fonts'):
        from .rendering import renderer, render_asset_check
        renderer(cfg)
        result['render_assets'] = render_asset_check(cfg)
    from .sources import resolved_roots
    roots = resolved_roots(index, cfg) if index is not None else cfg.get('roots', {})
    for alias, raw in roots.items():
        value = str(raw)
        if '${RCG_HOME}' in value:value = value.replace('${RCG_HOME}', str(project_home()))
        path = Path(value).expanduser()
        if not path.is_absolute():path = project_home() / path
        require(path.is_dir(), f'数据根目录不存在: {alias}={path}')
    checked = 0
    if index is not None:
        from .common import source_path
        for record in index['records']:
            require(source_path(record, roots).is_file(), '索引来源文件不存在')
            checked += 1
    result['input_roots_checked'] = len(roots); result['input_images_checked'] = checked
    if models:
        from .inference import Engine
        result['model_contracts'] = Engine(cfg).contracts()
    return result


def add_common(parser):
    parser.add_argument('--config', type=Path)
    parser.add_argument('--root', action='append', default=[], metavar='ALIAS=PATH')


def generation_args(parser):
    add_common(parser)
    parser.add_argument('--index', type=Path)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--count', type=int)
    parser.add_argument('--targets', choices=('single', 'all'), default='single')
    parser.add_argument('--target-id')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--reuse-blue')
    parser.add_argument('--reuse-green')
    parser.add_argument('--format', choices=('jpg', 'png'))
    parser.add_argument('--render-mode', choices=('material', 'full'))
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--no-crops', action='store_true')


def main(argv=None):
    parser = argparse.ArgumentParser(prog='rcg', description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('doctor'); p.add_argument('--models', action='store_true'); add_common(p); p.add_argument('--index', type=Path)
    p = subs.add_parser('prepare'); add_common(p); p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--source', choices=('auto','ccpd','manifest'), default='auto'); p.add_argument('--ocr', action='store_true')
    p = subs.add_parser('detect'); add_common(p); p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--ocr', action='store_true')
    p = subs.add_parser('rectify'); p.add_argument('--input', type=Path, required=True); p.add_argument('--quad', required=True); p.add_argument('--colour', choices=('blue','green'), required=True); p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('render'); add_common(p); p.add_argument('--text'); p.add_argument('--plate-type', choices=('blue','nev_small','nev_large')); p.add_argument('--count', type=int, default=1); p.add_argument('--seed', type=int, default=20261023); p.add_argument('--out', type=Path, required=True)
    p = subs.add_parser('plan'); generation_args(p); p.add_argument('--out', type=Path)
    p = subs.add_parser('generate'); generation_args(p); p.add_argument('--out', type=Path, required=True); p.add_argument('--workers', type=int, default=4); p.add_argument('--resume', action='store_true')
    for name in ('validate','report'):
        p = subs.add_parser(name); add_common(p); p.add_argument('--dataset', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
        if args.command == 'doctor':result = doctor(args.models, load_config(args.config, args.root), read_json(args.index) if args.index else None)
        elif args.command == 'prepare':
            from .sources import prepare
            doc = prepare(args.input, args.out, load_config(args.config, args.root), args.source, args.ocr)
            result = {k:doc[k] for k in ('source_image_count','qualified_image_count','rejection_count','rejection_reasons')}
        elif args.command == 'detect':
            from .inference import Engine
            cfg = load_config(args.config, args.root)
            engine = Engine(cfg); paths = sorted(p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png')) if args.input.is_dir() else [args.input]
            rows = []
            for path in paths:
                image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION); require(image is not None, '照片无法解码')
                rows.append(dict(background=str(path.resolve()), sha256=sha256(path), width=image.shape[1], height=image.shape[0], plates=engine.detect(image, args.ocr)))
            write_jsonl(args.out, rows); result = dict(images=len(rows), candidates=sum(len(r['plates']) for r in rows))
        elif args.command == 'rectify':
            from .geometry import rectify
            from .rendering import encode
            image = cv2.imread(str(args.input), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION); require(image is not None, '照片无法解码')
            face, crop, result = rectify(image, json.loads(args.quad), args.colour)
            atomic_bytes(args.out / 'rectified.png', encode(face, 'png')); atomic_bytes(args.out / 'native.png', encode(crop, 'png'))
            write_json(args.out / 'geometry.json', result)
        elif args.command == 'render':
            from .rules import make_samples, PLATE_TYPES, valid_text
            from .rendering import render_plate, encode
            cfg = load_config(args.config, args.root)
            require(args.count > 0, 'count 必须为正数')
            if args.text:
                kind = args.plate_type or next((k for k in PLATE_TYPES if valid_text(args.text,k)), None)
                require(kind and valid_text(args.text,kind), '号码不符合牌型规则')
                samples = [dict(text=args.text,plate_type=kind)]
            else:
                samples = make_samples(args.count * 8 if args.plate_type else args.count, args.seed)
                if args.plate_type:samples = [s for s in samples if s['plate_type']==args.plate_type][:args.count]
            require(not args.out.exists() or not any(args.out.iterdir()), 'render 输出目录必须为空')
            for i, s in enumerate(samples,1):atomic_bytes(args.out/f'{i:06d}.png', encode(render_plate(**s,seed=args.seed+i,cfg=cfg),'png'))
            write_jsonl(args.out/'labels.jsonl', [dict(image=f'{i:06d}.png',**s) for i,s in enumerate(samples,1)])
            result = dict(plates=len(samples), out=str(args.out))
        elif args.command in ('validate','report'):
            cfg = read_json(args.dataset/'config.json')
            for entry in args.root:
                key,value=entry.split('=',1);cfg['roots'][key]=str(Path(value).resolve())
            if args.config:
                extra=read_json(args.config);cfg.update({k:v for k,v in extra.items() if k!='roots'});cfg['roots'].update(extra.get('roots',{}))
            if args.command=='validate':
                from .validation import validate
                result=validate(args.dataset,cfg)
            else:
                from .reporting import report
                result=report(args.dataset,cfg)
        else:
            from .sources import prepare
            from .planner import build_plan
            cfg = load_config(args.config, args.root)
            if args.command=='generate' and args.resume:
                cfg=read_json(args.out/'config.json')
                if args.config:
                    extra=read_json(args.config);cfg.update({k:v for k,v in extra.items() if k!='roots'});cfg['roots'].update(extra.get('roots',{}))
                for entry in args.root:
                    key,value=entry.split('=',1);cfg['roots'][key]=str(Path(value).resolve())
                index=read_json(args.out/'index.json');plan=read_json(args.out/'plan.json')
            else:
                require(args.count is not None and (args.index or args.input), '需要 --count 和 --index 或 --input')
                for key in ('seed','reuse_blue','reuse_green','format','render_mode'):
                    value=getattr(args,key)
                    if value is not None:cfg[key]=value
                if args.no_augment:cfg['augment']=False
                if args.no_crops:cfg['export_crops']=False
                index=read_json(args.index) if args.index else prepare(args.input,None,cfg)
                plan=build_plan(index,args.count,cfg,args.targets,args.target_id)
            if args.command=='plan':
                if args.out:write_json(args.out,plan)
                result={k:v for k,v in plan.items() if k!='jobs'}
            else:
                from .pipeline import generate
                result=generate(index,plan,args.out,cfg,args.workers,args.resume)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,KeyError,InterruptedError) as exc:
        print(dumps(dict(status='failed',error=str(exc))),file=sys.stderr)
        return 2


if __name__ == '__main__':raise SystemExit(main())
