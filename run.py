#!/usr/bin/env python3
"""Read the local configuration and synthesize plates or replace them in photos."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from realistic_carplate_generation.common import load_config, read_json, require


def default_config():
    local = ROOT / 'config.local.json'
    return local if local.is_file() else ROOT / 'config.json'


def load_settings(path):
    path = Path(path).expanduser().resolve()
    raw = read_json(path)
    require(isinstance(raw, dict), '配置文件必须是 JSON 对象')
    job_keys = {'mode', 'input', 'output', 'count', 'plate_type', 'workers', 'resume', 'targets'}
    unknown = set(raw) - set(load_config()) - job_keys
    require(not unknown, '未知配置项: ' + ', '.join(sorted(unknown)))
    for key in ('fonts', 'models', 'roots'):
        require(key not in raw or isinstance(raw[key], dict), f'{key} 必须是 JSON 对象')
    cfg = load_config(path)
    for key, value in {'mode': 'render', 'input': 'input', 'output': 'outputs/plates',
                       'count': 100, 'plate_type': 'blue', 'workers': 2,
                       'resume': False, 'targets': 'single'}.items():
        cfg.setdefault(key, value)
    cfg['format'] = raw.get('format', 'png')
    cfg['reuse_blue'] = raw.get('reuse_blue', 'auto')
    cfg['reuse_green'] = raw.get('reuse_green', 'auto')
    require(cfg['mode'] in ('render', 'replace'), 'mode 只能是 render 或 replace')
    require(type(cfg['count']) is int and cfg['count'] > 0, 'count 必须是正整数')
    require(type(cfg['seed']) is int, 'seed 必须是整数')
    require(type(cfg['workers']) is int and 1 <= cfg['workers'] <= 12, 'workers 必须是 1–12 的整数')
    require(cfg['plate_type'] in ('all', 'blue', 'nev_small', 'nev_large'), 'plate_type 不受支持')
    require(cfg['format'] in ('png', 'jpg'), 'format 只能是 png 或 jpg')
    require(cfg['render_mode'] in ('material', 'full'), 'render_mode 只能是 material 或 full')
    require(cfg['targets'] in ('single', 'all'), 'targets 只能是 single 或 all')
    for key in ('resume', 'augment', 'export_crops', 'generate_report'):
        require(type(cfg[key]) is bool, f'{key} 必须是 true 或 false')
    require(not (cfg['mode'] == 'render' and cfg['resume']), 'resume 仅用于 replace 模式；正面生成请使用新输出目录')
    require(isinstance(cfg['output'], str) and bool(cfg['output'].strip()), 'output 不能为空')
    require(isinstance(cfg['input'], str), 'input 必须是路径字符串')
    require(cfg['mode'] != 'replace' or bool(cfg['input'].strip()), 'replace 模式需要 input 路径')
    for key in ('input', 'output'):
        value = Path(cfg[key]).expanduser()
        cfg[key] = str((value if value.is_absolute() else path.parent / value).resolve())
    value = Path(cfg['assets_dir']).expanduser() if cfg.get('assets_dir') else ROOT / 'assets'
    cfg['assets_dir'] = str((value if value.is_absolute() else path.parent / value).resolve())
    supplied_fonts = raw.get('fonts', {})
    require(not (set(supplied_fonts) - {'han', 'latin', 'preview'}), 'fonts 只支持 han、latin、preview')
    for role, value in cfg['fonts'].items():
        require(isinstance(value, (str, dict)), f'fonts.{role} 必须是路径或 path/index 对象')
        spec = {'path': value, 'index': 0} if isinstance(value, str) else dict(value)
        require(isinstance(spec.get('path'), str) and bool(spec['path'].strip()), f'fonts.{role}.path 不能为空')
        require(type(spec.get('index', 0)) is int and spec.get('index', 0) >= 0, f'fonts.{role}.index 必须是非负整数')
        font = Path(spec['path']).expanduser()
        base = path.parent if role in supplied_fonts else Path(cfg['assets_dir'])
        spec['path'] = str((font if font.is_absolute() else base / font).resolve())
        cfg['fonts'][role] = spec['path'] if isinstance(value, str) else spec
    return cfg


def engine_settings(settings):
    job_keys = {'mode', 'input', 'output', 'count', 'workers', 'resume', 'targets'}
    return {key: value for key, value in settings.items() if key not in job_keys}


def run_render(settings):
    import fcntl
    import cv2
    from realistic_carplate_generation.common import atomic_bytes, write_jsonl
    from realistic_carplate_generation.rendering import renderer, render_plate, encode
    from realistic_carplate_generation.rules import make_samples

    output = Path(settings['output'])
    require(not output.exists() or not any(output.iterdir()), '输出目录非空，请更换 output，避免覆盖已有图片')
    cfg = engine_settings(settings)
    renderer(cfg)  # Check both fonts before reserving the output directory.
    samples = make_samples(settings['count'], settings['seed'], settings['plate_type'])
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(not any(p.name != '.run.lock' for p in output.iterdir()), '输出目录非空，禁止覆盖')
        labels = []
        for index, sample in enumerate(samples, 1):
            image = render_plate(**sample, seed=settings['seed'] + index, cfg=cfg)
            data = encode(image, settings['format'], settings['jpeg_quality'])
            relative = f"images/{index:06d}.{settings['format']}"
            atomic_bytes(output / relative, data)
            saved = cv2.imread(str(output / relative))
            require(saved is not None and saved.shape == image.shape, '生成图片无法解码或尺寸异常')
            labels.append(dict(image=relative, **sample))
        write_jsonl(output / 'labels.jsonl', labels)
    return {'image_count': len(labels), 'output': str(output)}


def run_replace(settings):
    from realistic_carplate_generation.rendering import renderer
    from realistic_carplate_generation.sources import prepare
    from realistic_carplate_generation.planner import build_plan
    from realistic_carplate_generation.pipeline import generate

    output = Path(settings['output'])
    cfg = engine_settings(settings)
    if settings['resume']:
        require((output / 'config.json').is_file(), '没有可恢复的任务，请检查 output')
        index = read_json(output / 'index.json')
        plan = read_json(output / 'plan.json')
        require(plan['image_count'] == settings['count'] and plan['targets'] == settings['targets'],
                '恢复时不能修改 count 或 targets')
    else:
        require(not output.exists() or not any(output.iterdir()), '输出目录非空，请更换 output 或设置 resume 为 true')
        require(Path(settings['input']).exists(), 'input 不存在，请放入照片或填写四角标注文件路径')
        renderer(cfg)
        print('正在检查底图并规划生成任务……', flush=True)
        index = prepare(settings['input'], None, cfg)
        plan = build_plan(index, settings['count'], cfg, settings['targets'])
    return generate(index, plan, output, cfg, workers=settings['workers'], resume=settings['resume'])


def main(argv=None):
    parser = argparse.ArgumentParser(description='修改 config.json 后运行，支持正面车牌与照片号码替换。')
    parser.add_argument('--config', type=Path, default=default_config(),
                        help='配置文件；默认优先读取脚本同目录的 config.local.json，其次为 config.json')
    args = parser.parse_args(argv)
    try:
        import cv2
        cv2.setNumThreads(1); cv2.ocl.setUseOpenCL(False)
        settings = load_settings(args.config)
        result = run_render(settings) if settings['mode'] == 'render' else run_replace(settings)
        print(f"完成：{result['image_count']} 张图片，输出目录：{settings['output']}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f'错误：{exc}\n请检查配置文件：{args.config}', file=sys.stderr)
        return 2
    except ImportError as exc:
        print(f'缺少依赖：{exc}\n请先运行：python -m pip install -e ".[detection]"', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('已停止。照片替换任务可在同一配置中设置 resume 为 true 后继续。', file=sys.stderr)
        return 130


if __name__ == '__main__':
    raise SystemExit(main())
