"""Portable assets, immutable source identities and atomic output helpers."""
import hashlib
import json
import os
from importlib.resources import files
from pathlib import Path

import cv2
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def project_home():
    """Source checkout (or explicit home); installed wheels use the working directory."""
    if os.environ.get('RCG_HOME'):
        return Path(os.environ['RCG_HOME']).expanduser().resolve()
    for root in Path(__file__).resolve().parents:
        if (root / 'pyproject.toml').is_file() and (root / 'src/realistic_carplate_generation').is_dir():
            return root
    return Path.cwd().resolve()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_jsonl(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def write_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8'))


def write_jsonl(path, values):
    atomic_bytes(path, ''.join(dumps(value) + '\n' for value in values).encode('utf-8'))


def load_config(path=None, roots=()):
    cfg = json.loads(files('realistic_carplate_generation').joinpath('resources/default.json').read_text(encoding='utf-8'))
    if path:
        extra = read_json(path)
        for key, value in extra.items():
            if key in ('roots', 'fonts', 'models') and isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
    for item in roots:
        name, value = item.split('=', 1)
        cfg['roots'][name] = str(Path(value).expanduser().resolve())
    return cfg


def source_path(record, roots):
    raw_root = str(roots[record['root']]).replace('${RCG_HOME}', str(project_home()))
    root = Path(raw_root).expanduser()
    if not root.is_absolute():root = project_home() / root
    root = root.resolve()
    path = (root / record['path']).resolve()
    require(path.is_relative_to(root), '底图相对路径越过数据根目录')
    return path


def decode_source(record, roots, verify=True):
    path = source_path(record, roots)
    raw = path.read_bytes()
    if verify:
        require(hashlib.sha256(raw).hexdigest() == record['sha256'], f'来源哈希改变: {path}')
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    require(image is not None and image.shape[:2] == (record['height'], record['width']), '来源解码或尺寸错误')
    return image


def _asset_root(cfg=None):
    configured = (cfg or {}).get('assets_dir') or os.environ.get('RCG_ASSETS_DIR')
    return Path(configured).expanduser().resolve() if configured else (project_home() / 'assets').resolve()


def _asset_relative(relative):
    path = Path(relative)
    require(not path.is_absolute(), f'资源路径越界: {relative}')
    if path.parts and path.parts[0] == 'assets':
        path = Path(*path.parts[1:])
    require('..' not in path.parts, f'资源路径越界: {relative}')
    return path.as_posix()


def asset_path(relative, cfg=None):
    """Resolve within explicit assets_dir, RCG_ASSETS_DIR, or checkout assets."""
    root = _asset_root(cfg)
    path = (root / _asset_relative(relative)).resolve()
    require(path.is_relative_to(root), f'资源路径越界: {relative}')
    return path


def font_path(path, cfg=None):
    """Allow explicit local fonts; legacy relative paths keep asset boundaries."""
    expanded = Path(path).expanduser()
    return expanded.resolve() if expanded.is_absolute() else asset_path(path, cfg)


def asset_check(required=None, cfg=None):
    """Verify selected assets; None explicitly verifies the entire manifest.

    Unlisted user assets are fingerprinted, not assumed to have a known digest.
    Returned identities contain logical paths only, never deployment paths.
    """
    if required is not None:
        required = list(required)
        if not required:
            return {'schema_version': 1, 'files': []}
    manifest_path = _asset_root(cfg) / 'manifest.json'
    require(required is not None or manifest_path.is_file(),
            f'资源清单缺失: {manifest_path}; 请配置 assets_dir 或 RCG_ASSETS_DIR')
    manifest = read_json(manifest_path) if manifest_path.is_file() else {'schema_version': 1, 'files': []}
    registered = {_asset_relative(row['path']): row for row in manifest['files']}
    selected = registered if required is None else {_asset_relative(path): None for path in required}
    rows = []
    for relative in sorted(selected):
        p = asset_path(relative, cfg)
        require(p.is_file(), f'资源缺失: {relative} ({p}); 请配置 assets_dir 或 RCG_ASSETS_DIR')
        size, digest = p.stat().st_size, sha256(p)
        expected = registered.get(relative)
        if expected is not None:
            require(size == expected['bytes'] and digest == expected['sha256'], f'资源哈希改变: {relative}')
        rows.append({'path': 'assets/' + relative, 'bytes': size, 'sha256': digest})
    return {'schema_version': manifest.get('schema_version', 1), 'files': rows}
