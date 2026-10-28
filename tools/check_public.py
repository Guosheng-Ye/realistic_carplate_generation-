#!/usr/bin/env python3
"""Scan public source files, model bytes and preview metadata before sharing."""
import argparse
import io
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PATTERNS = {
    'personal_path': rb'/(?:home|Users)/[A-Za-z0-9_.-]+/|/root/(?:\.[A-Za-z]|[A-Za-z])',
    'windows_profile': rb'[A-Za-z]:[\\/]+(?:Users|Documents and Settings)[\\/]+[^\\/\s\x00\x22]+',
    'email': rb'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}',
    'private_network_address': rb'(?<![0-9.])(?:192\.168|10\.[0-9]{1,3}|172\.(?:1[6-9]|2[0-9]|3[01]))\.[0-9]{1,3}\.[0-9]{1,3}(?![0-9.])',
    'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'access_token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16})\b',
    'credential_assignment': rb'''(?i)\b(?:password|passwd|api_key|access_token|secret_key)["']?\s*[:=]\s*["'][^"'\s]{8,}["']''',
    'account_assignment': rb'''(?i)["'](?:username|user_name|login|account)["']\s*:\s*["'][A-Za-z0-9_.@-]+["']''',
    'personal_file_header': rb'(?im)^\s*(?:#|//|\*)?\s*@(?:Author|LastEditors|FilePath):',
}


def scan_bytes(name, data):
    findings = []
    for kind, pattern in PATTERNS.items():
        count = sum(1 for _ in re.finditer(pattern, data))
        if count:
            # Values can be credentials: report only the filename, category and count.
            findings.append({'file': name, 'kind': kind, 'count': count})
    if Path(name).suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
        from PIL import Image
        try:
            with Image.open(io.BytesIO(data)) as picture:
                if picture.getexif():
                    findings.append({'file': name, 'kind': 'image_exif', 'count': 1})
                if any(picture.info.get(k) for k in ('xmp', 'XML:com.adobe.xmp', 'comment', 'photoshop')):
                    findings.append({'file': name, 'kind': 'image_extra_metadata', 'count': 1})
                picture.verify()
        except (OSError, ValueError) as exc:
            raise ValueError('Cannot inspect public image: ' + name) from exc
    return findings


def public_paths(root=ROOT):
    root = Path(root)
    if (root / '.git').exists():
        raw = subprocess.check_output(['git', '-C', str(root), 'ls-files', '--cached',
                                       '--others', '--exclude-standard', '-z'])
        names = {p.decode('utf-8') for p in raw.split(b'\0') if p}
    elif (root / 'PUBLIC_MANIFEST.json').is_file():
        names = {row['path'] for row in json.loads((root / 'PUBLIC_MANIFEST.json').read_text())['files']}
    else:
        raise ValueError('Run in a Git checkout or an extracted public archive with PUBLIC_MANIFEST.json')
    for name in names:
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or (root / path).is_symlink():
            raise ValueError('Unsafe public file path')
    return sorted(names)


def audit(root=ROOT, archive=None):
    files, findings = [], []
    if archive:
        with zipfile.ZipFile(archive) as z:
            for name in sorted(z.namelist()):
                if name.endswith('/'):
                    continue
                files.append(name)
                findings.extend(scan_bytes(name, z.read(name)))
    else:
        for name in public_paths(root):
            files.append(name)
            findings.extend(scan_bytes(name, (Path(root) / name).read_bytes()))
    return {'status': 'passed' if not findings else 'failed', 'files_scanned': len(files),
            'models_scanned': sum(Path(n).suffix == '.onnx' for n in files),
            'preview_images_scanned': sum('/readme/' in n for n in files), 'findings': findings,
            'scope': 'known personal path, account, credential patterns and image metadata; real plate pixels retained'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    try:
        result = audit(archive=args.archive)
        text = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text, encoding='utf-8')
        print(text, end='')
        return 0 if result['status'] == 'passed' else 2
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as exc:
        print(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False))
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
