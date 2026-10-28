"""README contact cleanup, retained real previews, and project-license checks."""
from pathlib import Path
import importlib.util
import json
import re
import tomllib
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class PublicationTests(unittest.TestCase):
    def public_checker(self):
        spec = importlib.util.spec_from_file_location('rcg_check_public', ROOT / 'tools/check_public.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_public_sources_models_and_preview_metadata_have_no_private_patterns(self):
        result = self.public_checker().audit(ROOT)
        self.assertEqual(result['findings'], [])
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['models_scanned'], 4)

    def test_public_checker_inspects_binary_content_and_redacts_findings(self):
        checker = self.public_checker()
        path = ('/' + '/'.join(('home', 'sample_account', 'model.onnx'))).encode()
        findings = checker.scan_bytes('fixture.onnx', b'\x00' + path + b'\x00')
        self.assertEqual(findings[0]['kind'], 'personal_path')
        self.assertNotIn('sample_account', json.dumps(findings))
        account = json.dumps(dict(username='sample_account')).encode()
        self.assertEqual(checker.scan_bytes('fixture.json', account)[0]['kind'], 'account_assignment')

    def test_readme_has_no_personal_file_header_or_email(self):
        text = (ROOT / 'README.md').read_text(encoding='utf-8')
        self.assertNotRegex(text, r'@(Author|LastEditors|FilePath):')
        self.assertNotRegex(text, r'(?i)[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}')
        self.assertNotRegex(text, r'/(?:home|Users)/[A-Za-z0-9_.-]+/')

    def test_project_license_metadata_and_third_party_scope(self):
        doc = tomllib.loads((ROOT / 'pyproject.toml').read_text())
        self.assertEqual(doc['project']['license'], 'Apache-2.0')
        self.assertEqual(set(doc['project']['license-files']), {'LICENSE', 'NOTICE'})
        self.assertIn('Apache License', (ROOT / 'LICENSE').read_text())
        notice = (ROOT / 'NOTICE').read_text()
        self.assertIn('AGPL-3.0', notice)
        self.assertIn('does not relicense', notice)

    def test_real_preview_references_resolve_and_have_no_exif(self):
        text = (ROOT / 'README.md').read_text()
        references = set(re.findall(r'(assets/readme/[^)\s]+\.(?:jpg|png))', text))
        self.assertEqual(len(references), 11)
        self.assertIn('原始车辆照片', text)
        self.assertNotIn('以下图片全部由本工程的车牌渲染器和程序化几何场景生成', text)
        for relative in references:
            with self.subTest(image=relative), Image.open(ROOT / relative) as image:
                image.load()
                self.assertFalse(image.getexif())


if __name__ == '__main__':
    unittest.main()
