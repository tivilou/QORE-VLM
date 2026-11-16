"""Delivery regressions for fresh checkouts; stdlib only, no model/data access."""
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from scripts.check_top8_delivery import ROOT, check_delivery, dependency_files, main


class DeliveryTests(unittest.TestCase):
    def write(self, root, name, text):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def test_real_runner_closure_includes_guard_and_all_backbone_paths(self):
        files = check_delivery()
        for name in ('model_input_guards.py', 'top8_backbone_v001.py',
                     'top8_methods_v001.py', 'top8_portable_v001.py'):
            self.assertIn('applications/rag/' + name, files)

    def test_original_publication_omission_is_rejected_without_model_import(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in check_delivery():
                if name != 'applications/rag/model_input_guards.py':
                    self.write(root, name, (ROOT / name).read_text(encoding='utf-8-sig'))
            with self.assertRaisesRegex(RuntimeError, 'missing applications/rag/model_input_guards.py'):
                check_delivery(root)
            err = StringIO()
            with redirect_stderr(err):
                self.assertEqual(main(['--root', str(root)]), 2)
            self.assertIn('INCOMPLETE_SOURCE_DELIVERY', err.getvalue())

    def test_deferred_relative_and_namespace_imports_are_followed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'scripts/start.py', 'from applications.rag import adapter\n')
            self.write(root, 'applications/rag/adapter.py',
                       'def run():\n    from .guard import check\n    import torch\n')
            with self.assertRaisesRegex(RuntimeError, 'applications/rag/guard.py'):
                dependency_files(root, ['scripts/start.py'])
            self.write(root, 'applications/rag/guard.py', 'def check(): pass\n')
            self.assertIn('applications/rag/guard.py', dependency_files(root, ['scripts/start.py']))

    def test_unused_package_api_does_not_require_unrelated_selector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write(root, 'scripts/start.py', 'from applications.rag.guard import check\n')
            self.write(root, 'applications/rag/__init__.py',
                       'def other_api():\n    from .unrelated import operation\n')
            self.write(root, 'applications/rag/guard.py', 'def check(): pass\n')
            files = dependency_files(root, ['scripts/start.py'])
            self.assertNotIn('applications/rag/unrelated.py', files)

    def test_missing_direct_import_and_missing_entrypoint_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for entries in (['scripts/start.py'],):
                with self.assertRaisesRegex(RuntimeError, 'scripts/start.py'):
                    dependency_files(root, entries)
            self.write(root, 'scripts/start.py', 'import applications.rag.missing\n')
            with self.assertRaisesRegex(RuntimeError, 'applications/rag/missing.py'):
                dependency_files(root, ['scripts/start.py'])


if __name__ == '__main__':
    unittest.main()
