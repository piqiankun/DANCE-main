"""Regression checks for workflow dispatch and checkout-relative resources."""

import ast
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from dance import cli, paths

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_help_does_not_import_model_dependencies(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; from dance.cli import main; main([]); "
             "assert 'torch' not in sys.modules; assert 'openai' not in sys.modules"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        )
        self.assertIn("optimize-coalitions", result.stdout)

    def test_launcher_works_outside_the_checkout(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "run.py"), "--help"],
                cwd=directory, capture_output=True, text=True, check=True,
            )
        self.assertIn("discover-neurons", result.stdout)

    def test_every_command_forwards_arguments_and_restores_argv(self):
        original = sys.argv
        for command, (module, _) in cli.COMMANDS.items():
            with self.subTest(command=command):
                observed = []
                def execute(name, **kwargs):
                    observed.append((name, kwargs, sys.argv.copy()))
                with patch("dance.cli.runpy.run_module", side_effect=execute):
                    self.assertEqual(cli.main([command, "--example", "value with spaces"]), 0)
                self.assertEqual(observed, [(module,
                    {"run_name": "__main__", "alter_sys": True},
                    [module, "--example", "value with spaces"])])
                self.assertIs(sys.argv, original)
                self.assertTrue((ROOT / (module.replace(".", "/") + ".py")).is_file())

    def test_failed_command_restores_argv(self):
        original = sys.argv
        with patch("dance.cli.runpy.run_module", side_effect=SystemExit(7)):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["optimize-coalitions"])
        self.assertEqual(raised.exception.code, 7)
        self.assertIs(sys.argv, original)

    def test_unknown_command_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                cli.main(["missing-command"])
        self.assertEqual(raised.exception.code, 2)

    def test_resources_stay_in_the_checkout_from_another_cwd(self):
        code = (
            "import importlib.util, pathlib, sys; "
            "sys.path.insert(0, sys.argv[1]); from dance import paths; "
            "assert pathlib.Path(paths.DATA_DIR) == pathlib.Path(sys.argv[1]) / 'data'; "
            "assert pathlib.Path(paths.PROMPT_DIR) == pathlib.Path(sys.argv[1]) / 'prompts'; "
            "assert pathlib.Path(paths.RESULTS_DIR) == pathlib.Path(sys.argv[1]) / 'results'; "
            "assert (pathlib.Path(paths.PROMPT_DIR) / 'evaluate_s.txt').is_file()"
        )
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run([sys.executable, "-c", code, str(ROOT)], cwd=directory, check=True)

    def test_relocated_imports_and_all_python_syntax(self):
        for file in (ROOT / "dance").rglob("*.py"):
            source = file.read_text(encoding="utf-8")
            compile(source, str(file.relative_to(ROOT)), "exec")
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module == "kn_code" or node.module.startswith("kn_code."))
                    self.assertNotEqual(node.module, "llm_utils")
                    if node.level == 0 and node.module.startswith("dance"):
                        candidate = ROOT / node.module.replace(".", "/")
                        self.assertTrue(candidate.with_suffix(".py").is_file() or
                                        (candidate / "__init__.py").is_file(), node.module)


if __name__ == "__main__":
    unittest.main()
