"""The default Personal workflow must not require optional ML dependencies."""

import os
from pathlib import Path
import subprocess
import sys


def test_personal_shadow_hook_works_without_optional_ml_packages() -> None:
    code = """
import importlib.abc
import sys
class NoOptionalML(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'sklearn', 'torch', 'transformers', 'sentence_transformers'}:
            raise ModuleNotFoundError(f'Optional ML dependency unavailable: {fullname}')
sys.meta_path.insert(0, NoOptionalML())
import dashboard
dashboard.enqueue_web_candidate_shadow('Python required', 'Python evidence', workspace_mode='personal')
print('Personal hook passed without ML')
"""
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "JOB_COPILOT_WEB_SHADOW_ENABLED": "0"}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, env=env,
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "Personal hook passed without ML" in result.stdout
