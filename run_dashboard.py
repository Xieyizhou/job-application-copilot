"""Launch the dashboard with a stable PyArrow memory allocator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_PATH = PROJECT_ROOT / "src" / "dashboard.py"
SRC_DIR = PROJECT_ROOT / "src"
COMPANION_DIR = PROJECT_ROOT / "data" / "local_workspace" / "browser_companion"


def configure_arrow_memory_pool() -> str:
    """Select and verify PyArrow's system memory pool."""
    try:
        import pyarrow

        pyarrow.set_memory_pool(pyarrow.system_memory_pool())
        backend = pyarrow.default_memory_pool().backend_name
    except Exception as error:  # PyArrow may fail while loading native libraries.
        raise RuntimeError(f"Could not initialize PyArrow's system memory pool: {error}") from error

    if backend != "system":
        raise RuntimeError(
            f"Could not initialize PyArrow's system memory pool: active backend is {backend!r}."
        )
    return backend


def build_streamlit_argv(extra_args: Sequence[str] | None = None) -> list[str]:
    """Build Streamlit CLI arguments while preserving launcher arguments."""
    return ["streamlit", "run", str(DASHBOARD_PATH), *(extra_args or [])]


def main(extra_args: Sequence[str] | None = None) -> int:
    """Configure Arrow before importing and starting Streamlit."""
    try:
        configure_arrow_memory_pool()
    except RuntimeError as error:
        print(f"Dashboard launcher error: {error}", file=sys.stderr)
        return 1

    from streamlit.web import cli as streamlit_cli

    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    from browser_companion import DEFAULT_PORT, load_or_create_token, start_browser_companion
    from workspace import personal_workspace

    companion = None
    connection_path = COMPANION_DIR / "connection.json"
    connection_path.unlink(missing_ok=True)
    try:
        workspace = personal_workspace()
        token = load_or_create_token(COMPANION_DIR / "token")
        companion = start_browser_companion(workspace.jobs_dir, port=DEFAULT_PORT, token=token)
        COMPANION_DIR.mkdir(parents=True, exist_ok=True)
        connection_path.write_text(
            __import__("json").dumps(
                {
                    "endpoint": f"http://127.0.0.1:{companion.port}",
                    "token": companion.token,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        connection_path.chmod(0o600)
    except OSError as error:
        print(f"Browser companion unavailable: {error}", file=sys.stderr)

    sys.argv = build_streamlit_argv(sys.argv[1:] if extra_args is None else extra_args)
    try:
        streamlit_cli.main()
    finally:
        if companion is not None:
            companion.stop()
            connection_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
