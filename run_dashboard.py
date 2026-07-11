"""Launch the dashboard with a stable PyArrow memory allocator."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DASHBOARD_PATH = PROJECT_ROOT / "src" / "dashboard.py"


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

    sys.argv = build_streamlit_argv(sys.argv[1:] if extra_args is None else extra_args)
    streamlit_cli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
