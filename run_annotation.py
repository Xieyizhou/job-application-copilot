"""Launch the standalone local evidence annotation interface."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
ANNOTATION_DASHBOARD_PATH = PROJECT_ROOT / "src" / "ml" / "annotation_dashboard.py"


def configure_arrow_memory_pool() -> None:
    """Select Arrow's system allocator before Streamlit imports pandas."""
    try:
        import pyarrow
    except ModuleNotFoundError as error:
        raise RuntimeError("pyarrow is required. Install requirements.txt first.") from error
    pyarrow.set_memory_pool(pyarrow.system_memory_pool())


def build_streamlit_argv(extra_args: Sequence[str] | None = None) -> list[str]:
    """Build Streamlit arguments for the local annotation page."""
    return ["streamlit", "run", str(ANNOTATION_DASHBOARD_PATH), *(extra_args or [])]


def main(extra_args: Sequence[str] | None = None) -> int:
    """Configure Arrow and launch the annotation workspace."""
    try:
        configure_arrow_memory_pool()
    except RuntimeError as error:
        print(f"Annotation launcher error: {error}", file=sys.stderr)
        return 1
    from streamlit.web import cli as streamlit_cli

    sys.argv = build_streamlit_argv(sys.argv[1:] if extra_args is None else extra_args)
    streamlit_cli.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
