"""Create a local Chinese translation sidecar with Google Translate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import ssl
import sys
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_translation import (  # noqa: E402
    index_translation_rows,
    verified_task_translation,
)
from ml.annotation_translation_generation import (  # noqa: E402
    build_translation_sidecar,
)


GOOGLE_TRANSLATE_ENDPOINT = "https://translate.googleapis.com/translate_a/single"
SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
MARKER_TEMPLATE = "[[[JOB_COPILOT_TRANSLATION_{index}]]]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-language", default="zh-CN")
    parser.add_argument("--request-delay", type=float, default=0.10)
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=0,
        help="Translate at most this many pending tasks; zero means all.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a source-verified partial sidecar instead of overwriting it.",
    )
    return parser.parse_args()


def google_translate(
    text: str,
    *,
    target_language: str,
    request_delay: float,
) -> str:
    """Translate one public, authorized text string with bounded retries."""
    parameters = urlencode(
        {
            "client": "gtx",
            "sl": "en",
            "tl": target_language,
            "dt": "t",
            "q": text,
        }
    )
    request = Request(
        f"{GOOGLE_TRANSLATE_ENDPOINT}?{parameters}",
        headers={"User-Agent": "job-application-copilot/1.0"},
    )
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            with urlopen(  # noqa: S310
                request,
                timeout=20,
                context=SSL_CONTEXT,
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            segments = payload[0]
            translated = "".join(str(segment[0]) for segment in segments if segment[0])
            if translated:
                time.sleep(request_delay)
                return translated
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            last_error = error
            time.sleep(1 + attempt)
    raise SystemExit(f"Google translation failed after retries: {last_error}")


def google_translate_many(
    texts: list[str],
    *,
    target_language: str,
    request_delay: float,
) -> list[str]:
    """Translate one task's texts in a single request with stable separators."""
    if not texts:
        return []
    markers = [MARKER_TEMPLATE.format(index=index) for index in range(len(texts) - 1)]
    joined = texts[0]
    for marker, text in zip(markers, texts[1:], strict=True):
        joined += f"\n{marker}\n{text}"
    translated = google_translate(
        joined,
        target_language=target_language,
        request_delay=request_delay,
    )
    parts = translated.splitlines()
    output = [""]
    marker_index = 0
    for part in parts:
        if marker_index < len(markers) and part.strip() == markers[marker_index]:
            output.append("")
            marker_index += 1
        else:
            output[-1] = "\n".join(item for item in [output[-1], part] if item)
    if marker_index != len(markers) or len(output) != len(texts) or any(not item for item in output):
        raise SystemExit("Google translation did not preserve task text separators.")
    return output


def main() -> None:
    args = parse_args()
    if args.output.exists() and not args.resume:
        raise SystemExit("Translation sidecar exists; refusing overwrite.")
    queue = load_jsonl(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_jsonl(args.output) if args.output.exists() else []
    translations = index_translation_rows(existing)
    queue_by_id = {str(task["task_id"]): task for task in queue}
    if set(translations) - set(queue_by_id):
        raise SystemExit("Partial sidecar contains tasks outside the input queue.")
    for task_id, translation in translations.items():
        if verified_task_translation(queue_by_id[task_id], {task_id: translation}) is None:
            raise SystemExit(f"Partial sidecar source mismatch for {task_id}.")
    pending = [task for task in queue if str(task["task_id"]) not in translations]
    if args.max_tasks > 0:
        pending = pending[: args.max_tasks]
    for task in pending:
        sources = [
            str(task["requirement"]),
            *(str(candidate["evidence"]) for candidate in task["candidates"]),
        ]
        translated = google_translate_many(
            sources,
            target_language=args.target_language,
            request_delay=args.request_delay,
        )
        translation_by_source = dict(zip(sources, translated, strict=True))
        row = build_translation_sidecar(
            [task],
            translate=lambda text: translation_by_source[text],
        )[0]
        existing.append(row)
        _write_rows(args.output, existing)
    print(f"Translated this run: {len(pending)}")
    print(f"Translated total: {len(existing)}/{len(queue)}")
    print(f"Sidecar: {args.output}")


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
