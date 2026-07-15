"""Evaluate deterministic Role Fit scoring against the public benchmark."""

from __future__ import annotations

import argparse
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_job import score_job_texts  # noqa: E402


DEFAULT_FIXTURE = PROJECT_ROOT / "tests" / "fixtures" / "scoring_cases.yaml"
FIELDS = ("score", "eligibility", "confidence", "recommendation")


def load_cases(path: Path = DEFAULT_FIXTURE) -> list[dict[str, Any]]:
    """Load benchmark cases from YAML without applying scoring assumptions."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Benchmark fixture must contain a top-level YAML list.")
    if not all(isinstance(case, dict) for case in payload):
        raise ValueError("Every benchmark case must be a YAML mapping.")
    return payload


def evaluate_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Run the production scorer and calculate benchmark agreement metrics."""
    results: list[dict[str, Any]] = []
    family_totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, **{field: 0 for field in FIELDS}}
    )
    hard_constraint_false_negatives = 0
    unsafe_high_score_false_positives = 0

    for case in cases:
        actual = score_job_texts(case["job_text"], case["candidate_text"])
        expected = case["expected"]
        checks = {
            "score": expected["score_min"] <= actual["score"] <= expected["score_max"],
            "eligibility": actual["eligibility"]["status"] == expected["eligibility"],
            "confidence": actual["confidence"]["level"] == expected["confidence"],
            "recommendation": actual["recommendation"] == expected["recommendation"],
        }
        failures: list[str] = []
        if not checks["score"]:
            failures.append(
                f"score {actual['score']} outside {expected['score_min']}-{expected['score_max']}"
            )
        for field in ("eligibility", "confidence", "recommendation"):
            actual_value = (
                actual[field]["status"]
                if field == "eligibility"
                else actual[field]["level"]
                if field == "confidence"
                else actual[field]
            )
            if not checks[field]:
                failures.append(f"{field} {actual_value!r} != {expected[field]!r}")

        family = case["role_family"]
        family_totals[family]["total"] += 1
        for field, agreed in checks.items():
            family_totals[family][field] += int(agreed)

        if expected["eligibility"] == "failed" and actual["eligibility"]["status"] != "failed":
            hard_constraint_false_negatives += 1
        if (
            actual["score"] >= 80
            and actual["recommendation"] == "Apply"
            and (
                expected["eligibility"] != "passed"
                or expected["recommendation"] != "Apply"
            )
        ):
            unsafe_high_score_false_positives += 1

        results.append(
            {
                "id": case["id"],
                "role_family": family,
                "actual": actual,
                "expected": expected,
                "checks": checks,
                "failures": failures,
            }
        )

    total = len(results)
    agreements = {
        field: sum(int(result["checks"][field]) for result in results)
        for field in FIELDS
    }
    return {
        "total": total,
        "agreements": agreements,
        "rates": {
            field: (agreements[field] / total * 100 if total else 0.0)
            for field in FIELDS
        },
        "hard_constraint_false_negatives": hard_constraint_false_negatives,
        "unsafe_high_score_false_positives": unsafe_high_score_false_positives,
        "families": dict(sorted(family_totals.items())),
        "results": results,
        "failed": [result for result in results if result["failures"]],
    }


def _rate(agreed: int, total: int) -> str:
    percentage = agreed / total * 100 if total else 0.0
    return f"{agreed}/{total} ({percentage:.1f}%)"


def terminal_report(summary: dict[str, Any]) -> str:
    """Render a concise plain-text report for local runs and CI logs."""
    lines = [
        "Scoring Benchmark and Calibration V2",
        "=" * 36,
        f"Total cases: {summary['total']}",
        f"Score-range agreement: {_rate(summary['agreements']['score'], summary['total'])}",
        f"Eligibility agreement: {_rate(summary['agreements']['eligibility'], summary['total'])}",
        f"Confidence agreement: {_rate(summary['agreements']['confidence'], summary['total'])}",
        f"Recommendation agreement: {_rate(summary['agreements']['recommendation'], summary['total'])}",
        f"Hard-constraint false negatives: {summary['hard_constraint_false_negatives']}",
        f"Unsafe/high-score false positives: {summary['unsafe_high_score_false_positives']}",
        "",
        "Results by role family:",
    ]
    for family, metrics in summary["families"].items():
        total = metrics["total"]
        lines.append(
            f"- {family}: score {_rate(metrics['score'], total)}; "
            f"eligibility {_rate(metrics['eligibility'], total)}; "
            f"confidence {_rate(metrics['confidence'], total)}; "
            f"recommendation {_rate(metrics['recommendation'], total)}"
        )

    lines.extend(["", "Failed cases:"])
    if summary["failed"]:
        for result in summary["failed"]:
            lines.append(f"- {result['id']}: {'; '.join(result['failures'])}")
    else:
        lines.append("- None")
    return "\n".join(lines)


def markdown_report(summary: dict[str, Any]) -> str:
    """Render a stable Markdown report with no timestamp or personal paths."""
    lines = [
        "# Scoring Benchmark and Calibration V2",
        "",
        f"- Total cases: **{summary['total']}**",
        f"- Score-range agreement: **{_rate(summary['agreements']['score'], summary['total'])}**",
        f"- Eligibility agreement: **{_rate(summary['agreements']['eligibility'], summary['total'])}**",
        f"- Confidence agreement: **{_rate(summary['agreements']['confidence'], summary['total'])}**",
        f"- Recommendation agreement: **{_rate(summary['agreements']['recommendation'], summary['total'])}**",
        f"- Hard-constraint false negatives: **{summary['hard_constraint_false_negatives']}**",
        f"- Unsafe/high-score false positives: **{summary['unsafe_high_score_false_positives']}**",
        "",
        "## Results by role family",
        "",
        "| Role family | Cases | Score range | Eligibility | Confidence | Recommendation |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for family, metrics in summary["families"].items():
        total = metrics["total"]
        lines.append(
            f"| {family} | {total} | {_rate(metrics['score'], total)} | "
            f"{_rate(metrics['eligibility'], total)} | {_rate(metrics['confidence'], total)} | "
            f"{_rate(metrics['recommendation'], total)} |"
        )
    lines.extend(["", "## Failed cases", ""])
    if summary["failed"]:
        lines.extend(
            f"- `{result['id']}`: {'; '.join(result['failures'])}"
            for result in summary["failed"]
        )
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--markdown",
        nargs="?",
        const="__TEMP__",
        metavar="PATH",
        help="Write Markdown to PATH, or to a temporary file when PATH is omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_cases(load_cases(args.fixture))
    print(terminal_report(summary))

    if args.markdown:
        if args.markdown == "__TEMP__":
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".md", prefix="scoring-benchmark-", delete=False
            ) as report_file:
                report_file.write(markdown_report(summary))
                output_path = Path(report_file.name)
        else:
            output_path = Path(args.markdown).expanduser()
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(markdown_report(summary), encoding="utf-8")
        print(f"Markdown report: {output_path}")

    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
