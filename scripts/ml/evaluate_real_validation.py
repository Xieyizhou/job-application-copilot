"""Evaluate anonymous real-derived semantic and relevance validation manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.validation import (  # noqa: E402
    DEFAULT_REAL_PAIRS,
    DEFAULT_RELEVANCE_MANIFEST,
    DEFAULT_SEMANTIC_MANIFEST,
    evaluate_relevance_manifest,
    evaluate_semantic_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-manifest", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    parser.add_argument("--relevance-manifest", type=Path, default=DEFAULT_RELEVANCE_MANIFEST)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_REAL_PAIRS)
    parser.add_argument("--semantic-only", action="store_true")
    parser.add_argument("--require-semantic-f1", type=float, default=0.85)
    parser.add_argument("--json", action="store_true", help="Print one JSON object.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic = evaluate_semantic_manifest(args.semantic_manifest)
    relevance = None if args.semantic_only else evaluate_relevance_manifest(
        args.relevance_manifest,
        pairs_path=args.pairs_path,
    )
    report = {"semantic_evidence": semantic, "real_relevance": relevance}
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print("Anonymous Real-Data Validation")
        print("==============================")
        print(
            f"Semantic evidence: {semantic['cases']} cases, "
            f"accuracy {semantic['accuracy']:.1%}, F1 {semantic['f1']:.1%}"
        )
        print(f"Semantic failures: {', '.join(semantic['failed_case_ids']) or 'none'}")
        if relevance is not None:
            print(f"Real relevance diagnostic: {relevance['status']}")
            if relevance["status"] == "complete":
                print(
                    f"Probability range: {relevance['probability_min']:.6f}–"
                    f"{relevance['probability_max']:.6f}; collapsed: {relevance['collapsed']}"
                )
                print(
                    "Good Fit vs No Fit weak-label accuracy: "
                    f"{relevance['binary_accuracy_good_vs_no_fit']:.1%}"
                )
            else:
                print("The ignored local pair table or model is unavailable; semantic validation still ran.")
    if float(semantic["f1"]) < args.require_semantic_f1:
        raise SystemExit(
            f"Semantic evidence F1 {semantic['f1']:.3f} is below required {args.require_semantic_f1:.3f}."
        )


if __name__ == "__main__":
    main()
