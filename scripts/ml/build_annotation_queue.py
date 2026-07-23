"""Build a bias-reduced local requirement/evidence annotation queue."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PAIRS_PATH = PROJECT_ROOT / "data" / "ml" / "processed" / "canonical_pairs.parquet"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "data" / "ml" / "annotations" / "pilot_queue_v3.jsonl"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import write_queue  # noqa: E402
from ml.annotation_generation import build_tasks_from_records  # noqa: E402
from ml.annotation_scenarios_v3 import (  # noqa: E402
    expanded_fictional_records,
    scenario_design_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-path", type=Path, default=DEFAULT_PAIRS_PATH)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument(
        "--include-ats",
        action="store_true",
        help="Also sample local ATS research text; de-identify it before sharing.",
    )
    parser.add_argument("--unique-count", type=int, default=160)
    parser.add_argument("--blind-repeat-fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_ats_records(path: Path) -> list[dict[str, Any]]:
    """Load optional local research rows without making them a default dependency."""
    try:
        import pandas as pd
    except ModuleNotFoundError as error:
        raise RuntimeError("pandas and pyarrow are required for --include-ats.") from error
    frame = pd.read_parquet(path)
    required_columns = {"resume_text", "job_text", "resume_hash", "job_hash", "label"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical pairs are missing required columns: {sorted(missing)}")
    records = frame.to_dict("records")
    for record in records:
        record["source_dataset"] = "resume_ats_score_v1_en"
    return records


def main() -> None:
    args = parse_args()
    if args.unique_count < 8:
        raise ValueError("--unique-count must be at least 8 for balanced role coverage.")
    if not 0 <= args.blind_repeat_fraction <= 0.25:
        raise ValueError("--blind-repeat-fraction must be between 0 and 0.25.")

    challenge_records = expanded_fictional_records(random_state=args.random_state)
    records = list(challenge_records)
    if args.include_ats:
        records.extend(load_ats_records(args.pairs_path))
    tasks = build_tasks_from_records(
        records,
        unique_count=args.unique_count,
        blind_repeat_fraction=args.blind_repeat_fraction,
        random_state=args.random_state,
    )
    write_queue(tasks, args.queue_path)

    unique_tasks = [task for task in tasks if not task.get("blind_duplicate_of")]
    families = Counter(str(task["role_family"]) for task in unique_tasks)
    strata = Counter(str(record["sampling_stratum"]) for record in challenge_records)
    sources = Counter(str(task["source_dataset"]) for task in unique_tasks)
    design = scenario_design_audit(challenge_records)
    print(f"Saved {len(tasks)} local annotation tasks to {args.queue_path}")
    print(f"Unique tasks: {len(unique_tasks)}; blind repeats: {len(tasks) - len(unique_tasks)}")
    print("Role families: " + ", ".join(f"{name}={count}" for name, count in sorted(families.items())))
    print("Challenge design: " + ", ".join(f"{name}={count}" for name, count in sorted(strata.items())))
    print("Sources: " + ", ".join(f"{name}={count}" for name, count in sorted(sources.items())))
    print(
        "Surface design: "
        f"requirement styles={design['requirement_style_count']}, "
        f"candidate styles={design['candidate_surface_style_count']}, "
        "largest within-semantic-type style share="
        f"{design['largest_style_share_within_semantic_role']:.1%}"
    )
    print("Candidate order is randomized; retrieval rank and similarity are not stored.")


if __name__ == "__main__":
    main()
