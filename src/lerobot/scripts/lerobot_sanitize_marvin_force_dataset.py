#!/usr/bin/env python

"""Clone a Marvin LeRobot dataset and repair invalid gripper sentinel frames.

The source is never modified. Video files are hard-linked when possible; all
metadata and parquet files are copied before numeric repairs and stats rebuild.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from lerobot.datasets import LeRobotDataset, recompute_stats

logger = logging.getLogger(__name__)


def _copy_dataset_file(source: str, target: str) -> str:
    source_path = Path(source)
    if source_path.suffix.lower() in {".mp4", ".mkv", ".avi"}:
        try:
            os.link(source, target)
            return target
        except OSError:
            pass
    return shutil.copy2(source, target)


def _gripper_index(info: dict, key: str) -> int:
    names = info["features"][key].get("names")
    if not names:
        raise ValueError(f"Feature {key} has no dimension names")
    matches = [index for index, name in enumerate(names) if str(name).endswith("gripper.pos")]
    if len(matches) != 1:
        raise ValueError(f"Expected one gripper dimension in {key}, got {names}")
    return matches[0]


def _is_valid(value: float, lower: float, upper: float) -> bool:
    return math.isfinite(value) and lower <= value <= upper


def _repair_column(
    frame: pd.DataFrame,
    key: str,
    gripper_index: int,
    lower: float,
    upper: float,
) -> list[tuple[int, int, float, float]]:
    repairs: list[tuple[int, int, float, float]] = []
    for episode_index, episode_rows in frame.groupby("episode_index", sort=False).groups.items():
        row_indices = list(episode_rows)
        values = [float(frame.at[row, key][gripper_index]) for row in row_indices]
        valid_positions = [
            position for position, value in enumerate(values) if _is_valid(value, lower, upper)
        ]
        if not valid_positions and row_indices:
            raise ValueError(f"Episode {episode_index} has no valid {key} gripper sample")
        for position, (row_index, value) in enumerate(zip(row_indices, values, strict=True)):
            if _is_valid(value, lower, upper):
                continue
            replacement_position = min(
                valid_positions,
                key=lambda candidate: (abs(candidate - position), candidate < position),
            )
            replacement = values[replacement_position]
            vector = np.asarray(frame.at[row_index, key], dtype=np.float32).copy()
            vector[gripper_index] = replacement
            frame.at[row_index, key] = vector
            repairs.append((int(episode_index), int(row_index), value, replacement))
    return repairs


def audit_or_repair(
    source_root: Path,
    target_root: Path | None,
    repo_id: str,
    lower: float,
    upper: float,
) -> dict[str, list[tuple[int, int, float, float]]]:
    source_root = source_root.resolve()
    if target_root is None:
        working_root = source_root
    else:
        target_root = target_root.resolve()
        if target_root == source_root or source_root in target_root.parents:
            raise ValueError("Target must be a separate sibling directory, not the source or its child")
        if target_root.exists():
            raise FileExistsError(f"Refusing to overwrite existing target: {target_root}")
        shutil.copytree(source_root, target_root, copy_function=_copy_dataset_file)
        working_root = target_root

    import json

    info = json.loads((working_root / "meta" / "info.json").read_text(encoding="utf-8"))
    gripper_indices = {key: _gripper_index(info, key) for key in ("observation.state", "action")}
    all_repairs: dict[str, list[tuple[int, int, float, float]]] = {key: [] for key in gripper_indices}

    for parquet_path in sorted((working_root / "data").glob("*/*.parquet")):
        frame = pd.read_parquet(parquet_path)
        file_changed = False
        for key, index in gripper_indices.items():
            repairs = _repair_column(frame, key, index, lower, upper)
            all_repairs[key].extend(repairs)
            file_changed |= bool(repairs)
        if target_root is not None and file_changed:
            temporary = parquet_path.with_name(f".{parquet_path.name}.{uuid4().hex}.tmp")
            frame.to_parquet(temporary, index=False)
            os.replace(temporary, parquet_path)

    if target_root is None:
        return all_repairs

    dataset = LeRobotDataset(repo_id, root=working_root)
    recompute_stats(dataset, skip_image_video=True)

    # Reload from disk and fail closed if any sentinel survived.
    verification = audit_or_repair(working_root, None, repo_id, lower, upper)
    if any(verification.values()):
        raise RuntimeError(f"Sanitization verification failed: {verification}")
    return all_repairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--repo-id", default="hukewei/peg_optical_module_0726force")
    parser.add_argument("--valid-gripper-min", type=float, default=-2.0)
    parser.add_argument("--valid-gripper-max", type=float, default=2.0)
    args = parser.parse_args()
    # LeRobot imports may install a root handler before this CLI starts.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", force=True)
    repairs = audit_or_repair(
        source_root=args.source_root,
        target_root=args.target_root,
        repo_id=args.repo_id,
        lower=args.valid_gripper_min,
        upper=args.valid_gripper_max,
    )
    for key, items in repairs.items():
        logger.info("%s: %d invalid samples", key, len(items))
        for episode, row, old, new in items:
            logger.info("  episode=%d row=%d %.6f -> %.6f", episode, row, old, new)
    if args.target_root is None:
        logger.info("Audit only; pass --target-root to create a repaired clone")
    else:
        logger.info("Repaired dataset written to %s; source was not modified", args.target_root)


if __name__ == "__main__":
    main()
