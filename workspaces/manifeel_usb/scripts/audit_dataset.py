#!/usr/bin/env python3
"""Audit ManiFeel USB source data and an optional converted LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from convert_manifeel_to_lerobot import (  # noqa: E402
    ACTION,
    FPS,
    OBS_FORCE,
    OBS_STATE,
    OBS_WRIST,
    ManiFeelSource,
    build_features,
    episode_slices,
    open_source,
)
from plot_episode_force import plot_force_episode  # noqa: E402


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _finite_summary(name: str, array: np.ndarray, errors: list[str]) -> dict[str, Any]:
    finite = np.isfinite(array)
    bad_count = int(array.size - np.count_nonzero(finite))
    if bad_count:
        first_bad = np.argwhere(~finite)[0].tolist()
        errors.append(f"{name} contains {bad_count} NaN/Inf values; first local index={first_bad}")
    valid = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite": bad_count == 0,
        "non_finite_count": bad_count,
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "mean": float(valid.mean()) if valid.size else None,
    }


def _audit_source(
    source: ManiFeelSource,
    spans: list[slice],
    *,
    force_epsilon: float,
    errors: list[str],
    warnings: list[str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    episodes: list[dict[str, Any]] = []
    total_force_values = 0
    total_nonzero = 0
    total_changed = 0
    first_force: np.ndarray | None = None

    for episode_index, span in enumerate(spans):
        expected_length = span.stop - span.start
        state = np.asarray(source.state[span])
        action = np.asarray(source.action[span])
        force = np.asarray(source.force[span])
        wrist = np.asarray(source.wrist[span])
        expected_shapes = {
            "state": (expected_length, 7),
            "action": (expected_length, 6),
            "tactile_force_field_right": (expected_length, 10, 14, 3),
            "wrist": (expected_length, 256, 256, 3),
        }
        arrays = {
            "state": state,
            "action": action,
            "tactile_force_field_right": force,
            "wrist": wrist,
        }
        for key, expected_shape in expected_shapes.items():
            if arrays[key].shape != expected_shape:
                errors.append(
                    f"episode {episode_index} {key} shape {arrays[key].shape} != {expected_shape}; "
                    "modalities are not aligned to the same source frames"
                )

        summaries = {
            key: _finite_summary(f"episode {episode_index} {key}", value, errors)
            for key, value in arrays.items()
        }
        force_finite = np.isfinite(force)
        nonzero = int(np.count_nonzero(np.abs(force[force_finite]) > force_epsilon))
        changed = (
            int(np.count_nonzero(np.any(np.abs(np.diff(force, axis=0)) > force_epsilon, axis=(1, 2, 3))))
            if expected_length > 1 and force_finite.all()
            else 0
        )
        total_force_values += int(force.size)
        total_nonzero += nonzero
        total_changed += changed
        if nonzero == 0:
            warnings.append(f"episode {episode_index} has no force value above epsilon={force_epsilon:g}")
        if changed == 0:
            warnings.append(f"episode {episode_index} has no temporal force change above epsilon={force_epsilon:g}")
        episodes.append(
            {
                "episode_index": episode_index,
                "from_index": span.start,
                "to_index": span.stop,
                "length": expected_length,
                "arrays": summaries,
                "force_nonzero_count": nonzero,
                "force_nonzero_fraction": nonzero / force.size if force.size else 0.0,
                "force_changed_transitions": changed,
            }
        )
        if episode_index == 0:
            first_force = force.astype(np.float32, copy=False)

    if total_nonzero == 0:
        errors.append(f"All audited tactile-force values are zero within epsilon={force_epsilon:g}")
    if total_changed == 0:
        errors.append(f"No audited episode has time-varying tactile force within epsilon={force_epsilon:g}")

    selected_ends = [span.stop for span in spans]
    return (
        {
            "root": str(source.root),
            "fps": FPS,
            "total_source_episodes": source.num_episodes,
            "total_source_frames": source.num_frames,
            "audited_episodes": len(spans),
            "audited_frames": selected_ends[-1],
            "episode_ends": selected_ends,
            "episode_boundaries_valid": all(
                span.start == (0 if index == 0 else spans[index - 1].stop) and span.stop > span.start
                for index, span in enumerate(spans)
            ),
            "force": {
                "source_shape_per_frame": [10, 14, 3],
                "flattened_shape": [420],
                "flatten_order": "C",
                "components": ["normal", "shear_x", "shear_y"],
                "epsilon": force_epsilon,
                "nonzero_count": total_nonzero,
                "nonzero_fraction": total_nonzero / total_force_values if total_force_values else 0.0,
                "changed_transitions": total_changed,
            },
            "action_alignment": {
                "policy": "state[t], action[t], wrist[t], and tactile_force[t] use identical episode slices",
                "action_dimension": 6,
                "valid": not any("not aligned" in error for error in errors),
            },
            "episodes": episodes,
        },
        first_force,
    )


def _feature_matches(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    return actual.get("dtype") == expected["dtype"] and tuple(actual.get("shape", ())) == tuple(expected["shape"])


def _audit_converted(
    converted_root: Path,
    source: ManiFeelSource,
    spans: list[slice],
    *,
    compare_values: bool,
    errors: list[str],
) -> dict[str, Any]:
    converted_root = Path(converted_root).expanduser().resolve()
    info_path = converted_root / "meta" / "info.json"
    manifest_path = converted_root / "conversion_manifest.json"
    if not info_path.is_file():
        errors.append(f"Converted dataset is missing {info_path}")
        return {"root": str(converted_root), "metadata_valid": False}
    with info_path.open("r", encoding="utf-8") as stream:
        info = json.load(stream)
    manifest: dict[str, Any] = {}
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8") as stream:
            manifest = json.load(stream)
    else:
        errors.append(f"Converted dataset is missing {manifest_path}")

    image_storage = manifest.get("image_storage", info.get("features", {}).get(OBS_WRIST, {}).get("dtype", "video"))
    expected_features = build_features(image_storage)
    feature_results: dict[str, bool] = {}
    for key, expected in expected_features.items():
        actual = info.get("features", {}).get(key, {})
        feature_results[key] = _feature_matches(actual, expected)
        if not feature_results[key]:
            errors.append(f"Converted feature mismatch for {key}: {actual}; expected {expected}")

    expected_frames = spans[-1].stop
    expected_episodes = len(spans)
    if int(info.get("fps", -1)) != FPS:
        errors.append(f"Converted FPS is {info.get('fps')}; expected {FPS}")
    if int(info.get("total_episodes", -1)) != expected_episodes:
        errors.append(
            f"Converted episode count is {info.get('total_episodes')}; expected {expected_episodes}"
        )
    if int(info.get("total_frames", -1)) != expected_frames:
        errors.append(f"Converted frame count is {info.get('total_frames')}; expected {expected_frames}")

    values_match: bool | None = None
    max_abs_error: dict[str, float] = {}
    if compare_values:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            repo_id = manifest.get("repo_id")
            if not repo_id:
                raise RuntimeError("conversion_manifest.json does not contain repo_id")
            dataset = LeRobotDataset(repo_id=repo_id, root=converted_root, download_videos=False)
            raw = dataset.hf_dataset.with_format(None).select_columns([OBS_STATE, OBS_FORCE, ACTION])
            state_out = np.asarray(raw[OBS_STATE], dtype=np.float32)
            force_out = np.asarray(raw[OBS_FORCE], dtype=np.float32)
            action_out = np.asarray(raw[ACTION], dtype=np.float32)
            state_source = np.asarray(source.state[:expected_frames], dtype=np.float32)
            force_source = np.asarray(source.force[:expected_frames], dtype=np.float32).reshape(expected_frames, 420)
            action_source = np.asarray(source.action[:expected_frames], dtype=np.float32)
            pairs = {
                OBS_STATE: (state_out, state_source, (expected_frames, 7)),
                OBS_FORCE: (force_out, force_source, (expected_frames, 420)),
                ACTION: (action_out, action_source, (expected_frames, 6)),
            }
            values_match = True
            for key, (actual, expected, shape) in pairs.items():
                if actual.shape != shape:
                    errors.append(f"Converted {key} array shape {actual.shape} != {shape}")
                    values_match = False
                    continue
                if not np.isfinite(actual).all():
                    errors.append(f"Converted {key} contains NaN or Inf")
                    values_match = False
                difference = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
                max_abs_error[key] = difference
                if not np.array_equal(actual, expected):
                    errors.append(f"Converted {key} is not frame-aligned with source (max abs error {difference:g})")
                    values_match = False
        except Exception as exc:
            errors.append(f"Could not numerically audit converted LeRobot data: {type(exc).__name__}: {exc}")
            values_match = False

    return {
        "root": str(converted_root),
        "metadata_valid": all(feature_results.values()),
        "features": feature_results,
        "fps": info.get("fps"),
        "episodes": info.get("total_episodes"),
        "frames": info.get("total_frames"),
        "numeric_values_compared": compare_values,
        "numeric_values_match": values_match,
        "max_abs_error": max_abs_error,
    }


def audit_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    converted_root: Path | None = None,
    max_episodes: int | None = None,
    force_epsilon: float = 1e-12,
    canonical: bool = True,
    compare_values: bool = True,
) -> dict[str, Any]:
    if force_epsilon < 0:
        raise ValueError("force_epsilon must be non-negative")
    source = open_source(source_path, canonical=canonical)
    spans = episode_slices(source.episode_ends, max_episodes=max_episodes)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    warnings: list[str] = []

    source_report, first_force = _audit_source(
        source,
        spans,
        force_epsilon=force_epsilon,
        errors=errors,
        warnings=warnings,
    )
    plot_path = output_dir / "episode_000_tactile_force.png"
    if first_force is not None and np.isfinite(first_force).all():
        try:
            plot_force_episode(first_force, plot_path, episode_index=0, fps=FPS)
        except Exception as exc:
            errors.append(f"Failed to render first-episode force plot: {type(exc).__name__}: {exc}")
    else:
        errors.append("First episode force is unavailable/non-finite, so its curve could not be rendered")

    converted_report = None
    if converted_root is not None:
        converted_report = _audit_converted(
            converted_root,
            source,
            spans,
            compare_values=compare_values,
            errors=errors,
        )

    report = {
        "schema_version": 1,
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "source": source_report,
        "converted": converted_report,
        "artifacts": {"first_episode_force_plot": str(plot_path) if plot_path.is_file() else None},
    }
    atomic_write_json(output_dir / "audit_report.json", report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--converted-root", type=Path, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--force-epsilon", type=float, default=1e-12)
    parser.add_argument("--skip-value-compare", action="store_true")
    parser.add_argument("--allow-noncanonical-source", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = audit_dataset(
        args.source,
        args.output_dir,
        converted_root=args.converted_root,
        max_episodes=args.max_episodes,
        force_epsilon=args.force_epsilon,
        canonical=not args.allow_noncanonical_source,
        compare_values=not args.skip_value_compare,
    )
    report_path = args.output_dir.expanduser().resolve() / "audit_report.json"
    print(f"Audit report: {report_path}")
    print("PASS" if report["passed"] else "FAIL")
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
