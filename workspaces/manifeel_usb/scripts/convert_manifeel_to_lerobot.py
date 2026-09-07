#!/usr/bin/env python3
"""Convert the official ManiFeel USB Zarr dataset to LeRobot 0.6.1 format."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


FPS = 15
CANONICAL_FRAMES = 5976
CANONICAL_EPISODES = 50
STATE_KEY = "data/state"
ACTION_KEY = "data/action"
FORCE_KEY = "data/tactile_force_field_right"
WRIST_KEY = "data/wrist"
EPISODE_ENDS_KEY = "meta/episode_ends"

OBS_STATE = "observation.state"
OBS_WRIST = "observation.images.wrist"
OBS_FORCE = "observation.tactile_force"
ACTION = "action"

STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_qx",
    "eef_qy",
    "eef_qz",
    "eef_qw",
]
ACTION_NAMES = [
    "delta_x",
    "delta_y",
    "delta_z",
    "axis_angle_x",
    "axis_angle_y",
    "axis_angle_z",
]
FORCE_COMPONENTS = ("normal", "shear_x", "shear_y")


@dataclass(frozen=True)
class ManiFeelSource:
    root: Path
    group: Any
    state: Any
    action: Any
    force: Any
    wrist: Any
    episode_ends: np.ndarray

    @property
    def num_frames(self) -> int:
        return int(self.state.shape[0])

    @property
    def num_episodes(self) -> int:
        return int(self.episode_ends.size)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def resolve_zarr_root(source: Path) -> Path:
    source = Path(source).expanduser().resolve()
    candidates = [source, source / "usb_quan_Aug05"]
    for candidate in candidates:
        if (candidate / ".zgroup").is_file() and (candidate / "data" / "state" / ".zarray").is_file():
            return candidate
    if source.is_dir():
        matches = [
            path.parent
            for path in source.glob("*/.zgroup")
            if (path.parent / "data" / "state" / ".zarray").is_file()
        ]
        if len(matches) == 1:
            return matches[0].resolve()
    raise FileNotFoundError(
        f"Could not locate the ManiFeel Zarr root under {source}; expected data/state and meta/episode_ends"
    )


def _import_zarr() -> Any:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required; install the data-stage environment before conversion") from exc
    return zarr


def _get_array(group: Any, key: str) -> Any:
    try:
        return group[key]
    except (KeyError, IndexError) as exc:
        raise KeyError(f"Required Zarr array is missing: {key}") from exc


def open_source(source: Path, *, canonical: bool = True) -> ManiFeelSource:
    root = resolve_zarr_root(source)
    zarr = _import_zarr()
    group = zarr.open_group(str(root), mode="r")
    result = ManiFeelSource(
        root=root,
        group=group,
        state=_get_array(group, STATE_KEY),
        action=_get_array(group, ACTION_KEY),
        force=_get_array(group, FORCE_KEY),
        wrist=_get_array(group, WRIST_KEY),
        episode_ends=np.asarray(_get_array(group, EPISODE_ENDS_KEY)[:], dtype=np.int64),
    )
    validate_source_schema(result, canonical=canonical)
    return result


def validate_source_schema(source: ManiFeelSource, *, canonical: bool = True) -> None:
    expected_tails = {
        STATE_KEY: ((7,), source.state),
        ACTION_KEY: ((6,), source.action),
        FORCE_KEY: ((10, 14, 3), source.force),
        WRIST_KEY: ((256, 256, 3), source.wrist),
    }
    frame_count = int(source.state.shape[0])
    for key, (tail, array) in expected_tails.items():
        shape = tuple(int(value) for value in array.shape)
        if not shape or shape[0] != frame_count or shape[1:] != tail:
            raise ValueError(f"Invalid {key} shape {shape}; expected ({frame_count}, {', '.join(map(str, tail))})")
        if np.dtype(array.dtype) != np.dtype(np.float32):
            raise ValueError(f"Invalid {key} dtype {array.dtype}; expected float32")

    ends = source.episode_ends
    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("meta/episode_ends must be a non-empty one-dimensional array")
    if np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("meta/episode_ends must be positive and strictly increasing")
    if int(ends[-1]) != frame_count:
        raise ValueError(f"Last episode end {int(ends[-1])} does not equal frame count {frame_count}")
    if canonical and (frame_count != CANONICAL_FRAMES or ends.size != CANONICAL_EPISODES):
        raise ValueError(
            f"Not the pinned ManiFeel USB dataset: got {frame_count} frames/{ends.size} episodes; "
            f"expected {CANONICAL_FRAMES}/{CANONICAL_EPISODES}"
        )


def episode_slices(episode_ends: np.ndarray, max_episodes: int | None = None) -> list[slice]:
    ends = np.asarray(episode_ends, dtype=np.int64)
    if ends.ndim != 1 or ends.size == 0 or np.any(ends <= 0) or np.any(np.diff(ends) <= 0):
        raise ValueError("episode_ends must be a positive, strictly increasing one-dimensional array")
    count = int(ends.size) if max_episodes is None else int(max_episodes)
    if count <= 0 or count > int(ends.size):
        raise ValueError(f"max_episodes must be between 1 and {ends.size}, got {count}")
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    return [slice(int(starts[index]), int(ends[index])) for index in range(count)]


def flatten_force(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.shape != (10, 14, 3):
        raise ValueError(f"Expected tactile force shape (10, 14, 3), got {array.shape}")
    if np.dtype(array.dtype) != np.dtype(np.float32):
        array = array.astype(np.float32)
    if not np.isfinite(array).all():
        raise ValueError("Tactile force contains NaN or Inf")
    # C-order preserves the source order: ((row * 14 + column) * 3 + component).
    return np.ascontiguousarray(array.reshape(420), dtype=np.float32)


def normalize_wrist_rgb(frame: np.ndarray) -> np.ndarray:
    """Return HWC uint8 without changing the source channel order."""
    image = np.asarray(frame)
    if image.shape == (3, 256, 256):
        image = np.transpose(image, (1, 2, 0))
    if image.shape != (256, 256, 3):
        raise ValueError(f"Expected wrist RGB shape (256, 256, 3), got {image.shape}")
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if not np.issubdtype(image.dtype, np.floating):
        raise ValueError(f"Unsupported wrist RGB dtype: {image.dtype}")
    if not np.isfinite(image).all():
        raise ValueError("Wrist RGB contains NaN or Inf")
    minimum = float(image.min())
    maximum = float(image.max())
    if minimum < -1e-6 or maximum > 1.0 + 1e-6:
        raise ValueError(f"Float wrist RGB must be in [0, 1], got [{minimum}, {maximum}]")
    # This matches LeRobot's RGB writer semantics and avoids an implicit conversion.
    return np.ascontiguousarray((np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8))


def _force_names() -> list[str]:
    return [
        f"row_{row:02d}_col_{column:02d}_{component}"
        for row in range(10)
        for column in range(14)
        for component in FORCE_COMPONENTS
    ]


def build_features(image_storage: str = "video") -> dict[str, dict[str, Any]]:
    if image_storage not in {"video", "image"}:
        raise ValueError("image_storage must be 'video' or 'image'")
    return {
        OBS_STATE: {"dtype": "float32", "shape": (7,), "names": STATE_NAMES},
        OBS_WRIST: {
            "dtype": image_storage,
            "shape": (256, 256, 3),
            "names": ["height", "width", "channels"],
        },
        OBS_FORCE: {"dtype": "float32", "shape": (420,), "names": _force_names()},
        ACTION: {"dtype": "float32", "shape": (6,), "names": ACTION_NAMES},
    }


def _small_file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_identity(source: ManiFeelSource) -> dict[str, Any]:
    archive_sha = None
    download_manifest = source.root / "download_manifest.json"
    if download_manifest.is_file():
        manifest = _load_json(download_manifest)
        archive_sha = manifest.get("source", {}).get("sha256")
    metadata_files = [
        source.root.joinpath(*key.split("/"), ".zarray")
        for key in (STATE_KEY, ACTION_KEY, FORCE_KEY, WRIST_KEY, EPISODE_ENDS_KEY)
    ]
    metadata_hashes = {
        str(path.relative_to(source.root)).replace("\\", "/"): _small_file_sha256(path)
        for path in metadata_files
    }
    ends_sha = hashlib.sha256(source.episode_ends.astype("<i8", copy=False).tobytes()).hexdigest()
    return {
        "archive_sha256": archive_sha,
        "frames": source.num_frames,
        "episodes": source.num_episodes,
        "episode_ends_sha256": ends_sha,
        "zarr_metadata_sha256": metadata_hashes,
    }


def _conversion_spec(
    source: ManiFeelSource,
    *,
    repo_id: str,
    episode_count: int,
    image_storage: str,
    task: str,
) -> dict[str, Any]:
    selected_ends = source.episode_ends[:episode_count].astype(int).tolist()
    return {
        "schema_version": 1,
        "repo_id": repo_id,
        "fps": FPS,
        "robot_type": "manifeel_usb",
        "task": task,
        "image_storage": image_storage,
        "episodes": episode_count,
        "frames": selected_ends[-1],
        "selected_episode_ends": selected_ends,
        "source_identity": source_identity(source),
        "force_mapping": {
            "source": FORCE_KEY,
            "source_shape_per_frame": [10, 14, 3],
            "output": OBS_FORCE,
            "output_shape": [420],
            "formula": "source_force.reshape(420, order='C')",
            "components": list(FORCE_COMPONENTS),
            "units": "source-native ManiFeel/TacSL force-field value (no SI conversion)",
        },
        "rgb_mapping": {
            "source": WRIST_KEY,
            "source_shape_per_frame": [256, 256, 3],
            "output": OBS_WRIST,
            "channel_policy": "preserve source channel order; no BGR/RGB swap",
            "uint8_policy": "clip float [0,1], multiply by 255, then truncate to uint8",
        },
    }


def _signature(spec: dict[str, Any]) -> str:
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_lerobot_counts(root: Path) -> tuple[int, int]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise RuntimeError(f"Incomplete LeRobot staging directory (missing meta/info.json): {root}")
    info = _load_json(info_path)
    return int(info.get("total_episodes", -1)), int(info.get("total_frames", -1))


def _parquet_row_count(directory: Path) -> int | None:
    if not directory.is_dir():
        return None
    parquet_files = sorted(directory.rglob("*.parquet"))
    if not parquet_files:
        return 0
    try:
        import pyarrow.dataset as arrow_dataset
    except ImportError:
        # LeRobot's declared dataset dependencies include pyarrow. Keeping this
        # optional makes the pure conversion helpers testable in a minimal env.
        return None
    return int(arrow_dataset.dataset([str(path) for path in parquet_files], format="parquet").count_rows())


def _check_existing_info(root: Path, spec: dict[str, Any]) -> tuple[int, int]:
    info = _load_json(root / "meta" / "info.json")
    if int(info.get("fps", -1)) != FPS:
        raise RuntimeError(f"Staging dataset FPS mismatch: {info.get('fps')} != {FPS}")
    expected = build_features(spec["image_storage"])
    actual = info.get("features", {})
    for key, feature in expected.items():
        if key not in actual:
            raise RuntimeError(f"Staging dataset is missing feature {key}")
        if actual[key].get("dtype") != feature["dtype"] or tuple(actual[key].get("shape", ())) != tuple(
            feature["shape"]
        ):
            raise RuntimeError(f"Staging feature mismatch for {key}: {actual[key]}")
    episodes = int(info.get("total_episodes", -1))
    frames = int(info.get("total_frames", -1))
    if episodes > 0:
        episodes_directory = root / "meta" / "episodes"
        data_directory = root / "data"
        if not episodes_directory.is_dir() or not data_directory.is_dir():
            raise RuntimeError("LeRobot info.json records episodes, but committed parquet directories are missing")
        episode_rows = _parquet_row_count(episodes_directory)
        data_rows = _parquet_row_count(data_directory)
        if episode_rows is not None and episode_rows != episodes:
            raise RuntimeError(
                f"LeRobot episode metadata has {episode_rows} rows but info.json records {episodes} episodes"
            )
        if data_rows is not None and data_rows != frames:
            raise RuntimeError(f"LeRobot parquet data has {data_rows} rows but info.json records {frames} frames")
    return episodes, frames


def _load_lerobot_dataset_class() -> Any:
    try:
        from lerobot import __version__ as lerobot_version
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot 0.6.1 is not importable. Activate the branch environment or install this repository."
        ) from exc
    if lerobot_version != "0.6.1":
        raise RuntimeError(f"This converter targets LeRobot 0.6.1, but imported version {lerobot_version}")
    return LeRobotDataset


def _expected_frames_after(episode_ends: list[int], completed: int) -> int:
    return 0 if completed == 0 else int(episode_ends[completed - 1])


def _payload_files(root: Path) -> list[str]:
    """List episode payload files whose creation can safely be rolled aside."""
    result: list[str] = []
    for relative_root in (Path("data"), Path("videos"), Path("images"), Path("meta/episodes")):
        directory = root / relative_root
        if directory.is_dir():
            result.extend(
                str(path.relative_to(root)).replace("\\", "/")
                for path in directory.rglob("*")
                if path.is_file()
            )
    return sorted(result)


def _quarantine_uncommitted_payload(stage: Path, journal: dict[str, Any]) -> Path | None:
    before_value = journal.get("existing_payload_files")
    if not isinstance(before_value, list) or not all(isinstance(value, str) for value in before_value):
        raise RuntimeError("Pending episode journal has no safe payload snapshot; refusing an ambiguous resume")
    created = sorted(set(_payload_files(stage)) - set(before_value))
    if not created:
        return None
    quarantine = stage.with_name(
        f"{stage.name}.uncommitted-episode-{int(journal['episode_index']):03d}-{time.time_ns()}"
    )
    for relative in created:
        source = stage.joinpath(*relative.split("/"))
        destination = quarantine.joinpath(*relative.split("/"))
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)
    return quarantine


def _reconcile_progress(stage: Path, progress: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    actual_episodes, actual_frames = _check_existing_info(stage, spec)
    target = int(spec["episodes"])
    completed = int(progress.get("completed_episodes", -1))
    if completed < 0 or completed > target:
        raise RuntimeError(f"Invalid completed_episodes in progress file: {completed}")
    if actual_episodes < completed or actual_episodes > target:
        raise RuntimeError(
            f"Unsafe resume state: LeRobot has {actual_episodes} episodes but progress records {completed}"
        )
    expected_actual_frames = _expected_frames_after(spec["selected_episode_ends"], actual_episodes)
    if actual_frames != expected_actual_frames:
        raise RuntimeError(
            f"Unsafe resume state: {actual_episodes} committed episodes require {expected_actual_frames} "
            f"frames, but LeRobot metadata records {actual_frames}"
        )
    # Metadata is authoritative when a crash happened after finalize but before
    # the sidecar progress update.
    if actual_episodes > completed:
        progress = dict(progress)
        progress["completed_episodes"] = actual_episodes
        progress["completed_frames"] = actual_frames
        progress["status"] = "in_progress" if actual_episodes < target else "complete"
        atomic_write_json(stage / "conversion_progress.json", progress)
    return progress


def _initial_progress(spec: dict[str, Any], signature: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "in_progress",
        "spec_signature": signature,
        "completed_episodes": 0,
        "completed_frames": 0,
        "target_episodes": spec["episodes"],
        "target_frames": spec["frames"],
    }


def _load_or_initialize_progress(stage: Path, spec: dict[str, Any], signature: str) -> dict[str, Any]:
    progress_path = stage / "conversion_progress.json"
    if progress_path.is_file():
        progress = _load_json(progress_path)
        if progress.get("spec_signature") != signature:
            raise RuntimeError("Existing staging conversion was created with a different source or configuration")
        return _reconcile_progress(stage, progress, spec)

    episodes, frames = _check_existing_info(stage, spec)
    if episodes != 0 or frames != 0:
        raise RuntimeError("Staging dataset has committed data but no conversion progress sidecar")
    progress = _initial_progress(spec, signature)
    atomic_write_json(progress_path, progress)
    return progress


def _validate_numeric_episode(source: ManiFeelSource, span: slice) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    state = np.asarray(source.state[span], dtype=np.float32)
    action = np.asarray(source.action[span], dtype=np.float32)
    force = np.asarray(source.force[span], dtype=np.float32)
    length = span.stop - span.start
    if state.shape != (length, 7) or action.shape != (length, 6) or force.shape != (length, 10, 14, 3):
        raise RuntimeError(f"Source arrays became misaligned in frame range [{span.start}, {span.stop})")
    for key, array in ((STATE_KEY, state), (ACTION_KEY, action), (FORCE_KEY, force)):
        if not np.isfinite(array).all():
            bad = np.argwhere(~np.isfinite(array))[0].tolist()
            raise ValueError(f"{key} contains NaN/Inf in episode slice at local index {bad}")
    return state, action, force


def _iter_episode_frames(
    source: ManiFeelSource, span: slice
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    state, action, force = _validate_numeric_episode(source, span)
    wrist = np.asarray(source.wrist[span])
    if wrist.shape != (span.stop - span.start, 256, 256, 3):
        raise RuntimeError(f"Wrist images became misaligned in frame range [{span.start}, {span.stop})")
    for local_index in range(span.stop - span.start):
        yield (
            np.ascontiguousarray(state[local_index], dtype=np.float32),
            np.ascontiguousarray(action[local_index], dtype=np.float32),
            flatten_force(force[local_index]),
            normalize_wrist_rgb(wrist[local_index]),
        )


def convert_dataset(
    source_path: Path,
    output_root: Path,
    *,
    repo_id: str,
    max_episodes: int | None = None,
    task: str = "insert the USB plug into the socket",
    image_storage: str = "video",
    image_writer_threads: int = 4,
    canonical: bool = True,
    lerobot_dataset_class: Any | None = None,
) -> Path:
    source = open_source(source_path, canonical=canonical)
    spans = episode_slices(source.episode_ends, max_episodes=max_episodes)
    spec = _conversion_spec(
        source,
        repo_id=repo_id,
        episode_count=len(spans),
        image_storage=image_storage,
        task=task,
    )
    signature = _signature(spec)
    output_root = Path(output_root).expanduser().resolve()
    stage = output_root.with_name(f".{output_root.name}.inprogress")

    if output_root.exists():
        manifest_path = output_root / "conversion_manifest.json"
        if manifest_path.is_file():
            manifest = _load_json(manifest_path)
            if manifest.get("status") == "complete" and manifest.get("spec_signature") == signature:
                return output_root
        raise FileExistsError(f"Refusing to overwrite an existing output dataset: {output_root}")

    dataset_class = lerobot_dataset_class or _load_lerobot_dataset_class()
    features = build_features(image_storage)
    first_writer: Any | None = None
    if stage.exists():
        progress = _load_or_initialize_progress(stage, spec, signature)
        # A process killed before the first save leaves only create() metadata,
        # which LeRobot cannot resume because tasks/episodes parquet do not yet
        # exist. Preserve that incomplete tree for diagnosis and start clean;
        # no user destination or committed episode is deleted.
        if int(progress["completed_episodes"]) == 0:
            interrupted = stage.with_name(f"{stage.name}.interrupted-{time.time_ns()}")
            os.replace(stage, interrupted)
            first_writer = dataset_class.create(
                repo_id=repo_id,
                fps=FPS,
                features=features,
                root=stage,
                robot_type="manifeel_usb",
                use_videos=image_storage == "video",
                image_writer_processes=0,
                image_writer_threads=image_writer_threads,
                metadata_buffer_size=1,
            )
            progress = _initial_progress(spec, signature)
            atomic_write_json(stage / "conversion_progress.json", progress)
    else:
        stage.parent.mkdir(parents=True, exist_ok=True)
        first_writer = dataset_class.create(
            repo_id=repo_id,
            fps=FPS,
            features=features,
            root=stage,
            robot_type="manifeel_usb",
            use_videos=image_storage == "video",
            image_writer_processes=0,
            image_writer_threads=image_writer_threads,
            metadata_buffer_size=1,
        )
        progress = _initial_progress(spec, signature)
        atomic_write_json(stage / "conversion_progress.json", progress)

    progress = _reconcile_progress(stage, progress, spec)
    completed = int(progress["completed_episodes"])
    journal_path = stage / "episode_journal.json"
    if journal_path.is_file():
        journal = _load_json(journal_path)
        if journal.get("spec_signature") != signature:
            raise RuntimeError("Stale episode journal belongs to a different conversion")
        journal_episode = int(journal.get("episode_index", -1))
        if journal_episode < completed:
            journal_path.unlink()
        elif journal_episode != completed:
            raise RuntimeError(f"Unexpected pending episode journal index {journal_episode}; expected {completed}")
        else:
            # The episode never reached a committed LeRobot boundary. Move any
            # newly-created parquet/video/temp-image payload aside before retrying.
            _quarantine_uncommitted_payload(stage, journal)
            journal_path.unlink()

    for episode_index in range(completed, len(spans)):
        span = spans[episode_index]
        journal = {
            "schema_version": 1,
            "spec_signature": signature,
            "episode_index": episode_index,
            "source_from_index": span.start,
            "source_to_index": span.stop,
            "status": "writing",
            "existing_payload_files": _payload_files(stage),
        }
        atomic_write_json(journal_path, journal)
        if first_writer is not None and episode_index == 0:
            dataset = first_writer
            first_writer = None
        else:
            dataset = dataset_class.resume(
                repo_id=repo_id,
                root=stage,
                image_writer_processes=0,
                image_writer_threads=image_writer_threads,
                batch_encoding_size=1,
            )
        try:
            for state, action, force, wrist in _iter_episode_frames(source, span):
                dataset.add_frame(
                    {
                        OBS_STATE: state,
                        OBS_WRIST: wrist,
                        OBS_FORCE: force,
                        ACTION: action,
                        "task": task,
                    }
                )
            dataset.save_episode(parallel_encoding=False)
            dataset.finalize()
        except BaseException:
            try:
                dataset.finalize()
            except BaseException:
                pass
            raise
        finally:
            del dataset

        expected_episodes = episode_index + 1
        actual_episodes, actual_frames = _read_lerobot_counts(stage)
        expected_frames = int(source.episode_ends[episode_index])
        if (actual_episodes, actual_frames) != (expected_episodes, expected_frames):
            raise RuntimeError(
                "LeRobot commit did not reach the expected episode boundary: "
                f"got episodes={actual_episodes}, frames={actual_frames}; "
                f"expected {expected_episodes}, {expected_frames}"
            )
        progress = {
            **progress,
            "status": "in_progress" if expected_episodes < len(spans) else "complete",
            "completed_episodes": expected_episodes,
            "completed_frames": expected_frames,
        }
        atomic_write_json(stage / "conversion_progress.json", progress)
        journal_path.unlink(missing_ok=True)

    manifest = {
        **spec,
        "status": "complete",
        "spec_signature": signature,
        "source_path": str(source.root),
        "output_path": str(output_root),
        "features": build_features(image_storage),
        "alignment": "state[t], action[t], wrist[t], and tactile_force[t] share the same source frame t",
        "resume_policy": "episode progress advances only after save_episode() and finalize() succeed",
    }
    atomic_write_json(stage / "conversion_manifest.json", manifest)
    if output_root.exists():
        raise FileExistsError(f"Output appeared during conversion; refusing to overwrite: {output_root}")
    os.replace(stage, output_root)
    return output_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Extracted usb_quan_Aug05 Zarr root or its parent.")
    parser.add_argument("--output-root", type=Path, required=True, help="New LeRobot dataset directory.")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo id stored in metadata.")
    parser.add_argument("--max-episodes", type=int, default=None, help="Convert only the first N episodes.")
    parser.add_argument("--task", default="insert the USB plug into the socket")
    parser.add_argument("--image-storage", choices=("video", "image"), default="video")
    parser.add_argument("--image-writer-threads", type=int, default=4)
    parser.add_argument(
        "--allow-noncanonical-source",
        action="store_true",
        help="Permit a schema-compatible subset/test Zarr instead of the pinned 50-episode source.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.image_writer_threads < 0:
        raise ValueError("--image-writer-threads must be non-negative")
    output = convert_dataset(
        args.source,
        args.output_root,
        repo_id=args.repo_id,
        max_episodes=args.max_episodes,
        task=args.task,
        image_storage=args.image_storage,
        image_writer_threads=args.image_writer_threads,
        canonical=not args.allow_noncanonical_source,
    )
    print(f"LeRobot dataset ready: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
