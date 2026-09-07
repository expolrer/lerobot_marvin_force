from __future__ import annotations

import hashlib
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import audit_dataset as audit  # noqa: E402
import convert_manifeel_to_lerobot as convert  # noqa: E402
import download_dataset as download  # noqa: E402
import plot_episode_force as plot_force  # noqa: E402


try:
    import zarr
except ImportError:
    zarr = None


def make_source(root: Path, *, with_nan: bool = False) -> tuple[Path, dict[str, np.ndarray]]:
    if zarr is None:
        pytest.skip("zarr is required for synthetic source tests")
    source = root / "usb_quan_Aug05"
    group = zarr.open_group(str(source), mode="w")
    data = group.require_group("data")
    meta = group.require_group("meta")
    frames = 5
    state = np.arange(frames * 7, dtype=np.float32).reshape(frames, 7) / 10
    if with_nan:
        state[1, 0] = np.nan
    action = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6) / 20
    force = np.arange(frames * 420, dtype=np.float32).reshape(frames, 10, 14, 3) / 1000
    wrist = np.linspace(0.0, 1.0, frames * 256 * 256 * 3, dtype=np.float32).reshape(
        frames, 256, 256, 3
    )
    data.create_dataset("state", data=state, chunks=(2, 7))
    data.create_dataset("action", data=action, chunks=(2, 6))
    data.create_dataset("tactile_force_field_right", data=force, chunks=(2, 10, 14, 3))
    data.create_dataset("wrist", data=wrist, chunks=(1, 256, 256, 3))
    meta.create_dataset("episode_ends", data=np.asarray([2, 5], dtype=np.int64), chunks=(2,))
    return source, {"state": state, "action": action, "force": force, "wrist": wrist}


def test_schema_alignment_flatten_order_and_rgb(tmp_path: Path) -> None:
    source_path, arrays = make_source(tmp_path)
    source = convert.open_source(source_path, canonical=False)
    assert [value.stop - value.start for value in convert.episode_slices(source.episode_ends)] == [2, 3]
    assert convert.FPS == 15

    flattened = convert.flatten_force(arrays["force"][0])
    np.testing.assert_array_equal(flattened, arrays["force"][0].reshape(420, order="C"))
    assert flattened[3] == arrays["force"][0, 0, 1, 0]

    image = convert.normalize_wrist_rgb(arrays["wrist"][0])
    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(
        image,
        (np.clip(arrays["wrist"][0], 0.0, 1.0) * 255.0).astype(np.uint8),
    )
    features = convert.build_features("video")
    assert features[convert.OBS_WRIST]["shape"] == (256, 256, 3)
    assert features[convert.OBS_FORCE]["shape"] == (420,)
    assert features[convert.ACTION]["shape"] == (6,)
    assert len(features[convert.OBS_FORCE]["names"]) == 420


def test_bad_episode_boundary_is_rejected(tmp_path: Path) -> None:
    source_path, _ = make_source(tmp_path)
    group = zarr.open_group(str(source_path), mode="a")
    group["meta/episode_ends"][:] = np.asarray([3, 3], dtype=np.int64)
    with pytest.raises(ValueError, match="strictly increasing"):
        convert.open_source(source_path, canonical=False)


def test_audit_reports_nan_and_writes_report(tmp_path: Path) -> None:
    source_path, _ = make_source(tmp_path, with_nan=True)
    report_dir = tmp_path / "audit"
    report = audit.audit_dataset(source_path, report_dir, canonical=False)
    assert not report["passed"]
    assert any("NaN/Inf" in error for error in report["errors"])
    assert (report_dir / "audit_report.json").is_file()
    # Force itself is finite, so the diagnostic curve is still generated.
    assert (report_dir / "episode_000_tactile_force.png").stat().st_size > 0


def test_audit_passes_and_force_plot_has_curves(tmp_path: Path) -> None:
    source_path, arrays = make_source(tmp_path)
    report_dir = tmp_path / "audit"
    report = audit.audit_dataset(source_path, report_dir, max_episodes=1, canonical=False)
    assert report["passed"], report["errors"]
    assert report["source"]["action_alignment"]["valid"]
    assert report["source"]["force"]["nonzero_count"] > 0
    assert report["source"]["force"]["changed_transitions"] > 0
    assert Path(report["artifacts"]["first_episode_force_plot"]).stat().st_size > 0

    curves = plot_force.compute_force_curves(arrays["force"][:2])
    assert curves["component_mean"].shape == (2, 3)
    assert curves["marker_l2_heatmap"].shape == (140, 2)


class FakeLeRobotDataset:
    frames: list[dict[str, Any]] = []
    fail_once_on_resume = False

    def __init__(self, root: Path):
        self.root = Path(root)
        self.buffer: list[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.frames = []
        cls.fail_once_on_resume = False

    @classmethod
    def create(cls, *, root: Path, fps: int, features: dict[str, Any], **kwargs: Any) -> "FakeLeRobotDataset":
        root = Path(root)
        root.mkdir(parents=True, exist_ok=False)
        (root / "meta").mkdir()
        info = {
            "fps": fps,
            "total_episodes": 0,
            "total_frames": 0,
            "features": features,
        }
        (root / "meta" / "info.json").write_text(json.dumps(info), encoding="utf-8")
        return cls(root)

    @classmethod
    def resume(cls, *, root: Path, **kwargs: Any) -> "FakeLeRobotDataset":
        instance = cls(Path(root))
        if cls.fail_once_on_resume:
            cls.fail_once_on_resume = False
            instance.raise_on_add = True
        else:
            instance.raise_on_add = False
        return instance

    def add_frame(self, frame: dict[str, Any]) -> None:
        if getattr(self, "raise_on_add", False):
            (self.root / "data").mkdir(exist_ok=True)
            (self.root / "data" / "orphan.bin").write_bytes(b"uncommitted")
            raise RuntimeError("synthetic interruption")
        self.buffer.append(
            {key: np.array(value, copy=True) if key != "task" else value for key, value in frame.items()}
        )

    def save_episode(self, parallel_encoding: bool = True) -> None:
        info_path = self.root / "meta" / "info.json"
        info = json.loads(info_path.read_text(encoding="utf-8"))
        type(self).frames.extend(self.buffer)
        (self.root / "data").mkdir(exist_ok=True)
        (self.root / "meta" / "episodes").mkdir(exist_ok=True)
        info["total_episodes"] += 1
        info["total_frames"] += len(self.buffer)
        info_path.write_text(json.dumps(info), encoding="utf-8")
        self.buffer = []

    def finalize(self) -> None:
        return None


def test_conversion_is_frame_aligned_and_resumes_by_episode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path, arrays = make_source(tmp_path)
    output = tmp_path / "lerobot"
    FakeLeRobotDataset.reset()
    FakeLeRobotDataset.fail_once_on_resume = True
    monkeypatch.setattr(convert, "_parquet_row_count", lambda directory: None)

    # Episode zero commits, then the synthetic interruption happens at episode one.
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        convert.convert_dataset(
            source_path,
            output,
            repo_id="test/manifeel_usb",
            image_storage="image",
            canonical=False,
            lerobot_dataset_class=FakeLeRobotDataset,
        )
    stage = output.with_name(f".{output.name}.inprogress")
    progress = json.loads((stage / "conversion_progress.json").read_text(encoding="utf-8"))
    assert progress["completed_episodes"] == 1
    assert progress["completed_frames"] == 2

    result = convert.convert_dataset(
        source_path,
        output,
        repo_id="test/manifeel_usb",
        image_storage="image",
        canonical=False,
        lerobot_dataset_class=FakeLeRobotDataset,
    )
    assert result == output.resolve()
    assert len(FakeLeRobotDataset.frames) == 5
    for index, frame in enumerate(FakeLeRobotDataset.frames):
        np.testing.assert_array_equal(frame[convert.OBS_STATE], arrays["state"][index])
        np.testing.assert_array_equal(frame[convert.ACTION], arrays["action"][index])
        np.testing.assert_array_equal(frame[convert.OBS_FORCE], arrays["force"][index].reshape(420))
    manifest = json.loads((output / "conversion_manifest.json").read_text(encoding="utf-8"))
    assert manifest["fps"] == 15
    assert manifest["episodes"] == 2
    assert manifest["frames"] == 5
    assert manifest["force_mapping"]["formula"] == "source_force.reshape(420, order='C')"
    assert not stage.exists()
    assert not (output / "data" / "orphan.bin").exists()
    quarantines = list(tmp_path.glob(".lerobot.inprogress.uncommitted-episode-001-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "data" / "orphan.bin").read_bytes() == b"uncommitted"

    # A completed conversion is idempotent and is never overwritten.
    assert (
        convert.convert_dataset(
            source_path,
            output,
            repo_id="test/manifeel_usb",
            image_storage="image",
            canonical=False,
            lerobot_dataset_class=FakeLeRobotDataset,
        )
        == output.resolve()
    )


class RangeHandler(BaseHTTPRequestHandler):
    payload = b""
    ranges: list[str | None] = []

    def do_GET(self) -> None:  # noqa: N802
        range_header = self.headers.get("Range")
        type(self).ranges.append(range_header)
        start = 0
        if range_header:
            match = re.fullmatch(r"bytes=(\d+)-", range_header)
            assert match
            start = int(match.group(1))
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")
        else:
            self.send_response(200)
        body = self.payload[start:]
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        return None


def test_download_resumes_and_verifies_sha256(tmp_path: Path) -> None:
    payload = (b"manifeel-usb-range-test-" * 1024) + b"done"
    RangeHandler.payload = payload
    RangeHandler.ranges = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        destination = tmp_path / "dataset.zip"
        part = destination.with_name("dataset.zip.part")
        part.write_bytes(payload[:333])
        result = download.download_with_resume(
            f"http://127.0.0.1:{server.server_port}/dataset.zip",
            destination,
            expected_size=len(payload),
            expected_sha256=hashlib.sha256(payload).hexdigest(),
            retries=1,
            timeout_s=5,
        )
        assert result.read_bytes() == payload
        assert RangeHandler.ranges[0] == "bytes=333-"
        assert not part.exists()
    finally:
        server.shutdown()
        server.server_close()


def test_download_never_overwrites_bad_existing_file(tmp_path: Path) -> None:
    destination = tmp_path / "dataset.zip"
    destination.write_bytes(b"wrong")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        download.download_with_resume(
            "http://127.0.0.1:1/not-used",
            destination,
            expected_size=5,
            expected_sha256=hashlib.sha256(b"right").hexdigest(),
            retries=0,
        )
    assert destination.read_bytes() == b"wrong"
