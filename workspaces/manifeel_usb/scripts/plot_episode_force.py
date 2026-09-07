#!/usr/bin/env python3
"""Plot ManiFeel USB tactile-force curves for one source episode."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np


FPS = 15
FORCE_COMPONENTS = ("normal", "shear_x", "shear_y")


def force_grid(force: np.ndarray) -> np.ndarray:
    array = np.asarray(force, dtype=np.float32)
    if array.ndim == 2 and array.shape[1] == 420:
        array = array.reshape(array.shape[0], 10, 14, 3)
    if array.ndim != 4 or array.shape[1:] != (10, 14, 3):
        raise ValueError(f"Expected [T,10,14,3] or [T,420] force data, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError("Force data contains NaN or Inf")
    return array


def compute_force_curves(force: np.ndarray) -> dict[str, np.ndarray]:
    grid = force_grid(force)
    marker_l2 = np.linalg.norm(grid, axis=-1)
    return {
        "component_mean": grid.mean(axis=(1, 2)),
        "component_abs_mean": np.abs(grid).mean(axis=(1, 2)),
        "marker_l2_mean": marker_l2.mean(axis=(1, 2)),
        "marker_l2_max": marker_l2.max(axis=(1, 2)),
        "marker_l2_heatmap": marker_l2.reshape(grid.shape[0], 140).T,
    }


def plot_force_episode(
    force: np.ndarray,
    output: Path,
    *,
    episode_index: int = 0,
    fps: int = FPS,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render force curves") from exc

    curves = compute_force_curves(force)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    time_s = np.arange(curves["component_mean"].shape[0], dtype=np.float64) / fps

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)
    for component_index, label in enumerate(FORCE_COMPONENTS):
        axes[0].plot(time_s, curves["component_mean"][:, component_index], label=label, linewidth=1.4)
    axes[0].set_title(f"ManiFeel USB episode {episode_index}: signed spatial mean")
    axes[0].set_ylabel("simulator force-field value")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(time_s, curves["marker_l2_mean"], label="mean marker L2", linewidth=1.5)
    axes[1].plot(time_s, curves["marker_l2_max"], label="max marker L2", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("L2 magnitude")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best")

    image = axes[2].imshow(
        curves["marker_l2_heatmap"],
        aspect="auto",
        origin="lower",
        extent=(0.0, max(float(time_s[-1]) if time_s.size else 0.0, 1.0 / fps), 0, 140),
        interpolation="nearest",
    )
    axes[2].set_title("Per-cell tactile-force magnitude (10 x 14 grid flattened by row)")
    axes[2].set_xlabel("time (s)")
    axes[2].set_ylabel("grid cell")
    figure.colorbar(image, ax=axes[2], label="L2 magnitude")

    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    figure.savefig(temporary, dpi=160)
    plt.close(figure)
    os_replace(temporary, output)
    return output


def os_replace(source: Path, destination: Path) -> None:
    # Kept as a tiny seam so atomic output replacement is unit-testable.
    import os

    os.replace(source, destination)


def _resolve_zarr_root(source: Path) -> Path:
    source = Path(source).expanduser().resolve()
    for candidate in (source, source / "usb_quan_Aug05"):
        if (candidate / ".zgroup").is_file() and (candidate / "meta" / "episode_ends" / ".zarray").is_file():
            return candidate
    raise FileNotFoundError(f"Could not find the ManiFeel Zarr group under {source}")


def _open_group(source: Path) -> Any:
    try:
        import zarr
    except ImportError as exc:
        raise RuntimeError("zarr is required to read the source dataset") from exc
    return zarr.open_group(str(_resolve_zarr_root(source)), mode="r")


def plot_source_episode(source: Path, output: Path, *, episode_index: int = 0) -> Path:
    group = _open_group(source)
    ends = np.asarray(group["meta/episode_ends"][:], dtype=np.int64)
    if episode_index < 0 or episode_index >= ends.size:
        raise IndexError(f"episode_index must be between 0 and {ends.size - 1}")
    start = 0 if episode_index == 0 else int(ends[episode_index - 1])
    end = int(ends[episode_index])
    force = np.asarray(group["data/tactile_force_field_right"][start:end], dtype=np.float32)
    return plot_force_episode(force, output, episode_index=episode_index, fps=FPS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = plot_source_episode(args.source, args.output, episode_index=args.episode)
    print(f"Force plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
