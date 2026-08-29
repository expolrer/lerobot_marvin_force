#!/usr/bin/env python3
"""Audit the Marvin force dataset and render the first-episode force curves."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.dataset as pads
import yaml


def expand(value: str, repo_root: Path) -> Path:
    return Path(value.replace("${repo_root}", str(repo_root))).expanduser().resolve()


def vector(table, key: str) -> np.ndarray:
    return np.asarray(table[key].combine_chunks().to_pylist(), dtype=np.float32)


def shape_text(feature: dict[str, Any]) -> str:
    return "[" + ", ".join(str(value) for value in feature["shape"]) + "]"


def write_feature_table(path: Path, info: dict[str, Any]) -> None:
    lines = [
        "| Feature | dtype | shape | Names / meaning |",
        "|---|---:|---:|---|",
    ]
    for key, feature in info["features"].items():
        names = feature.get("names")
        if isinstance(names, list):
            names_text = ", ".join(str(name) for name in names)
        else:
            names_text = "-"
        lines.append(f"| `{key}` | `{feature['dtype']}` | `{shape_text(feature)}` | {names_text} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    repo_root = args.config.resolve().parents[3]
    dataset_root = expand(config["dataset_root"], repo_root)
    report_dir = expand(config["report_dir"], repo_root)
    plot_path = expand(config["plot_path"], repo_root)
    episode_index = int(config["episode_index"])

    info = json.loads((dataset_root / "meta" / "info.json").read_text(encoding="utf-8"))
    stats = json.loads((dataset_root / "meta" / "stats.json").read_text(encoding="utf-8"))
    expected = config["contract"]
    if info.get("codebase_version") != expected["codebase_version"]:
        raise ValueError("Dataset schema is not the reviewed LeRobot v3.0 schema")
    if int(info.get("fps")) != int(expected["fps"]):
        raise ValueError(f"Expected {expected['fps']} FPS, got {info.get('fps')}")
    for key, required in expected["features"].items():
        actual = info["features"].get(key)
        if actual is None or actual.get("dtype") != required["dtype"] or actual.get("shape") != required["shape"]:
            raise ValueError(f"Feature contract failed for {key}: {actual}")

    cart = stats["observation.cart_force"]
    cart_stats_all_zero = all(float(value) == 0.0 for value in [*cart["min"], *cart["max"]])

    dataset = pads.dataset(str(dataset_root / "data"), format="parquet")
    columns = [
        "timestamp",
        "frame_index",
        "episode_index",
        "observation.joint_torque",
        "observation.joint_force",
    ]
    all_table = dataset.to_table(
        columns=[
            "episode_index",
            "observation.joint_torque",
            "observation.joint_force",
            "observation.cart_force",
        ]
    )
    if all_table.num_rows != int(info["total_frames"]):
        raise ValueError(
            f"Metadata declares {info['total_frames']} frames, parquet contains {all_table.num_rows}"
        )
    all_episode_indices = np.asarray(
        all_table["episode_index"].combine_chunks().to_pylist(), dtype=np.int64
    )
    all_torque = vector(all_table, "observation.joint_torque")
    all_force = vector(all_table, "observation.joint_force")
    all_cart_force = vector(all_table, "observation.cart_force")
    if all_torque.shape[1:] != (7,) or all_force.shape[1:] != (7,):
        raise ValueError(f"Unexpected full-dataset force shapes: torque={all_torque.shape}, force={all_force.shape}")
    if not np.isfinite(all_torque).all() or not np.isfinite(all_force).all():
        raise ValueError("Dataset contains NaN or Inf in force channels")
    cart_all_zero = cart_stats_all_zero and bool(np.all(all_cart_force == 0.0))
    if not cart_all_zero:
        raise ValueError("cart_force is not all-zero in parquet; review the model input contract")

    unique_episodes = np.unique(all_episode_indices)
    if unique_episodes.size != int(info["total_episodes"]):
        raise ValueError(
            f"Metadata declares {info['total_episodes']} episodes, parquet contains {unique_episodes.size}"
        )
    episode_force_means = np.stack(
        [all_force[all_episode_indices == index].mean(axis=0) for index in unique_episodes]
    )
    episode_torque_means = np.stack(
        [all_torque[all_episode_indices == index].mean(axis=0) for index in unique_episodes]
    )
    episode_force_stds = np.stack(
        [all_force[all_episode_indices == index].std(axis=0) for index in unique_episodes]
    )
    baseline_std = episode_force_means.std(axis=0)
    typical_within_episode_std = np.median(episode_force_stds, axis=0)
    dynamics_to_baseline_ratio = typical_within_episode_std / np.maximum(baseline_std, 1e-8)
    force_torque_correlation = []
    for joint in range(7):
        correlation = np.corrcoef(all_torque[:, joint], all_force[:, joint])[0, 1]
        force_torque_correlation.append(float(correlation))

    table = dataset.to_table(
        columns=columns,
        filter=pads.field("episode_index") == episode_index,
    ).sort_by("frame_index")
    if table.num_rows == 0:
        raise ValueError(f"Episode {episode_index} is empty")
    timestamps = np.asarray(table["timestamp"].combine_chunks().to_pylist(), dtype=np.float64)
    frame_indices = np.asarray(table["frame_index"].combine_chunks().to_pylist(), dtype=np.int64)
    torque = vector(table, "observation.joint_torque")
    force = vector(table, "observation.joint_force")
    if torque.shape[1:] != (7,) or force.shape[1:] != (7,):
        raise ValueError(f"Unexpected force shapes: torque={torque.shape}, force={force.shape}")
    if not np.isfinite(torque).all() or not np.isfinite(force).all():
        raise ValueError("Episode contains NaN or Inf in force channels")

    report = {
        "dataset_root": str(dataset_root),
        "schema": info["codebase_version"],
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "total_episodes": info["total_episodes"],
        "total_frames": info["total_frames"],
        "default_model_force_feature": config["default_model_force_feature"],
        "cart_force_all_zero": cart_all_zero,
        "cross_episode": {
            "episode_count": int(unique_episodes.size),
            "joint_force_episode_mean_min": episode_force_means.min(axis=0).tolist(),
            "joint_force_episode_mean_max": episode_force_means.max(axis=0).tolist(),
            "joint_force_episode_mean_std": episode_force_means.std(axis=0).tolist(),
            "joint_torque_episode_mean_min": episode_torque_means.min(axis=0).tolist(),
            "joint_torque_episode_mean_max": episode_torque_means.max(axis=0).tolist(),
            "joint_torque_episode_mean_std": episode_torque_means.std(axis=0).tolist(),
            "joint_force_median_within_episode_std": typical_within_episode_std.tolist(),
            "joint_force_dynamics_to_baseline_ratio": dynamics_to_baseline_ratio.tolist(),
            "joint_torque_vs_joint_force_correlation": force_torque_correlation,
            "interpretation": (
                "Episode means are not identical. Ratios above one mean typical "
                "within-episode dynamics exceed cross-episode baseline drift."
            ),
        },
        "episode": {
            "index": episode_index,
            "frames": int(table.num_rows),
            "first_frame": int(frame_indices[0]),
            "last_frame": int(frame_indices[-1]),
            "duration_s": float(timestamps[-1] - timestamps[0]),
            "joint_torque_min": torque.min(axis=0).tolist(),
            "joint_torque_max": torque.max(axis=0).tolist(),
            "joint_torque_mean": torque.mean(axis=0).tolist(),
            "joint_force_min": force.min(axis=0).tolist(),
            "joint_force_max": force.max(axis=0).tolist(),
            "joint_force_mean": force.mean(axis=0).tolist(),
        },
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0

    report_dir.mkdir(parents=True, exist_ok=True)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    (report_dir / "dataset_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_feature_table(report_dir / "feature_table.md", info)
    with (report_dir / "episode_force_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["episode_index"]
            + [f"joint_torque_mean_{index}" for index in range(1, 8)]
            + [f"joint_force_mean_{index}" for index in range(1, 8)]
        )
        for row, episode in enumerate(unique_episodes):
            writer.writerow(
                [int(episode), *episode_torque_means[row], *episode_force_means[row]]
            )
    with (report_dir / f"episode_{episode_index:03d}_force.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["timestamp", "frame_index"]
            + [f"joint_torque_{index}" for index in range(1, 8)]
            + [f"joint_force_{index}" for index in range(1, 8)]
        )
        for row in range(table.num_rows):
            writer.writerow([timestamps[row], frame_indices[row], *torque[row], *force[row]])

    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    for index in range(7):
        axes[0].plot(timestamps, torque[:, index], label=f"J{index + 1}", color=colors[index], linewidth=1.0)
        axes[1].plot(timestamps, force[:, index], label=f"J{index + 1}", color=colors[index], linewidth=1.0)
    axes[0].set_title(f"Episode {episode_index}: feedback joint torque (m_FB_Joint_SToq)")
    axes[0].set_ylabel("Torque (Nm)")
    axes[1].set_title(f"Episode {episode_index}: estimated external joint force (m_EST_Joint_Force)")
    axes[1].set_ylabel("Estimated force (Nm)")
    axes[1].set_xlabel("Time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(ncol=7, loc="upper right")
    figure.suptitle("peg_optical_module_0726force — first episode", fontsize=15)
    figure.savefig(plot_path, dpi=150)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
