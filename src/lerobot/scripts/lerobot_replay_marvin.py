#!/usr/bin/env python3
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Replays the actions of an episode from a dataset on a Marvin robot.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Examples:

```shell
lerobot-replay-marvin \
    --robot.type=marvin \
    --robot.ip=10.121.40.62 \
    --robot.enable_torque_mode=false \
    --dataset.repo_id=/home/marvin/hhw/peg_optical_module_0707 \
    --dataset.episode=0
```

With custom fps:
```shell
lerobot-replay-marvin \
    --robot.type=marvin \
    --robot.ip=10.121.40.62 \
    --dataset.repo_id=/home/marvin/hhw/peg_optical_module_0707 \
    --dataset.episode=0 \
    --dataset.fps=30
```
"""

import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
)
from lerobot.robots.marvin import MarvinRobotConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)


@dataclass
class DatasetReplayConfig:
    # Dataset identifier or local path
    repo_id: str
    # Episode to replay
    episode: int
    # Root directory where the dataset will be stored. If None, uses repo_id as path for local datasets.
    root: str | Path | None = None
    # Limit the frames per second. By default, uses the dataset fps.
    fps: int | None = None


@dataclass
class ReplayConfig:
    robot: MarvinRobotConfig
    dataset: DatasetReplayConfig
    # Use vocal synthesis to read events
    play_sounds: bool = False


@parser.wrap()
def replay(cfg: ReplayConfig):
    init_logging()
    logging.info("=" * 80)
    logging.info("Marvin Robot Replay Configuration:")
    logging.info("=" * 80)
    logging.info(pformat(asdict(cfg)))
    logging.info("=" * 80)

    # Load dataset
    dataset_path = cfg.dataset.repo_id
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    logging.info(f"Loading dataset from: {dataset_path}")
    dataset = LeRobotDataset(
        dataset_path,
        root=cfg.dataset.root,
        episodes=[cfg.dataset.episode]
    )

    # Use dataset fps or override
    replay_fps = cfg.dataset.fps if cfg.dataset.fps is not None else dataset.fps
    logging.info(f"Replay FPS: {replay_fps}")
    logging.info(f"Episode {cfg.dataset.episode}: {dataset.num_frames} frames")

    # Extract actions
    actions = dataset.select_columns(ACTION)
    action_names = dataset.features[ACTION]["names"]

    logging.info(f"Action names: {action_names}")

    # Connect to robot
    logging.info(f"Connecting to Marvin robot at {cfg.robot.ip}...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        log_say(f"Replaying episode {cfg.dataset.episode}", cfg.play_sounds, blocking=True)

        for idx in range(dataset.num_frames):
            start_t = time.perf_counter()

            # Get action from dataset
            action_array = actions[idx][ACTION]
            action = {}
            for i, name in enumerate(action_names):
                action[name] = float(action_array[i])

            # Send action to robot
            robot.send_action(action)

            # Log progress every 30 frames
            if idx % 30 == 0:
                logging.info(f"Frame {idx}/{dataset.num_frames} ({100*idx/dataset.num_frames:.1f}%)")

            # Maintain fps
            dt_s = time.perf_counter() - start_t
            precise_sleep(max(1.0 / replay_fps - dt_s, 0.0))

        logging.info("Replay completed successfully!")
        log_say("Replay completed", cfg.play_sounds, blocking=True)

    except KeyboardInterrupt:
        logging.info("Replay interrupted by user")
    except Exception as e:
        logging.error(f"Replay failed: {e}")
        raise
    finally:
        logging.info("Disconnecting robot...")
        robot.disconnect()


def main():
    register_third_party_plugins()
    replay()


if __name__ == "__main__":
    main()
