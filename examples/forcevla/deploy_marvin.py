#!/usr/bin/env python

"""Deploy a remote ForceVLA-JointForce server on a Marvin B arm."""

import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.robots.config import RobotConfig
from lerobot.robots.marvin import MarvinRobot, MarvinRobotConfig
from lerobot.transport.openpi_websocket_client import OpenPIWebSocketClient
from lerobot.utils.robot_utils import precise_sleep

logger = logging.getLogger(__name__)

JOINT_KEYS = [f"joint_{index}.pos" for index in range(1, 8)]
FORCE_KEYS = [f"joint_{index}.force" for index in range(1, 8)]
CAMERA_KEYS = ("up_cam", "left_close", "right_close")


@dataclass
class ForceVLADeployConfig:
    robot: RobotConfig = field(default_factory=MarvinRobotConfig)
    server_host: str = "127.0.0.1"
    server_port: int = 8000
    api_key: str | None = None
    prompt: str = "Insert the optical peg module"
    fps: int = 30
    replan_every: int = 3
    max_response_age_s: float = 1.0
    duration_s: float = 0.0
    dry_run: bool = True
    force_limits: list[float] = field(
        default_factory=lambda: [6.5, 22.7, 6.0, 9.5, 1.2, 5.2, 2.4]
    )

    def __post_init__(self) -> None:
        if self.fps <= 0 or self.replan_every <= 0:
            raise ValueError("fps and replan_every must be positive")
        if len(self.force_limits) != 7 or min(self.force_limits) <= 0:
            raise ValueError("force_limits must contain seven positive magnitudes")


def _validate_robot_config(config: RobotConfig) -> None:
    if not isinstance(config, MarvinRobotConfig):
        raise TypeError("ForceVLA deployment requires --robot.type=marvin")
    if config.use_arm != "B":
        raise ValueError("ForceVLA deployment controls only Marvin arm B; set --robot.use_arm=B")
    if not config.use_gripper:
        raise ValueError("The 8D checkpoint requires --robot.use_gripper=true")
    if not config.use_force_feedback or "joint_force" not in config.force_feedback_types:
        raise ValueError(
            "Enable --robot.use_force_feedback=true and "
            "--robot.force_feedback_types='[\"joint_force\"]'"
        )
    missing_cameras = set(CAMERA_KEYS) - set(config.cameras)
    if missing_cameras:
        raise ValueError(f"Missing training cameras: {sorted(missing_cameras)}")


def _request_from_observation(observation: dict, prompt: str) -> dict:
    state = np.asarray([observation[key] for key in [*JOINT_KEYS, "gripper.pos"]], dtype=np.float32)
    joint_force = np.asarray([observation[key] for key in FORCE_KEYS], dtype=np.float32)
    images = [np.asarray(observation[key]) for key in CAMERA_KEYS]
    if state.shape != (8,) or joint_force.shape != (7,):
        raise ValueError(f"Bad Marvin vector shapes: state={state.shape}, force={joint_force.shape}")
    if not np.isfinite(state).all() or not np.isfinite(joint_force).all():
        raise ValueError("Marvin observation contains NaN or Inf")
    if any(image.ndim != 3 or image.shape[-1] != 3 for image in images):
        raise ValueError(f"Expected HWC RGB images, got {[image.shape for image in images]}")
    return {
        "state": state,
        "joint_force": joint_force,
        "image": images[0].astype(np.uint8, copy=False),
        "left_wrist_image": images[1].astype(np.uint8, copy=False),
        "right_wrist_image": images[2].astype(np.uint8, copy=False),
        "prompt": prompt,
    }


def _validated_actions(response: dict) -> np.ndarray:
    actions = np.asarray(response.get("actions"), dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 8 or not len(actions):
        raise ValueError(f"ForceVLA response actions must be [H,8], got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError("ForceVLA response contains NaN or Inf")
    return actions


@parser.wrap()
def deploy(config: ForceVLADeployConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _validate_robot_config(config.robot)
    robot = MarvinRobot(config.robot)
    client = OpenPIWebSocketClient(
        config.server_host, config.server_port, api_key=config.api_key
    )
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ForceVLAClient")
    pending: Future | None = None
    pending_started = 0.0
    action_queue: deque[np.ndarray] = deque()
    steps_since_request = config.replan_every
    started = time.monotonic()

    try:
        robot.connect()
        while config.duration_s <= 0 or time.monotonic() - started < config.duration_s:
            tick_started = time.perf_counter()
            observation = robot.get_observation()
            force = np.asarray([observation[key] for key in FORCE_KEYS], dtype=np.float32)
            over_force = np.any(np.abs(force) > np.asarray(config.force_limits, dtype=np.float32))

            if over_force:
                action_queue.clear()
                logger.error("Force envelope exceeded; discarding the open-loop chunk and holding")

            if pending is not None and pending.done():
                age = time.monotonic() - pending_started
                try:
                    response = pending.result()
                    actions = _validated_actions(response)
                    if over_force:
                        logger.warning("Discarded ForceVLA response while force envelope is exceeded")
                    elif age <= config.max_response_age_s:
                        action_queue = deque(actions)
                        logger.info("Accepted ForceVLA chunk %s in %.3fs", actions.shape, age)
                    else:
                        logger.warning("Discarded stale ForceVLA response (%.3fs)", age)
                except Exception:
                    logger.exception("ForceVLA request failed; holding until a valid response")
                pending = None

            if not over_force and pending is None and steps_since_request >= config.replan_every:
                request = _request_from_observation(observation, config.prompt)
                pending_started = time.monotonic()
                pending = executor.submit(client.infer, request)
                steps_since_request = 0

            if action_queue and not over_force:
                action = action_queue.popleft()
                action_dict = {key: float(action[index]) for index, key in enumerate(JOINT_KEYS)}
                action_dict["gripper.pos"] = float(action[7])
            else:
                action_dict = {key: float(observation[key]) for key in JOINT_KEYS}
                action_dict["gripper.pos"] = float(observation["gripper.pos"])
            if config.dry_run:
                if steps_since_request % config.fps == 0:
                    logger.info("DRY RUN: action computed but not sent to Marvin")
            else:
                robot.send_action(action_dict)
            steps_since_request += 1

            elapsed = time.perf_counter() - tick_started
            if elapsed < 1.0 / config.fps:
                precise_sleep(1.0 / config.fps - elapsed)
            else:
                logger.warning("ForceVLA control tick overrun: %.1f ms", elapsed * 1000)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        client.close()
        if robot.is_connected:
            robot.disconnect()


def main() -> None:
    deploy()


if __name__ == "__main__":
    main()
