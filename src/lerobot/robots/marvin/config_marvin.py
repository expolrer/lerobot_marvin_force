#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("marvin")
@dataclass
class MarvinRobotConfig(RobotConfig):
    """Configuration class for Marvin dual-arm robot.

    The Marvin robot is a bimanual 7DOF collaborative robot controlled via UDP
    through a native SDK (libMarvinSDK.so / libMarvinSDK.dll).

    Attributes:
        ip: IP address of the robot controller (default: 192.168.1.190).
        use_arm: Which arm(s) to use — "A" (left), "B" (right), or "AB" (both).
        control_mode: Control mode — "position" (high stiffness tracking) or
            "impedance" (compliant joint impedance control).
        vel_ratio: Velocity limit as percentage (1–100).
        acc_ratio: Acceleration limit as percentage (1–100).
        impedance_k: Joint impedance stiffness (Nm/deg) per joint, used when
            control_mode="impedance".
        impedance_d: Joint impedance damping (Nm/(deg/s)) per joint, used when
            control_mode="impedance".
        use_gripper: Whether to enable DM4310 gripper motors via 485/CAN.
        left_gripper_id: CAN SlaveID for the left gripper motor.
        left_gripper_master_id: CAN MasterID for the left gripper motor.
        right_gripper_id: CAN SlaveID for the right gripper motor.
        right_gripper_master_id: CAN MasterID for the right gripper motor.
        gripper_open_pos: Gripper fully open position (radians).
        gripper_close_pos: Gripper fully closed position (radians).
        disable_torque_on_disconnect: Whether to power down servos on disconnect.
        cameras: Camera configurations (keyed by name).
    """

    ip: str = "192.168.1.190"
    use_arm: str = "AB"
    control_mode: str = "impedance" # "position" or "impedance"

    vel_ratio: int = 10
    acc_ratio: int = 10

    # if self.cyr_k is None:
    #         object.__setattr__(self, "cyr_k", [400, 400, 400, 400, 300, 300, 300])
    #     if self.cyr_d is None:
    #         object.__setattr__(self, "cyr_d", [10, 10, 10, 10, 10, 10, 10])
    # Impedance parameters for policy inference (lower stiffness for smoother control)
    impedance_k: list[float] = field(default_factory=lambda: [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])
    impedance_d: list[float] = field(default_factory=lambda: [0.5, 0.5,0.5, 0.5,0.5, 0.5, 0.5])

    use_gripper: bool = False
    left_gripper_id: int = 0x01
    left_gripper_master_id: int = 0x11
    right_gripper_id: int = 0x02
    right_gripper_master_id: int = 0x12

    # Gripper position range (in radians)
    # These define the physical limits of the gripper mechanism
    # Adjust based on your actual gripper hardware
    gripper_open_pos: float = -0.5   # Fully open (radians)
    gripper_close_pos: float = 0.5   # Fully closed (radians)
    # DM4310 commands at 30 Hz can starve EtherCAT. 5 gives ~6 Hz at a 30 Hz
    # policy loop; set to 1 only after validating the hardware link.
    gripper_update_divisor: int = 5

    # Standalone policy safety. Commands are absolute joint positions in degrees.
    # The step limiter is applied against the most recent hardware observation.
    max_policy_joint_step_deg: float | None = 2.0
    # Policy checkpoints in peg_optical_module_0726force use approximately
    # [0.3, 1.35] for the gripper state/action. Keep these separate from the
    # calibration helpers above, whose historical defaults do not match this
    # dataset.
    min_policy_gripper_pos: float | None = 0.3
    max_policy_gripper_pos: float | None = 1.4
    # Optional external joint-force envelope (Nm). None disables force-triggered
    # hold because the correct threshold is payload/task specific.
    max_policy_abs_joint_force: float | None = None
    # Per-joint envelope is preferred for the 0726 dataset because channel
    # magnitudes differ substantially. Values are intentionally a little above
    # the observed training range and must be revalidated after payload changes.
    max_policy_abs_joint_force_by_joint: list[float] | None = field(
        default_factory=lambda: [6.5, 22.7, 6.0, 9.5, 1.2, 5.2, 2.4]
    )

    disable_torque_on_disconnect: bool = True

    # Tool gravity compensation parameters (for policy inference mode)
    # These are applied during robot.connect() when skip_configure=False
    enable_gravity_compensation: bool = True

    # A arm (left) tool parameters
    tool_mass_a: float = 0.7  # kg (gripper + typical payload)
    tool_com_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 50.0])  # mm (X, Y, Z center of mass in tool frame)
    tool_inertia_a: list[float] = field(default_factory=lambda: [0.004, 0.0, 0.0, 0.004, 0.0, 0.03])  # kg·m² (Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
    tool_kine_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # mm, degrees (X, Y, Z, Rx, Ry, Rz)

    # B arm (right) tool parameters
    tool_mass_b: float = 1.1  # kg (gripper + typical payload)
    tool_com_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 50.0])  # mm (X, Y, Z center of mass in tool frame)
    tool_inertia_b: list[float] = field(default_factory=lambda: [0.004, 0.0, 0.0, 0.004, 0.0, 0.03])  # kg·m² (Ixx, Ixy, Ixz, Iyy, Iyz, Izz)
    tool_kine_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # mm, degrees (X, Y, Z, Rx, Ry, Rz)

    # Force sensor configuration (Marvin-specific)
    # First layer: whether to use force feedback data
    use_force_feedback: bool = False  # Default: only record joint positions

    # Second layer: specific force feedback types to record (only effective when use_force_feedback=True)
    # Valid options: ["joint_vel", "joint_torque", "cart_force", "joint_force"]
    # - "joint_vel": joint velocities (rad/s or deg/s)
    # - "joint_torque": joint sensor torques (Nm)
    # - "cart_force": end-effector Cartesian space forces [Fx, Fy, Fz, Mx, My, Mz]
    # - "joint_force": joint space external forces (Nm)
    force_feedback_types: list[str] = field(default_factory=list)

    cameras: dict[str, CameraConfig] = field(default_factory=dict)
