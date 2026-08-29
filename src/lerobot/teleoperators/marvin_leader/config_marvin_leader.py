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

from dataclasses import dataclass

from ..config import TeleoperatorConfig
from dataclasses import dataclass, field


@TeleoperatorConfig.register_subclass("marvin_leader")
@dataclass
class MarvinLeaderConfig(TeleoperatorConfig):
    """Configuration for a Marvin arm used as a teleoperation leader.

    The leader arm is put in torque + joint-impedance + drag mode so the user
    can freely move it by hand.  Joint positions are read and forwarded as
    actions to the follower robot.

    Attributes:
        ip: IP address of the leader robot controller.
        use_arm: Which arm(s) to use as leader — "A", "B", or "AB".
        vel_ratio: Velocity limit as percentage (1–100) for alignment motion.
        acc_ratio: Acceleration limit as percentage (1–100) for alignment motion.
        drag_k: Joint impedance stiffness for drag mode (low = easy to move).
        drag_d: Joint impedance damping for drag mode.
        cyr_k: Joint impedance stiffness for CYR follower mode (high = precise tracking).
            Default: [400, 400, 400, 400, 300, 300, 300]
        cyr_d: Joint impedance damping for CYR follower mode.
            Default: [10, 10, 10, 10, 10, 10, 10]
        align_on_connect: Whether to align leader and follower to the same
            initial position before entering drag mode. Required for teleoperation.
        align_position: Initial position for alignment (7 joint angles in degrees).
            If None, aligns to follower's current position.
        align_timeout: Maximum time (seconds) to wait for alignment.
        enable_gravity_compensation: Whether to enable gravity compensation for end-effector.
        tool_mass: Mass of the end-effector tool in kg (for gravity compensation).
        tool_com: Center of mass of the tool [X, Y, Z] in mm relative to flange frame.
        tool_inertia: Inertia tensor [Ixx, Ixy, Ixz, Iyy, Iyz, Izz] in kg·m².
        tool_kine: Tool kinematic parameters [X, Y, Z, A, B, C] (mm, degrees) relative to flange.
    """

    ip: str = "192.168.1.190"
    use_arm: str = "AB"

    # Velocity and acceleration limits for alignment (percentage 1-100)
    vel_ratio: int = 20  # 20% max speed for safe alignment
    acc_ratio: int = 20  # 20% max acceleration

    drag_k: list[float] = None
    drag_d: list[float] = None
    cyr_k: list[float] = None
    cyr_d: list[float] = None
    align_on_connect: bool = True
    # align_position: list[float] = None  # e.g., [0, 0, 0, -90, 0, 0, 0]
    # align_position: list[float] = [0, 0, 0, -50, 0, 0, 0]  # e.g., [0, 0, 0, -90, 0, 0, 0]
    align_position: list[float] = field(default_factory=lambda: [0, 0, 0, -100, 0, -40, 0])
    align_timeout: float = 10.0

    # Gravity compensation parameters (A arm - leader, draggable)
    enable_gravity_compensation: bool = True
    tool_mass_a: float = 0.7 # kg (gripper + payload for A arm)
    tool_com_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 50.0])  # mm (X, Y, Z center of mass for A arm)
    tool_inertia_a: list[float] = field(default_factory=lambda: [0.004, 0.0, 0.0, 0.004, 0.0, 0.03])  # kg·m² for A arm
    tool_kine_a: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # mm, degrees for A arm

    # Gravity compensation parameters (B arm - follower)
    tool_mass_b: float = 1  # kg (gripper + payload for B arm)
    tool_com_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 50.0])  # mm (X, Y, Z center of mass for B arm)
    tool_inertia_b: list[float] = field(default_factory=lambda: [0.004, 0.0, 0.0, 0.004, 0.0, 0.03])  # kg·m² for B arm
    tool_kine_b: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # mm, degrees for B arm

    def __post_init__(self):
        if self.drag_k is None:
            object.__setattr__(self, "drag_k", [2.0] * 7)
        if self.drag_d is None:
            object.__setattr__(self, "drag_d", [0.5] * 7)
        if self.cyr_k is None:
            object.__setattr__(self, "cyr_k", [400, 400, 400, 400, 300, 300, 300])
        if self.cyr_d is None:
            object.__setattr__(self, "cyr_d", [10, 10, 10, 10, 10, 10, 10])
