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

import logging
import math
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from .config_marvin import MarvinRobotConfig
from .sdk_pool import get_sdk, get_gripper_imports, is_shared, release_sdk, set_latest_data, _refcount

logger = logging.getLogger(__name__)

_ARM_OUT_IDX = {"A": 0, "B": 1}
_JOINT_NAMES = [f"joint_{i+1}" for i in range(7)]


class MarvinRobot(Robot):
    """
    Marvin dual-arm 7DOF collaborative robot.

    Communicates via UDP to a native SDK (`libMarvinSDK.so` / `.dll`).
    When a MarvinLeader teleoperator uses the same IP, the SDK connection is
    shared automatically; a lock serialises all SDK calls.
    """

    config_class = MarvinRobotConfig
    name = "marvin"

    def __init__(self, config: MarvinRobotConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._gripper = None
        self._motor_left = None
        self._motor_right = None
        self._last_joint_positions: list[float] | None = None
        self._last_joint_forces: list[float] | None = None
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _arms(self) -> list[str]:
        return ["A", "B"] if self.config.use_arm == "AB" else [self.config.use_arm]

    def _arm_prefix(self, arm: str) -> str:
        if self.config.use_arm != "AB":
            return ""
        return "left_" if arm == "A" else "right_"

    @property
    def _motors_ft(self) -> dict[str, type]:
        """
        Motor features for observation and action.
        In teleoperation mode:
        - Observation: B arm (follower, executing robot)
        - Action: B arm targets (same dimensions)

        Force feedback (if enabled):
        - joint_vel: Joint velocities
        - joint_torque: Joint sensor torques
        - joint_force: Joint space external forces
        - cart_force: End-effector Cartesian space forces
        """
        ft = {}
        # Only B arm (follower robot) is included in observation/action
        # A arm is the leader (teleoperation device) and not part of robot state

        # ========== Always include: Joint positions ==========
        for i in range(7):
            ft[f"joint_{i+1}.pos"] = float

        # ========== Optional: Force feedback data ==========
        if self.config.use_force_feedback:
            for i in range(7):
                if "joint_vel" in self.config.force_feedback_types:
                    ft[f"joint_{i+1}.vel"] = float

                if "joint_torque" in self.config.force_feedback_types:
                    ft[f"joint_{i+1}.torque"] = float

                if "joint_force" in self.config.force_feedback_types:
                    ft[f"joint_{i+1}.force"] = float

            # End-effector Cartesian space forces [Fx, Fy, Fz, Mx, My, Mz]
            if "cart_force" in self.config.force_feedback_types:
                ft["cart_force.fx"] = float
                ft["cart_force.fy"] = float
                ft["cart_force.fz"] = float
                ft["cart_force.mx"] = float
                ft["cart_force.my"] = float
                ft["cart_force.mz"] = float

        # ========== Gripper position ==========
        if self.config.use_gripper:
            ft["gripper.pos"] = float

        return ft

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        # Action only contains positions for B arm (follower)
        ft = {}
        for name in _JOINT_NAMES:
            ft[f"joint_{name.split('_')[1]}.pos"] = float
        if self.config.use_gripper:
            ft["gripper.pos"] = float
        return ft

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True, skip_configure: bool = False) -> None:
        self._sdk, self._sdk_lock, already_connected = get_sdk(self.config.ip)

        if not already_connected:
            with self._sdk_lock:
                if not self._sdk.connect(self.config.ip):
                    raise ConnectionError(f"Failed to connect to Marvin at {self.config.ip}")
                time.sleep(0.1)

                # 关闭SDK底层日志打印
                self._sdk.log_switch('0')
                self._sdk.local_log_switch('0')

                self._sdk.clear_error("A")
                self._sdk.clear_error("B")
                time.sleep(0.1)
                self._verify_connection()
        else:
            logger.info(f"Reusing existing SDK connection to {self.config.ip}")

        # If SDK already connected, the robot is already configured by the leader
        # (in teleoperation mode). Don't reconfigure, just use the existing setup.
        # Also skip configuration if skip_configure=True (e.g., when teleop will connect next).
        if not already_connected and not skip_configure:
            with self._sdk_lock:
                for arm in self._arms:
                    self._configure_arm(arm)
        elif skip_configure:
            logger.info(f"Skipping arm configuration (teleop will configure)")


        if self.config.use_gripper:
            self._init_gripper()

        for cam in self.cameras.values():
            cam.connect()

        self._connected = True

        # Populate shared data pool for teleoperation alignment
        # This ensures the leader can read the follower's position during its connect()
        with self._sdk_lock:
            data = self._sdk.subscribe()
        set_latest_data(self.config.ip, data)

        logger.info(f"{self} connected.")

    def _verify_connection(self) -> None:
        # Must be called with lock held
        seen = set()
        for _ in range(10):
            data = self._sdk.subscribe()
            fid = data["outputs"][0]["frame_serial"]
            if fid != 0:
                seen.add(fid)
            if len(seen) >= 3:
                return
            time.sleep(0.1)
        raise ConnectionError("Marvin: frames not updating — check network connection.")

    def _configure_arm(self, arm: str) -> None:
        # Must be called with lock held
        arm_idx = _ARM_OUT_IDX[arm]

        # Read current position before configuring any mode to prevent jumps
        data = self._sdk.subscribe()
        current_joints = data["outputs"][arm_idx]["fb_joint_pos"]

        # Set tool gravity compensation (if enabled)
        # This should be done before switching to any control mode
        if self.config.enable_gravity_compensation:
            tool_mass = self.config.tool_mass_a if arm == "A" else self.config.tool_mass_b
            tool_com = self.config.tool_com_a if arm == "A" else self.config.tool_com_b
            tool_inertia = self.config.tool_inertia_a if arm == "A" else self.config.tool_inertia_b
            tool_kine = self.config.tool_kine_a if arm == "A" else self.config.tool_kine_b

            tool_dyn = [
                tool_mass,
                tool_com[0], tool_com[1], tool_com[2],
                tool_inertia[0],  # Ixx
                tool_inertia[1],  # Ixy
                tool_inertia[2],  # Ixz
                tool_inertia[3],  # Iyy
                tool_inertia[4],  # Iyz
                tool_inertia[5],  # Izz
            ]

            self._sdk.clear_set()
            ret = self._sdk.set_tool(arm=arm, kineParams=tool_kine, dynamicParams=tool_dyn)
            self._sdk.send_cmd()
            logger.info(f"Tool compensation for arm {arm}: mass={tool_mass}kg, com={tool_com}mm → {ret}")
            time.sleep(0.1)

        if self.config.control_mode == "position":
            # Position control mode (state=1): robot tracks target position with high stiffness
            self._sdk.clear_set()
            self._sdk.set_state(arm=arm, state=1)
            self._sdk.set_vel_acc(arm=arm, velRatio=self.config.vel_ratio, AccRatio=self.config.acc_ratio)
            self._sdk.set_joint_cmd_pose(arm=arm, joints=current_joints)
            self._sdk.send_cmd()
        else:
            # Impedance control mode (state=3): robot uses K/D params for compliant control
            # CRITICAL: Must send initial target position to prevent undefined state oscillation
            self._sdk.clear_set()
            self._sdk.set_state(arm=arm, state=3)
            self._sdk.set_impedance_type(arm=arm, type=1)
            self._sdk.set_vel_acc(arm=arm, velRatio=self.config.vel_ratio, AccRatio=self.config.acc_ratio)
            self._sdk.set_joint_kd_params(arm=arm, K=self.config.impedance_k, D=self.config.impedance_d)
            # Send current position as initial target to avoid jump/oscillation
            self._sdk.set_joint_cmd_pose(arm=arm, joints=current_joints)
            self._sdk.send_cmd()
        time.sleep(0.3)

    def _init_gripper(self) -> None:
        KMGripperControl, Motor, KM_Motor_Type, Control_Type = get_gripper_imports()
        self._gripper = KMGripperControl(robot=self._sdk)
        self._motor_left = Motor(KM_Motor_Type.DM4310, self.config.left_gripper_id, self.config.left_gripper_master_id)
        self._motor_right = Motor(KM_Motor_Type.DM4310, self.config.right_gripper_id, self.config.right_gripper_master_id)
        self._gripper.addMotor(self._motor_left)
        self._gripper.add_to_ch(self._motor_left, "left")
        self._gripper.addMotor(self._motor_right)
        self._gripper.add_to_ch(self._motor_right, "right")
        for motor in (self._motor_left, self._motor_right):
            self._gripper.disable(motor)
        time.sleep(0.5)
        for motor in (self._motor_left, self._motor_right):
            self._gripper.switchControlMode(motor, Control_Type.MIT)
            self._gripper.enable(motor)
        time.sleep(0.5)

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """
        Get observation from B arm (follower robot).
        A arm is the teleoperation device (leader), not part of robot observation.
        """
        with self._sdk_lock:
            data = self._sdk.subscribe()

        obs = {}

        # Only read B arm (follower) state
        out = data["outputs"][_ARM_OUT_IDX['B']]

        # ========== Always include: Joint positions ==========
        for i in range(7):
            obs[f"joint_{i+1}.pos"] = float(out["fb_joint_pos"][i])

        # ========== Optional: Force feedback data ==========
        if self.config.use_force_feedback:
            for i in range(7):
                if "joint_vel" in self.config.force_feedback_types:
                    obs[f"joint_{i+1}.vel"] = float(out["fb_joint_vel"][i])

                if "joint_torque" in self.config.force_feedback_types:
                    obs[f"joint_{i+1}.torque"] = float(out["fb_joint_sToq"][i])

                if "joint_force" in self.config.force_feedback_types:
                    obs[f"joint_{i+1}.force"] = float(out["est_joint_force"][i])

            # End-effector Cartesian space forces [Fx, Fy, Fz, Mx, My, Mz]
            if "cart_force" in self.config.force_feedback_types:
                cart_fn = out["est_cart_fn"]
                obs["cart_force.fx"] = float(cart_fn[0])
                obs["cart_force.fy"] = float(cart_fn[1])
                obs["cart_force.fz"] = float(cart_fn[2])
                obs["cart_force.mx"] = float(cart_fn[3])
                obs["cart_force.my"] = float(cart_fn[4])
                obs["cart_force.mz"] = float(cart_fn[5])

        # ========== Gripper position ==========
        if self.config.use_gripper and self._gripper is not None:
            # Receive gripper feedback
            self._gripper.recv()
            # Only B arm (right) gripper
            right_gripper_pos = float(self._motor_right.getPosition())
            obs["gripper.pos"] = right_gripper_pos

            # Save both gripper positions to shared data for Leader to read A arm gripper
            left_gripper_pos = float(self._motor_left.getPosition())
            data["left_gripper_pos"] = left_gripper_pos
            data["right_gripper_pos"] = right_gripper_pos

        set_latest_data(self.config.ip, data)

        # ========== Camera images ==========
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()

        self._last_joint_positions = [obs[f"joint_{i+1}.pos"] for i in range(7)]
        if self.config.use_force_feedback and "joint_force" in self.config.force_feedback_types:
            self._last_joint_forces = [obs[f"joint_{i+1}.force"] for i in range(7)]

        return obs

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """
        Send action to B arm (follower robot).
        In teleoperation mode, A arm is draggable (leader) and B arm follows via CYR.
        In policy mode, B arm executes predicted actions.
        """
        action = dict(action)

        # Dynamically check whether a teleop also holds a reference to this SDK.
        # record.py connects robot first, then teleop — so at connect() time the
        # refcount is still 1.  By the time send_action runs the teleop has
        # connected too, so refcount > 1 means we are in teleoperation mode.
        already_connected = is_shared(self.config.ip)

        # Initialize gripper update counter for rate limiting
        if not hasattr(self, '_gripper_update_counter'):
            self._gripper_update_counter = 0

        self._gripper_update_counter += 1

        # CRITICAL: Gripper control at 30Hz causes EtherCAT communication loss.
        # Reduce gripper update rate to every 5 frames (~6Hz at 30fps control loop)
        # to prevent CPU starvation and maintain EtherCAT real-time performance.
        gripper_update_divisor = self.config.gripper_update_divisor
        if gripper_update_divisor < 1:
            raise ValueError("gripper_update_divisor must be at least 1")

        if self.config.use_gripper and self._gripper is not None:
            if self._gripper_update_counter % gripper_update_divisor == 0:
                if already_connected:
                    # ============ Teleoperation mode ============
                    # Left gripper (A arm leader) is draggable, right gripper (B arm follower) follows left
                    # Read current positions from hardware (real-time dragging)
                    m1_pos = self._motor_left.getPosition()
                    m2_target = m1_pos + 0.1  # Right follows left with offset

                    # M1: Low stiffness (easy to drag by hand)
                    self._gripper.controlMIT(
                        self._motor_left,
                        0.15,      # Low stiffness
                        0.15,      # Low damping
                        m1_pos,    # Target = current (free dragging)
                        0.0,
                        0.0
                    )

                    # M2: High stiffness (precise following)
                    self._gripper.controlMIT(
                        self._motor_right,
                        8.0,       # High stiffness
                        0.20,      # High damping
                        m2_target, # Follow M1
                        0.0,
                        0.0
                    )
                else:
                    # ============ Policy inference mode ============
                    # Use gripper target from action (model predictions for B arm)
                    gripper_target = action.get("gripper.pos")

                    if gripper_target is not None:
                        gripper_target = float(gripper_target)
                        if not math.isfinite(gripper_target):
                            raise ValueError(f"Refusing non-finite gripper target: {gripper_target}")
                        low = self.config.min_policy_gripper_pos
                        high = self.config.max_policy_gripper_pos
                        if low is not None and high is not None and low >= high:
                            raise ValueError("min_policy_gripper_pos must be smaller than max_policy_gripper_pos")
                        if low is not None:
                            gripper_target = max(gripper_target, low)
                        if high is not None:
                            gripper_target = min(gripper_target, high)
                        action["gripper.pos"] = gripper_target
                        # Only control B arm (right) gripper in policy mode
                        self._gripper.controlMIT(
                            self._motor_right,
                            8.0,
                            0.20,
                            gripper_target,
                            0.0,
                            0.0
                        )

        if not already_connected:
            # Standalone mode (policy inference): send position commands to B arm
            joints = [float(action[f"joint_{i+1}.pos"]) for i in range(7)]
            if not all(math.isfinite(value) for value in joints):
                raise ValueError(f"Refusing non-finite Marvin joint target: {joints}")

            force_limit = self.config.max_policy_abs_joint_force
            force_limits = self.config.max_policy_abs_joint_force_by_joint
            if force_limits is not None and len(force_limits) != 7:
                raise ValueError("max_policy_abs_joint_force_by_joint must contain seven values")
            force_exceeded = self._last_joint_forces is not None and (
                (force_limit is not None and max(abs(value) for value in self._last_joint_forces) > force_limit)
                or (
                    force_limits is not None
                    and any(
                        abs(value) > limit
                        for value, limit in zip(self._last_joint_forces, force_limits, strict=True)
                    )
                )
            )
            if force_exceeded:
                if self._last_joint_positions is None:
                    raise RuntimeError("Force limit exceeded before a valid Marvin observation was captured")
                logger.error(
                    "External joint force exceeded the configured envelope; holding the latest observed pose"
                )
                joints = list(self._last_joint_positions)

            step_limit = self.config.max_policy_joint_step_deg
            if step_limit is not None and self._last_joint_positions is not None:
                if step_limit <= 0:
                    raise ValueError("max_policy_joint_step_deg must be positive or None")
                joints = [
                    current + min(max(target - current, -step_limit), step_limit)
                    for target, current in zip(joints, self._last_joint_positions, strict=True)
                ]
            for i, target in enumerate(joints, start=1):
                action[f"joint_{i}.pos"] = target

            with self._sdk_lock:
                self._sdk.clear_set()
                # Only control B arm
                self._sdk.set_joint_cmd_pose(arm='B', joints=joints)
                self._sdk.send_cmd()
        # else:
        #     # CRITICAL: In CYR teleoperation mode, DO NOT send any position commands
        #     # to B arm — it is already following A arm via hardware-level CYR control.
        #     # Sending additional commands will conflict with CYR and cause the arm to "fly".
        #     pass

        return {k: v for k, v in action.items() if k in self.action_features}

    @check_if_not_connected
    def align_to_position(self, target_position: list[float], wait_time: float = 3.0) -> None:
        """
        Align both arms to a target position.

        This method temporarily switches to position control mode, moves both arms
        to the target position, then switches back to the original control mode.

        CRITICAL: This should ONLY be called when teleoperation is active.
        The teleoperator must disable CYR mode before calling this method.

        Args:
            target_position: Joint angles in degrees [j1, j2, j3, j4, j5, j6, j7]
            wait_time: Time to wait for movement to complete (seconds)
        """
        logger.info(f"Aligning to position: {target_position}")

        with self._sdk_lock:
            # Step 1: Switch to position control mode for both arms
            for arm in self._arms:
                self._sdk.clear_set()
                self._sdk.set_state(arm=arm, state=1)  # Position control
                self._sdk.set_vel_acc(arm=arm, velRatio=self.config.vel_ratio, AccRatio=self.config.acc_ratio)
                self._sdk.send_cmd()
            time.sleep(0.3)

            # Step 2: Send target position commands
            for arm in self._arms:
                self._sdk.clear_set()
                self._sdk.set_joint_cmd_pose(arm=arm, joints=target_position)
                self._sdk.send_cmd()

            # Step 3: Wait for movement to complete
            time.sleep(wait_time)

            # Step 4: Verify alignment
            data = self._sdk.subscribe()
            for arm in self._arms:
                arm_idx = _ARM_OUT_IDX[arm]
                current = data["outputs"][arm_idx]["fb_joint_pos"]
                error = max(abs(c - t) for c, t in zip(current, target_position))
                logger.info(f"Arm {arm} alignment error: {error:.3f} deg")

            # Step 5: Switch back to impedance mode for both arms
            for arm in self._arms:
                # Read current position to prevent jump
                current_joints = data["outputs"][_ARM_OUT_IDX[arm]]["fb_joint_pos"]

                self._sdk.clear_set()
                self._sdk.set_state(arm=arm, state=3)  # Impedance control
                self._sdk.set_impedance_type(arm=arm, type=1)
                self._sdk.set_vel_acc(arm=arm, velRatio=self.config.vel_ratio, AccRatio=self.config.acc_ratio)
                self._sdk.set_joint_kd_params(arm=arm, K=self.config.impedance_k, D=self.config.impedance_d)
                self._sdk.set_joint_cmd_pose(arm=arm, joints=current_joints)
                self._sdk.send_cmd()
            time.sleep(0.3)

        logger.info("Alignment completed")

    @check_if_not_connected
    def align_b_arm_to_position(self, target_position: list[float], max_align_time: float = 5.0, align_threshold: float = 2.0) -> dict:
        """
        Align only B arm (follower) to a target position in policy mode.
        A arm is not moved.

        This method checks the current state and only proceeds if B arm is in
        state 1 (position mode) or state 3 (impedance mode). The arm stays in
        its current mode during alignment.

        Args:
            target_position: Joint angles in degrees [j1, j2, j3, j4, j5, j6, j7]
            max_align_time: Maximum time to wait for alignment (seconds)
            align_threshold: Maximum allowed error in degrees

        Returns:
            dict with keys: success, final_error, time_elapsed

        Raises:
            RuntimeError: If B arm is not in state 1 or 3, or if alignment fails
        """
        logger.info(f"Aligning B arm to position: {target_position}")

        with self._sdk_lock:
            # Step 1: Check B arm state (must be 1 or 3)
            data = self._sdk.subscribe()
            b_state = data['states'][1]['cur_state']

            if b_state not in (1, 3):
                raise RuntimeError(
                    f"Cannot align B arm: invalid state {b_state}. "
                    f"Expected state 1 (position mode) or 3 (impedance mode). "
                    f"Current state may be 0 (powered down) or another invalid state."
                )

            logger.info(f"B arm in valid state: {b_state} ({'position' if b_state == 1 else 'impedance'}) - proceeding with alignment")

            # Step 2: Send target position command (works in both state 1 and state 3)
            logger.info(f"Sending alignment command to B arm: {target_position}")
            self._sdk.clear_set()
            self._sdk.set_joint_cmd_pose(arm='B', joints=target_position)
            self._sdk.send_cmd()
            time.sleep(2.0)

        # Step 3: Wait and verify alignment (outside lock to allow other threads)
        logger.info(f"Waiting for B arm alignment (max {max_align_time}s, threshold {align_threshold} deg)")
        align_start_t = time.perf_counter()
        consecutive_success_count = 0
        required_consecutive_success = 3
        max_error = 999.0

        while time.perf_counter() - align_start_t < max_align_time:
            time.sleep(0.1)

            with self._sdk_lock:
                data = self._sdk.subscribe()
                b_pos = data["outputs"][1]["fb_joint_pos"]
                b_state = data['states'][1]['cur_state']

            # Calculate alignment error
            b_error = max(abs(c - t) for c, t in zip(b_pos, target_position))
            max_error = b_error

            # Check alignment success (accept both state 1 and state 3)
            if b_error < align_threshold and b_state in (1, 3):
                consecutive_success_count += 1
                logger.info(f"Alignment check {consecutive_success_count}/{required_consecutive_success}: "
                           f"error={b_error:.3f} deg, state={b_state}")

                if consecutive_success_count >= required_consecutive_success:
                    logger.info(f"B arm alignment successful after {time.perf_counter() - align_start_t:.2f}s: "
                               f"error={b_error:.3f} deg")
                    break
            else:
                if consecutive_success_count > 0:
                    logger.info(f"Alignment check failed (resetting counter): error={b_error:.3f} deg, state={b_state}")
                consecutive_success_count = 0
        else:
            # Timeout: alignment failed
            with self._sdk_lock:
                data = self._sdk.subscribe()
                b_pos = data["outputs"][1]["fb_joint_pos"]
                b_state = data['states'][1]['cur_state']

            b_error = max(abs(c - t) for c, t in zip(b_pos, target_position))

            raise RuntimeError(
                f"B arm alignment timeout after {max_align_time:.1f}s. "
                f"Final error: {b_error:.3f} deg. "
                f"State: {b_state}. "
                f"Threshold: {align_threshold:.1f} deg. "
                f"Please check for mechanical issues or obstructions."
            )

        result = {
            "success": True,
            "final_error": max_error,
            "time_elapsed": time.perf_counter() - align_start_t,
        }
        logger.info(f"B arm alignment completed: {result}")
        return result

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        Disconnect and cleanup: disable gripper motors, disable teleoperation mode,
        and optionally power down the robot arms.

        Based on the cleanup sequence in the official Marvin SDK demo.
        """
        # Step 1: Disable gripper motors
        if self.config.use_gripper and self._gripper is not None:
            for motor in (self._motor_left, self._motor_right):
                try:
                    self._gripper.disable(motor)
                except Exception:
                    pass

        # Step 2: Check if we're the last user of the SDK
        # Only the last user should perform cleanup to avoid duplicate operations
        is_last_user = _refcount.get(self.config.ip, 0) <= 1
        logger.info(f"Disconnect: is_last_user={is_last_user}, refcount={_refcount.get(self.config.ip, 0)}, "
                    f"disable_torque_on_disconnect={self.config.disable_torque_on_disconnect}")

        # Step 3: Release SDK connection with cleanup flags
        # Cleanup is performed inside release_sdk() BEFORE calling release_robot()
        # to avoid pipe errors if the SDK crashes during release
        release_sdk(
            self.config.ip,
            disable_cyr=is_last_user,  # Disable teleoperation if we're the last user
            power_down=is_last_user and self.config.disable_torque_on_disconnect
        )
        self._connected = False

        # Step 4: Disconnect cameras
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
