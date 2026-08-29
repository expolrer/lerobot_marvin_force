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
import time
from typing import Callable, Optional

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .config_marvin_leader import MarvinLeaderConfig

logger = logging.getLogger(__name__)

_ARM_OUT_IDX = {"A": 0, "B": 1}
_JOINT_NAMES = [f"joint_{i+1}" for i in range(7)]


def _get_pool():
    from lerobot.robots.marvin.sdk_pool import get_sdk, release_sdk, get_latest_data
    return get_sdk, release_sdk, get_latest_data


class MarvinLeader(Teleoperator):
    """
    Marvin arm used as a teleoperation leader (torque + joint-drag mode).
    When the follower robot uses the same IP, the SDK connection is shared
    and a lock serialises all SDK calls.
    """

    config_class = MarvinLeaderConfig
    name = "marvin_leader"

    def __init__(self, config: MarvinLeaderConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @property
    def _arms(self) -> list[str]:
        return ["A", "B"] if self.config.use_arm == "AB" else [self.config.use_arm]

    @property
    def action_features(self) -> dict[str, type]:
        """
        Action features for A arm (leader, teleoperation device).
        Returns positions that will be used as targets for B arm (follower).
        """
        ft = {}
        # Only A arm (leader) positions
        for name in _JOINT_NAMES:
            joint_name = name.split('_')[1]  # "joint_1" -> "1"
            ft[f"joint_{joint_name}.pos"] = float
        # Add gripper feature (matches Robot's action_features)
        ft["gripper.pos"] = float
        return ft

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        get_sdk, _, get_latest_data = _get_pool()
        self._sdk, self._sdk_lock, already_connected = get_sdk(self.config.ip)

        if not already_connected:
            with self._sdk_lock:
                if not self._sdk.connect(self.config.ip):
                    raise ConnectionError(f"Failed to connect to Marvin leader at {self.config.ip}")
                time.sleep(0.1)

                # 关闭SDK底层日志打印
                self._sdk.log_switch('0')
                self._sdk.local_log_switch('0')

                self._sdk.clear_error("A")
                self._sdk.clear_error("B")
                time.sleep(0.1)
                seen = set()
                for _ in range(10):
                    data = self._sdk.subscribe()
                    fid = data["outputs"][0]["frame_serial"]
                    if fid != 0:
                        seen.add(fid)
                    if len(seen) >= 3:
                        break
                    time.sleep(0.1)
                else:
                    raise ConnectionError("Marvin leader: frames not updating.")
        else:
            logger.info(f"Reusing existing SDK connection to {self.config.ip}")

        # Configure robot for built-in teleoperation mode (based on joint_drag_gripper.py)
        with self._sdk_lock:
            # Step 1: Clear errors
            self._sdk.clear_set()
            self._sdk.clear_error('A')
            self._sdk.clear_error('B')
            self._sdk.send_cmd()
            time.sleep(0.1)

            # Step 1.5: Check if arms are powered down, if not, power them down first
            data = self._sdk.subscribe()
            a_state = data['states'][0]['cur_state']
            b_state = data['states'][1]['cur_state']

            if a_state != 0 or b_state != 0:
                logger.info(f"Arms not in idle state (A={a_state}, B={b_state}), powering down first...")
                self._sdk.clear_set()
                self._sdk.set_state(arm='A', state=0)
                self._sdk.set_state(arm='B', state=0)
                self._sdk.send_cmd()
                time.sleep(0.3)

                # Verify power down
                data = self._sdk.subscribe()
                a_state = data['states'][0]['cur_state']
                b_state = data['states'][1]['cur_state']
                logger.info(f"After power down: A state={a_state}, B state={b_state}")
            time.sleep(0.5)

            # Step 2: Switch to position mode (for gravity compensation setup)
            self._sdk.clear_set()
            self._sdk.set_state(arm='A', state=1)  # Position mode
            self._sdk.set_state(arm='B', state=1)
            self._sdk.send_cmd()
            time.sleep(0.1)

            # Verify both arms are in position mode (retry up to 1 second)
            max_retry_time = 1.0
            retry_interval = 0.1
            retry_start = time.perf_counter()
            position_mode_ok = False

            while time.perf_counter() - retry_start < max_retry_time:
                data = self._sdk.subscribe()
                a_state = data['states'][0]['cur_state']
                b_state = data['states'][1]['cur_state']

                if a_state == 1 and b_state == 1:
                    logger.info(f"Both arms in position mode: A state={a_state}, B state={b_state}")
                    position_mode_ok = True
                    break
                else:
                    logger.warning(f"Waiting for position mode: A state={a_state}, B state={b_state}")
                    time.sleep(retry_interval)

            if not position_mode_ok:
                raise RuntimeError(
                    f"Failed to enter position mode after {max_retry_time}s. "
                    f"Final states: A={a_state}, B={b_state}. "
                    f"Expected state=1 for both arms."
                )

            # Step 3: Set tool dynamics for gravity compensation
            # Note: Gravity compensation parameters are persistent and work in any mode.
            # We set them here in position mode before alignment for simplicity.
            if self.config.enable_gravity_compensation:
                # A arm (leader, draggable)
                tool_dyn_a = [
                    self.config.tool_mass_a,
                    self.config.tool_com_a[0],
                    self.config.tool_com_a[1],
                    self.config.tool_com_a[2],
                    self.config.tool_inertia_a[0],  # Ixx
                    self.config.tool_inertia_a[1],  # Ixy
                    self.config.tool_inertia_a[2],  # Ixz
                    self.config.tool_inertia_a[3],  # Iyy
                    self.config.tool_inertia_a[4],  # Iyz
                    self.config.tool_inertia_a[5],  # Izz
                ]

                # B arm (follower)
                tool_dyn_b = [
                    self.config.tool_mass_b,
                    self.config.tool_com_b[0],
                    self.config.tool_com_b[1],
                    self.config.tool_com_b[2],
                    self.config.tool_inertia_b[0],  # Ixx
                    self.config.tool_inertia_b[1],  # Ixy
                    self.config.tool_inertia_b[2],  # Ixz
                    self.config.tool_inertia_b[3],  # Iyy
                    self.config.tool_inertia_b[4],  # Iyz
                    self.config.tool_inertia_b[5],  # Izz
                ]

                self._sdk.clear_set()
                ret_a = self._sdk.set_tool(arm='A', kineParams=self.config.tool_kine_a, dynamicParams=tool_dyn_a)
                ret_b = self._sdk.set_tool(arm='B', kineParams=self.config.tool_kine_b, dynamicParams=tool_dyn_b)
                self._sdk.send_cmd()
                logger.info(f"Gravity compensation enabled:")
                logger.info(f"  A arm: mass={self.config.tool_mass_a}kg, com={self.config.tool_com_a}mm → {ret_a}")
                logger.info(f"  B arm: mass={self.config.tool_mass_b}kg, com={self.config.tool_com_b}mm → {ret_b}")
                time.sleep(0.1)

        # Mark as connected before align_and_enable_cyr (it has @check_if_not_connected)
        self._connected = True

        # Step 4: Perform alignment and enable CYR teleoperation mode
        if self.config.align_on_connect:
            init_pos = self.config.align_position if self.config.align_position else [0, 0, 0, -90, 0, 0, 0]

            result = self.align_and_enable_cyr(
                target_position=init_pos,
                record_callback=None,  # No recording during initial connection
                max_align_time=10.0,
                align_threshold=5.0,
                required_consecutive_success=3,
                drag_wait_time=0.5,
                cyr_kd_wait_time=0.8,
                cyr_on_wait_time=0.2,
            )

            logger.info(f"Initial alignment and CYR enabled successfully: "
                       f"error={result['final_error']:.3f} deg, time={result['time_elapsed']:.2f}s")

        logger.info(f"{self} connected (CYR mode active — A arm draggable, B arm follows).")

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_not_connected
    def align_and_enable_cyr(
        self,
        target_position: list[float],
        record_callback: Optional[Callable[[], bool]] = None,
        max_align_time: float = 10.0,
        align_threshold: float = 5.0,
        required_consecutive_success: int = 3,
        drag_wait_time: float = 0.5,
        cyr_kd_wait_time: float = 0.8,
        cyr_on_wait_time: float = 0.2,
    ) -> dict:
        """
        完整的对齐 + CYR 启用流程。

        流程：
        1. 切换到位置模式（state=1）
        2. 发送对齐位置命令
        3. 等待并验证对齐完成
        4. 切换到阻抗模式（state=3）+ 发送当前位置作为目标
        5. 验证阻抗模式
        6. 启用拖动模式
        7. 设置 CYR 参数
        8. 启用 CYR

        Args:
            target_position: 目标关节角度 [7 个值]，单位：度
            record_callback: 可选的录制回调函数 () -> should_continue
                           在对齐过程中每 0.1 秒调用一次，如果返回 False 则停止对齐循环
            max_align_time: 最大对齐时间（秒）
            align_threshold: 对齐误差阈值（度）
            required_consecutive_success: 连续成功次数要求
            drag_wait_time: 启用拖动模式后的等待时间（秒）
            cyr_kd_wait_time: 设置 CYR 参数后的等待时间（秒）
            cyr_on_wait_time: 启用 CYR 后的等待时间（秒）

        Returns:
            {
                "success": bool,
                "final_error": float,
                "time_elapsed": float,
                "frames_recorded": int,  # 如果有 record_callback
            }

        Raises:
            RuntimeError: 对齐失败或模式切换失败
        """
        logger.info(f"Starting align_and_enable_cyr to position: {target_position}")

        with self._sdk_lock:
            # ============ 步骤 1: 切换到位置模式 ============
            logger.info("Step 1: Switching to position mode (state=1)")
            self._sdk.clear_set()
            self._sdk.set_state(arm='A', state=1)
            self._sdk.set_state(arm='B', state=1)
            self._sdk.set_vel_acc(arm='A', velRatio=self.config.vel_ratio, AccRatio=self.config.acc_ratio)
            self._sdk.set_vel_acc(arm='B', velRatio=self.config.vel_ratio, AccRatio=self.config.acc_ratio)
            self._sdk.send_cmd()
            time.sleep(0.1)

            # 验证位置模式（带重试）
            max_retry_time = 1.0
            retry_interval = 0.1
            retry_start = time.perf_counter()
            position_mode_ok = False

            while time.perf_counter() - retry_start < max_retry_time:
                data = self._sdk.subscribe()
                a_state = data['states'][0]['cur_state']
                b_state = data['states'][1]['cur_state']

                if a_state == 1 and b_state == 1:
                    logger.info(f"Both arms in position mode: A state={a_state}, B state={b_state}")
                    position_mode_ok = True
                    break
                else:
                    logger.warning(f"Waiting for position mode: A state={a_state}, B state={b_state}")
                    time.sleep(retry_interval)

            if not position_mode_ok:
                raise RuntimeError(
                    f"Failed to enter position mode after {max_retry_time}s. "
                    f"Final states: A={a_state}, B={b_state}. "
                    f"Expected state=1 for both arms."
                )

            # ============ 步骤 2: 发送对齐位置命令 ============
            logger.info(f"Step 2: Sending alignment command to position: {target_position}")
            self._sdk.clear_set()
            self._sdk.set_joint_cmd_pose(arm='A', joints=target_position)
            self._sdk.set_joint_cmd_pose(arm='B', joints=target_position)
            self._sdk.send_cmd()
            time.sleep(2.0)  # 等待运动开始

        # ============ 步骤 3: 等待并验证对齐完成 ============
        logger.info(f"Step 3: Waiting for alignment (max {max_align_time}s, threshold {align_threshold} deg)")
        align_start_t = time.perf_counter()
        consecutive_success_count = 0
        frame_count = 0
        frames_recorded = 0
        max_error = 999.0  # 初始值

        while time.perf_counter() - align_start_t < max_align_time:
            time.sleep(0.1)
            frame_count += 1

            # 如果提供了录制回调，每帧都调用
            if record_callback is not None:
                should_continue = record_callback()
                frames_recorded += 1
                if not should_continue:
                    logger.warning("Record callback requested stop")
                    break

            # 每 10 帧检查一次对齐状态（减少 SDK 调用开销）
            if frame_count % 10 != 0:
                continue

            with self._sdk_lock:
                data = self._sdk.subscribe()
                a_pos = data["outputs"][0]["fb_joint_pos"]
                b_pos = data["outputs"][1]["fb_joint_pos"]
                a_state = data['states'][0]['cur_state']
                b_state = data['states'][1]['cur_state']

            # 计算对齐误差
            a_error = max(abs(c - t) for c, t in zip(a_pos, target_position))
            b_error = max(abs(c - t) for c, t in zip(b_pos, target_position))
            max_error = max(a_error, b_error)

            # 检查是否对齐成功（误差小于阈值 且 仍在位置模式）
            if max_error < align_threshold and a_state == 1 and b_state == 1:
                consecutive_success_count += 1
                logger.info(f"Alignment check {consecutive_success_count}/{required_consecutive_success}: "
                           f"error={max_error:.3f} deg, states A={a_state} B={b_state}")

                if consecutive_success_count >= required_consecutive_success:
                    logger.info(f"Alignment successful after {time.perf_counter() - align_start_t:.2f}s: "
                               f"A error={a_error:.3f} deg, B error={b_error:.3f} deg")
                    break
            else:
                # 重置计数器
                if consecutive_success_count > 0:
                    logger.info(f"Alignment check failed (resetting counter): error={max_error:.3f} deg, "
                               f"states A={a_state} B={b_state}")
                consecutive_success_count = 0
        else:
            # 超时：对齐失败
            with self._sdk_lock:
                data = self._sdk.subscribe()
                a_pos = data["outputs"][0]["fb_joint_pos"]
                b_pos = data["outputs"][1]["fb_joint_pos"]
                a_state = data['states'][0]['cur_state']
                b_state = data['states'][1]['cur_state']

            a_error = max(abs(c - t) for c, t in zip(a_pos, target_position))
            b_error = max(abs(c - t) for c, t in zip(b_pos, target_position))

            raise RuntimeError(
                f"Alignment timeout after {max_align_time:.1f}s. "
                f"Final errors: A={a_error:.3f} deg, B={b_error:.3f} deg. "
                f"States: A={a_state}, B={b_state}. "
                f"Threshold: {align_threshold:.1f} deg. "
                f"Please check for mechanical issues or obstructions."
            )

        # 如果有录制回调，继续录制 0.5 秒（捕捉停止动作）
        if record_callback is not None:
            logger.info("Recording stop motion for 0.5s...")
            stop_record_time = 0.5
            stop_start = time.perf_counter()
            while time.perf_counter() - stop_start < stop_record_time:
                should_continue = record_callback()
                frames_recorded += 1
                if not should_continue:
                    break
                time.sleep(0.1)

        with self._sdk_lock:
            # ============ 步骤 4: 切换到阻抗模式 + 发送当前位置作为目标 ============
            logger.info("Step 4: Switching to impedance mode (state=3) with current position as target")

            # 验证位置模式
            data = self._sdk.subscribe()
            a_state = data['states'][0]['cur_state']
            b_state = data['states'][1]['cur_state']
            if a_state != 1 or b_state != 1:
                raise RuntimeError(
                    f"Cannot proceed: arms not in position mode (state=1). "
                    f"Current states: A={a_state}, B={b_state}"
                )
            logger.info(f"Verified position mode: armA state={a_state}, armB state={b_state}")

            # 读取当前位置
            current_pos_A = data["outputs"][0]["fb_joint_pos"]
            current_pos_B = data["outputs"][1]["fb_joint_pos"]

            # 切换到阻抗模式
            self._sdk.clear_set()
            self._sdk.set_state(arm='A', state=3)
            self._sdk.set_state(arm='B', state=3)
            self._sdk.set_impedance_type(arm='A', type=1)
            self._sdk.set_impedance_type(arm='B', type=1)
            self._sdk.set_joint_cmd_pose(arm='A', joints=current_pos_A)  # ⬅️ 关键！必须发送当前位置作为目标
            self._sdk.set_joint_cmd_pose(arm='B', joints=current_pos_B)
            self._sdk.send_cmd()
            time.sleep(0.1)

            # ============ 步骤 5: 验证阻抗模式 ============
            logger.info("Step 5: Verifying impedance mode")
            data = self._sdk.subscribe()
            a_state = data['states'][0]['cur_state']
            b_state = data['states'][1]['cur_state']
            if a_state != 3 or b_state != 3:
                raise RuntimeError(
                    f"Cannot enable CYR: arms not in impedance mode (state=3). "
                    f"Current states: A={a_state}, B={b_state}"
                )
            logger.info(f"Verified impedance mode before CYR: armA state={a_state}, armB state={b_state}")

            # ============ 步骤 6: 启用拖动模式 ============
            logger.info(f"Step 6: Enabling drag mode (wait {drag_wait_time}s)")
            self._sdk.clear_set()
            self._sdk.set_drag_space(arm='A', dgType=1)  # 1: 关节空间拖动
            self._sdk.set_drag_space(arm='B', dgType=1)
            self._sdk.send_cmd()
            time.sleep(drag_wait_time)

            # ============ 步骤 7: 设置 CYR 参数 ============
            logger.info(f"Step 7: Setting CYR KD params (wait {cyr_kd_wait_time}s)")
            self._sdk.clear_set()
            ret = self._sdk.set_cyr_kd_params(K=self.config.cyr_k, D=self.config.cyr_d)
            self._sdk.send_cmd()
            logger.info(f"Set CYR KD params (K={self.config.cyr_k}, D={self.config.cyr_d}): {ret}")
            time.sleep(cyr_kd_wait_time)

            # 🆕 额外等待：让 CPU 实时任务稳定下来
            logger.info("Waiting for real-time task to stabilize...")
            time.sleep(2.0)  # 额外 2 秒缓冲

            # ============ 步骤 8: 启用 CYR ============
            logger.info(f"Step 8: Enabling CYR teleoperation mode (wait {cyr_on_wait_time}s)")
            self._sdk.clear_set()
            ret = self._sdk.set_cyr_state_on(state=1)
            self._sdk.send_cmd()
            logger.info(f"Enabled CYR teleoperation mode: {ret}")
            time.sleep(cyr_on_wait_time)

            # 验证 CYR 已启用
            data = self._sdk.subscribe()
            a_state = data['states'][0]['cur_state']
            b_state = data['states'][1]['cur_state']
            logger.info(f"After CYR on: armA state={a_state}, armB state={b_state}")

        result = {
            "success": True,
            "final_error": max_error,
            "time_elapsed": time.perf_counter() - align_start_t,
            "frames_recorded": frames_recorded,
        }
        logger.info(f"align_and_enable_cyr completed: {result}")
        return result

    @check_if_not_connected
    def disable_cyr(self) -> None:
        """
        Disable CYR teleoperation mode.
        This allows the robot to accept position commands without conflict.
        """
        with self._sdk_lock:
            self._sdk.clear_set()
            ret = self._sdk.set_cyr_state_on(state=0)
            self._sdk.send_cmd()
            logger.info(f"Disabled CYR teleoperation mode: {ret}")
            time.sleep(0.5)

    @check_if_not_connected
    @check_if_not_connected
    def get_action(self) -> dict[str, float]:
        """
        Get action from A arm (leader, teleoperation device).
        These positions will be used as targets for B arm (follower).
        """
        _, _, get_latest_data = _get_pool()
        data = get_latest_data(self.config.ip)
        if not data:
            # Fallback: no robot has subscribed yet, call subscribe once
            with self._sdk_lock:
                data = self._sdk.subscribe()

        action = {}

        # Read A arm (leader) joint positions
        out = data["outputs"][_ARM_OUT_IDX['A']]
        for i, name in enumerate(_JOINT_NAMES):
            joint_name = name.split('_')[1]  # "joint_1" -> "1"
            action[f"joint_{joint_name}.pos"] = float(out["fb_joint_pos"][i])

        # Read A arm (left) gripper position from shared data
        if "left_gripper_pos" in data:
            action["gripper.pos"] = data["left_gripper_pos"]

        return action

    def send_feedback(self, feedback: dict) -> None:
        raise NotImplementedError

    @check_if_not_connected
    def disconnect(self) -> None:
        """
        Disconnect teleoperator and release SDK reference.

        NOTE: Actual cleanup (disabling CYR mode and powering down) is handled by
        the Robot class's disconnect() method, which is called after this.
        This avoids duplicate cleanup operations that can cause pipe errors.
        """
        _, release_sdk, _ = _get_pool()
        release_sdk(self.config.ip)
        self._connected = False
        logger.info(f"{self} disconnected.")
