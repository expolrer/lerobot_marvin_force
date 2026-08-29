"""ROS2 订阅 /joint_states 的位置控制脚本.

修改：
1. 原先固定往返控制改为从 /joint_states 话题获取目标位置 (pos_des)。
2. 使用 Timer 以固定频率(默认50Hz)下发 control_pos_force。
3. 加入软限位 (基于初始化时扫描得到的 min/max)，防止越界。
4. 优雅关闭：节点销毁时关闭串口。
"""

import math
import time
import threading
from collections import deque

import numpy as np
import matplotlib.pyplot as plt
# 安装后入口点方式运行需要包内相对导入；源码直接运行 fallback
try:
    from .KM_CAN import *  # type: ignore  # noqa
    from .fx_robot import Marvin_Robot  # type: ignore  # noqa
except ImportError:
    from KM_CAN import *  # type: ignore  # noqa
    from fx_robot import Marvin_Robot  # type: ignore  # noqa

# 全局（保持与原结构兼容）
Motor1, Motor2, grippers = None, None, KMGripperControl()
Motor1_min_pos, Motor1_max_pos = 0.0, 0.0
Motor2_min_pos, Motor2_max_pos = 0.0, 0.0
auto_calibrate = False


class GripperMotorNode():
    def __init__(self):
        self.q1 = 0.0
        self.q2 = 0.0  # Motor2 目标位置缓存
        self.vel_cmd = 0
        self.i_des = 0
        self.current_desired_pos_m1 = 0.0
        self.current_desired_pos_m2 = 0.0
        self.have_joint_state = False
        self.left_id = 1
        self.right_id = 2
        self.robot = Marvin_Robot()
        self.robot.connect('192.168.10.190')
        self.robot.log_switch('0') #全局日志开关
        self.robot.local_log_switch('0') # 主要日志
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._latest = {
            "q_cmd_L": 0.0,
            "q_cmd_R": 0.0,
            "q_fb_L": float('nan'),
            "q_fb_R": float('nan'),
        }
        self._init_plot()
        self._init_motors()
        self._start_threads()
        print('Gripper motor node started. Waiting joint states...')

    def reset_motors_callback(self, request, response):
        grippers.disable(Motor1)
        grippers.disable(Motor2)
        time.sleep(0.1)
        grippers.enable(Motor1)
        grippers.enable(Motor2)
        response.success = True
        response.message = "Gripper motors have been reset."
        return response

    def control_timer_callback(self):
        self.q1 = 0.5+math.sin(time.time()*5)*0.5  # 模拟数据，范围0~1
        self.q2 = 0.5+math.sin(time.time()*5)*0.5
        grippers.controlMIT(Motor1, 10.0, 0.12, self.q1, self.vel_cmd, self.i_des)
        grippers.controlMIT(Motor2, 10.0, 0.12, self.q2, self.vel_cmd, self.i_des)
        with self._state_lock:
            self._latest["q_cmd_L"] = self.q1
            self._latest["q_cmd_R"] = self.q2

    # ---------------- 初始化与标定 ----------------
    def _init_motors(self):
        global Motor1, Motor2, grippers
        grippers = KMGripperControl(robot=self.robot)
        Motor1 = Motor(KM_Motor_Type.DM4310, self.left_id, self.left_id+0x10)
        Motor2 = Motor(KM_Motor_Type.DM4310, self.right_id, self.right_id+0x10)

        grippers.addMotor(Motor1)
        grippers.add_to_ch(Motor1, 'left')

        grippers.addMotor(Motor2)
        grippers.add_to_ch(Motor2, 'right')
        
        grippers.enable(Motor1)
        # self.motor1_send()
        grippers.enable(Motor2)
        # self.motor2_send()
        print("Gripper motors initialized.")
        # MotorControl1.set_zero_position(Motor1)
        # MotorControl1.set_zero_position(Motor2)
    def _clamp(self, v, vmin, vmax):
        v = v*(vmax - vmin) + vmin
        return v

    # ---------------- 线程 ----------------
    def _start_threads(self):
        self._control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self._feedback_thread = threading.Thread(target=self._feedback_loop, daemon=True)
        self._control_thread.start()
        self._feedback_thread.start()

    def _control_loop(self):
        # period = 1.0 / 1000.0
        while not self._stop_event.is_set():
            # t0 = time.time()
            self.control_timer_callback()
            # dt = time.time() - t0
            time.sleep(0.01)

    def _feedback_loop(self):
        while not self._stop_event.is_set():
            grippers.recv()
            q_fb_R = Motor2.getPosition()
            q_fb_L = Motor1.getPosition()
            print(f"GripperR pos_fb:{q_fb_R:.4f}")
            print(f"GripperL pos_fb:{q_fb_L:.4f}")
            with self._state_lock:
                self._latest["q_fb_L"] = q_fb_L
                self._latest["q_fb_R"] = q_fb_R
            # time.sleep(0.0001)

    # ---------------- 绘图 ----------------
    def _init_plot(self):
        self.plot_len = 600
        self.plot_stride = 2
        self._plot_counter = 0
        self._ts = deque(maxlen=self.plot_len)
        self._cmd_L = deque(maxlen=self.plot_len)
        self._fb_L = deque(maxlen=self.plot_len)
        self._cmd_R = deque(maxlen=self.plot_len)
        self._fb_R = deque(maxlen=self.plot_len)

        plt.ion()
        self.fig, (self.ax_L, self.ax_R) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
        self.line_cmd_L, = self.ax_L.plot([], [], label='L cmd')
        self.line_fb_L, = self.ax_L.plot([], [], label='L fb')
        self.line_cmd_R, = self.ax_R.plot([], [], label='R cmd')
        self.line_fb_R, = self.ax_R.plot([], [], label='R fb')
        self.ax_L.set_ylabel('Position')
        self.ax_R.set_ylabel('Position')
        self.ax_R.set_xlabel('Time (s)')
        self.ax_L.legend(loc='upper right')
        self.ax_R.legend(loc='upper right')
        self.fig.tight_layout()

    def _update_plot(self, q_cmd_L, q_cmd_R, q_fb_L, q_fb_R):
        self._plot_counter += 1
        if self._plot_counter % self.plot_stride != 0:
            return
        t = time.time()
        self._ts.append(t)
        self._cmd_L.append(q_cmd_L)
        self._fb_L.append(q_fb_L)
        self._cmd_R.append(q_cmd_R)
        self._fb_R.append(q_fb_R)

        ts_rel = np.asarray(self._ts) - self._ts[0]
        self.line_cmd_L.set_data(ts_rel, self._cmd_L)
        self.line_fb_L.set_data(ts_rel, self._fb_L)
        self.line_cmd_R.set_data(ts_rel, self._cmd_R)
        self.line_fb_R.set_data(ts_rel, self._fb_R)
        self.ax_L.relim()
        self.ax_L.autoscale_view()
        self.ax_R.relim()
        self.ax_R.autoscale_view()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        # plt.pause(0.001)

    
    def destroy_node(self):  # 资源释放
        self._stop_event.set()
        if hasattr(self, "_control_thread"):
            self._control_thread.join(timeout=1.0)
        if hasattr(self, "_feedback_thread"):
            self._feedback_thread.join(timeout=1.0)
        try:
            grippers.disable(Motor1)
            grippers.disable(Motor2)
        except Exception:
            pass

if __name__ == '__main__':  # 兼容直接运行
    gripper_node = GripperMotorNode()
    try:
        while True:
            with gripper_node._state_lock:
                q_cmd_L = gripper_node._latest["q_cmd_L"]
                q_cmd_R = gripper_node._latest["q_cmd_R"]
                q_fb_L = gripper_node._latest["q_fb_L"]
                q_fb_R = gripper_node._latest["q_fb_R"]
            gripper_node._update_plot(q_cmd_L, q_cmd_R, q_fb_L, q_fb_R)
            # time.sleep(0.001)
    except KeyboardInterrupt:
        gripper_node.destroy_node()
