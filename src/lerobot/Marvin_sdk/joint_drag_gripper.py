"""
关节拖动 + 夹爪遥操作 Demo (Python 版)
对应 C++ demo: joint_drag_Gripper.cpp

功能：
  1. 连接机器人，切换到扭矩模式 + 关节阻抗 + 关节空间拖动
  2. 设置遥操作阻抗参数，开启遥操作
  3. 初始化两个 DM4310 夹爪电机（M1 主动、M2 从动）
  4. 主循环：M1 低刚度易拖动，M2 高刚度跟随 M1 位置
     M2 夹物时力矩反向反馈给 M1，产生夹持感
  5. Ctrl+C 退出清理

用法：
  python joint_drag_gripper.py
"""

import time
import signal
import sys
import os

# 保证能找到上层模块
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

try:
    from fx_robot import Marvin_Robot, DCSS
    from KM_CAN import (
        KMGripperControl, Motor, KM_Motor_Type,
        Control_Type, ERROR_MESSAGES,
    )
except ImportError:
    from fx_robot import Marvin_Robot, DCSS
    from KM_CAN import (
        KMGripperControl, Motor, KM_Motor_Type,
        Control_Type, ERROR_MESSAGES,
    )

# ──────────────────────── 全局状态 ────────────────────────
stop_flag = False


def signal_handler(sig, frame):
    global stop_flag
    print("\nReceived interrupt signal. Stopping...")
    stop_flag = True


def err_to_str(err):
    """电机错误码 → 可读字符串"""
    return ERROR_MESSAGES.get(err, f"Unknown(0x{err:X})")


# ══════════════════════════════════════════════════════════
#                          主流程
# ══════════════════════════════════════════════════════════
def main():
    global stop_flag
    signal.signal(signal.SIGINT, signal_handler)

    robot = Marvin_Robot()
    dcss = DCSS()

    # ── 1. 连接机器人 ──────────────────────────────────────
    init = robot.connect('192.168.1.190')
    if not init:
        print("failed: 端口占用，连接失败!")
        return -1

    # 关闭SDK底层日志打印
    robot.log_switch('0')
    robot.local_log_switch('0')
    time.sleep(0.1)

    # 防总线通信异常，先清错
    # 注意：clear_error 内部会自行 clear_set + send，不需要再单独 send_cmd
    time.sleep(0.1)
    robot.clear_error('A')
    robot.clear_error('B')
    time.sleep(0.1)

    # 校验通信是否生效（通过帧序号刷新判断）
    motion_tag = 0
    frame_update = 0
    for i in range(5):
        sub_data = robot.subscribe(dcss)
        frame_serial = sub_data['outputs'][0]['frame_serial']
        print(f"connect frames: {frame_serial}")
        if frame_serial != 0 and frame_update != frame_serial:
            motion_tag += 1
            frame_update = frame_serial
        time.sleep(0.1)

    if motion_tag > 0:
        print("success: 机器人连接成功!")
    else:
        print("failed: 机器人连接失败!")
        robot.release_robot()
        return -1

    # ── 2. 清错，准备切换模式 ──────────────────────────────
    robot.clear_set()
    robot.clear_error('A')
    robot.clear_error('B')
    robot.send_cmd()
    time.sleep(0.1)

    # ── 3. 切换到扭矩模式（拖动必须在扭矩模式下）──────────
    robot.clear_set()
    robot.set_state(arm='A', state=3)       # 3: 扭矩模式
    robot.set_state(arm='B', state=3)
    robot.set_impedance_type(arm='A', type=1)  # 1: 关节阻抗
    robot.set_impedance_type(arm='B', type=1)
    robot.send_cmd()
    time.sleep(0.1)

    # ── 4. 设置工具动力学参数（用于重力补偿）──────────────
    # 运动学参数: [X, Y, Z, A, B, C] (mm, degree) - 工具坐标系相对法兰
    tool_kine = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 动力学参数: [M, Cx, Cy, Cz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz]
    # M: 质量(kg), C: 质心(mm), I: 惯量(kg·m²)
    tool_dyn = [
        0.6,    # 质量 0.6kg（夹爪+负载）
        0.0,    # 质心 X
        0.0,    # 质心 Y
        50.0,   # 质心 Z（沿工具轴向50mm）
        0.004,  # Ixx
        0.0,    # Ixy
        0.0,    # Ixz
        0.004,  # Iyy
        0.0,    # Iyz
        0.03    # Izz
    ]

    robot.clear_set()
    r_tool_a = robot.set_tool(arm='A', kineParams=tool_kine, dynamicParams=tool_dyn)
    r_tool_b = robot.set_tool(arm='B', kineParams=tool_kine, dynamicParams=tool_dyn)
    robot.send_cmd()
    print(f"[DEBUG] set_tool A={r_tool_a}, B={r_tool_b} (重力补偿已配置)")
    time.sleep(0.1)

    # ── 5. 开启关节空间拖动 ────────────────────────────────
    robot.clear_set()
    r1 = robot.set_drag_space(arm='A', dgType=1)  # 1: 关节空间拖动
    r2 = robot.set_drag_space(arm='B', dgType=1)
    robot.send_cmd()
    print(f"[DEBUG] set_drag_space A={r1}, B={r2}")
    time.sleep(1.0)  # 不能遥操改成3秒，默认0.1秒

    joints = [0.000000,30.000000,0.000000,-135.000000,0.000000,-30.000000,0.000000]
    robot.clear_set()
    r3 = robot.set_joint_cmd_pose(arm='A', joints=joints)
    r4 = robot.set_joint_cmd_pose(arm='B', joints=joints)
    robot.send_cmd()
    print(f"[DEBUG] set_joint_cmd_pose A={r3}, B={r4}")
    time.sleep(3.0)

    # 验证两臂是否到达目标位置
    sub_data = robot.subscribe(dcss)
    a_pos = sub_data['outputs'][0]['fb_joint_pos']
    b_pos = sub_data['outputs'][1]['fb_joint_pos']
    print(f"[DEBUG] arm A pos: {['%.1f' % p for p in a_pos]}")
    print(f"[DEBUG] arm B pos: {['%.1f' % p for p in b_pos]}")
    print(f"[DEBUG] arm A state: {sub_data['states'][0]}")
    print(f"[DEBUG] arm B state: {sub_data['states'][1]}")

    # ── 6. 设置遥操作阻抗参数 ─────────────────────────────
    cyr_k = [400, 400, 400, 400, 300, 300, 300]
    cyr_d = [10, 10, 10, 10, 10, 10, 10]
    robot.clear_set()
    r5 = robot.set_cyr_kd_params(K=cyr_k, D=cyr_d)
    robot.send_cmd()
    print(f"[DEBUG] set_cyr_kd_params ret={r5}")
    time.sleep(0.1)

    # ── 7. 开启遥操作控制 ──────────────────────────────────
    robot.clear_set()
    r6 = robot.set_cyr_state_on(state=1)
    robot.send_cmd()
    print(f"[DEBUG] set_cyr_state_on ret={r6}")
    time.sleep(0.5)

    # 验证遥操作是否生效
    sub_data = robot.subscribe(dcss)
    a_state = sub_data['states'][0]['cur_state']
    b_state = sub_data['states'][1]['cur_state']
    print(f"[DEBUG] after CYR on: armA cur_state={a_state}, armB cur_state={b_state}")

    # ── 8. 电机初始化 ─────────────────────────────────────
    grippers = KMGripperControl(robot=robot)
    Motor1 = Motor(KM_Motor_Type.DM4310, 0x01, 0x11)  # 主动端
    Motor2 = Motor(KM_Motor_Type.DM4310, 0x02, 0x12)  # 从动端

    grippers.addMotor(Motor1)
    grippers.add_to_ch(Motor1, 'left')  # M1 通过 A 臂 485 通道

    grippers.addMotor(Motor2)
    grippers.add_to_ch(Motor2, 'right')  # M2 通过 B 臂 485 通道

    grippers.disable(Motor1)
    time.sleep(0.1)
    grippers.disable(Motor2)
    time.sleep(1.0)

    # 切换到 MIT 模式
    m1_mode_ok = grippers.switchControlMode(Motor1, Control_Type.MIT)
    m2_mode_ok = grippers.switchControlMode(Motor2, Control_Type.MIT)
    print(f"M1 MIT_MODE: {'Success' if m1_mode_ok else 'Classical'}")
    print(f"M2 MIT_MODE: {'Success' if m2_mode_ok else 'Classical'}")

    # 使能电机
    if m1_mode_ok:
        grippers.enable(Motor1)
    else:
        grippers.enable_old(Motor1, Control_Type.MIT)
    time.sleep(0.1)
    if m2_mode_ok:
        grippers.enable(Motor2)
    else:
        grippers.enable_old(Motor2, Control_Type.MIT)
    time.sleep(1.0)

    print("初始化完成！拖动M1，M2跟随；M2夹物时，M1有夹持感")
    print("Press Ctrl+C to stop...")

    # ── 软启动：读取初始位置，避免突然弹跳 ─────────────────
    grippers.recv()
    m1_init_pos = Motor1.getPosition()
    m2_init_pos = Motor2.getPosition()
    print(f"[初始位置] M1: {m1_init_pos:.3f}, M2: {m2_init_pos:.3f}")

    # 先让 M2 保持当前位置 1 秒（软启动）
    print("软启动中，M2 保持当前位置...")
    for _ in range(100):  # 100 次 × 10ms = 1 秒
        grippers.recv()
        m1_pos = Motor1.getPosition()
        m2_pos = Motor2.getPosition()

        # M1 和 M2 都保持当前位置
        grippers.controlMIT(Motor1, 0.15, 0.15, m1_pos, 0.0, 0.0)
        grippers.controlMIT(Motor2, 2.0, 0.20, m2_pos, 0.0, 0.0)  # 低刚度保持
        time.sleep(0.01)

    print("✓ 软启动完成，开始正常控制")

    # ── 9. 主循环：实时控制 ───────────────────────────────
    while not stop_flag:
        # 刷新机器人状态
        sub_data = robot.subscribe(dcss)

        # ──────── 读取 B 臂（从臂）状态信息 ────────
        b_output = sub_data['outputs'][1]

        # 关节位置和速度
        b_joints = b_output['fb_joint_pos']       # 7个关节位置（度）
        b_vel = b_output['fb_joint_vel']          # 7个关节速度（度/秒）

        # 关节力矩
        b_torque_calc = b_output['fb_joint_cToq']  # 计算力矩（Nm）
        b_torque_sensor = b_output['fb_joint_sToq']  # 传感器力矩（Nm）

        # 关节外力估计
        b_joint_force = b_output['est_joint_force']  # 7个关节外力（Nm）

        # 末端笛卡尔外力/力矩
        b_cart_fn = b_output['est_cart_fn']  # [Fx, Fy, Fz, Mx, My, Mz]
        Fx, Fy, Fz = b_cart_fn[0], b_cart_fn[1], b_cart_fn[2]
        Mx, My, Mz = b_cart_fn[3], b_cart_fn[4], b_cart_fn[5]

        # 计算末端外力大小（力的模）
        force_magnitude = (Fx**2 + Fy**2 + Fz**2)**0.5
        torque_magnitude = (Mx**2 + My**2 + Mz**2)**0.5

        # 接收电机反馈
        grippers.recv()

        m1_pos = Motor1.getPosition()
        m2_pos = Motor2.getPosition()
        m2_tau = Motor2.getTorque()

        # ==================== 核心控制逻辑 ====================
        # M2 目标位置 = M1 当前位置（校准后）
        m2_target = m1_pos + 0.1

        # 力反馈：M2 的力矩反向传给 M1（产生夹持感）
        # 系数 0.8 是反馈增益，越大夹持感越强
        feedback_tau = -m2_tau * 0.8

        # ==================== MIT 模式控制 ====================
        # M1：低刚度（易拖动） + 力反馈（有夹持感）
        grippers.controlMIT(
            Motor1,
            0.15,       # 低刚度：轻松手动拖动
            0.15,       # 阻尼：防止抖动
            m1_pos,     # 目标位置 = 当前位置（随动）
            0.0,        # 速度前馈
            0.0         # 力反馈：M2的力矩反向叠加（按需加，影响遥操手感）
        )

        # M2：高刚度（精准跟随）
        grippers.controlMIT(
            Motor2,
            8.0,        # 高刚度：精准跟随
            0.20,       # 高阻尼：无抖动
            m2_target,  # 目标位置 = M1位置 + 偏移
            0.0,
            0.0         # 无额外力矩
        )

        # ──────── 打印状态（优化显示） ────────
        print("\033[2J\033[1;1H")  # 清屏
        print("=" * 80)
        print("                   遥操作实时状态监控 - B臂（从臂）")
        print("=" * 80)

        # 夹爪状态
        print(f"\n【夹爪状态】")
        print(f"  M1(主动) 位置: {m1_pos:6.3f} rad | 反馈力矩: {feedback_tau:6.3f} Nm | {err_to_str(Motor1.getErr())}")
        print(f"  M2(从动) 位置: {m2_pos:6.3f} rad | 目标: {m2_target:6.3f} rad | 夹持力: {m2_tau:6.3f} Nm | {err_to_str(Motor2.getErr())}")

        # B臂关节位置
        print(f"\n【B臂关节位置】(度)")
        print(f"  J1:{b_joints[0]:7.2f}  J2:{b_joints[1]:7.2f}  J3:{b_joints[2]:7.2f}  J4:{b_joints[3]:7.2f}")
        print(f"  J5:{b_joints[4]:7.2f}  J6:{b_joints[5]:7.2f}  J7:{b_joints[6]:7.2f}")

        # B臂关节速度
        print(f"\n【B臂关节速度】(度/秒)")
        print(f"  J1:{b_vel[0]:7.2f}  J2:{b_vel[1]:7.2f}  J3:{b_vel[2]:7.2f}  J4:{b_vel[3]:7.2f}")
        print(f"  J5:{b_vel[4]:7.2f}  J6:{b_vel[5]:7.2f}  J7:{b_vel[6]:7.2f}")

        # B臂关节力矩（传感器）
        print(f"\n【B臂关节力矩】(Nm, 传感器)")
        print(f"  J1:{b_torque_sensor[0]:7.2f}  J2:{b_torque_sensor[1]:7.2f}  J3:{b_torque_sensor[2]:7.2f}  J4:{b_torque_sensor[3]:7.2f}")
        print(f"  J5:{b_torque_sensor[4]:7.2f}  J6:{b_torque_sensor[5]:7.2f}  J7:{b_torque_sensor[6]:7.2f}")

        # B臂关节外力估计
        print(f"\n【B臂关节外力估计】(Nm)")
        print(f"  J1:{b_joint_force[0]:7.2f}  J2:{b_joint_force[1]:7.2f}  J3:{b_joint_force[2]:7.2f}  J4:{b_joint_force[3]:7.2f}")
        print(f"  J5:{b_joint_force[4]:7.2f}  J6:{b_joint_force[5]:7.2f}  J7:{b_joint_force[6]:7.2f}")

        # B臂末端外力（重点）
        print(f"\n【B臂末端外力】(笛卡尔空间)")
        print(f"  线性力 (N)  : Fx={Fx:7.2f}  Fy={Fy:7.2f}  Fz={Fz:7.2f}  |F|={force_magnitude:7.2f}")
        print(f"  力矩 (Nm)   : Mx={Mx:7.2f}  My={My:7.2f}  Mz={Mz:7.2f}  |M|={torque_magnitude:7.2f}")

        # 外力警告
        if force_magnitude > 5.0:
            print(f"\n  ⚠️  检测到较大外力: {force_magnitude:.2f} N")

        print("\n" + "=" * 80)
        print("按 Ctrl+C 停止")

        time.sleep(0.01)  # 10ms 刷新

    # ── 10. 退出清理：关闭拖动、下电、释放连接 ─────────────
    print("Stopping motor and cleaning up...")
    grippers.disable(Motor1)
    grippers.disable(Motor2)

    robot.clear_set()
    robot.set_cyr_state_off(state=0)
    robot.send_cmd()
    time.sleep(0.1)

    robot.clear_set()
    robot.set_state(arm='A', state=0)   # 0: 空闲/下电
    robot.set_state(arm='B', state=0)
    robot.send_cmd()
    time.sleep(0.1)

    robot.release_robot()
    return 0


if __name__ == '__main__':
    main()
