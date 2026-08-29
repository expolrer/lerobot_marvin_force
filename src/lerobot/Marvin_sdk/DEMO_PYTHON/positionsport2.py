import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
current_file_path = os.path.abspath(__file__)
current_path = os.path.dirname(current_file_path)
from SDK_PYTHON.fx_robot import Marvin_Robot, DCSS
import time
import logging

logging.basicConfig(format='%(message)s')
logger = logging.getLogger('debug_printer')
logger.setLevel(logging.DEBUG)

# ------------------------------
# 1. 读取轨迹文件 z_trace.txt
# ------------------------------
traj_file = "F-converted_joints8.txt"
#F-converted_joints8.txt,上升到固定高度，F-converted_joints7.txt，下降到固定高度。
#F-converted_joints32.txt,上升到固定高度，F-converted_joints31.txt，下降到固定高度。
# if not os.path.exists(traj_file):
#     raise FileNotFoundError("未找到 z_trace.txt！")

joint_traj = []
# with open(traj_file, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         vals = line.strip().split(",")
#         joints = [float(v) for v in vals]
#         joint_traj.append(joints)
# with open(traj_file, "r") as f:
#     lines = f.readlines()
#     for line in lines:
#         vals = line.strip().split(",")  # 修改这里
#         joints = [float(v) for v in vals]
#         joint_traj.append(joints)
# print(f"轨迹点数量：{len(joint_traj)}")

with open(traj_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        vals = [v.strip() for v in line.strip().split(",") if v.strip() != ""]
        joints = [float(v) for v in vals]
        joint_traj.append(joints)



# ------------------------------
# 2. 连接机器人
# ------------------------------
robot = Marvin_Robot()
robot.connect('192.168.1.190')

robot.log_switch('1')
robot.local_log_switch('1')


robot.clear_set()
robot.clear_error('B')
robot.send_cmd()
time.sleep(0.5)

# 3. 设置位置模式 + 速度
robot.clear_set()
robot.set_state(arm='B', state=1)  # 位置模式
robot.set_vel_acc(arm='B', velRatio=5, AccRatio=5)
robot.send_cmd()
time.sleep(0.2)

# ------------------------------
# 4. 以 100 Hz（10 ms）下发轨迹
# ------------------------------
a = 0
while True:
    a = a + 1
    control_period = 0.01  # 100 Hz

    print("开始下发轨迹（100Hz）...")

    for idx, joints in enumerate(joint_traj):

        robot.clear_set()
        robot.set_joint_cmd_pose(arm='B', joints=joints)
        robot.send_cmd()

        time.sleep(control_period)
    if a >= 1:
        break
    print("轨迹下发完成！")

# ------------------------------
# 5. 可选：下伺服
# ------------------------------
robot.clear_set()
robot.set_state(arm='B', state=0)
robot.send_cmd()
print("左臂已下伺服")
