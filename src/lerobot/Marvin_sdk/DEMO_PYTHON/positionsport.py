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

# 配置日志系统
logging.basicConfig(format='%(message)s')
logger = logging.getLogger('debug_printer')
logger.setLevel(logging.INFO)# 一键关闭所有调试打印
logger.setLevel(logging.DEBUG)  # 默认开启DEBUG级
'''初始化订阅数据的结构体'''
dcss=DCSS()

'''初始化机器人接口'''
robot=Marvin_Robot()

'''查验连接是否成功'''
init = robot.connect('192.168.10.190')
if init==0:
    logger.error('failed:端口占用，连接失败!')
    exit(0)
else:
    '''防总线通信异常,先清错'''
    time.sleep(0.5)
    robot.clear_set()
    robot.clear_error('B')
    robot.send_cmd()
    time.sleep(0.5)

    motion_tag = 0
    frame_update = None
    for i in range(5):
        sub_data = robot.subscribe(dcss)
        print(f"connect frames :{sub_data['outputs'][0]['frame_serial']}")
        if sub_data['outputs'][0]['frame_serial'] != 0 and frame_update != sub_data['outputs'][0]['frame_serial']:
            motion_tag += 1
            frame_update = sub_data['outputs'][0]['frame_serial']
        time.sleep(0.1)
    if motion_tag > 0:
        logger.info('success:机器人连接成功!')
    else:
        logger.error('failed:机器人连接失败!')
        exit(0)
'''开启日志以便检查'''
robot.log_switch('1') #全局日志开关
robot.local_log_switch('1') # 主要日志
'''清错'''
robot.clear_set()
robot.clear_error('B')
robot.send_cmd()
time.sleep(1)


'''设置扭矩模式,关节阻抗模式,速度加速度百分比'''
robot.clear_set()
robot.set_state(arm='B',state=3)#state=3扭矩模式
robot.set_impedance_type(arm='B',type=2) #type = 1 关节阻抗;type = 2 坐标阻抗;type = 3 力控
robot.set_vel_acc(arm='B',velRatio=20, AccRatio=20)
robot.send_cmd()
time.sleep(0.5)


'''阻抗参数'''
robot.clear_set()
robot.set_cart_kd_params(arm='B',K=[1500,3000,1500,40,20,40,20], D=[0.6,0.6,0.6,0.3,0.3,0.3,1], type=2) #预设参考。
robot.send_cmd()
time.sleep(0.5)

'''订阅数据查看是否设置'''
sub_data=robot.subscribe(dcss)
logger.info(f"current state{sub_data['states'][0]['cur_state']}")
logger.info(f"cmd state:{sub_data['states'][0]['cmd_state']}")
logger.info(f"arm error code:{sub_data['states'][0]['err_code']}")
logger.info(f'set vel={sub_data["inputs"][0]["joint_vel_ratio"]}, acc={sub_data["inputs"][0]["joint_acc_ratio"]}')
logger.info(f'set card k={sub_data["inputs"][0]["cart_k"][:]}, d={sub_data["inputs"][0]["cart_k"][:]}')
logger.info(f'set impedance type={sub_data["inputs"][0]["imp_type"]}')

# ------------------------------
# 1. 读取轨迹文件 F-converted_joints.txt
# ------------------------------
traj_file = "11.txt"  
 #-converted_joints3.txt上升到固定高度。
#F-converted_joints.txt，下降到放料点。
#F-converted_joints8.txt,上升到固定高度，F-converted_joints7.txt，下降到固定高度。
#F-converted_joints32.txt,上升到固定高度，F-converted_joints31.txt，下降到固定高度。
joint_traj = []
with open(traj_file, "r") as f:
    lines = f.readlines()
    for line in lines:
        vals = [v.strip() for v in line.strip().split(",") if v.strip() != ""]
        joints = [float(v) for v in vals]
        joint_traj.append(joints)
# ------------------------------
# 3. 以 100 Hz（10 ms）下发轨迹
# ------------------------------
a = 0
while True:
    a = a + 1
    control_period = 0.01  # 100 Hz

    print("开始下发轨迹（100Hz）...")

    for idx, joints in enumerate(joint_traj):

        robot.clear_set()
        robot.set_joint_cmd_pose(arm='B',joints=joints)
        robot.send_cmd()
        time.sleep(control_period)
    if a >= 1 :
        break
    print("轨迹下发完成！")
    
# ------------------------------
# 5. 可选：下伺服
# ------------------------------
robot.clear_set()
robot.set_state(arm='B', state=0)
robot.send_cmd()
print("左臂已下伺服")
