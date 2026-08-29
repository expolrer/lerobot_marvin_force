import requests
import json
import threading
import time
from datetime import datetime
import logging
import os
import sys
import websockets
import asyncio

# Windows系统使用msvcrt替代tty/termios
if os.name == 'nt':
    import msvcrt
else:
    # 如果不是Windows，提示不支持
    print("此版本仅支持Windows系统")
    sys.exit(1)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agv_control.log')
    ]
)
logger = logging.getLogger(__name__)


class AGVController:
    """AGV控制器类 - Windows版本"""
    def __init__(self, base_url, vehicle_id):
        """
        初始化AGV控制器
        :param base_url: API基础URL，例如：http://6.6.6.6:6002
        :param vehicle_id: 车体编号：18
        """
        self.base_url = base_url.rstrip('/')
        self.vehicle_id = vehicle_id
        self.session = requests.Session()
        self.session.timeout = 2
        self.websocket = None
        
        # 状态变量
        self.is_manual_mode = False
        self.is_mapping = False
        self.is_auto_mode = False
        self.running = False
        self.lock = threading.Lock()
        
        # 当前目标速度
        self.target_linear_speed = 0
        self.target_angular_speed = 0
        
        # 最后发送速度的时间
        self.last_send_time = None
        self.last_sent_speed = (0, 0)  # 记录最后发送的速度，避免重复发送
        
        # 速度配置
        self.max_linear_speed = 50  # mm/s
        self.max_angular_speed = 30  # °/s
        
        # API接口
        self.get_state_url = f"{self.base_url}/api/AMR/GetState"
        self.set_manual_mode_url = f"{self.base_url}/api/AMR/SetManualMode"
        self.set_auto_mode_url = f"{self.base_url}/api/AMR/SetAutoMode"
        self.start_mapping_url = f"{self.base_url}/api/AMR/StartMapping"
        self.stop_mapping_url = f"{self.base_url}/api/AMR/StopMapping"
        self.update_map_url = f"{self.base_url}/api/Map/UpdateFloorMap"
        self.move_to_target_url = f"{self.base_url}/api/AMR/MoveToTargetPose"
        self.move_relative_url = f"{self.base_url}/api/AMR/MoveRelative2"
        self.manual_move_url = f"{self.base_url}/api/AMR/ManualMove"

        # WebSocket URL for manual control
        self.ws_url = f"ws://{base_url.split('://')[1].split(':')[0]}:6001/api/AMR/ManualMove"
        
        # 线程控制
        self.speed_thread = None
        self.monitor_thread = None
        self.keyboard_thread = None
        
    async def connect_websocket(self):
        """连接WebSocket"""
        try:
            self.websocket = await websockets.connect(self.ws_url)
            print(f"✅ WebSocket客户端连接到服务器: {self.ws_url}")
            return True
        except Exception as e:
            print(f"❌ WebSocket连接失败: {e}")
            return False

    def get_robot_state(self):
        """获取机器人状态 - /api/AMR/GetState"""
        try:
            print(f"获取机器人状态: {self.get_state_url}")
            response = self.session.get(self.get_state_url)
            
            print(f"响应状态码: {response.status_code}")
            if response.status_code == 200:
                state_data = response.json()
                print(f"机器人状态: {json.dumps(state_data, indent=2, ensure_ascii=False)}")
                return state_data
            else:
                print(f"❌ 获取状态失败: HTTP {response.status_code}")
                return None
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return None
        except Exception as e:
            print(f"❌ 获取状态异常: {e}")
            return None

    def set_manual_mode(self):
        """设置为手动模式 - /api/AMR/SetManualMode"""
        try:
            data = {
                "id": self.vehicle_id,
                "manualMode": True
            }
            
            headers = {'Content-Type': 'application/json'}
            print(f"发送手动模式请求: {self.set_manual_mode_url}")
            print(f"请求数据: {json.dumps(data)}")

            response = self.session.post(self.set_manual_mode_url, json=data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                self.is_manual_mode = True
                self.is_auto_mode = False
                print(f"✅ 车辆{self.vehicle_id}已设置为手动模式")
                return True
            else:
                print(f"❌ 设置手动模式失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 设置手动模式异常: {e}")
            return False

    def set_auto_mode(self):
        """设置为自动模式 - /api/AMR/SetAutoMode"""
        try:
            # 先停止速度发送线程
            self.stop_speed_thread()
            
            data = {
                "id": self.vehicle_id,
                "autoMode": True
            }
            
            headers = {'Content-Type': 'application/json'}
            print(f"发送自动模式请求: {self.set_auto_mode_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(self.set_auto_mode_url, json=data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                self.is_auto_mode = True
                self.is_manual_mode = False
                print(f"✅ 车辆{self.vehicle_id}已设置为自动模式")
                return True
            else:
                print(f"❌ 设置自动模式失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 设置自动模式异常: {e}")
            return False
    
    def start_mapping(self):
        """开始建图 - /api/AMR/StartMapping"""
        try:
            data = {
                "id": 5,
                "param": {
                    "resolution": 0.02,
                    "updateDist": 0.2,
                    "updateRadian": 0.05,
                    "useRssi": True,
                    "minIntensity": 200,
                    "radiusOfCylinder": 0.042,
                    "maxLoopbackDistance": 0.6
                }
            }
            
            headers = {'Content-Type': 'application/json'}
            print(f"开始建图请求: {self.start_mapping_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(self.start_mapping_url, json=data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                self.is_mapping = True
                print(f"✅ 车辆{self.vehicle_id}开始建图")
                return True
            else:
                print(f"❌ 开始建图失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 开始建图异常: {e}")
            return False

    def stop_mapping(self):
        """停止建图 - /api/AMR/StopMapping"""
        try:
            data = {
                "id": self.vehicle_id,
                "stopMapping": True
            }
            
            headers = {'Content-Type': 'application/json'}
            print(f"停止建图请求: {self.stop_mapping_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(self.stop_mapping_url, json=data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                self.is_mapping = False
                print(f"✅ 车辆{self.vehicle_id}停止建图")
                return True
            else:
                print(f"❌ 停止建图失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 停止建图异常: {e}")
            return False

    def update_map(self, map_name="default_map", map_data=None):
        """上传地图 - /api/Map/UpdateFloorMap"""
        try:
            if map_data is None:
                map_data = {
                    "id": 6,
                    "floorId": 1,
                    "mapType": 0,
                    "data": {
                        "id": 2,
                        "uuid": "string",
                        "name": "yes",
                        "geometry": {
                            "x": 36000,
                            "y": 43000,
                            "width": 0,
                            "height": 0
                        },
                        "data": "string"
                    }
                }
            headers = {'Content-Type': 'application/json'}
            print(f"上传地图请求: {self.update_map_url}")
            print(f"请求数据: {json.dumps(map_data)}")
            
            response = self.session.post(self.update_map_url, json=map_data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                print(f"✅ 地图上传成功: {map_name}")
                return True
            else:
                print(f"❌ 地图上传失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 上传地图异常: {e}")
            return False

    def move_to_target(self,theta,x, y,z):
    
        """发送目标点任务 - /api/AMR/MoveToTargetPose"""
        try:
            data = {
                "id": self.vehicle_id,
                "theta":0,
                "x": 2000,
                "y":0,
                "z": 1 }
            
            headers = {'Content-Type': 'application/json'}
            print(f"发送目标点请求: {self.move_to_target_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(self.move_to_target_url, json=data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                print(f"✅ 目标点任务发送成功: theta={theta},x={x}, y={y}, z={z}")
                return True
            else:
                print(f"❌ 发送目标点失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 发送目标点异常: {e}")
            return False
        
    def move_relative(self,theta,x, y,z):
    
        """发送目标点任务 - /api/AMR/MoveRelative"""
        try:
            data = {
                "id": self.vehicle_id,
                "theta":0,
                "x": 1000,
                "y":0,
                "z": 0 }
            
            headers = {'Content-Type': 'application/json'}
            print(f"发送目标点请求: {self.move_relative_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(self.move_relative_url, json=data, headers=headers)

            print(f"响应状态码: {response.status_code}")
            print(f"响应内容: {response.text}")

            if response.status_code == 200:
                print(f"✅ 目标点任务发送成功: theta={theta},x={x}, y={y}, z={z}")
                return True
            else:
                print(f"❌ 发送目标点失败: HTTP {response.status_code}")
                return False      
        except requests.exceptions.ConnectionError:
            print(f"❌ 无法连接到服务器: {self.base_url}")
            return False
        except Exception as e:
            print(f"❌ 发送目标点异常: {e}")
            return False    
    async def send_speed_command(self, linear_speed, angular_speed, silent=False):
        """
        下发速度指令 - WebSocket
        :param silent: 是否静默模式（不输出警告）
        """
        if not self.is_manual_mode:
            if not silent:
                print("⚠️ 未处于手动模式，无法发送速度指令")
            return False
        try:
            data = {
                "linearSpeed": linear_speed,
                "angularSpeed": angular_speed
            }
            
            if self.websocket:
                await self.websocket.send(json.dumps(data))
                self.last_send_time = datetime.now()
                self.last_sent_speed = (linear_speed, angular_speed)
                return True
            else:
                if not silent:
                    print("❌ WebSocket未连接")
                return False
                
        except Exception as e:
            if not silent:
                print(f"❌ 速度指令发送异常: {e}")
            return False
    
    def _speed_sender_thread(self):
        """速度发送线程"""
        async def _send_vel():
            print("🚀 速度发送线程已启动 (100ms间隔)")
            
            while self.running:
                try:
                    with self.lock:
                        linear = self.target_linear_speed
                        angular = self.target_angular_speed
                    
                    # 使用 silent=True 避免在非手动模式时输出警告
                    await self.send_speed_command(linear, angular, silent=True)
                    await asyncio.sleep(0.1)  # 100ms
                    
                except Exception as e:
                    print(f"速度发送线程异常: {e}")
                    await asyncio.sleep(0.1)
        
        asyncio.run(_send_vel())
    
    def start_speed_thread(self):
        """启动速度发送线程"""
        if self.speed_thread and self.speed_thread.is_alive():
            print("速度发送线程已在运行")
            return
        
        self.running = True
        self.speed_thread = threading.Thread(target=self._speed_sender_thread, daemon=True)
        self.speed_thread.start()
        print("速度发送线程已启动")
    
    def stop_speed_thread(self):
        """停止速度发送线程"""
        # 先停止运行标志
        self.running = False
        
        # 等待线程结束
        if self.speed_thread and self.speed_thread.is_alive():
            self.speed_thread.join(timeout=1.0)
            print("速度发送线程已停止")
        
        # 重置速度
        with self.lock:
            self.target_linear_speed = 0
            self.target_angular_speed = 0
        
        # 发送停止指令
        try:
            # 创建新的事件循环或使用现有循环
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.send_speed_command(0, 0, silent=True))
            loop.close()
        except Exception as e:
            print(f"发送停止指令异常: {e}")
    
    def restart_speed_thread(self):
        """重启速度发送线程"""
        self.stop_speed_thread()
        time.sleep(0.1)  # 确保线程完全停止
        self.running = True
        self.start_speed_thread()
    
    def _timeout_monitor_thread(self):
        """超时监控线程"""
        logger.info("⏱️ 超时监控线程已启动 (500ms阈值)")
        
        while self.running:
            try:
                time.sleep(0.1)
                
                with self.lock:
                    if self.last_send_time is None:
                        continue
                    
                    time_since_last = (datetime.now() - self.last_send_time).total_seconds() * 1000
                    if time_since_last > 500 and (self.target_linear_speed != 0 or self.target_angular_speed != 0):
                        print(f"⚠️ 检测到{time_since_last:.0f}ms未收到速度指令，自动停止")
                        self.emergency_stop()
                        
            except Exception as e:
                print(f"超时监控线程异常: {e}")
    
    def emergency_stop(self):
        """紧急停止"""
        with self.lock:
            self.target_linear_speed = 0
            self.target_angular_speed = 0
        
        # 在新线程中发送停止指令，使用静默模式
        def send_stop():
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.send_speed_command(0, 0, silent=True))
                loop.close()
            except:
                pass
        
        threading.Thread(target=send_stop, daemon=True).start()
        print("🛑 紧急停止")
    
    async def start(self, auto_start_speed_thread=True):
        """启动控制器
        :param auto_start_speed_thread: 是否自动启动速度发送线程
        """
        print("正在启动AGV控制器...")
        
        # 1. 连接WebSocket
        if not await self.connect_websocket():
            print("❌ WebSocket连接失败")
            return False
        
        # 2. 获取机器人状态
        self.get_robot_state()
        time.sleep(1)
        
        # 3. 设置为手动模式
        if not self.set_manual_mode():
            print("❌ 无法设置为手动模式，启动失败")
            return False
        
        self.running = True
        self.last_send_time = datetime.now()
        
        # 4. 根据需要启动速度发送线程
        if auto_start_speed_thread:
            self.start_speed_thread()
        
        print(f"🚀 AGV控制器已启动 - 车辆ID: {self.vehicle_id}")
        return True
    
    def stop(self):
        """停止控制器"""
        print("正在停止AGV控制器...")
        
        # 停止速度发送线程
        self.stop_speed_thread()
        
        # 发送停止指令
        self.emergency_stop()
        
        # 关闭WebSocket
        if self.websocket:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.websocket.close())
                loop.close()
            except:
                pass
        
        print("🛑 AGV控制器已停止")
    
    async def set_speed(self, linear_speed, angular_speed):
        """设置目标速度"""
        # 限制速度范围
        linear_speed = max(-self.max_linear_speed, min(self.max_linear_speed, linear_speed))
        angular_speed = max(-self.max_angular_speed, min(self.max_angular_speed, angular_speed))
        
        with self.lock:
            self.target_linear_speed = linear_speed
            self.target_angular_speed = angular_speed
        
        # 立即发送一次，提高响应速度
        await self.send_speed_command(linear_speed, angular_speed)
    
    def get_current_speed(self):
        """获取当前速度"""
        with self.lock:
            return self.target_linear_speed, self.target_angular_speed


class WindowsKeyboardControl:
    """Windows系统的键盘控制类"""
    
    def __init__(self, controller):
        self.controller = controller
        self.running = False
        
        self.max_linear_speed = 100
        self.max_angular_speed = 30
        
        # 当前速度
        self.linear_speed = 0
        self.angular_speed = 0
        
        # 按键状态
        self.key_states = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
            'w': False,
            's': False,
            'a': False,
            'd': False
        }
        
        # 方向键映射
        self.arrow_keys = {
            b'H': 'up',    # 上箭头
            b'P': 'down',  # 下箭头
            b'K': 'left',  # 左箭头
            b'M': 'right'  # 右箭头
        }
    
    def stop(self):
        """停止键盘控制"""
        self.running = False
        # 重置所有按键状态
        for k in self.key_states:
            self.key_states[k] = False
        print("键盘控制已停止")
    
    def _get_key_windows(self):
        """Windows系统获取按键"""
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key == b'\xe0':  # 方向键或功能键
                key2 = msvcrt.getch()
                return ('arrow', key2)
            return ('char', key)
        return None
    
    async def _update_speed(self):
        """根据按键状态更新速度"""
        # 计算线速度（前进/后退）
        if (self.key_states['up'] or self.key_states['w']) and not (self.key_states['down'] or self.key_states['s']):
            self.linear_speed = self.max_linear_speed
        elif (self.key_states['down'] or self.key_states['s']) and not (self.key_states['up'] or self.key_states['w']):
            self.linear_speed = -self.max_linear_speed
        else:
            self.linear_speed = 0
        
        # 计算角速度（左转/右转）
        if (self.key_states['left'] or self.key_states['a']) and not (self.key_states['right'] or self.key_states['d']):
            self.angular_speed = -self.max_angular_speed
        elif (self.key_states['right'] or self.key_states['d']) and not (self.key_states['left'] or self.key_states['a']):
            self.angular_speed = self.max_angular_speed
        else:
            self.angular_speed = 0
        
        # 设置速度
        await self.controller.set_speed(self.linear_speed, self.angular_speed)
        self._display_status()
  
    def _display_status(self):
        # 方向指示
        direction = []
        if self.linear_speed > 0:
            direction.append("前进")
        elif self.linear_speed < 0:
            direction.append("后退")
        
        if self.angular_speed > 0:
            direction.append("右转")
        elif self.angular_speed < 0:
            direction.append("左转")
        if not direction:
            direction = ["停止"]
        print(f"  运动状态: {' + '.join(direction)}")
    
    async def run(self):
        """运行键盘控制"""
        self.running = True
        self._display_status()
        
        print("\n控制已启动，按方向键或WASD控制小车")
        try:
            while self.running:
                key_info = self._get_key_windows()
                
                if key_info:
                    key_type, key = key_info
                    
                    if key_type == 'char':
                        # 字符键按下
                        if key == b' ':  # 空格键
                            print("⚠️ 紧急停止")
                            for k in self.key_states:
                                self.key_states[k] = False
                            await self._update_speed()
                        
                        elif key == b'q' or key == b'Q':  # Q键退出键盘控制
                            print("👋 退出键盘控制")
                            break
                        
                        elif key == b'w' or key == b'W':
                            self.key_states['w'] = True
                            self.key_states['up'] = False
                            await self._update_speed()
                            
                        elif key == b's' or key == b'S':
                            self.key_states['s'] = True
                            self.key_states['down'] = False
                            await self._update_speed()
                        
                        elif key == b'a' or key == b'A':
                            self.key_states['a'] = True
                            self.key_states['left'] = False
                            await self._update_speed()
                        
                        elif key == b'd' or key == b'D':
                            self.key_states['d'] = True
                            self.key_states['right'] = False
                            await self._update_speed()
                        
                        elif key == b'\r':  # 回车键 - 释放所有按键
                            for k in self.key_states:
                                self.key_states[k] = False
                            await self._update_speed()
                    
                    elif key_type == 'arrow':
                        # 方向键按下
                        if key in self.arrow_keys:
                            arrow_name = self.arrow_keys[key]
                            # 重置对应的WASD键
                            if arrow_name == 'up':
                                self.key_states['w'] = False
                            elif arrow_name == 'down':
                                self.key_states['s'] = False
                            elif arrow_name == 'left':
                                self.key_states['a'] = False
                            elif arrow_name == 'right':
                                self.key_states['d'] = False
                            
                            self.key_states[arrow_name] = True
                            await self._update_speed()
                
                # 检查按键释放（简化处理）
                await asyncio.sleep(0.1)
                # 重置所有按键状态（简化处理）
                for k in self.key_states:
                    self.key_states[k] = False

        except Exception as e:
            logger.error(f"键盘控制异常: {e}")
        finally:
            # 停止小车
            self.stop()
            self.controller.emergency_stop()


def test_connection(base_url, vehicle_id):
    """测试连接"""
    print(f"测试连接到{base_url}")
    test_data = {
        "id": vehicle_id,
        "manualMode": True
    }
    try:
        response = requests.post(
            f"{base_url}/api/AMR/SetManualMode",
            json=test_data,
            timeout=3
        )
        print(f"连接测试结果: {response.status_code}")
        if response.status_code == 200:
            print("✅ 连接成功")
            return True
        else:
            print(f"❌ 连接失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 连接异常: {e}")
        return False

async def main():
    """主函数"""
    print("=" * 60)
    print("AGV 控制系统 - Windows版本")
    print("=" * 60)
    # 配置参数
    API_BASE_URL = input("请输入API地址 (默认: http://6.6.6.6:6002): ").strip()
    if not API_BASE_URL:
        API_BASE_URL = "http://6.6.6.6:6002"
    
    VEHICLE_ID = input("请输入车体编号 (默认: 18): ").strip()
    if not VEHICLE_ID:
        VEHICLE_ID = "18"
    VEHICLE_ID = int(VEHICLE_ID)
    
    # 测试连接
    if not test_connection(API_BASE_URL, VEHICLE_ID):
        print("无法连接到服务器，请检查网络和API地址")
        input("按回车键退出...")
        return
    
    # 创建控制器
    controller = AGVController(API_BASE_URL, VEHICLE_ID)
    
    try:
        # 启动控制器（包括WebSocket连接和手动模式设置）
        if not await controller.start(auto_start_speed_thread=True):
            logger.error("❌ 控制器启动失败")
            input("按回车键退出...")
            return
        
        while True:
            print("\n" + "=" * 60)
            print("请选择操作模式:")
            print("1. 获取机器人状态")
            print("2. 手动模式 + 键盘控制")
            print("3. 开始建图")
            print("4. 停止建图")
            print("5. 上传地图")
            print("6. 切换自动模式")
            print("7. 发送目标点任务")
            print("8. 发送相对移动目标")
            print("9. 退出程序")
            print("=" * 60)
            
            choice = input("请输入选项 (1-8): \n").strip()
            
            if choice == '1':
                # 获取机器人状态
                controller.get_robot_state()
                
            elif choice == '2':
                # 手动模式 + 键盘控制
                if not controller.is_manual_mode:
                    controller.set_manual_mode()
                    time.sleep(1)
                
                # 确保速度发送线程在运行
                if not controller.running:
                    controller.running = True
                    controller.start_speed_thread()
                
                keyboard = WindowsKeyboardControl(controller)
                await keyboard.run()
                
            elif choice == '3':
                # 开始建图
                if not controller.is_manual_mode:
                    print("请先切换到手动模式")
                else:
                    controller.start_mapping()
                    
            elif choice == '4':
                # 停止建图
                controller.stop_mapping()
                
            elif choice == '5':
                # 上传地图
                map_name = input("请输入地图名称 (默认: map_" + time.strftime("%Y%m%d_%H%M%S") + "): ").strip()
                if not map_name:
                    map_name = "map_" + time.strftime("%Y%m%d_%H%M%S")
                controller.update_map(map_name)
                
            elif choice == '6':
                # 切换自动模式
                print("切换到自动模式...")
                
                # 停止速度发送线程
                controller.stop_speed_thread()
                
                # 设置为自动模式
                controller.set_auto_mode()
                
            elif choice == '7':
                # 发送目标点任务
                if not controller.is_auto_mode:
                    print("请先切换到自动模式")
                else:
                    try:
                        theta = 0
                        x = 0
                        y = 0
                        z = 1
                        theta = float(theta) if theta else 0
                        controller.move_to_target(theta,x, y,z)
                    except ValueError:
                        print("❌ 输入格式错误，请输入数字")
            elif choice == '8':
                # 发送目标点任务
                    try:
                        theta = 0
                        x = 0
                        y = 0
                        z = 0
                        theta = float(theta) if theta else 0
                        controller.move_relative(theta,x, y,z)
                    except ValueError:
                        print("❌ 输入格式错误，请输入数字")   
            elif choice == '9':
                # 退出程序
                print("正在退出程序...")
                break 
            else:
                print("❌ 无效选项，请重新选择")
            
            # 暂停一下，让用户看到结果
            if choice != '2':  # 键盘控制模式下不需要暂停
                input("\n按回车键继续...")
                
    except KeyboardInterrupt:
        print("\n收到中断信号")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        controller.stop()
        print("程序结束")


if __name__ == "__main__":
    asyncio.run(main())