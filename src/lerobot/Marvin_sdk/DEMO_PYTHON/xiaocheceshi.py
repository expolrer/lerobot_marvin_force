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
        :param vehicle_id: 车体编号
        """
        self.base_url = base_url.rstrip('/')
        self.vehicle_id = vehicle_id
        self.session = requests.Session()
        self.session.timeout = 2

        self.websocket = None
        
        
        # 状态变量

        self.is_manual_mode = False
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
        self.set_manual_mode_url = f"{self.base_url}/api/AMR/SetManualMode"
        self.set_speed_url = f"{self.base_url}/api/AMR/ManualMove"
        
        # 线程控制
        self.speed_thread = None
        self.monitor_thread = None
        self.keyboard_thread = None
        
    async def connect(self):
        try:
            self.websocket = await websockets.connect("ws://6.6.6.6:6001/api/AMR/ManualMove")
            print("控制客户端连接到服务器")
        except Exception as e:
            print(f"连接失败: {e}") 


    def set_manual_mode(self):
        """设置为手动模式 - /api/AMR/SetManualMode"""
        try:
            data = {
                "id": self.vehicle_id,
                "manualMode": True  # True表示手动模式
            }
            
            headers = {'Content-Type': 'application/json'}
            print(f"发送手动模式请求: {self.set_manual_mode_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(
                self.set_manual_mode_url,#"http://6.6.6.6:6002/api/AMR/SetManualMode"
                data=json.dumps(data),
                headers=headers
            )

            print(f"响应状态码: {response.status_code} 响应内容: {response.text}")

            
            if response.status_code == 200:
                self.is_manual_mode = True
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
    
    async def send_speed_command(self, linear_speed, angular_speed, angle_offset=0):
        """
        下发速度指令 - /api/AMR/SetSpeed
        要求格式:
        {
            "id": 5,
            "linearSpeed": 600,
            "angularSpeed": 30,
            "angleOffset": 0
        }
        """

        if not self.is_manual_mode:
            logger.warning("⚠️ 未处于手动模式，无法发送速度指令")
            return False
        
        # 检查是否和上次发送的速度相同，避免重复发送
        current_speed = (linear_speed, angular_speed)
        #if current_speed == self.last_sent_speed:
        #    return True
        
        try:

            #self.websocket = await websockets.connect("ws://6.6.6.6:6001")
            data = {
                "id": self.vehicle_id,
                "linearSpeed": linear_speed,
                "angularSpeed": angular_speed,
                "angleOffset": angle_offset
            }
            await self.websocket.send(json.dumps(data))
            #print(f"请求数据: {json.dumps(data)}")
            

            '''
            headers = {'Content-Type': 'application/json'}

            print(f"发送手动模式请求: {self.set_speed_url}")
            print(f"请求数据: {json.dumps(data)}")
            
            response = self.session.post(
                self.set_speed_url,
                data=json.dumps(data),
                headers=headers
            )

            print(f"响应状态码: {response.status_code} 响应内容: {response.text}")

            
            if response.status_code == 200:
                with self.lock:
                    
                    self.last_send_time = datetime.now()
                    self.last_sent_speed = current_speed
                    print(f"📤 last_send_time={self.last_send_time}")
                
                if linear_speed != 0 or angular_speed != 0:
                    print(f"📤 速度指令: 线速度={linear_speed}mm/s, 角速度={angular_speed}°/s")
                return True
            else:
                print(f"❌ 速度指令发送失败: {response.status_code}")
                return False
            '''
                
        except Exception as e:
            print(f"❌ 速度指令发送异常: {e}")
            return False
    
    def _speed_sender_thread(self):
        async def _send_vel():
            """
            速度发送线程 - 每100ms下发一次
            要求：100ms一次下发速度
            """
            print("🚀 速度发送线程已启动 (100ms间隔)")
            
            while self.running:
                try:
                    with self.lock:
                        linear = self.target_linear_speed
                        angular = self.target_angular_speed
                    await self.send_speed_command(linear, angular, 0)
                    #print("send_speed_command")
                    #time.sleep(0.1)  # 100ms
                    await asyncio.sleep(0.1)
                except Exception as e:
                    print(f"速度发送线程异常: {e}")
                    #time.sleep(0.1)
                    await asyncio.sleep(0.1)
        asyncio.run(_send_vel())
    
    def _timeout_monitor_thread(self):
        """
        超时监控线程 - 500ms内收不到速度则自动停止
        要求：如果500ms内收不到速度则会自动减速停止
        """
        logger.info("⏱️ 超时监控线程已启动 (500ms阈值)")
        
        while self.running:
            try:
                time.sleep(0.1)  # 100ms检查一次
                
                with self.lock:
                    if self.last_send_time is None:
                        continue
                        
                    time_since_last = (datetime.now() - self.last_send_time).total_seconds() * 1000
                    
                    if time_since_last > 500:
                        print(f"⚠️ 检测到{time_since_last:.0f}ms未收到速度指令，自动停止")
                        # 在新线程中发送停止指令
                        threading.Thread(target=self.emergency_stop, daemon=True).start()
            except Exception as e:
                print(f"超时监控线程异常: {e}")
    
    def emergency_stop(self):
        """
        紧急停止 - 发送速度为0
        要求：主动停止需要发送线速度角速度均为0
        """
        with self.lock:
            self.target_linear_speed = 0
            self.target_angular_speed = 0
            self.last_sent_speed = (0, 0)
        
        self.send_speed_command(0, 0, 0)
        print("🛑 紧急停止")
    
    async def start(self):
        """
        启动控制器
        步骤：
        1. 设置为手动模式
        2. 启动速度发送线程（100ms）
        3. 启动超时监控线程（500ms）
        """
        print("正在启动AGV控制器...")
        
        # 1. 设置为手动模式
        if not self.set_manual_mode():
            print("❌ 无法设置为手动模式，启动失败")
            return False
        
        self.running = True
        self.last_send_time = datetime.now()
        
        # 2. 启动速度发送线程
        self.speed_thread = threading.Thread(target=self._speed_sender_thread, daemon=True)
        self.speed_thread.start()
        
        # 3. 启动超时监控线程
        #self.monitor_thread = threading.Thread(target=self._timeout_monitor_thread, daemon=True)
        #self.monitor_thread.start()
        
        print(f"🚀 AGV控制器已启动 - 车辆ID: {self.vehicle_id}")
        return True
    
    def stop(self):
        """停止控制器"""
        print("正在停止AGV控制器...")
        self.running = False
        
        # 发送停止指令
        self.emergency_stop()
        
        # 等待线程结束
        if self.speed_thread and self.speed_thread.is_alive():
            self.speed_thread.join(timeout=1)
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1)
        
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
        await self.send_speed_command(linear_speed, angular_speed, 0)
    
    def get_current_speed(self):
        """获取当前速度"""
        with self.lock:
            return self.target_linear_speed, self.target_angular_speed


class WindowsKeyboardControl:
    """Windows系统的键盘控制类 - 使用msvcrt"""
    
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
        #print(f"  linear_speed: {self.linear_speed}  angular_speed: {self.angular_speed}")
        #self._display_status()
    
    def _display_status(self):
        """显示状态"""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 60)
        print("              AGV 键盘控制系统 (Windows)")
        print("=" * 60)
        print("\n📋 控制说明:")
        print("  W / ↑ - 前进     S / ↓ - 后退")
        print("  A / ← - 左转     D / → - 右转")
        print("  空格键 - 紧急停止    Q - 退出程序")
        print("\n" + "-" * 60)
        print("📊 当前状态:")
        print(f"  线速度: {self.linear_speed:4d} mm/s")
        print(f"  角速度: {self.angular_speed:4d} °/s")
        
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
        
        # 按键状态
        print("\n🔘 按键状态:")
        print(f"  W/↑: {'●' if self.key_states['up'] or self.key_states['w'] else '○'}  ", end="")
        print(f"  S/↓: {'●' if self.key_states['down'] or self.key_states['s'] else '○'}  ", end="")
        print(f"  A/←: {'●' if self.key_states['left'] or self.key_states['a'] else '○'}  ", end="")
        print(f"  D/→: {'●' if self.key_states['right'] or self.key_states['d'] else '○'}")
        
        print("\n🔒 安全状态:")
        print(f"  手动模式: {'✅' if self.controller.is_manual_mode else '❌'}")
        print(f"  速度发送: 100ms间隔")
        print(f"  超时保护: 500ms阈值")
        print("=" * 60)
        print("按 Q 退出程序")
    
    async def run(self):
        """运行键盘控制"""
        self.running = True
        #self._display_status()
        
        print("\n控制已启动，按方向键或WASD控制小车")

        
        try:
            while self.running:
                key_info = self._get_key_windows()
                
                if key_info:
                    key_type, key = key_info
                    #print(f"  key_type: {key_type}  key: {key}")

                    if key_type == 'char':
                        # 字符键处理
                        if key == b' ':  # 空格键
                            logger.info("⚠️ 紧急停止")
                            for k in self.key_states:
                                self.key_states[k] = False
                            await self._update_speed()
                        
                        elif key == b'q' or key == b'Q':  # Q键退出
                            logger.info("👋 退出程序")
                            break
                        
                        elif key == b'w' or key == b'W':
                            self.key_states['w'] = True
                            await self._update_speed()

                        elif key == b's' or key == b'S':
                            self.key_states['s'] = True
                            await self._update_speed()
                        
                        elif key == b'a' or key == b'A':
                            self.key_states['a'] = True
                            await self._update_speed()
                        
                        elif key == b'd' or key == b'D':
                            self.key_states['d'] = True
                            await self._update_speed()
                    
                    elif key_type == 'arrow':
                        # 方向键处理
                        if key in self.arrow_keys:
                            arrow_name = self.arrow_keys[key]
                            self.key_states[arrow_name] = True
                            await self._update_speed()

                
                # 简单处理按键释放 - 假设按键按下后很快释放
                # 在实际应用中，可能需要更复杂的按键状态管理

                await asyncio.sleep(0.1)
                
                # 重置所有按键状态（简化处理）
                # 注意：这样会失去持续按键的效果
                for k in self.key_states:
                    self.key_states[k] = False
                #self._update_speed()
                
                #time.sleep(0.05)
                
        except Exception as e:
            logger.error(f"键盘控制异常: {e}")
        finally:
            self.controller.emergency_stop()


def test_connection(base_url, vehicle_id):
    """测试连接"""
    logger.info(f"测试连接到 {base_url}")
    
    # 测试手动模式设置
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
        logger.error(f"❌ 连接异常: {e}")
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
    
    VEHICLE_ID = input("请输入车体编号 (默认: 5): ").strip()
    if not VEHICLE_ID:
        VEHICLE_ID = "5"
    VEHICLE_ID = int(VEHICLE_ID)
    
    # 测试连接
    
    if not test_connection(API_BASE_URL, VEHICLE_ID):
        print("无法连接到服务器，请检查网络和API地址")
        input("按回车键退出...")
        return
    
    # 创建控制器
    controller = AGVController(API_BASE_URL, VEHICLE_ID)
    await controller.connect()
    
    try:
        # 启动控制器
        if not await controller.start():
            logger.error("❌ 控制器启动失败")
            input("按回车键退出...")
            return
        
        print("\n请选择控制模式:")
        print("1. 键盘控制 (实时控制)")
        print("2. 命令控制 (输入命令)")
        print("3. 仅启动控制器 (后台运行)")
        print("4. 退出")

        choice = input("请选择 (1/2/3/4): ").strip()
        
        if choice == '1':
            keyboard = WindowsKeyboardControl(controller)
            await keyboard.run()

        elif choice == '2':
            print("\n可用命令:")
            print("  f - 前进")
            print("  b - 后退")
            print("  l - 左转")
            print("  r - 右转")
            print("  s - 停止")
            print("  q - 退出")
            print("  fr3 - 前进3秒 (f+时间)")
            print("  bl2 - 后退2秒")
            print("  lr3 - 左转3秒")
            print("  rr2 - 右转2秒")
            
            while True:
                cmd = input("\n输入命令: ").strip().lower()
                
                if cmd == 'f':
                    controller.set_speed(600, 0)
                    print("前进")
                elif cmd == 'b':
                    controller.set_speed(-600, 0)
                    print("后退")
                elif cmd == 'l':
                    controller.set_speed(0, -30)
                    print("左转")
                elif cmd == 'r':
                    controller.set_speed(0, 30)
                    print("右转")
                elif cmd == 's':
                    controller.emergency_stop()
                    print("停止")
                elif cmd == 'q':
                    break
                elif len(cmd) >= 3 and cmd[0] in ['f', 'b', 'l', 'r'] and cmd[1] in ['r', 'l']:
                    # 处理时间命令，如 fr3 (前进右转3秒)
                    direction = cmd[0]
                    turn = cmd[1]
                    try:
                        duration = float(cmd[2:])
                        if direction == 'f':
                            linear = 300
                        elif direction == 'b':
                            linear = -300
                        else:
                            linear = 0
                        
                        if turn == 'r':
                            angular = 15
                        elif turn == 'l':
                            angular = -15
                        else:
                            angular = 0
                        
                        controller.set_speed(linear, angular)
                        print(f"执行 {duration}秒")
                        time.sleep(duration)
                        controller.emergency_stop()
                    except:
                        print("命令格式错误")
                else:
                    print("未知命令")
        elif choice == '3':
            logger.info("控制器已启动，按Ctrl+C停止")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            logger.info("退出程序")
        
            
    except KeyboardInterrupt:
        logger.info("收到中断信号")
    except Exception as e:
        logger.error(f"程序异常: {e}")
    finally:
        controller.stop()
        logger.info("程序结束")


'''
if __name__ == "__main__":
    # 检查依赖
    try:
        import requests
    except ImportError:
        print("请安装requests库: pip install requests")
        sys.exit(1)
'''    
    # 运行主程序
asyncio.run(main())