'''
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import websockets
import json
import threading
import time
import msvcrt


class AGVController:
    def __init__(self, base_url, vehicle_id):
        # ... 其他初始化代码保持不变 ...
        self.websocket = None
        self.loop = None  # 保存事件循环
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.vehicle_id = 5
        
    async def connect(self):
        """建立websocket连接"""
        try:
            self.websocket = await websockets.connect("ws://6.6.6.6:6001/api/AMR/ManualMove")
            self.loop = asyncio.get_running_loop()  # 保存当前事件循环
            print("✅ 控制客户端连接到服务器")
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            
    async def send_speed_command(self, linear_speed, angular_speed, angle_offset=0):
        """下发速度指令"""
        
        try:
            data = {
                "id": self.vehicle_id,
                "linearSpeed": linear_speed,
                "angularSpeed": angular_speed,
                "angleOffset": angle_offset
            }
            
            if self.websocket:
                await self.websocket.send(json.dumps(data))
                self.last_sent_speed = (linear_speed, angular_speed)
                return True
            else:
                print("❌ websocket未连接")
                return False
                
        except Exception as e:
            print(f"❌ 速度指令发送异常: {e}")
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
    

    
    # 创建控制器
    controller = AGVController(API_BASE_URL, VEHICLE_ID)
    await controller.connect()
    
    try:
        last_kbhit = False
        
        while True:
            kbhit = msvcrt.kbhit()  # 检查是否有按键

            if kbhit:  # 检查是否有按键
                key = msvcrt.getch()  # 获取按键
                if key == b'w' or key == b'W':
                    await controller.send_speed_command(50, 0, 0)
                elif key == b's' or key == b'S':
                    await controller.send_speed_command(-50, 0, 0)
                elif key == b'a' or key == b'A':
                    await controller.send_speed_command(0, -5, 0)
                elif key == b'd' or key == b'D':
                    await controller.send_speed_command(0, 5, 0)
            
            elif(last_kbhit):
                await controller.send_speed_command(0, 0, 0)
            last_kbhit = kbhit
            await asyncio.sleep(0.3)
            
        
        
            
    except KeyboardInterrupt:
        print("收到中断信号")
    except Exception as e:
        print(f"程序异常: {e}")
    finally:
        controller.stop()
        print("程序结束")

 
    # 运行主程序
asyncio.run(main())
'''



import asyncio
import websockets
import json
import msvcrt
import time

class SimpleAGVController:
    def __init__(self, vehicle_id=5):
        self.vehicle_id = vehicle_id
        self.websocket = None
        self.current_linear = 0
        self.current_angular = 0
        
    async def connect(self):
        self.websocket = await websockets.connect("ws://6.6.6.6:6001/api/AMR/ManualMove")
        print("✅ 连接成功")
        
    async def send_current_speed(self):
        """发送当前速度"""
        if self.websocket:
            data = {
                "id": self.vehicle_id,
                "linearSpeed": self.current_linear,
                "angularSpeed": self.current_angular,
                "angleOffset": 0
            }
            await self.websocket.send(json.dumps(data))
            
    async def run(self):
        await self.connect()
        print("WASD控制，空格键停止，Q退出")
        
        last_sent_time = time.time()
        send_interval = 0.05  # 50ms发送一次
        
        try:
            last_kbhit = False
            while True:
                # 检查按键
                kbhit = msvcrt.kbhit()
                if kbhit:
                    key = msvcrt.getch()
                    if key == b'q' or key == b'Q':
                        break
                    
                    # 实时更新速度
                    if key == b'w' or key == b'W':
                        self.current_linear = 50
                    elif key == b's' or key == b'S':
                        self.current_linear = -50
                    elif key == b'a' or key == b'A':
                        self.current_angular = 5
                    elif key == b'd' or key == b'D':
                        self.current_angular = -5
                    elif key == b' ':  # 空格键停止
                        self.current_linear = 0
                        self.current_angular = 0
                    
                    # 立即发送
                    await self.send_current_speed()
                    #print(f"速度: L={self.current_linear:3d}, A={self.current_angular:3d}")
                    last_sent_time = time.time()
                elif last_kbhit:
                    self.current_linear = 0
                    self.current_angular = 0
                    await self.send_current_speed()
                    #print(f"速度: L={self.current_linear:3d}, A={self.current_angular:3d}")
                    last_sent_time = time.time()

                last_kbhit = kbhit

                
                # 按固定频率发送，确保服务器持续收到命令
                current_time = time.time()
                if current_time - last_sent_time >= send_interval:
                    await self.send_current_speed()
                    last_sent_time = current_time
                
                await asyncio.sleep(0.10)  # 10ms轮询
                
        finally:
            # 发送停止命令
            self.current_linear = 0
            self.current_angular = 0
            await self.send_current_speed()
            await self.websocket.close()

# 运行
asyncio.run(SimpleAGVController(5).run())