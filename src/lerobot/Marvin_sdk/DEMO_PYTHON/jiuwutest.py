# 上位机：HTTP 服务器（监听下位机请求）
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import threading

# 服务器配置
HOST = '0.0.0.0'  # 允许所有设备访问
PORT = 7273        # 监听端口


class MyHandler(BaseHTTPRequestHandler):
    # 统一返回 JSON 格式
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json; charset=utf-8')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))

    # 处理 POST 请求（下位机上传数据）
    def do_POST(self):
        try:
            # 获取数据长度
            content_length = int(self.headers['Content-Length'])
            # 读取下位机发送的 JSON 数据
            post_data = self.rfile.read(content_length).decode('utf-8')
            data = json.loads(post_data)

            # ======================
            # 上位机收到数据，在这里处理
            # ======================
            print("=" * 50)
            print("【上位机收到下位机数据】")
            print(f"客户端地址: {self.client_address}")
            print(f"数据内容: {data}")
            print("=" * 50)

            # 给下位机返回响应
            response = {
                "code": 200,
                "msg": "数据接收成功",
                "data": {
                    "status": "OK",
                    "receive": data
                }
            }
            self.send_json_response(response)

        except Exception as e:
            response = {"code": 500, "msg": f"服务器错误: {str(e)}", "data": None}
            self.send_json_response(response)

    # 处理 GET 请求
    def do_GET(self):
        response = {
            "code": 200,
            "msg": "上位机服务正常运行",
            "data": {
                "server": "Python 上位机",
                "port": PORT
            }
        }
        self.send_json_response(response)

    # 关闭日志输出
    def log_message(self, format, *args):
        pass


# 启动服务器
def start_server():
    server = HTTPServer((HOST, PORT), MyHandler)
    print(f"✅ 上位机 HTTP 服务器已启动")
    print(f"监听地址: {HOST}:{PORT}")
    print(f"下位机可通过 http://上位机IP:7273连接")
    print("=" * 50)
    server.serve_forever()


if __name__ == '__main__':
    # 启动服务（后台线程运行）
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # 保持程序运行
    while True:
        input("按 Ctrl+C 关闭服务器\n")