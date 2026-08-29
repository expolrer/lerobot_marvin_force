import requests
import json

# ====================== 全局配置：车辆信息 ======================
# 在这里统一配置你的车辆ID和名称，所有接口都会自动携带
VEHICLE_ID = "ubot_001"          # 车辆唯一ID，比如设备序列号、编号
# ==============================================================

# 机器人服务的基础地址（对应图中的 http://10.8.8.8:8283）
BASE_URL = "http://10.8.8.8:7273"

# 通用POST请求封装（统一处理请求、响应、异常 + 自动携带车辆信息）
def send_post_request(endpoint: str, payload: dict = None, timeout: int = 10) -> dict:
    """
    发送POST请求到机器人接口，自动携带车辆ID和名称1
    :param endpoint: 接口路径，如 "/mission"
    :param payload: 原始请求体（字典格式，会自动转为JSON）
    :param timeout: 超时时间（秒）
    :return: 接口返回的JSON数据，失败返回错误信息
    """
    url = f"{BASE_URL}{endpoint}"
    headers = {"Content-Type": "application/json"}

    # 🔴 核心修改：自动在请求体中添加车辆ID和名称
    # 如果没有传入payload，就创建一个新的字典；否则在原有字典基础上追加
    final_payload = payload.copy() if payload else {}
    final_payload.update({
        "vehicle_id": VEHICLE_ID
    })
    
    try:
        response = requests.post(
            url,
            json=final_payload,
            headers=headers,
            timeout=timeout
        )
        # 检查HTTP状态码是否为200
        response.raise_for_status()
        return {
            "success": True,
            "status_code": response.status_code,
            "data": response.json() if response.content else {}
        }
    except requests.exceptions.RequestException as e:
        return {
            "success": False,
            "error": str(e),
            "status_code": response.status_code if 'response' in locals() else None
        }

# ------------------------------
# 1. 下发任务给机器人（/mission）
# ------------------------------
def send_mission_to_robot(mission_data: dict) -> dict:
    """
    向机器人发送任务，自动携带车辆ID和名称
    :param mission_data: 任务数据（根据机器人协议定义，示例为导航任务）
    """
    return send_post_request("/mission", payload=mission_data)

# ------------------------------
# 2. 暂停当前任务（/mission/pause）
# ------------------------------
def pause_mission() -> dict:
    """暂停机器人正在执行的任务，自动携带车辆ID和名称"""
    return send_post_request("/mission/pause")

# ------------------------------
# 3. 恢复暂停的任务（/mission/resume）
# ------------------------------
def resume_mission() -> dict:
    """恢复机器人暂停的任务，自动携带车辆ID和名称"""
    return send_post_request("/mission/resume")

# ------------------------------
# 4. 取消当前任务（/mission/cancel）
# ------------------------------
def cancel_mission() -> dict:
    """取消机器人正在执行的任务，自动携带车辆ID和名称"""
    return send_post_request("/mission/cancel")

# ------------------------------
# 示例：完整的任务控制流程
# ------------------------------
if __name__ == "__main__":
    # 1. 定义一个示例任务（请根据你的机器人实际协议修改！）
    example_mission = {
        "robot_id": "ubot_001",
        "target":"pose",
        "x": 10.5, 
        "y": 20.3,
        "t": 90.0,   
    }
    print("=== 1. 下发任务给机器人（自动携带车辆信息） ===")
    send_result = send_mission_to_robot(example_mission)
    print(json.dumps(send_result, indent=2, ensure_ascii=False))

    if send_result["success"]:
        print("\n=== 2. 暂停任务（自动携带车辆信息） ===")
        pause_result = pause_mission()
        print(json.dumps(pause_result, indent=2, ensure_ascii=False))

        print("\n=== 3. 恢复任务（自动携带车辆信息） ===")
        resume_result = resume_mission()
        print(json.dumps(resume_result, indent=2, ensure_ascii=False))

        print("\n=== 4. 取消任务（自动携带车辆信息） ===")
        cancel_result = cancel_mission()
        print(json.dumps(cancel_result, indent=2, ensure_ascii=False))
    else:
        print("\n任务下发失败，跳过后续操作")