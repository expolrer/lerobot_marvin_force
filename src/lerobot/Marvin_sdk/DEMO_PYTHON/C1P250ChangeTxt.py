import os
import re

def convert_point_data(input_file, output_file):
    """
    将z70.txt中的点数据转换为七关节角度格式
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        print(f"当前工作目录：{os.getcwd()}")
        print(f"尝试的完整路径：{os.path.abspath(input_file)}")
        return
    except Exception as e:
        print(f"读取文件时出错：{e}")
        return
    
    # 分割成单独的点数据行
    lines = content.strip().split('\n')
    
    # 存储转换后的数据
    converted_points = []
    
    # 正则表达式匹配坐标数据
    pattern = r'X\s+([-\d.]+)\$Y\s+([-\d.]+)\$Z\s+([-\d.]+)\$A\s+([-\d.]+)\$B\s+([-\d.]+)\$C\s+([-\d.]+)\$U\s+([-\d.]+)\$V\s+([-\d.]+)\$W\s+([-\d.]+)'
    
    for line in lines:
        # 跳过不是数据行的内容
        if not line.startswith('X '):
            continue
            
        match = re.search(pattern, line)
        if match:
            # 提取所有坐标值
            x, y, z, a, b, c, u, v, w = match.groups()
            
            # 创建转换后的字符串
            converted = f"{x},{y},{z},{a},{b},{c},{u}"
            converted_points.append(converted)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 写入输出文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(converted_points))
    except Exception as e:
        print(f"写入输出文件时出错：{e}")
        return
    
    print(f"转换完成！共转换了 {len(converted_points)} 个点")
    print(f"输入文件: {os.path.abspath(input_file)}")
    print(f"输出文件: {os.path.abspath(output_file)}")
    
    # 显示前几个转换结果作为示例
    if converted_points:
        print("\n前5个转换结果示例:")
        for i in range(min(5, len(converted_points))):
            print(converted_points[i])

def main():
    # 使用绝对路径
    input_file = r"C:\Users\T0001541\Music\Desktop\TJ_FX_ROBOT_CONTRL_SDK-master\\testkj.txt"
    output_file = r"C:\Users\T0001541\Music\Desktop\TJ_FX_ROBOT_CONTRL_SDK-master\\11.txt"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件不存在")
        print(f"请检查路径: {input_file}")
        return
    
    # 执行转换
    convert_point_data(input_file, output_file)

if __name__ == "__main__":
    main()