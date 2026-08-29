import json
import math
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import sys

# 尝试导入PIL，如果失败则给出友好提示
try:
    from PIL import Image
    import numpy as np
    PIL_AVAILABLE = True
except ImportError as e:
    PIL_AVAILABLE = False
    print("=" * 60)
    print("错误: 缺少必要的图像处理库")
    print("请运行以下命令安装:")
    print("  pip install Pillow numpy")
    print("=" * 60)
    # 创建一个模拟的Image类，避免程序崩溃
    class Image:
        @staticmethod
        def new(*args, **kwargs):
            raise ImportError("PIL库未安装，请运行: pip install Pillow")
        def open(*args, **kwargs):
            raise ImportError("PIL库未安装，请运行: pip install Pillow")

@dataclass
class Point2D:
    """二维点类"""
    x: float
    y: float
    
    def __add__(self, other):
        return Point2D(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point2D(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Point2D(self.x * scalar, self.y * scalar)
    
    def __truediv__(self, scalar):
        return Point2D(self.x / scalar, self.y / scalar)
    
    def to_point(self):
        """转换为整数点"""
        return Point2D(int(round(self.x)), int(round(self.y)))
    
    def __repr__(self):
        return f"Point2D({self.x}, {self.y})"

class AgvBmapAttribute:
    """AGV地图属性类"""
    
    def __init__(self):
        self.image_data: Optional[Image.Image] = None
        self.raw_data: Dict[str, Any] = {}
        self.start_pos: Point2D = Point2D(0, 0)
        self.scale: float = 50
        self.size: Point2D = Point2D(0, 0)
        self.is_raw_data_latest: bool = False
        
    def check_pil_available(self):
        """检查PIL是否可用"""
        if not PIL_AVAILABLE:
            raise ImportError("请先安装Pillow库: pip install Pillow")
    
    def map_from_image(self, x: int, y: int) -> Point2D:
        """
        将图像坐标转换为世界坐标
        
        Args:
            x: 图像x坐标（像素）
            y: 图像y坐标（像素）
            
        Returns:
            世界坐标点（米）
        """
        world_x = self.start_pos.x / 1000.0 + x * (self.scale / 1000.0)
        world_y = self.start_pos.y / 1000.0 + y * (self.scale / 1000.0)
        return Point2D(world_x, world_y)
    
    def map_to_image(self, point: Point2D, min_pos: Point2D, resolution: float) -> Tuple[int, int]:
        """
        将世界坐标转换为图像坐标
        
        Args:
            point: 世界坐标点（米）
            min_pos: 最小世界坐标（米）
            resolution: 分辨率（米/像素）
            
        Returns:
            图像坐标 (x, y)
        """
        x = int(round((point.x - min_pos.x) / resolution))
        y = int(round((point.y - min_pos.y) / resolution))
        return (x, y)
    
    def generate_raw_data(self):
        """
        PNG转BMAP：将图像数据转换为BMAP格式
        """
        self.check_pil_available()
        
        if self.image_data is not None:
            self.raw_data.clear()
            
            # 创建header信息
            map_header = {}
            min_pos_data = {}
            max_pos_data = {}
            
            # 设置最小位置坐标（转换为米）
            min_pos_data["x"] = self.start_pos.x / 1000.0
            min_pos_data["y"] = self.start_pos.y / 1000.0
            
            # 计算图像右下角对应的世界坐标
            end_pos = self.map_from_image(
                self.image_data.width - 1, 
                self.image_data.height - 1
            )
            max_pos_data["x"] = end_pos.x
            max_pos_data["y"] = end_pos.y
            
            # 组装header
            map_header["minPos"] = min_pos_data
            map_header["maxPos"] = max_pos_data
            map_header["resolution"] = self.scale / 1000.0
            
            self.raw_data["header"] = map_header
            
            # 存储可通行点
            normal_pos_list = []
            
            # 获取图像数据
            try:
                # 转换为RGB模式以便处理
                if self.image_data.mode != 'RGB':
                    img_rgb = self.image_data.convert('RGB')
                else:
                    img_rgb = self.image_data
                
                # 遍历图像像素
                width, height = img_rgb.size
                for x in range(width):
                    for y in range(height):
                        pixel = img_rgb.getpixel((x, y))
                        
                        # 检查是否为白色像素
                        if pixel == (255, 255, 255):
                            point = self.map_from_image(x, y)
                            normal_pos_list.append({
                                "x": round(point.x, 6),  # 保留6位小数
                                "y": round(point.y, 6)
                            })
            except Exception as e:
                print(f"处理图像时出错: {e}")
                return
            
            self.raw_data["normalPosList"] = normal_pos_list
            self.is_raw_data_latest = True
            print(f"成功转换 {len(normal_pos_list)} 个点")
    
    def generate_image_data(self):
        """
        BMAP转PNG：将BMAP格式数据转换为图像数据
        """
        self.check_pil_available()
        
        if not self.raw_data:
            print("错误: 没有BMAP数据")
            return
        
        try:
            # 解析header
            map_header = self.raw_data.get("header", {})
            if not map_header:
                print("错误: BMAP数据缺少header")
                return
                
            min_pos_data = map_header.get("minPos", {})
            max_pos_data = map_header.get("maxPos", {})
            
            if not min_pos_data or not max_pos_data:
                print("错误: header缺少位置信息")
                return
            
            # 获取位置信息
            min_pos = Point2D(
                float(min_pos_data.get("x", 0)),
                float(min_pos_data.get("y", 0))
            )
            max_pos = Point2D(
                float(max_pos_data.get("x", 0)),
                float(max_pos_data.get("y", 0))
            )
            
            # 获取分辨率
            resolution = float(map_header.get("resolution", 0.05))
            
            # 更新比例尺（转换为毫米/像素）
            self.scale = round(resolution * 1000)
            
            # 计算图像尺寸
            size_diff = max_pos - min_pos
            d = (size_diff / resolution + Point2D(1.5, 1.5)).to_point()
            image_size = (max(1, d.x), max(1, d.y))  # 确保尺寸至少为1
            
            # 更新起始位置（毫米）
            self.start_pos = min_pos * 1000
            
            # 更新实际物理尺寸（毫米）
            self.size = Point2D(
                image_size[0] * self.scale,
                image_size[1] * self.scale
            )
            
            # 创建透明背景图像
            self.image_data = Image.new('RGBA', image_size, (0, 0, 0, 0))
            
            # 获取像素访问对象
            pixels = self.image_data.load()
            
            # 绘制可通行点
            normal_pos_list = self.raw_data.get("normalPosList", [])
            point_count = 0
            
            for pos in normal_pos_list:
                if not isinstance(pos, dict):
                    continue
                    
                pos_data = Point2D(
                    float(pos.get("x", 0)),
                    float(pos.get("y", 0))
                )
                
                # 转换世界坐标到图像坐标
                x = int(round((pos_data.x - min_pos.x) / resolution))
                y = int(round((pos_data.y - min_pos.y) / resolution))
                
                # 检查坐标是否在图像范围内
                if 0 <= x < image_size[0] and 0 <= y < image_size[1]:
                    pixels[x, y] = (255, 255, 255, 255)
                    point_count += 1
            
            print(f"成功绘制 {point_count} 个点")
            
        except Exception as e:
            print(f"生成图像时出错: {e}")
    
    def load_image(self, image_path: str):
        """
        从文件加载PNG图像
        """
        self.check_pil_available()
        try:
            self.image_data = Image.open(image_path)
            print(f"成功加载图像: {image_path}")
            print(f"图像尺寸: {self.image_data.size}")
            print(f"图像模式: {self.image_data.mode}")
        except Exception as e:
            print(f"加载图像失败: {e}")
    
    def save_image(self, image_path: str):
        """
        保存图像到文件
        """
        self.check_pil_available()
        if self.image_data:
            try:
                self.image_data.save(image_path)
                print(f"图像已保存: {image_path}")
            except Exception as e:
                print(f"保存图像失败: {e}")
        else:
            print("错误: 没有图像数据可保存")
    
    def load_bmap(self, bmap_path: str):
        """
        从文件加载BMAP数据
        """
        try:
            with open(bmap_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            print(f"成功加载BMAP: {bmap_path}")
            print(f"header: {self.raw_data.get('header', {})}")
            point_count = len(self.raw_data.get('normalPosList', []))
            print(f"点数: {point_count}")
        except Exception as e:
            print(f"加载BMAP失败: {e}")
    
    def save_bmap(self, bmap_path: str):
        """
        保存BMAP数据到文件
        """
        try:
            with open(bmap_path, 'w', encoding='utf-8') as f:
                json.dump(self.raw_data, f, ensure_ascii=False, indent=2)
            print(f"BMAP已保存: {bmap_path}")
        except Exception as e:
            print(f"保存BMAP失败: {e}")
    
    def set_parameters(self, start_x: float, start_y: float, scale: float):
        """
        设置地图参数
        
        Args:
            start_x: 起始X坐标（毫米）
            start_y: 起始Y坐标（毫米）
            scale: 比例尺（毫米/像素）
        """
        self.start_pos = Point2D(start_x, start_y)
        self.scale = scale
        print(f"参数已设置: 起始点=({start_x}, {start_y}), 比例尺={scale} mm/像素")


# 使用示例
def example_png_to_bmap():
    """PNG转BMAP示例"""
    print("\n=== PNG转BMAP示例 ===")
    agv_map = AgvBmapAttribute()
    
    # 设置参数
    agv_map.set_parameters(start_x=0, start_y=0, scale=50)
    
    # 加载PNG图像
    agv_map.load_image("input.png")
    
    if agv_map.image_data:
        # 转换为BMAP
        agv_map.generate_raw_data()
        
        # 保存BMAP文件
        agv_map.save_bmap("output.bmap")
        
        print("PNG转BMAP完成")

def example_bmap_to_png():
    """BMAP转PNG示例"""
    print("\n=== BMAP转PNG示例 ===")
    agv_map = AgvBmapAttribute()
    
    # 加载BMAP文件
    agv_map.load_bmap("input.bmap")
    
    if agv_map.raw_data:
        # 转换为PNG
        agv_map.generate_image_data()
        
        # 保存PNG图像
        agv_map.save_image("output.png")
        
        print("BMAP转PNG完成")

def create_test_image():
    """创建测试图像"""
    try:
        from PIL import Image, ImageDraw
        
        # 创建100x100的透明图像
        img = Image.new('RGBA', (100, 100), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # 画一些白色点
        for i in range(10, 90, 10):
            for j in range(10, 90, 10):
                draw.point((i, j), fill=(255, 255, 255, 255))
        
        # 保存测试图像
        img.save("test_input.png")
        print("测试图像已创建: test_input.png")
        return img
    except Exception as e:
        print(f"创建测试图像失败: {e}")
        return None

def example_bidirectional_conversion():
    """双向转换示例"""
    print("\n=== 双向转换测试 ===")
    
    # 创建测试图像
    test_image = create_test_image()
    if not test_image:
        return
    
    # PNG → BMAP
    agv_map = AgvBmapAttribute()
    agv_map.set_parameters(start_x=0, start_y=0, scale=50)
    agv_map.image_data = test_image
    
    print("\n[步骤1] PNG转BMAP...")
    agv_map.generate_raw_data()
    
    if agv_map.raw_data:
        print(f"BMAP数据点数: {len(agv_map.raw_data['normalPosList'])}")
        
        # 保存中间结果
        agv_map.save_bmap("test.bmap")
        
        # BMAP → PNG
        print("\n[步骤2] BMAP转PNG...")
        new_map = AgvBmapAttribute()
        new_map.load_bmap("test.bmap")
        new_map.generate_image_data()
        
        if new_map.image_data:
            new_map.save_image("test_output.png")
            
            # 验证结果
            print("\n[步骤3] 验证结果:")
            print(f"原始图像尺寸: {test_image.size}")
            print(f"转换后图像尺寸: {new_map.image_data.size}")
            print(f"起始位置: {new_map.start_pos}")
            print(f"比例尺: {new_map.scale} mm/像素")
            
            # 检查点数量
            original_points = len(agv_map.raw_data['normalPosList'])
            print(f"原始点数: {original_points}")
            
            # 由于舍入误差，可能会有微小差异
            print("\n转换完成！")
    else:
        print("转换失败")

def main():
    """主函数"""
    print("=" * 50)
    print("AGV地图格式转换工具")
    print("=" * 50)
    
    # 检查PIL
    if not PIL_AVAILABLE:
        print("\n请先安装必要的库:")
        print("  pip install Pillow numpy")
        return
    
    # 运行双向转换测试
    example_bidirectional_conversion()
    
    print("\n" + "=" * 50)
    print("提示: 可以使用以下函数单独转换:")
    print("  example_png_to_bmap()  # PNG转BMAP")
    print("  example_bmap_to_png()  # BMAP转PNG")

if __name__ == "__main__":
    main()