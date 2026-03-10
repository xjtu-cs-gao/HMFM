import os
from PIL import Image
import numpy as np
from tqdm import tqdm

def is_mostly_black(image_path, threshold=0.1):
    """
    检测图片是否超过指定比例的纯黑区域
    
    Args:
        image_path: 图片路径
        threshold: 黑色区域比例阈值，默认0.1(10%)
    
    Returns:
        bool: 如果黑色区域超过阈值返回True，否则返回False
        float: 黑色区域的实际比例
    """
    try:
        # 打开图片并转换为RGB模式
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # 检测纯黑像素 (RGB都为0)
        black_pixels = np.all(img_array == [0, 0, 0], axis=2)
        
        # 计算黑色像素比例
        black_ratio = np.sum(black_pixels) / black_pixels.size
        
        return black_ratio > threshold, black_ratio
        
    except Exception as e:
        print(f"\n处理图片 {image_path} 时出错: {e}")
        return False, 0.0

def find_black_images(folder_path, threshold=0.1, extensions=None):
    """
    查找文件夹中黑色区域超过指定比例的图片
    
    Args:
        folder_path: 文件夹路径
        threshold: 黑色区域比例阈值
        extensions: 支持的图片扩展名列表，默认为常见图片格式
    
    Returns:
        list: 符合条件的图片路径列表
        int: 符合条件的图片数量
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']
    
    black_images = []
    count = 0
    
    print(f"开始检测文件夹: {folder_path}")
    print(f"黑色区域阈值: {threshold*100}%")
    print("-" * 50)
    
    # 首先收集所有图片文件
    all_image_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                all_image_files.append(file_path)
    
    # 使用tqdm显示进度条
    for file_path in tqdm(all_image_files, desc="检测图片", unit="张"):
        # 检测图片
        is_black, black_ratio = is_mostly_black(file_path, threshold)
        
        if is_black:
            black_images.append(file_path)
            count += 1
            # 在进度条下方显示检测到的黑色图片信息
            tqdm.write(f"[{count}] {file_path} - 黑色区域: {black_ratio*100:.2f}%")
    
    print("-" * 50)
    print(f"检测完成! 共找到 {count} 张黑色区域超过 {threshold*100}% 的图片")
    
    return black_images, count

def main():
    # 设置文件夹路径和阈值
    folder_path = "./satellite/"
    
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return
    
    # 设置阈值（可以调整）
    threshold = 0.1  # 10%
    
    # 执行检测
    black_images, count = find_black_images(folder_path, threshold)
    
    # 可选：将结果保存到文件
    if count > 0:
        output_file = "black_images_list.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"黑色区域超过 {threshold*100}% 的图片列表 (共 {count} 张):\n")
            f.write("=" * 60 + "\n")
            for i, path in enumerate(black_images, 1):
                f.write(f"{i}. {path}\n")
        
        print(f"\n结果已保存到: {output_file}")

if __name__ == "__main__":
    main()