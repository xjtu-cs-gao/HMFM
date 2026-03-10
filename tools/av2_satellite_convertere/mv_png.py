# 遍历satellite文件夹，将除了black list之外的png文件移动到av2_satellite文件夹中，并且重命名，删除城市名

import os
import shutil
import re
from tqdm import tqdm

def load_blacklist(blacklist_file):
    """读取黑名单文件，提取文件名"""
    blacklist = set()
    
    if not os.path.exists(blacklist_file):
        print(f"黑名单文件不存在: {blacklist_file}")
        return blacklist
    
    with open(blacklist_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        # 跳过标题行和分隔线
        if not line or line.startswith('黑色区域') or line.startswith('===') or line.startswith('共') or line.startswith('张'):
            continue
        
        # 提取文件名，格式如: "1. ./satellite/CityName.WDC_xxx.png"
        match = re.search(r'\./satellite/(.+)', line)
        if match:
            filename = match.group(1)
            blacklist.add(filename)
    
    print(f"加载了 {len(blacklist)} 个黑名单文件")
    return blacklist

def remove_city_prefix(filename):
    """删除文件名中的城市名前缀（CityName.）"""
    if filename.startswith('CityName.'):
        return filename[13:]  # 删除 "CityName." 前缀
    return filename

def move_png_files():
    """主函数：移动PNG文件"""
    # 路径设置
    satellite_dir = './satellite'
    av2_satellite_dir = './av2_satellite'
    blacklist_file = './black_images_list.txt'
    
    # 创建目标目录
    if not os.path.exists(av2_satellite_dir):
        os.makedirs(av2_satellite_dir)
        print(f"创建目录: {av2_satellite_dir}")
    
    # 加载黑名单
    blacklist = load_blacklist(blacklist_file)
    
    # 统计变量
    moved_count = 0
    skipped_count = 0
    error_count = 0
    
    # 遍历satellite文件夹
    if not os.path.exists(satellite_dir):
        print(f"源目录不存在: {satellite_dir}")
        return
    
    print(f"开始处理 {satellite_dir} 目录...")
    
    # 获取所有PNG文件列表
    png_files = [f for f in os.listdir(satellite_dir) if f.lower().endswith('.png')]
    print(f"找到 {len(png_files)} 个PNG文件")
    
    # 使用tqdm显示进度条
    for filename in tqdm(png_files, desc="处理PNG文件", unit="文件"):
        # 检查是否在黑名单中
        if filename in blacklist:
            skipped_count += 1
            continue
        
        # 生成新文件名（删除城市名前缀）
        new_filename = remove_city_prefix(filename)
        
        # 源文件和目标文件路径
        src_path = os.path.join(satellite_dir, filename)
        dst_path = os.path.join(av2_satellite_dir, new_filename)
        
        try:
            # 复制文件（而不是移动）
            shutil.copy2(src_path, dst_path)
            moved_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"复制文件失败 {filename}: {e}")
    
    # 输出统计结果
    print(f"\n处理完成!")
    print(f"成功移动: {moved_count} 个文件")
    print(f"跳过黑名单: {skipped_count} 个文件")
    print(f"错误: {error_count} 个文件")

if __name__ == '__main__':
    move_png_files()