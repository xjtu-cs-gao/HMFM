import mmcv
import os
from tqdm import tqdm

# 定义要处理的数据集划分
splits = ['train', 'val']

print("Starting to filter annotation files...")

for s in splits:
    pkl_path = f'./annotation/av2_map_infos_{s}.pkl'
    
    if not os.path.exists(pkl_path):
        print(f"Warning: Annotation file not found at {pkl_path}. Skipping.")
        continue

    print(f"\nProcessing file: {pkl_path}")
    
    # 使用 mmcv.load 直接读取文件，更简洁
    data = mmcv.load(pkl_path)
    
    # 检查 'samples' 键是否存在
    if 'samples' not in data:
        print(f"Error: 'samples' key not found in {pkl_path}. Skipping.")
        continue
        
    samples = data['samples']
    original_count = len(samples)
    
    filtered_samples = []
    
    # 使用 tqdm 创建进度条
    print(f"Filtering {original_count} samples for '{s}' split...")
    for sample in tqdm(samples, desc=f"Filtering {s}"):
        # 假设卫星图像存储在 ./data/av2_satellite/ 目录下
        sat_path = f"./data/av2_satellite/{sample['token']}.png"
        if os.path.exists(sat_path):
            filtered_samples.append(sample)
            
    filtered_count = len(filtered_samples)
    
    # 更新数据字典中的 samples 列表
    data['samples'] = filtered_samples
    
    # 使用 mmcv.dump 直接写回文件，它会自动处理 pickle 协议
    new_pkl_path = pkl_path.replace('.pkl', '_filtered.pkl')
    mmcv.dump(data, new_pkl_path)
    
    print(f"Finished processing '{s}' split.")
    print(f"Result: Kept {filtered_count} out of {original_count} samples.")

print("\nAll files have been processed and updated.")

