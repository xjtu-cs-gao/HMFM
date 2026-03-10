import pickle
import numpy as np
from av2.geometry.utm import convert_city_coords_to_utm, CityName, convert_city_coords_to_wgs84
from pathlib import Path
import os
from tqdm import tqdm


def detect_city_from_log_id(log_id, data_root=None):
    """从log_id中检测城市名称"""
    from av2.geometry.utm import CityName

    # 如果无法从log_id中检测到，尝试从map文件路径中检测
    if data_root is not None:
        try:
            from pathlib import Path
            from os import path as osp
            
            log_map_dirpath = Path(osp.join(data_root, log_id, "map"))
            vector_data_fnames = sorted(log_map_dirpath.glob("log_map_archive_*.json"))
            
            if len(vector_data_fnames) == 1:
                vector_data_json_path = vector_data_fnames[0]
                
                # 从文件路径中检测城市名称
                for city_enum in CityName:
                    if city_enum in str(vector_data_json_path):
                        return city_enum
                        
        except Exception as e:
            print(f"从map文件路径检测城市失败: {e}")
    
    # 如果都失败了，返回None
    return None


def enhance_sample_with_pose(sample, data_root=None):
    """为单个样本添加city、utm_coords、wgs84_coords信息"""
    # 获取e2g_translation
    e2g_translation = sample['e2g_translation']
    
    # 从log_id中检测城市
    log_id = sample['log_id']
    city = detect_city_from_log_id(log_id, data_root)
    
    if city is None:
        print(f"警告: 无法从log_id {log_id} 中检测到城市名称，跳过该样本")
        return sample
    
    # 坐标转换：城市坐标系 -> UTM -> WGS84
    # 只使用前两个维度 (x, y)，忽略 z 坐标
    e2g_translation_2d = np.array([e2g_translation[:2]])
    
    try:
        utm_coords = convert_city_coords_to_utm(e2g_translation_2d, city)
        wgs84_coords = convert_city_coords_to_wgs84(e2g_translation_2d, city)
        
        # 添加新的键到样本中
        sample['city'] = str(city)
        sample['utm_coords'] = utm_coords[0]  # UTM坐标 (x, y)
        sample['wgs84_coords'] = wgs84_coords[0]  # WGS84坐标 (longitude, latitude)
        
    except Exception as e:
        print(f"警告: 坐标转换失败 for log_id {log_id}, city {city}: {e}")
        # 即使转换失败，也添加city信息
        sample['city'] = str(city)
        sample['utm_coords'] = None
        sample['wgs84_coords'] = None
    
    return sample


def enhance_pkl_with_pose(pkl_path, output_path=None, data_root=None):
    """增强pkl文件，添加pose相关信息"""
    print(f"处理文件: {pkl_path}")
    
    # 加载数据
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"数据类型: {type(data)}")
    samples = data['samples']
    print(f"样本数量: {len(samples)}")
    
    # 处理每个样本
    enhanced_samples = []
    for sample in tqdm(samples, desc="处理样本", unit="sample"):
        enhanced_sample = enhance_sample_with_pose(sample, data_root)
        enhanced_samples.append(enhanced_sample)
    
    # 更新数据
    data['samples'] = enhanced_samples
    
    # 保存增强后的数据
    if output_path is None:
        output_path = pkl_path.replace('.pkl', '_enhanced.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"增强后的数据已保存到: {output_path}")

if __name__ == "__main__":
    splits = ['train']
    
    # 设置数据根目录，这里需要根据实际情况调整
    # 如果pkl文件中已经包含了完整的数据路径信息，可以设置为None
    data_root = './data/argoverse2/sensor/'  # 例如: "/path/to/argoverse2/data"
    
    for split in splits:
        pkl_path = f'./annotation/av2_map_infos_{split}.pkl'
        
        if not os.path.exists(pkl_path):
            print(f"文件不存在: {pkl_path}")
            continue
            
        enhance_pkl_with_pose(pkl_path, data_root=data_root+split)
