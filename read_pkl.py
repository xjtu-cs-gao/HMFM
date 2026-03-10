import pickle
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
with open('./annotation/av2_map_infos_val_enhanced.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())
# samples = data['samples']
# sample = None
# for sam in samples:
#     if 'PIT' in sam['city']:
#         sample = sam
#         break

# annotation = sample['annotation']

# # 创建图形
# fig, ax = plt.subplots(figsize=(12, 10))

# # 定义颜色映射
# colors = {
#     'divider': 'yellow',
#     'ped_crossing': 'red', 
#     'boundary': 'blue',
#     'centerline': 'green'
# }

# # 绘制每种类型的vector
# for vector_type, vectors in annotation.items():
#     if vector_type in colors:
#         color = colors[vector_type]
#         for i, vector in enumerate(vectors):
#             if len(vector) > 0:
#                 # 提取x, y坐标
#                 x_coords = vector[:, 0]
#                 y_coords = vector[:, 1]
#                 # 绘制线条，只在第一个vector时添加标签
#                 ax.plot(x_coords, y_coords, color=color, linewidth=2, label=vector_type if i == 0 else "")

# # 设置图形属性
# ax.set_xlabel('X坐标')
# ax.set_ylabel('Y坐标')
# ax.set_title('Map Vector可视化')
# ax.legend()
# ax.grid(True, alpha=0.3)
# ax.set_aspect('equal')

# # 保存图像
# plt.tight_layout()
# plt.savefig('map_vector_visualization.png', dpi=300, bbox_inches='tight')
# print("可视化图像已保存为: map_vector_visualization.png")

# # 显示图像
# plt.show()

# print(sample['wgs84_coords'])
# print(sample['utm_coords'])
# print(sample['e2g_rotation'])
