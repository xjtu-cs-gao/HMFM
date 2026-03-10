# av2_satellite_convertere

这个工具用于把 OpenSatMap 的大图 tile，按 Argoverse2 每个样本的经纬度和朝向裁成小卫星图，作为数据增强输入。

## 保留的核心脚本
- `extract_av2_satellite.py`: 主入口。读取 `av2_map_infos_*_enhanced.pkl`，批量导出 patch。
- `tile_sampler.py`: 底层采样逻辑。根据 `*_info.json` 的 tile 边界，从多张 tile 中拼接并旋转裁剪。

## 可选后处理脚本
- `detect_black.py`: 检测 `./satellite` 中黑边比例过高的图片并写入 `black_images_list.txt`。
- `mv_png.py`: 把非黑名单图片复制到 `./av2_satellite`，并去掉文件名前缀 `CityName.`。

## 依赖
```bash
pip install -r requirements.txt
```

## 数据准备
需要两类输入：
- AV2 增强后的 pkl（每条样本至少有 `wgs84_coords`、`e2g_rotation`、`city`、`token`）
- OpenSatMap 对应城市的 `*_info.json`（包含每个 tile 的经纬度边界和本地图片路径）

## 快速使用
示例（MIA）：

```bash
python extract_av2_satellite.py \
  --pkl av2_map_infos_val_enhanced.pkl av2_map_infos_train_enhanced.pkl \
  --tiles-info MIA_info.json \
  --city-contains MIA \
  --width-m 60 --height-m 30 \
  --output-size 400x200 \
  --out-dir ./satellite \
  --preload
```

输出文件名格式：`<city>_<token>.png`（例如 `CityName.MIA_xxx.png`）。

## 说明
- `--yaw-sign` 默认为 `-1`，用于保持旧流程的角度方向；如果旋转方向反了可改为 `+1`。
- tile 名支持如 `PIT_-1_-4_sat.png`、`WDC_-1_-1_sat.png`、`Miami_-1_-1_sat.png`。
