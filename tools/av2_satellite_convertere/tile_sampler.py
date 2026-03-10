from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from pyproj import Proj


TILE_SIZE_DEFAULT = 4096


def _parse_tile_indices(tile_name: str) -> Optional[Tuple[int, int]]:
    """Parse OpenSatMap tile filename like `PIT_-1_-4_sat.png` or `Miami_-1_-4_sat.png`."""
    m = re.search(r"_(-?\d+)_(-?\d+)_sat\.png$", tile_name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _guess_utm_zone(lon: float) -> int:
    # UTM zones are 1..60.
    zone = int((lon + 180.0) / 6.0) + 1
    return max(1, min(60, zone))


def load_tiles_info(json_path: str) -> List[dict]:
    """Load tile metadata json (e.g. `MIA_info.json`) into a normalized tile list."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    tiles: List[dict] = []
    for name, info in data.items():
        parsed = _parse_tile_indices(name)
        if parsed is None:
            continue
        idx_x, idx_y = parsed
        tiles.append(
            {
                "name": name,
                "path": info["path"],
                "idx_x": idx_x,
                "idx_y": idx_y,
                # NOTE: the json uses x=lon, y=lat
                "center_lon": info["center_x"],
                "center_lat": info["center_y"],
                "min_lon": info["min_x_bound"],
                "max_lon": info["max_x_bound"],
                "min_lat": info["min_y_bound"],
                "max_lat": info["max_y_bound"],
            }
        )

    if not tiles:
        raise ValueError(f"No valid tiles found in: {json_path}")

    return tiles


@dataclass(frozen=True)
class PixelCoordSystem:
    tile_size: int
    min_idx_x: int
    max_idx_x: int
    min_idx_y: int
    max_idx_y: int
    global_min_lon: float
    global_max_lon: float
    global_min_lat: float
    global_max_lat: float
    grid_width: int
    grid_height: int
    meter_per_pixel: float


def build_pixel_coordinate_system(tiles: Iterable[dict], tile_size: int = TILE_SIZE_DEFAULT) -> PixelCoordSystem:
    tiles = list(tiles)

    min_idx_x = min(t["idx_x"] for t in tiles)
    max_idx_x = max(t["idx_x"] for t in tiles)
    min_idx_y = min(t["idx_y"] for t in tiles)
    max_idx_y = max(t["idx_y"] for t in tiles)

    global_min_lon = min(t["min_lon"] for t in tiles)
    global_max_lon = max(t["max_lon"] for t in tiles)
    global_min_lat = min(t["min_lat"] for t in tiles)
    global_max_lat = max(t["max_lat"] for t in tiles)

    grid_width = (max_idx_x - min_idx_x + 1) * tile_size
    grid_height = (max_idx_y - min_idx_y + 1) * tile_size

    center_lon = (global_min_lon + global_max_lon) / 2.0
    center_lat = (global_min_lat + global_max_lat) / 2.0
    zone = _guess_utm_zone(center_lon)

    projector = Proj(proj="utm", zone=zone, ellps="WGS84", datum="WGS84", units="m")

    corners = [
        (global_min_lon, global_min_lat),
        (global_max_lon, global_min_lat),
        (global_min_lon, global_max_lat),
        (global_max_lon, global_max_lat),
    ]

    utm = [projector(lon, lat) for lon, lat in corners]

    width_bottom = utm[1][0] - utm[0][0]
    width_top = utm[3][0] - utm[2][0]
    avg_width_m = (width_bottom + width_top) / 2.0

    height_left = utm[2][1] - utm[0][1]
    height_right = utm[3][1] - utm[1][1]
    avg_height_m = (height_left + height_right) / 2.0

    meter_per_pixel_x = avg_width_m / float(grid_width)
    meter_per_pixel_y = avg_height_m / float(grid_height)
    meter_per_pixel = (meter_per_pixel_x + meter_per_pixel_y) / 2.0

    return PixelCoordSystem(
        tile_size=tile_size,
        min_idx_x=min_idx_x,
        max_idx_x=max_idx_x,
        min_idx_y=min_idx_y,
        max_idx_y=max_idx_y,
        global_min_lon=global_min_lon,
        global_max_lon=global_max_lon,
        global_min_lat=global_min_lat,
        global_max_lat=global_max_lat,
        grid_width=grid_width,
        grid_height=grid_height,
        meter_per_pixel=meter_per_pixel,
    )


def lonlat_to_global_pixel(lon: float, lat: float, cs: PixelCoordSystem) -> Tuple[float, float]:
    lon_ratio = (lon - cs.global_min_lon) / (cs.global_max_lon - cs.global_min_lon)
    lat_ratio = (lat - cs.global_min_lat) / (cs.global_max_lat - cs.global_min_lat)

    global_px = lon_ratio * cs.grid_width
    global_py = (1.0 - lat_ratio) * cs.grid_height  # image y-axis is downward
    return global_px, global_py


class TileSampler:
    """Sample patches from a grid of OpenSatMap tiles described by `*_info.json`."""

    def __init__(self, tiles: List[dict], tile_size: int = TILE_SIZE_DEFAULT, preload: bool = False):
        self.tiles = tiles
        self.tile_size = tile_size
        self.cs = build_pixel_coordinate_system(tiles, tile_size=tile_size)
        self._tile_info_by_idx: Dict[Tuple[int, int], dict] = {(t["idx_x"], t["idx_y"]): t for t in tiles}
        self._tile_img_cache: Dict[Tuple[int, int], np.ndarray] = {}

        if preload:
            for k in self._tile_info_by_idx.keys():
                self._load_tile_image(k)

    def _load_tile_image(self, key: Tuple[int, int]) -> Optional[np.ndarray]:
        if key in self._tile_img_cache:
            return self._tile_img_cache[key]

        info = self._tile_info_by_idx.get(key)
        if info is None:
            return None

        path = info["path"]
        if not os.path.exists(path):
            return None

        try:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img)
        except Exception:
            return None

        # Allow non-4096 tile size, but require consistent size.
        if arr.shape[0] != self.tile_size or arr.shape[1] != self.tile_size:
            raise ValueError(
                f"Tile size mismatch for {path}: got {arr.shape[1]}x{arr.shape[0]}, expected {self.tile_size}x{self.tile_size}"
            )

        self._tile_img_cache[key] = arr
        return arr

    def sample_patch(
        self,
        *,
        center_lon: float,
        center_lat: float,
        width_m: float,
        height_m: float,
        angle_deg: float = 0.0,
        output_size: Tuple[int, int] = (512, 512),
        fill_color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Optional[Image.Image]:
        """Return a rotated patch (nearest-neighbor) or None if center is outside tile coverage."""

        if (
            center_lon < self.cs.global_min_lon
            or center_lon > self.cs.global_max_lon
            or center_lat < self.cs.global_min_lat
            or center_lat > self.cs.global_max_lat
        ):
            return None

        out_w, out_h = output_size

        center_px, center_py = lonlat_to_global_pixel(center_lon, center_lat, self.cs)

        width_px = width_m / self.cs.meter_per_pixel
        height_px = height_m / self.cs.meter_per_pixel

        px_grid, py_grid = np.meshgrid(np.arange(out_w, dtype=np.float32), np.arange(out_h, dtype=np.float32))

        local_px = (px_grid - out_w / 2.0) * (width_px / out_w)
        local_py = (out_h / 2.0 - py_grid) * (height_px / out_h)

        rad = math.radians(angle_deg)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        rotated_px = local_px * cos_a - local_py * sin_a
        rotated_py = local_px * sin_a + local_py * cos_a

        global_px = rotated_px + center_px
        global_py = rotated_py + center_py

        ts = float(self.tile_size)
        grid_x = np.floor(global_px / ts).astype(np.int32)
        grid_y = np.floor(global_py / ts).astype(np.int32)

        tile_x = grid_x + self.cs.min_idx_x
        tile_y = grid_y + self.cs.min_idx_y

        local_u = np.rint(global_px - grid_x.astype(np.float32) * ts).astype(np.int32)
        local_v = np.rint(global_py - grid_y.astype(np.float32) * ts).astype(np.int32)

        # Clamp to valid range.
        local_u = np.clip(local_u, 0, self.tile_size - 1)
        local_v = np.clip(local_v, 0, self.tile_size - 1)

        result = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        result[:, :] = np.asarray(fill_color, dtype=np.uint8)

        # Only fill pixels whose tiles exist.
        uniq_tiles = np.unique(np.stack([tile_x, tile_y], axis=-1).reshape(-1, 2), axis=0)
        for x, y in uniq_tiles:
            key = (int(x), int(y))
            tile_img = self._load_tile_image(key)
            if tile_img is None:
                continue
            mask = (tile_x == x) & (tile_y == y)
            if not np.any(mask):
                continue
            result[mask] = tile_img[local_v[mask], local_u[mask]]

        return Image.fromarray(result)
