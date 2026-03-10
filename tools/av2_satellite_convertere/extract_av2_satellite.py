from __future__ import annotations

import argparse
import os
import pickle
from typing import Iterable, Tuple

import numpy as np

from tile_sampler import TileSampler, load_tiles_info


def rotation_matrix_to_yaw_deg(R: Iterable[Iterable[float]]) -> float:
    """Return yaw (Z rotation) in degrees from a 3x3 rotation matrix.

    This matches the previous scripts' behavior (ZYX convention).
    """
    R = np.asarray(R, dtype=float)
    sy = float(np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0]))
    singular = sy < 1e-6
    if singular:
        yaw = 0.0
    else:
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return float(np.degrees(yaw))


def parse_size(s: str) -> Tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except Exception as e:
        raise argparse.ArgumentTypeError(f"Invalid size '{s}', expected like 400x200") from e


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract rotated satellite patches for Argoverse2 samples using OpenSatMap tiles",
    )
    ap.add_argument("--pkl", nargs="+", required=True, help="Input pkl(s), e.g. av2_map_infos_val_enhanced.pkl")
    ap.add_argument("--tiles-info", required=True, help="City tile index json, e.g. MIA_info.json")
    ap.add_argument("--out-dir", default="./satellite", help="Output directory")
    ap.add_argument("--city-contains", default=None, help="Only process samples whose city contains this string (e.g. MIA)")
    ap.add_argument("--width-m", type=float, default=60.0)
    ap.add_argument("--height-m", type=float, default=30.0)
    ap.add_argument("--output-size", type=parse_size, default=(400, 200), help="Output size WxH, e.g. 400x200")
    ap.add_argument(
        "--yaw-sign",
        type=float,
        default=-1.0,
        help="Multiply yaw by this sign (default -1 keeps legacy behavior)",
    )
    ap.add_argument("--preload", action="store_true", help="Preload all tiles into memory (faster, more RAM)")

    args = ap.parse_args()

    tiles = load_tiles_info(args.tiles_info)
    sampler = TileSampler(tiles, preload=args.preload)

    os.makedirs(args.out_dir, exist_ok=True)

    saved = 0
    failed = 0

    for pkl_path in args.pkl:
        if not os.path.exists(pkl_path):
            print(f"[skip] not found: {pkl_path}")
            continue

        print(f"\n=== Processing {pkl_path} ===")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        samples = data.get("samples", [])
        print(f"samples: {len(samples)}")

        for sample in samples:
            city = str(sample.get("city", ""))
            if args.city_contains and args.city_contains not in city:
                continue

            wgs84 = sample.get("wgs84_coords")
            if not wgs84:
                failed += 1
                continue

            # In this repo's enhanced pkls: wgs84_coords is [lat, lon]
            center_lat = float(wgs84[0])
            center_lon = float(wgs84[1])

            yaw = rotation_matrix_to_yaw_deg(sample.get("e2g_rotation"))
            patch = sampler.sample_patch(
                center_lon=center_lon,
                center_lat=center_lat,
                width_m=args.width_m,
                height_m=args.height_m,
                angle_deg=args.yaw_sign * yaw,
                output_size=args.output_size,
            )
            if patch is None:
                failed += 1
                continue

            token = sample.get("token", "")
            out_name = f"{city}_{token}.png" if token else f"{city}.png"
            patch.save(os.path.join(args.out_dir, out_name))
            saved += 1

            if saved % 100 == 0:
                print(f"saved: {saved}, failed: {failed}")

    print(f"\nDone. saved={saved}, failed={failed}")


if __name__ == "__main__":
    main()
