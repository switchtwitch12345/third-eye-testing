#!/usr/bin/env python3
"""
Convert a saved LiDAR depth frame + intrinsics into a point cloud and visualize it.

Usage:
    python3 process_scan.py --latest
    python3 process_scan.py <timestamp>
"""
import sys
import os
import json
import numpy as np
import open3d as o3d

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_saved")


def find_latest_timestamp():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith("_intrinsics.json")]
    if not files:
        raise SystemExit(
            "No frames with intrinsics found. Rebuild the app in Xcode and capture a "
            "new short scan first — older captures don't have intrinsics saved."
        )
    timestamps = sorted(int(f.split("_")[0]) for f in files)
    return timestamps[-1]


def load_frame(timestamp):
    depth_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(f"{timestamp}_depth_")]
    if not depth_files:
        raise SystemExit(f"No depth file found for timestamp {timestamp}")
    depth_filename = depth_files[0]
    dims = depth_filename.replace(f"{timestamp}_depth_", "").replace(".bin", "")
    depth_w, depth_h = map(int, dims.split("x"))

    depth_path = os.path.join(SAVE_DIR, depth_filename)
    depth = np.fromfile(depth_path, dtype=np.float32).reshape((depth_h, depth_w))

    intrinsics_path = os.path.join(SAVE_DIR, f"{timestamp}_intrinsics.json")
    with open(intrinsics_path) as f:
        intr = json.load(f)

    return depth, depth_w, depth_h, intr


def depth_to_pointcloud(depth, depth_w, depth_h, intr):
    # Intrinsics are given at RGB resolution — scale down to the depth map's resolution
    scale_x = depth_w / intr["rgb_width"]
    scale_y = depth_h / intr["rgb_height"]
    fx = intr["fx"] * scale_x
    fy = intr["fy"] * scale_y
    cx = intr["cx"] * scale_x
    cy = intr["cy"] * scale_y

    us, vs = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    z = depth
    valid = z > 0.05  # drop invalid/zero depth

    x = (us - cx) * z / fx
    y = (vs - cy) * z / fy

    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points


def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--latest":
        timestamp = int(sys.argv[1])
    else:
        timestamp = find_latest_timestamp()

    print(f"Loading frame {timestamp}...")
    depth, depth_w, depth_h, intr = load_frame(timestamp)

    valid_depth = depth[depth > 0]
    print(f"Depth map: {depth_w}x{depth_h}, range {valid_depth.min():.2f}m - {valid_depth.max():.2f}m")

    points = depth_to_pointcloud(depth, depth_w, depth_h, intr)
    print(f"Generated {len(points):,} 3D points")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"Bounding box (camera-space, meters): X={extent[0]:.3f}  Y={extent[1]:.3f}  Z={extent[2]:.3f}")
    print("(this is camera-relative, not world-aligned — Y is 'up in the frame', not true vertical yet)")

    o3d.visualization.draw_geometries([pcd], window_name=f"Scan {timestamp}")


if __name__ == "__main__":
    main()
