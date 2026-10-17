#!/usr/bin/env python3
"""
Fuse multiple captured LiDAR frames — using each frame's tracked camera pose —
into one combined, aligned point cloud. This is what actually gets you
Polycam-like density/detail: many overlapping viewpoints merged together,
instead of looking at a single raw 256x192 depth frame.

Usage:
    python3 fuse_scan.py                  # fuse ALL frames in iphone_saved/
    python3 fuse_scan.py --last 100       # fuse only the 100 most recent frames
    python3 fuse_scan.py --save scan.ply  # also save the result to a file
"""
import argparse
import os
import json
import numpy as np
import open3d as o3d

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_saved")


def list_timestamps():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith("_intrinsics.json")]
    return sorted(int(f.split("_")[0]) for f in files)


def load_meta(timestamp):
    with open(os.path.join(SAVE_DIR, f"{timestamp}_intrinsics.json")) as f:
        return json.load(f)


def pick_dominant_session(timestamps):
    """Group frames by session_id (one per app launch) and pick whichever
    session has the most frames — almost certainly the real scan, since
    stray frames from earlier test launches are usually just a handful."""
    counts = {}
    for ts in timestamps:
        meta = load_meta(ts)
        sid = meta.get("session_id", "unknown")
        counts.setdefault(sid, []).append(ts)
    best_sid = max(counts, key=lambda k: len(counts[k]))
    print(f"Found {len(counts)} session(s); using session with {len(counts[best_sid])} frames "
          f"(discarding {len(timestamps) - len(counts[best_sid])} frames from other launches)")
    return counts[best_sid]


def load_frame(timestamp):
    depth_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(f"{timestamp}_depth_")]
    if not depth_files:
        return None
    depth_filename = depth_files[0]
    dims = depth_filename.replace(f"{timestamp}_depth_", "").replace(".bin", "")
    depth_w, depth_h = map(int, dims.split("x"))

    depth = np.fromfile(os.path.join(SAVE_DIR, depth_filename), dtype=np.float32).reshape((depth_h, depth_w))
    meta = load_meta(timestamp)

    if "pose_column_major" not in meta:
        return None  # captured before pose data was added — skip it

    pose = np.array(meta["pose_column_major"], dtype=np.float64).reshape((4, 4), order="F")
    return depth, depth_w, depth_h, meta, pose


def frame_to_world_points(depth, depth_w, depth_h, meta, pose):
    scale_x = depth_w / meta["rgb_width"]
    scale_y = depth_h / meta["rgb_height"]
    fx = meta["fx"] * scale_x
    fy = meta["fy"] * scale_y
    cx = meta["cx"] * scale_x
    cy = meta["cy"] * scale_y

    us, vs = np.meshgrid(np.arange(depth_w), np.arange(depth_h))
    d = depth
    valid = d > 0.05

    # ARKit camera-local convention: X right, Y up, Z backward (camera looks down -Z)
    x_cam = (us - cx) * d / fx
    y_cam = -(vs - cy) * d / fy
    z_cam = -d

    pts_cam = np.stack(
        [x_cam[valid], y_cam[valid], z_cam[valid], np.ones(np.count_nonzero(valid))], axis=-1
    )
    pts_world = (pose @ pts_cam.T).T[:, :3]
    return pts_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--last", type=int, default=None, help="Only fuse the N most recent frames")
    parser.add_argument("--save", type=str, default=None, help="Save the fused cloud to this .ply file")
    parser.add_argument("--voxel", type=float, default=0.005, help="Voxel downsample size in meters (default 5mm)")
    args = parser.parse_args()

    timestamps = list_timestamps()
    if not timestamps:
        raise SystemExit("No captured frames found in " + SAVE_DIR)

    timestamps = pick_dominant_session(timestamps)

    if args.last:
        timestamps = timestamps[-args.last:]

    print(f"Fusing {len(timestamps)} frames...")

    all_points = []
    skipped = 0
    for ts in timestamps:
        loaded = load_frame(ts)
        if loaded is None:
            skipped += 1
            continue
        depth, depth_w, depth_h, meta, pose = loaded
        all_points.append(frame_to_world_points(depth, depth_w, depth_h, meta, pose))

    if skipped:
        print(f"Skipped {skipped} frame(s) with no pose data — rebuild the app and re-scan to include them.")
    if not all_points:
        raise SystemExit("No usable frames with pose data. Rebuild the app in Xcode and capture a new scan first.")

    points = np.concatenate(all_points, axis=0)
    print(f"Raw fused points: {len(points):,}")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    pcd = pcd.voxel_down_sample(voxel_size=args.voxel)
    print(f"After {args.voxel*1000:.0f}mm voxel downsample: {len(pcd.points):,} points")

    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"After outlier removal: {len(pcd.points):,} points")

    bbox = pcd.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    print(f"Bounding box (world-space, meters): X={extent[0]:.3f}  Y={extent[1]:.3f}  Z={extent[2]:.3f}")

    if args.save:
        o3d.io.write_point_cloud(args.save, pcd)
        print(f"Saved to {args.save}")

    o3d.visualization.draw_geometries([pcd], window_name=f"Fused scan ({len(timestamps)} frames)")


if __name__ == "__main__":
    main()
