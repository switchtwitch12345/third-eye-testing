#!/usr/bin/env python3
"""
TSDF volumetric fusion — merges many depth frames (with their tracked poses)
into ONE smooth, clean mesh, averaging out per-frame pose jitter instead of
just concatenating raw points. This is the technique that actually gets you
Polycam-like clean surfaces instead of streaky "comb" artifacts.

Usage:
    python3 tsdf_fuse.py --save mesh.ply
"""
import argparse
import os
import json
import numpy as np
import open3d as o3d

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_saved")

# ARKit camera-local convention (X right, Y up, Z backward) -> OpenCV/Open3D
# convention (X right, Y down, Z forward) that TSDF integration expects.
ARKIT_TO_CV = np.diag([1, -1, -1, 1]).astype(np.float64)


def list_timestamps():
    files = [f for f in os.listdir(SAVE_DIR) if f.endswith("_intrinsics.json")]
    return sorted(int(f.split("_")[0]) for f in files)


def load_meta(timestamp):
    with open(os.path.join(SAVE_DIR, f"{timestamp}_intrinsics.json")) as f:
        return json.load(f)


def pick_dominant_session(timestamps):
    counts = {}
    for ts in timestamps:
        sid = load_meta(ts).get("session_id", "unknown")
        counts.setdefault(sid, []).append(ts)
    best_sid = max(counts, key=lambda k: len(counts[k]))
    print(f"Found {len(counts)} session(s); using session with {len(counts[best_sid])} frames "
          f"(discarding {len(timestamps) - len(counts[best_sid])} frames from other launches)")
    return counts[best_sid]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", type=str, default="fused_mesh.ply")
    parser.add_argument("--voxel", type=float, default=0.02,
                         help="TSDF voxel size in meters (default 20mm — below this, single-frame LiDAR noise "
                              "exceeds voxel resolution and no surface extracts at all)")
    parser.add_argument("--depth_trunc", type=float, default=0.6,
                         help="Ignore depth beyond this range (meters) — keeps only the nearby object, drops room background")
    args = parser.parse_args()

    timestamps = list_timestamps()
    if not timestamps:
        raise SystemExit("No captured frames found in " + SAVE_DIR)
    timestamps = pick_dominant_session(timestamps)
    timestamps = [ts for ts in timestamps if "pose_column_major" in load_meta(ts)]
    if not timestamps:
        raise SystemExit("No frames with pose data in the dominant session.")

    print(f"Integrating {len(timestamps)} frames into TSDF volume (voxel={args.voxel*1000:.1f}mm)...")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel,
        sdf_trunc=args.voxel * 4,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    intrinsic_o3d = None
    integrated = 0
    for i, ts in enumerate(timestamps):
        depth_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(f"{ts}_depth_")]
        if not depth_files:
            continue
        depth_filename = depth_files[0]
        dims = depth_filename.replace(f"{ts}_depth_", "").replace(".bin", "")
        w, h = map(int, dims.split("x"))

        depth = np.fromfile(os.path.join(SAVE_DIR, depth_filename), dtype=np.float32).reshape((h, w))

        meta = load_meta(ts)
        scale_x = w / meta["rgb_width"]
        scale_y = h / meta["rgb_height"]
        fx = meta["fx"] * scale_x
        fy = meta["fy"] * scale_y
        cx = meta["cx"] * scale_x
        cy = meta["cy"] * scale_y

        if intrinsic_o3d is None:
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        pose_arkit = np.array(meta["pose_column_major"], dtype=np.float64).reshape((4, 4), order="F")
        pose_cv = pose_arkit @ ARKIT_TO_CV   # camera-to-world, OpenCV/Open3D axis convention
        extrinsic = np.linalg.inv(pose_cv)   # Open3D wants world-to-camera

        color_dummy = np.full((h, w, 3), 128, dtype=np.uint8)  # geometry matters here, not color
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color_dummy),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=args.depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic_o3d, extrinsic)
        integrated += 1

        if integrated % 20 == 0:
            print(f"  {integrated}/{len(timestamps)} frames integrated...")

    print("Extracting mesh...")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")

    o3d.io.write_triangle_mesh(args.save, mesh)
    print(f"Saved to {args.save}")

    o3d.visualization.draw_geometries([mesh], window_name="TSDF fused mesh", mesh_show_back_face=True)


if __name__ == "__main__":
    main()
