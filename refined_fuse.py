#!/usr/bin/env python3
"""
Fuse captured LiDAR frames with ICP pose refinement.

ARKit's tracked poses are good frame-to-frame (~5mm) but drift over a scan,
which smears the fused model. This uses ARKit's pose as an initial guess and
then registers each frame against the model built so far, correcting drift
before TSDF integration.

Usage:
    python3 refined_fuse.py --save mesh.ply
"""
import argparse
import os
import json
import numpy as np
import open3d as o3d

SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iphone_saved")
ARKIT_TO_CV = np.diag([1, -1, -1, 1]).astype(np.float64)


def load_frames():
    files = sorted(f for f in os.listdir(SAVE_DIR) if f.endswith("_intrinsics.json"))
    frames = []
    for f in files:
        ts = f.split("_")[0]
        df = [x for x in os.listdir(SAVE_DIR) if x.startswith(f"{ts}_depth_")]
        if not df:
            continue
        dims = df[0].split("_depth_")[1].replace(".bin", "")
        w, h = map(int, dims.split("x"))
        raw = np.fromfile(os.path.join(SAVE_DIR, df[0]), dtype=np.float32)
        if raw.size != w * h:
            continue  # truncated upload
        meta = json.load(open(os.path.join(SAVE_DIR, f)))
        if "pose_column_major" not in meta:
            continue
        frames.append({
            "ts": ts, "w": w, "h": h,
            "depth": raw.reshape((h, w)),
            "meta": meta,
            "pose": np.array(meta["pose_column_major"], dtype=np.float64).reshape((4, 4), order="F"),
        })
    # keep only the session with the most frames
    by_session = {}
    for fr in frames:
        by_session.setdefault(fr["meta"].get("session_id", "unknown"), []).append(fr)
    best = max(by_session.values(), key=len)
    print(f"{len(frames)} usable frames, using session with {len(best)}")
    return best


def scaled_intrinsics(fr):
    sx = fr["w"] / fr["meta"]["rgb_width"]
    sy = fr["h"] / fr["meta"]["rgb_height"]
    return (fr["meta"]["fx"] * sx, fr["meta"]["fy"] * sy,
            fr["meta"]["cx"] * sx, fr["meta"]["cy"] * sy)


def frame_cloud(fr, pose, depth_max, voxel):
    """Point cloud in world space under the given pose."""
    fx, fy, cx, cy = scaled_intrinsics(fr)
    us, vs = np.meshgrid(np.arange(fr["w"]), np.arange(fr["h"]))
    d = fr["depth"]
    m = (d > 0.05) & (d < depth_max)
    # ARKit camera-local: X right, Y up, Z backward
    x = (us - cx) * d / fx
    y = -(vs - cy) * d / fy
    z = -d
    pts = np.stack([x[m], y[m], z[m], np.ones(m.sum())], axis=-1)
    world = (pose @ pts.T).T[:, :3]
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(world)
    pc = pc.voxel_down_sample(voxel)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 4, max_nn=30))
    return pc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", default="refined_mesh.ply")
    p.add_argument("--voxel", type=float, default=0.008, help="TSDF voxel size (m)")
    p.add_argument("--depth_trunc", type=float, default=0.8, help="Max depth to keep (m)")
    p.add_argument("--icp_dist", type=float, default=0.03, help="ICP correspondence distance (m)")
    p.add_argument("--no_icp", action="store_true", help="Skip refinement (compare raw ARKit poses)")
    args = p.parse_args()

    frames = load_frames()
    if not frames:
        raise SystemExit("No usable frames.")

    reg_voxel = 0.005
    refined_poses = []
    correction = np.eye(4)
    model = None
    total_corr = 0.0

    for i, fr in enumerate(frames):
        init_pose = correction @ fr["pose"]
        if model is None or args.no_icp:
            pose = init_pose
        else:
            src = frame_cloud(fr, init_pose, args.depth_trunc, reg_voxel)
            if len(src.points) < 100:
                pose = init_pose
            else:
                reg = o3d.pipelines.registration.registration_icp(
                    src, model, args.icp_dist, np.eye(4),
                    o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                if reg.fitness > 0.3:
                    correction = reg.transformation @ correction
                    total_corr += np.linalg.norm(reg.transformation[:3, 3])
                    pose = reg.transformation @ init_pose
                else:
                    pose = init_pose
        refined_poses.append(pose)

        pc = frame_cloud(fr, pose, args.depth_trunc, reg_voxel)
        model = pc if model is None else model + pc
        if i % 10 == 0:
            model = model.voxel_down_sample(reg_voxel)
            model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=reg_voxel * 4, max_nn=30))
            print(f"  aligned {i+1}/{len(frames)} frames...")

    print(f"Total ICP drift correction applied: {total_corr*1000:.1f}mm")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=args.voxel, sdf_trunc=args.voxel * 5,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

    intr = None
    for fr, pose in zip(frames, refined_poses):
        fx, fy, cx, cy = scaled_intrinsics(fr)
        if intr is None:
            intr = o3d.camera.PinholeCameraIntrinsic(fr["w"], fr["h"], fx, fy, cx, cy)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.full((fr["h"], fr["w"], 3), 128, dtype=np.uint8)),
            o3d.geometry.Image(fr["depth"]),
            depth_scale=1.0, depth_trunc=args.depth_trunc, convert_rgb_to_intensity=False)
        volume.integrate(rgbd, intr, np.linalg.inv(pose @ ARKIT_TO_CV))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    ext = mesh.get_axis_aligned_bounding_box().get_extent() if len(mesh.vertices) else [0, 0, 0]
    print(f"Mesh: {len(mesh.vertices):,} vertices, {len(mesh.triangles):,} triangles")
    print(f"Bounding box (m): {np.round(ext, 3)}")

    if len(mesh.vertices):
        o3d.io.write_triangle_mesh(args.save, mesh)
        print(f"Saved to {args.save}")
        o3d.visualization.draw_geometries([mesh], window_name="Refined mesh", mesh_show_back_face=True)


if __name__ == "__main__":
    main()
