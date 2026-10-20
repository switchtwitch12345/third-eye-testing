#!/usr/bin/env python3
"""
Does ARKit's fused mesh resolve the tower?  (blocks are 15 x 25 x 75 mm; a pulled block is a 10-15 mm recess)

The app (mesh switched on from the viewer) writes <ts>_mesh.bin:
    "MSH1" | u32 anchors | 16 x f32 camera transform (column-major, metres)
    per anchor: u32 vertexCount | u32 faceCount | vertexCount x (f32 x,y,z world, metres) | faceCount x (u32 a,b,c)

We put the mesh into the marker-defined tower frame (same pose chain as the live tracker), then measure:
  * triangle edge length near the tower  -> to see 15 mm layers it must be well under ~8 mm
  * vertices that land on each tower face, per layer
  * flatness of the desk (noise)

    .venv/bin/python mesh_test.py [folder]           analyse recorded mesh files
    .venv/bin/python mesh_test.py --selftest         check the maths on a synthetic mesh
"""
import glob
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FLIP = np.diag([1.0, -1.0, -1.0, 1.0])
HALF, HEIGHT = 37.5, 15.0


def load_mesh(path):
    """-> (camera_transform 4x4 [metres], vertices Nx3 [metres, world], faces Mx3)"""
    b = open(path, "rb").read()
    assert b[:4] == b"MSH1", "not a mesh file"
    n = struct.unpack_from("<I", b, 4)[0]
    cam = np.array(struct.unpack_from("<16f", b, 8), np.float64).reshape(4, 4, order="F")
    off, V, F, base = 72, [], [], 0
    for _ in range(n):
        vc, fc = struct.unpack_from("<II", b, off)
        off += 8
        V.append(np.frombuffer(b, np.float32, vc * 3, off).reshape(vc, 3).astype(np.float64))
        off += vc * 12
        F.append(np.frombuffer(b, np.uint32, fc * 3, off).reshape(fc, 3).astype(np.int64) + base)
        off += fc * 12
        base += vc
    if not V:
        return cam, np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    return cam, np.concatenate(V), np.concatenate(F)


def tower_pose_from_frame(folder, ts_ms):
    """World<-tower transform (mm) from the nearest iPhone frame where the markers are visible."""
    import cv2
    import face_crops as fc
    stamps = sorted(int(os.path.basename(f).split("_")[0]) for f in glob.glob(f"{folder}/*_intrinsics.json"))
    for ts in sorted(stamps, key=lambda s: abs(s - ts_ms))[:40]:
        img = cv2.imread(f"{folder}/{ts}_rgb.jpg")
        if img is None:
            continue
        meta = json.load(open(f"{folder}/{ts}_intrinsics.json"))
        sol = fc.solve_pose(img, fc.iphone_K(meta, img.shape[1]))
        if not sol:
            continue
        R, t, err = sol
        C = np.eye(4); C[:3, :3] = R; C[:3, 3] = t
        P = np.array(meta["pose_column_major"], np.float64).reshape(4, 4, order="F")
        P[:3, 3] *= 1000.0
        return P @ FLIP @ C, err
    return None, None


def analyse(V_world_m, F, W_T_tower):
    """Print resolution metrics.  V in metres (world), W_T_tower in mm."""
    Vh = np.c_[V_world_m * 1000.0, np.ones(len(V_world_m))]
    X = (np.linalg.inv(W_T_tower) @ Vh.T).T[:, :3]                 # tower frame, mm
    near = (np.abs(X[:, 0]) < 100) & (np.abs(X[:, 1]) < 100) & (X[:, 2] > -20) & (X[:, 2] < 175)
    res = {"vertices_total": len(X), "vertices_near_tower": int(near.sum())}

    ok = near[F].all(axis=1)
    if ok.any():
        T = F[ok]
        e = np.concatenate([np.linalg.norm(X[T[:, i]] - X[T[:, (i + 1) % 3]], axis=1) for i in range(3)])
        res["edge_mm"] = (float(np.percentile(e, 10)), float(np.median(e)), float(np.percentile(e, 90)))
    on_face = (X[:, 2] > 0) & (X[:, 2] < 150) & near & (
        ((np.abs(np.abs(X[:, 0]) - HALF) < 6) & (np.abs(X[:, 1]) < HALF + 6)) |
        ((np.abs(np.abs(X[:, 1]) - HALF) < 6) & (np.abs(X[:, 0]) < HALF + 6)))
    res["on_tower_faces"] = int(on_face.sum())
    res["per_layer"] = [int(((X[:, 2] >= HEIGHT * l) & (X[:, 2] < HEIGHT * (l + 1)) & on_face).sum()) for l in range(10)]
    ring = (np.abs(X[:, 2]) < 25) & (np.maximum(np.abs(X[:, 0]), np.abs(X[:, 1])) > 80) & (np.abs(X[:, 0]) < 300) & (np.abs(X[:, 1]) < 300)
    res["desk_n"] = int(ring.sum())
    if ring.sum() > 30:
        res["desk_z"] = (float(np.median(X[ring, 2])), float(X[ring, 2].std()))
    return res


def report(res, label=""):
    print(f"--- {label}")
    print(f"vertices: {res['vertices_total']} total, {res['vertices_near_tower']} within 10 cm of the tower, "
          f"{res['on_tower_faces']} on its faces")
    if "edge_mm" in res:
        p10, med, p90 = res["edge_mm"]
        print(f"triangle edge near tower: median {med:.1f} mm  (p10 {p10:.1f}, p90 {p90:.1f})")
        need = 7.5
        print(f"  -> to resolve a 15 mm layer pitch it needs edges <= ~{need} mm: "
              f"{'YES, fine enough' if med <= need else 'NO, about %.0fx too coarse' % (med / need)}")
    print("vertices on the faces, per layer 1..10:", res["per_layer"])
    layers_ok = sum(1 for n in res["per_layer"] if n >= 4)
    print(f"  -> {layers_ok}/10 layers have >= 4 vertices on their face")
    if "desk_z" in res:
        print(f"desk flatness: median z {res['desk_z'][0]:+.1f} mm, scatter {res['desk_z'][1]:.1f} mm  (n={res['desk_n']})")


# ------------------------------------------------------------------------------------------------
def selftest():
    import tempfile
    rng = np.random.default_rng(0)

    def synth(spacing):
        pts, tri = [], []
        def grid(origin, u, v, nu, nv):
            base = len(pts)
            for i in range(nu + 1):
                for j in range(nv + 1):
                    pts.append(origin + u * i / nu + v * j / nv)
            for i in range(nu):
                for j in range(nv):
                    a = base + i * (nv + 1) + j
                    tri.append((a, a + 1, a + nv + 1)); tri.append((a + 1, a + nv + 2, a + nv + 1))
        n = int(round(75 / spacing)); nz = int(round(150 / spacing))
        for sx in (-1, 1):
            grid(np.array([sx * 37.5, -37.5, 0.0]), np.array([0, 75.0, 0]), np.array([0, 0, 150.0]), n, nz)
            grid(np.array([-37.5, sx * 37.5, 0.0]), np.array([75.0, 0, 0]), np.array([0, 0, 150.0]), n, nz)
        nd = int(round(500 / max(spacing, 25)))
        grid(np.array([-250.0, -250.0, 0.0]), np.array([500.0, 0, 0]), np.array([0, 500.0, 0]), nd, nd)
        return np.array(pts), np.array(tri)

    # a made-up world<-tower transform (rotation about x and y, translation), in mm
    ax, ay = 0.3, -0.5
    Rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)], [0, 1, 0], [-np.sin(ay), 0, np.cos(ay)]])
    W = np.eye(4); W[:3, :3] = Ry @ Rx; W[:3, 3] = [800, -200, 400]
    ok = True
    for spacing in (30.0, 5.0):
        P, T = synth(spacing)
        P += rng.normal(0, 1.0, P.shape)                             # 1 mm noise
        Vw = ((W @ np.c_[P, np.ones(len(P))].T).T[:, :3]) / 1000.0     # world, metres
        # write exactly what the app writes
        b = bytearray(b"MSH1" + struct.pack("<I", 1))
        b += struct.pack("<16f", *np.eye(4).reshape(-1, order="F"))
        b += struct.pack("<II", len(Vw), len(T)) + Vw.astype(np.float32).tobytes() + T.astype(np.uint32).tobytes()
        path = os.path.join(tempfile.mkdtemp(), "x_mesh.bin"); open(path, "wb").write(bytes(b))
        cam, V, F = load_mesh(path)
        assert V.shape == Vw.shape and F.shape == T.shape, "parse mismatch"
        res = analyse(V, F, W)
        report(res, f"synthetic mesh, true spacing {spacing:g} mm")
        med = res["edge_mm"][1]
        good = abs(med - spacing * (1.0 if spacing < 10 else 1.2)) < spacing * 0.45
        deskd = res.get("desk_z", (99, 99))
        good &= abs(deskd[0]) < 3
        print(f"  check: edge reading plausible and desk at z=0: {'PASS' if good else 'FAIL'}\n")
        ok &= good
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


def main():
    if "--selftest" in sys.argv:
        sys.exit(0 if selftest() else 1)
    folder = next((a for a in sys.argv[1:] if not a.startswith("-")), os.path.join(HERE, "iphone_saved"))
    files = sorted(glob.glob(f"{folder}/*_mesh.bin"))
    if not files:
        sys.exit(f"no *_mesh.bin in {folder}. Turn 'ARKit mesh' ON in the viewer and let it run ~30 s on the tower.")
    print(f"{len(files)} mesh files in {folder}")
    last = files[-1]
    ts = int(os.path.basename(last).split("_")[0])
    W, err = tower_pose_from_frame(folder, ts)
    if W is None:
        sys.exit("could not find the markers in any iPhone frame near the newest mesh file "
                 "(the iPhone must be recording so RGB frames are saved too)")
    cam, V, F = load_mesh(last)
    report(analyse(V, F, W), f"newest mesh {os.path.basename(last)} (marker reproj {err:.1f} px)")


if __name__ == "__main__":
    main()
