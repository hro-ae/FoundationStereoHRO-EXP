"""
Convert a disparity map to a 3D point cloud (PLY) using an arbitrary baseline.

Output coordinates are in "baseline units" — i.e., the *shape* of the cloud is
correct, but it is scaled by an unknown factor. Multiply XYZ by the true
baseline (in meters) afterwards if you ever measure it.

Coloring:
    --color rgb       use the rectified RGB (default)
    --color residual  use the plane-fit residual heatmap (red = above board)

Usage:
    python scripts/disp_to_pointcloud.py \
        --disp ./test_outputs/disp.npy \
        --rgb  ./rectified_calib/Image__2026-04-30__16-39-33_rect.png \
        --K    ./rectified_calib/K_rect.txt \
        --out  ./test_outputs/cloud_relative.ply \
        --color rgb

Note: K_rect.txt was written assuming the original full-res image (the
rectification step uses the original size). disp.npy is at the *scaled*
resolution from run_demo.py, so we rescale K by --scale to match.
"""

import argparse
import os

import cv2
import numpy as np
import open3d as o3d


def load_K(path):
    nums = np.loadtxt(path, max_rows=1)
    if nums.size != 9:
        raise ValueError(f"{path}: expected 9 numbers on line 1, got {nums.size}")
    return nums.reshape(3, 3).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disp", required=True)
    ap.add_argument("--rgb", required=True, help="rectified left image used for the run")
    ap.add_argument("--K", required=True, help="K_rect.txt (intrinsics for the FULL-res rectified image)")
    ap.add_argument("--out", required=True, help="output PLY path")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="--scale value used in run_demo.py (rescales K to match disp resolution)")
    ap.add_argument("--baseline", type=float, default=1.0,
                    help="placeholder baseline; output XYZ is in these units")
    ap.add_argument("--color", choices=["rgb", "residual"], default="rgb")
    ap.add_argument("--residual_npy", default=None,
                    help="path to residual.npy (required if --color residual)")
    ap.add_argument("--max_z", type=float, default=None,
                    help="clip points farther than this (in baseline units)")
    ap.add_argument("--remove_invisible", type=int, default=1,
                    help="drop pixels whose right-image match falls outside the right image")
    args = ap.parse_args()

    disp = np.load(args.disp).astype(np.float32)
    H, W = disp.shape
    rgb = cv2.imread(args.rgb)
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))

    K = load_K(args.K).copy()
    K[:2] *= args.scale  # K was for full-res; disp is at args.scale of that
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = np.isfinite(disp) & (disp > 0)

    if args.remove_invisible:
        # Match run_demo.py logic: drop pixels whose right-image x falls < 0
        u, _ = np.meshgrid(np.arange(W), np.arange(H))
        right_x = u - disp
        valid &= right_x >= 0

    Z = np.where(valid, fx * args.baseline / disp, 0.0).astype(np.float32)
    if args.max_z is not None:
        valid &= Z < args.max_z

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    mask = valid.reshape(-1)

    if args.color == "rgb":
        # OpenCV is BGR; Open3D expects RGB in [0, 1]
        col = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0
    elif args.color == "residual":
        if args.residual_npy is None:
            raise ValueError("--color residual requires --residual_npy")
        residual = np.load(args.residual_npy).astype(np.float32)
        if residual.shape != (H, W):
            raise ValueError(f"residual shape {residual.shape} != disp shape {(H, W)}")
        valid_res = np.isfinite(residual) & np.isfinite(disp)
        p99 = float(np.percentile(np.abs(residual[valid_res]), 99)) or 1.0
        norm = np.clip(residual / p99, -1, 1)
        u8 = ((norm + 1) / 2 * 255).astype(np.uint8)
        bgr = cv2.applyColorMap(u8, cv2.COLORMAP_JET)
        col = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3) / 255.0

    xyz = xyz[mask]
    col = col[mask]
    print(f"Points after masking: {xyz.shape[0]:,}")
    print(f"Z range: {xyz[:,2].min():.3f} .. {xyz[:,2].max():.3f}  (baseline units)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    o3d.io.write_point_cloud(args.out, pcd)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
