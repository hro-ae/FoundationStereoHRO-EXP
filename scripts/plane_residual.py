"""
Subtract a fitted plane from a disparity map and visualize the residual.

For a flat board photographed from an angle, the dominant signal in disparity
is the board itself; components only perturb it by a few px. Fitting a plane
(via RANSAC) and subtracting it yields a "height above board" map, which is
much more informative for inspection than raw depth.

Usage:
    python scripts/plane_residual.py \
        --disp ./test_outputs/disp.npy \
        --rgb  ./rectified_calib/Image__2026-04-30__16-39-33_rect.png \
        --out_dir ./test_outputs

Math: for any planar surface, disparity d(u,v) = a*u + b*v + c. RANSAC fits
that on inlier pixels (the board), so the residual r = d - (a*u + b*v + c)
is non-zero only where the surface deviates from the plane (components,
solder joints, cables, etc.). Sign convention: r > 0  ⇒  closer to camera.
"""

import argparse
import os

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--disp", required=True, help="path to disp.npy from run_demo.py")
    ap.add_argument("--rgb", required=True, help="rectified left image used for the run")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--threshold_pct", type=float, default=85,
                    help="percentile of |residual| to use as 'sticking up' threshold")
    ap.add_argument("--max_disp", type=float, default=None,
                    help="ignore disparities above this value (filters bad matches near edges)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    disp = np.load(args.disp).astype(np.float32)
    rgb = cv2.imread(args.rgb)
    H, W = disp.shape
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))

    valid = np.isfinite(disp) & (disp > 0)
    if args.max_disp is not None:
        valid &= disp < args.max_disp
    print(f"Valid pixels: {valid.sum()}/{disp.size}")

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = np.stack([u[valid].ravel(), v[valid].ravel()], axis=-1).astype(np.float32)
    y = disp[valid].ravel()

    # RANSAC plane fit: d = a*u + b*v + c. residual_threshold in disparity (px).
    # 0.5 px is tight; raise if scene is rough.
    ransac = RANSACRegressor(residual_threshold=0.5, max_trials=200, random_state=0)
    ransac.fit(X, y)
    a, b = ransac.estimator_.coef_
    c = ransac.estimator_.intercept_
    inlier_frac = ransac.inlier_mask_.mean()
    print(f"Plane fit:  d = {a:+.4f}*u {b:+.4f}*v {c:+.2f}")
    print(f"Inlier fraction: {inlier_frac*100:.1f}%")

    # Plane disparity at every pixel
    plane_disp = a * u + b * v + c
    residual = disp - plane_disp
    residual[~valid] = 0

    # Symmetric color around 0; use 99th-percentile of |residual| to clip
    p99 = float(np.percentile(np.abs(residual[valid]), 99)) or 1.0
    res_norm = np.clip(residual / p99, -1, 1)
    res_uint8 = ((res_norm + 1) / 2 * 255).astype(np.uint8)
    res_color = cv2.applyColorMap(res_uint8, cv2.COLORMAP_JET)
    res_color[~valid] = 0

    # Highlight: pixels significantly above the plane (closer to camera)
    pos_thr = float(np.percentile(residual[valid], args.threshold_pct))
    above = (residual > pos_thr) & valid
    overlay = rgb.copy()
    overlay[above] = (overlay[above] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    # Side-by-side: rgb | residual map | overlay
    sbs = np.hstack([rgb, res_color, overlay])
    out_sbs = os.path.join(args.out_dir, "plane_residual.png")
    cv2.imwrite(out_sbs, sbs)

    np.save(os.path.join(args.out_dir, "residual.npy"), residual)

    print(f"Wrote {out_sbs}")
    print(f"Wrote {os.path.join(args.out_dir, 'residual.npy')}")
    print(f"Threshold for 'above plane' (p{args.threshold_pct}): {pos_thr:+.3f} px disparity")


if __name__ == "__main__":
    main()
