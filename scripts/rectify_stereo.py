"""
Stereo rectification — calibrated (with K) or uncalibrated.

Usage (uncalibrated, fast path):
    python scripts/rectify_stereo.py \
        --left  path/to/left.jpg  \
        --right path/to/right.jpg \
        --out_dir ./rectified

Usage (calibrated — give intrinsics K via --K_file or --K_inline):
    python scripts/rectify_stereo.py \
        --left  ... --right ... --out_dir ... \
        --K_inline 2925.5 0 1332 0 2925.7 1152 0 0 1

The K_file format is a plain text file with 9 whitespace-separated numbers
(row-major), one matrix.

Optional lens undistortion (applied to both images *before* rectification):
  * --dist          full OpenCV model, as returned by cv2.calibrateCamera,
                    i.e. [k1 k2 p1 p2 k3 ...] (4, 5, 8, 12, or 14 values)
  * --dist_radial   radial-only model [k1 k2], [k1 k2 k3], or
                    [k1 k2 k3 k4 k5 k6] — tangential terms are set to 0
Undistortion requires K (it needs the camera matrix), and is skipped entirely
when neither flag is given. The intrinsics K are unchanged by undistortion, so
the rest of the pipeline (and the written K_rect.txt) stay valid.

When K is given the script:
  * recovers R, t between the two views via the essential matrix,
  * uses cv2.stereoRectify (orthonormal, geometrically sound),
  * writes a K_rect.txt for FoundationStereo (intrinsics post-rectification).

Output depth from FoundationStereo is metric only if you also know the
baseline ||t|| in world units. ||t|| is recovered up to scale here, so the
generated K_rect.txt sets baseline=0 — fix it manually if you measure t.
"""

import argparse
import os

import cv2
import numpy as np


def detect_and_match(g1, g2, ratio=0.75):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(g1, None)
    kp2, des2 = sift.detectAndCompute(g2, None)
    if des1 is None or des2 is None:
        raise RuntimeError("SIFT found no descriptors in one of the images")

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(des1, des2, k=2)

    pts1, pts2 = [], []
    for m, n in knn:
        if m.distance < ratio * n.distance:
            pts1.append(kp1[m.queryIdx].pt)
            pts2.append(kp2[m.trainIdx].pt)
    return np.float32(pts1), np.float32(pts2)


def parse_K(args):
    if args.K_inline is not None:
        vals = args.K_inline
        if len(vals) != 9:
            raise ValueError("--K_inline needs exactly 9 numbers")
        return np.array(vals, dtype=np.float64).reshape(3, 3)
    if args.K_file is not None:
        nums = np.loadtxt(args.K_file).flatten()
        if nums.size != 9:
            raise ValueError(f"{args.K_file} must contain 9 numbers, got {nums.size}")
        return nums.reshape(3, 3)
    return None


def parse_dist(args):
    """Return an OpenCV distortion vector (np.float64) or None.

    --dist        : full model, used as-is ([k1 k2 p1 p2 k3 ...]).
    --dist_radial : radial-only; tangential p1=p2 forced to 0 and the
                    coefficients packed into OpenCV order [k1 k2 0 0 k3 ...].
    """
    if args.dist is not None and args.dist_radial is not None:
        raise ValueError("Pass only one of --dist or --dist_radial")
    if args.dist is not None:
        d = np.array(args.dist, dtype=np.float64)
        if d.size not in (4, 5, 8, 12, 14):
            raise ValueError(
                f"--dist needs 4, 5, 8, 12, or 14 coefficients (got {d.size})")
        return d
    if args.dist_radial is not None:
        r = list(args.dist_radial)
        if len(r) not in (2, 3, 6):
            raise ValueError(
                "--dist_radial needs 2 (k1 k2), 3 (k1 k2 k3), or "
                f"6 (k1 k2 k3 k4 k5 k6) values; got {len(r)}")
        # OpenCV order is [k1, k2, p1, p2, k3, k4, k5, k6]; tangential = 0
        d = [r[0], r[1], 0.0, 0.0] + r[2:]
        return np.array(d, dtype=np.float64)
    return None


def undistort_pair(img1, img2, K, dist):
    # newCameraMatrix defaults to K, so intrinsics are preserved
    u1 = cv2.undistort(img1, K, dist)
    u2 = cv2.undistort(img2, K, dist)
    return u1, u2


def rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F, size):
    w, h = size
    ok, H1, H2 = cv2.stereoRectifyUncalibrated(pts1_in, pts2_in, F, (w, h))
    if not ok:
        raise RuntimeError("stereoRectifyUncalibrated failed")
    rect1 = cv2.warpPerspective(img1, H1, (w, h))
    rect2 = cv2.warpPerspective(img2, H2, (w, h))
    return rect1, rect2, None  # no metric K_rect


def rectify_calibrated(img1, img2, pts1_in, pts2_in, K, size):
    w, h = size
    dist = np.zeros(5)  # images are already undistorted upstream (see --dist/--dist_radial)
    E, mask = cv2.findEssentialMat(pts1_in, pts2_in, K, cv2.RANSAC, 0.999, 1.0)
    inl = mask.ravel().astype(bool)
    pts1_e = pts1_in[inl]
    pts2_e = pts2_in[inl]
    print(f"Essential-matrix inliers: {inl.sum()}/{len(pts1_in)}")

    n_in, R, t, _ = cv2.recoverPose(E, pts1_e, pts2_e, K)
    print(f"recoverPose inliers: {n_in}")
    print(f"R =\n{R}")
    print(f"t (unit) =\n{t.ravel()}")
    angle_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    print(f"Rotation between views: {angle_deg:.3f} deg")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, t,
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0
    )
    map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    rect1 = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)
    K_rect = P1[:3, :3]
    return rect1, rect2, K_rect


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--ratio", type=float, default=0.75,
                    help="Lowe's ratio test threshold (lower = stricter)")
    ap.add_argument("--K_file", type=str, default=None,
                    help="Text file with 9 numbers (row-major K)")
    ap.add_argument("--K_inline", type=float, nargs=9, default=None,
                    help="9 numbers: fx 0 cx 0 fy cy 0 0 1")
    ap.add_argument("--dist", type=float, nargs="+", default=None,
                    help="OpenCV distortion coefficients k1 k2 p1 p2 [k3 ...] "
                         "(4, 5, 8, 12, or 14 values, as returned by cv2.calibrateCamera)")
    ap.add_argument("--dist_radial", type=float, nargs="+", default=None,
                    help="Radial-only distortion: k1 k2 [k3] or k1 k2 k3 k4 k5 k6 "
                         "(tangential terms set to 0)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    K = parse_K(args)
    dist = parse_dist(args)

    img1 = cv2.imread(args.left)
    img2 = cv2.imread(args.right)
    if img1 is None:
        raise FileNotFoundError(args.left)
    if img2 is None:
        raise FileNotFoundError(args.right)

    h, w = img1.shape[:2]
    if img2.shape[:2] != (h, w):
        img2 = cv2.resize(img2, (w, h))

    # Undistort before anything else, so matching/rectification see clean images.
    if dist is not None:
        if K is None:
            raise ValueError(
                "Undistortion needs K — pass --K_file or --K_inline alongside "
                "the distortion coefficients")
        print(f"Undistorting both images with dist = {dist.tolist()}")
        img1, img2 = undistort_pair(img1, img2, K, dist)
    else:
        print("No distortion coefficients given — skipping undistortion")

    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    pts1, pts2 = detect_and_match(g1, g2, ratio=args.ratio)
    if len(pts1) < 30:
        raise RuntimeError(
            f"Only {len(pts1)} good matches — try a less strict --ratio "
            f"(e.g. 0.85) or check that both images view the same scene"
        )

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)
    if F is None:
        raise RuntimeError("findFundamentalMat failed")
    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]
    print(f"Matches: {len(pts1)}, fundamental-matrix inliers: {len(pts1_in)}")

    if K is None:
        print("Mode: uncalibrated (no K provided)")
        rect1, rect2, K_rect = rectify_uncalibrated(img1, img2, pts1_in, pts2_in, F, (w, h))
    else:
        print(f"Mode: calibrated\nK =\n{K}")
        rect1, rect2, K_rect = rectify_calibrated(img1, img2, pts1_in, pts2_in, K, (w, h))

    base1 = os.path.splitext(os.path.basename(args.left))[0]
    base2 = os.path.splitext(os.path.basename(args.right))[0]
    out1 = os.path.join(args.out_dir, f"{base1}_rect.png")
    out2 = os.path.join(args.out_dir, f"{base2}_rect.png")
    cv2.imwrite(out1, rect1)
    cv2.imwrite(out2, rect2)

    preview = np.hstack([rect1, rect2])
    for y in range(0, h, max(1, h // 20)):
        cv2.line(preview, (0, y), (preview.shape[1], y), (0, 255, 0), 1)
    preview_path = os.path.join(args.out_dir, "preview_epilines.png")
    cv2.imwrite(preview_path, preview)

    if K_rect is not None:
        k_path = os.path.join(args.out_dir, "K_rect.txt")
        # FoundationStereo K.txt format: line 1 = 9 flat values, line 2 = baseline (m)
        flat = " ".join(f"{v:.6f}" for v in K_rect.flatten())
        with open(k_path, "w") as f:
            f.write(flat + "\n")
            f.write("0.0\n")  # baseline unknown — replace once measured
        print(f"Wrote {k_path}  (baseline placeholder=0.0; set this if you know it)")

    print(f"Wrote {out1}")
    print(f"Wrote {out2}")
    print(f"Wrote {preview_path}  (corresponding points should sit on the same green line)")


if __name__ == "__main__":
    main()
