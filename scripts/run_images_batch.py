import os
import sys
import cv2
import imageio
import numpy as np
import torch
import logging
import open3d as o3d

from imageio.core import Array
from omegaconf import OmegaConf

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import set_logging_format, set_seed, vis_disparity
# from Utils import set_logging_format, set_seed, vis_disparity, depth2xyzmap, toOpen3dCloud
from core.foundation_stereo import FoundationStereo
from core.utils.utils import InputPadder


def undistort_pair(img1, img2, K, dist):
    # newCameraMatrix defaults to K, so intrinsics are preserved
    u1 = cv2.undistort(img1, K, dist)
    u2 = cv2.undistort(img2, K, dist)
    return u1, u2

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

set_logging_format()
set_seed(0)
torch.autograd.set_grad_enabled(False)

ckpt_dir = r"C:\Users\Marian\PycharmProjects\FoundationStereo\pretrained_models\23-51-11\model_best_bp2-001.pth"
cfg = OmegaConf.load(f'{os.path.dirname(ckpt_dir)}/cfg.yaml')
if 'vit_size' not in cfg:
    cfg['vit_size'] = 'vitl'

args = OmegaConf.create(cfg)
logging.info(f"args:\n{args}")
logging.info(f"Using pretrained model from {ckpt_dir}")

model = FoundationStereo(args)

ckpt = torch.load(ckpt_dir)
logging.info(f"ckpt global_step:{ckpt['global_step']}, epoch:{ckpt['epoch']}")
model.load_state_dict(ckpt['model'])

model.cuda()
model.eval()

left_images_dir = r"C:\Users\Marian\Desktop\HVH PINS\first_pos_fs"
right_images_dir = r"C:\Users\Marian\Desktop\HVH PINS\second_pos_fs"

output_path = r"C:\Users\Marian\Desktop\HVH PINS\fs_clouds"

left_image_paths = []
right_image_paths = []

K = np.array([2197.042, 0, 1335.617, 0, 2193.379, 1132.879, 0, 0, 1], dtype=np.float64).reshape(3, 3)
dist = np.array([1.2714, 58.2875, 0.0005, -0.0021, 11.1459, 1.4880, 58.7450, 23.8935, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

hiera = 0
valid_iters = 16

os.makedirs(output_path, exist_ok=True)

for f in os.listdir(left_images_dir):
    left_image_paths.append(os.path.join(left_images_dir, f))

left_image_paths.sort(key=lambda f: os.path.getmtime(f))

for f in os.listdir(right_images_dir):
    right_image_paths.append(os.path.join(right_images_dir, f))

right_image_paths.sort(key=lambda f: os.path.getmtime(f))

for l_img_path, r_img_path in zip(left_image_paths, right_image_paths):
    img_left = cv2.imread(l_img_path)
    img_right = cv2.imread(r_img_path)

    h, w = img_left.shape[:2]

    img_left, img_right = undistort_pair(img_left, img_right, K, dist)

    g1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    pts1, pts2 = detect_and_match(g1, g2, ratio=0.75)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

    if F is None:
        raise RuntimeError("findFundamentalMat failed")
    inliers = mask.ravel().astype(bool)
    pts1_in = pts1[inliers]
    pts2_in = pts2[inliers]

    rect1, rect2, K_rect = rectify_calibrated(img_left, img_right, pts1_in, pts2_in, K, (w, h))

    print(f"K rect: {K_rect}")

    rect1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB)
    rect2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2RGB)

    img0 = Array(rect1)
    img1 = Array(rect2)
    scale = 0.5
    assert scale <= 1, "scale must be <=1"
    img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
    img1 = cv2.resize(img1, fx=scale, fy=scale, dsize=None)
    H, W = img0.shape[:2]
    img0_ori = img0.copy()
    logging.info(f"img0: {img0.shape}")

    img0 = torch.as_tensor(img0).cuda().float()[None].permute(0, 3, 1, 2)
    img1 = torch.as_tensor(img1).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0.shape, divis_by=32, force_square=False)
    img0, img1 = padder.pad(img0, img1)

    with torch.cuda.amp.autocast(True):
        if not hiera:
            disp = model.forward(img0, img1, iters=valid_iters, test_mode=True)
        else:
            disp = model.run_hierachical(img0, img1, iters=valid_iters, test_mode=True, small_ratio=0.5)
    disp = padder.unpad(disp.float())
    disp = disp.data.cpu().numpy().reshape(H, W)
    vis = vis_disparity(disp)
    vis = np.concatenate([img0_ori, vis], axis=1)
    # imageio.imwrite(f'{args.out_dir}/vis.png', vis)
    # np.save(f'{args.out_dir}/disp.npy', disp)
    # logging.info(f"Outputs saved to {args.out_dir}")

    H, W = disp.shape
    print(f"PC H: {H}, PC W: {W}")
    rgb = rect1.copy()
    if rgb.shape[:2] != (H, W):
        rgb = cv2.resize(rgb, (W, H))

    K_rect[:2] *= scale  # K was for full-res; disp is at args.scale of that
    fx, fy = K_rect[0, 0], K_rect[1, 1]
    cx, cy = K_rect[0, 2], K_rect[1, 2]

    valid = np.isfinite(disp) & (disp > 0)

    if 1:
        # Match run_demo.py logic: drop pixels whose right-image x falls < 0
        u, _ = np.meshgrid(np.arange(W), np.arange(H))
        right_x = u - disp
        valid &= right_x >= 0

    Z = np.where(valid, fx * 0.005 / disp, 0.0).astype(np.float32)

    valid &= Z < 50

    u, v = np.meshgrid(np.arange(W), np.arange(H))
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy

    xyz = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    mask = valid.reshape(-1)

    col = rgb.reshape(-1, 3) / 255.0

    xyz = xyz[mask]
    col = col[mask]
    print(f"Points after masking: {xyz.shape[0]:,}")
    print(f"Z range: {xyz[:, 2].min():.3f} .. {xyz[:, 2].max():.3f}  (baseline units)")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(col.astype(np.float64))

    left_img_name = l_img_path.split("\\")[-1].split(".")[0]
    right_img_name = r_img_path.split("\\")[-1].split(".")[0]

    out_path = os.path.join(output_path, f"{left_img_name}_{right_img_name}.ply")

    # o3d.io.write_point_cloud(out_path, pcd)
    # imageio.imwrite(out_path.replace('.ply', '.png'), rect1)
    cv2.imwrite("rectified_left.png", cv2.cvtColor(rect1, cv2.COLOR_RGB2BGR))
    cv2.imwrite("rectified_right.png", cv2.cvtColor(rect2, cv2.COLOR_RGB2BGR))
    break

