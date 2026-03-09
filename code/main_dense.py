"""
===================================================
CSCI 1430 - Brown University
Camera Geometry - main_dense.py

Part A: Calibrated Dense Stereo via Plane Sweeping
===================================================

Our stencil code that calls estimate_camera_matrix (Task 1) to compute
camera matrices from ArUco-marked images, then calls back_project and
plane_sweep_stereo (Task 2) to produce a dense depth map and 3D point cloud.

Usage:
    python main_dense.py
    python main_dense.py --dataset dollar
    python main_dense.py --n-planes 64 --win-size 9      (faster, coarser)
    python main_dense.py --n-planes 128 --max-views 6    (slower, denser)
"""

import argparse
import os
import time
import numpy as np
import cv2

# Path setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_dense')
os.makedirs(OUTPUT_DIR, exist_ok=True)

from helpers import (
    get_markers,
    detect_aruco_points,
    compute_lambda_range,
    select_nearest_views,
    save_reprojections,
    save_depth_visualization,
    save_dense_cloud,
    save_point_cloud_ply,
)

from student import (
    estimate_camera_matrix,
    back_project,
    plane_sweep_stereo,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Part A: Calibrated Dense Stereo")
    parser.add_argument('--dataset', type=str, default='cards',
                        choices=['cards', 'dollar', 'mikeandikes'],
                        help='Which image set to use (default: cards)')
    parser.add_argument('--n-planes', type=int, default=96,
                        help='Number of depth planes to sweep (default: 96)')
    parser.add_argument('--win-size', type=int, default=11,
                        help='NCC window size (default: 11)')
    parser.add_argument('--max-views', type=int, default=4,
                        help='Number of nearest views to use (default: 4)')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Image scale factor for speed (default: 0.5)')
    parser.add_argument('--ncc-threshold', type=float, default=0.4,
                        help='Minimum NCC for 3D output (default: 0.4)')
    return parser.parse_args()


def main():
    args = parse_args()

    print("  Part A: Calibrated Dense Stereo via Plane Sweeping")

    # 1. Load images and compute M matrices (Task 1)

    markers = get_markers(os.path.join(DATA_DIR, 'markers.txt'))
    image_dir = os.path.join(DATA_DIR, args.dataset)

    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpeg', '.jpg'))
    ])
    print(f"\n  Loading {len(image_paths)} {args.dataset} images...")

    images_bgr = []
    Ms = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        # ArUco detection -> projection matrix estimation
        points2d, points3d = detect_aruco_points(img, markers)
        M, res = estimate_camera_matrix(points2d, points3d)

        name = os.path.basename(path)
        print(f"    {name}: residual={res:.1f}")
        images_bgr.append(img)
        Ms.append(M)

    # 1b. Reprojection check — validates M and student's project()
    reproj_path = os.path.join(OUTPUT_DIR, 'reprojections.png')
    save_reprojections(images_bgr, Ms, markers, reproj_path)
    print(f"  Reprojection check saved -> {reproj_path}")

    # 2. Prepare images at working resolution

    scale = args.scale
    S = np.diag([scale, scale, 1.0])   # scale the M matrices too
    Ms_work = [S @ M for M in Ms]

    grays = []
    rgbs = []
    for img in images_bgr:
        h, w = img.shape[:2]
        sm = cv2.resize(img, (int(w * scale), int(h * scale)))
        grays.append(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY).astype(np.float32))
        rgbs.append(cv2.cvtColor(sm, cv2.COLOR_BGR2RGB))

    # 3. Select reference view and neighbors

    ref_idx = len(images_bgr) // 2
    neighbors = select_nearest_views(ref_idx, Ms, args.max_views)

    print(f"\n  Reference: image{ref_idx:02d}")
    print(f"  Using {len(neighbors)} nearest views: "
          + ", ".join(f"image{j:02d} (d={d:.2f})" for j, d in neighbors))

    other_grays = [grays[j] for j, _ in neighbors]
    Ms_other_work = [Ms_work[j] for j, _ in neighbors]

    # 4. Compute depth range from markers

    lam_lo, lam_hi = compute_lambda_range(Ms_work[ref_idx], markers)
    if lam_lo > lam_hi:
        lam_lo, lam_hi = lam_hi, lam_lo
    lambdas = np.linspace(lam_lo, lam_hi, args.n_planes).astype(np.float32)
    print(f"  Depth range: [{lam_lo:.3f}, {lam_hi:.3f}], {args.n_planes} planes")

    # 5. Run plane-sweep stereo (Task 2)

    print(f"\n  Running plane-sweep stereo...")
    t0 = time.time()
    lambda_map = plane_sweep_stereo(
        grays[ref_idx], other_grays,
        Ms_work[ref_idx], Ms_other_work,
        lambdas, args.win_size, args.ncc_threshold)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s")

    # 6. Statistics

    valid = np.isfinite(lambda_map)
    n_valid = int(np.sum(valid))
    print(f"\n  Valid depth pixels: "
          f"{n_valid:,} / {valid.size:,} ({100*n_valid/valid.size:.1f}%)")

    # 7. Back-project to 3D

    ys, xs = np.where(valid)
    points2d = np.column_stack([xs.astype(float), ys.astype(float)])
    pts3d = back_project(Ms_work[ref_idx], points2d, lambda_map[valid])

    print(f"  3D points: {len(pts3d):,}")

    # 8. Filter to world bounding box (ArUco stage bounds)

    if len(pts3d) > 0:
        keep = np.all(np.isfinite(pts3d), axis=1)
        keep &= np.all(pts3d >= -1.0, axis=1)
        keep &= (pts3d[:, 0] <= 8.0) & (pts3d[:, 1] <= 8.0) & (pts3d[:, 2] <= 3.0)

        # Outlier removal
        if np.sum(keep) > 100:
            med = np.median(pts3d[keep], axis=0)
            dists = np.linalg.norm(pts3d - med, axis=1)
            keep &= dists < np.percentile(dists[keep], 98)

        pts3d = pts3d[keep]

        # Colors from reference image
        colors = rgbs[ref_idx][ys, xs] / 255.0
        colors = colors[keep]

        print(f"  After filtering: {len(pts3d):,} points")
        if len(pts3d) > 0:
            print(f"  Range: X=[{pts3d[:,0].min():.1f},{pts3d[:,0].max():.1f}] "
                  f"Y=[{pts3d[:,1].min():.1f},{pts3d[:,1].max():.1f}] "
                  f"Z=[{pts3d[:,2].min():.1f},{pts3d[:,2].max():.1f}]")
    else:
        colors = np.zeros((0, 3))

    # 9. Visualize

    depth_path = os.path.join(OUTPUT_DIR, 'dense_depth.png')
    save_depth_visualization(lambda_map, rgbs[ref_idx], depth_path)
    print(f"  Saved -> {depth_path}")

    cloud_path = os.path.join(OUTPUT_DIR, 'dense_cloud.png')
    save_dense_cloud(pts3d, colors, Ms, cloud_path)
    print(f"  Saved -> {cloud_path}")

    ply_path = os.path.join(OUTPUT_DIR, 'dense_cloud.ply')
    save_point_cloud_ply(pts3d, colors, ply_path)
    print(f"  Saved -> {ply_path}")

    print(f"\n  Output directory: {OUTPUT_DIR}")
    print(f"  View your point cloud: "
          f"https://browncsci1430.github.io/resources/pointcloud_viewer/")


if __name__ == '__main__':
    main()
