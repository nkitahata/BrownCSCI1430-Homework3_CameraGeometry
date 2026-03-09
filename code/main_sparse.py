"""
===================================================
CSCI 1430 - Brown University
Camera Geometry - main_sparse.py

Part B: Uncalibrated Stereo
====================================================

Our stencil code that runs SIFT matching on a selected image pair,
filters with ransac_fundamental_matrix (Task 4) to estimate F,
then uses F to rectify the stereo pair and compute a disparity map
via compute_disparity_map (Task 5).

Usage:
    python main_sparse.py
    python main_sparse.py --dataset notredame
    python main_sparse.py --dataset dollar
    python main_sparse.py --ransac-iters 500
    python main_sparse.py --num-keypoints 8000
"""

import argparse
import os
import numpy as np
import cv2
from skimage import img_as_float32

# Path setup

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'data'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_sparse')
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATASET_DIRS = {
    'cards': 'cards',
    'dollar': 'dollar',
    'mikeandikes': 'mikeandikes',
    'notredame': 'notredame',
}
CALIBRATED_DATASETS = {'cards', 'dollar', 'mikeandikes'}

from helpers import (
    get_markers,
    get_matches,
    show_matches,
    detect_aruco_points,
    compute_fundamental_from_cameras,
    compare_fundamental_matrices,
    save_rectified_pair,
    save_disparity_visualization,
)

from student import (
    estimate_camera_matrix,
    ransac_fundamental_matrix,
    compute_disparity_map,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Part B: Uncalibrated Stereo")
    parser.add_argument('--dataset', type=str, default='cards',
                        choices=list(DATASET_DIRS.keys()),
                        help='Which image set to use (default: cards)')
    parser.add_argument('--ransac-iters', type=int, default=500,
                        help='Number of RANSAC iterations (default: 500)')
    parser.add_argument('--num-keypoints', type=int, default=10000,
                        help='Number of SIFT keypoints (default: 10000)')
    parser.add_argument('--max-disparity', type=int, default=64,
                        help='Maximum disparity to search (default: 64)')
    parser.add_argument('--no-intermediate-vis', action='store_true',
                        help='Suppress intermediate match visualizations')
    parser.add_argument('--visualize-ransac', action='store_true',
                        help='Show RANSAC convergence plots')
    return parser.parse_args()


# Rectification + disparity helper

def rectify_and_compute_disparity(img1, img2, F, inliers1, inliers2,
                                  max_disparity, label, win_size=11):
    """
    Rectify a stereo pair using F and compute disparity.

    Returns (rect_left, rect_right, disparity) or None if rectification fails.
    """
    h, w = img1.shape[:2]

    # Rectify using OpenCV's uncalibrated rectification (Hartley 1999)
    pts1 = inliers1.astype(np.float64)
    pts2 = inliers2.astype(np.float64)
    ret, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, (w, h))

    if not ret:
        print(f"    Rectification failed for {label}")
        return None

    # Warp images
    rect_left = cv2.warpPerspective(img1, H1, (w, h))
    rect_right = cv2.warpPerspective(img2, H2, (w, h))

    # Report rectification quality: transform inlier points, check y-offsets
    ones = np.ones((len(pts1), 1))
    pts1_h = np.hstack([pts1, ones])
    pts2_h = np.hstack([pts2, ones])
    pts1_rect = (H1 @ pts1_h.T).T
    pts2_rect = (H2 @ pts2_h.T).T
    pts1_rect = pts1_rect[:, :2] / pts1_rect[:, 2:3]
    pts2_rect = pts2_rect[:, :2] / pts2_rect[:, 2:3]
    dy = np.abs(pts1_rect[:, 1] - pts2_rect[:, 1])
    print(f"    Rectification quality: mean |dy|={dy.mean():.2f}px  "
          f"median={np.median(dy):.2f}px  max={dy.max():.1f}px")

    # Save rectified pair visualization
    rect_path = os.path.join(OUTPUT_DIR, f'{label}_rectified.png')
    save_rectified_pair(rect_left, rect_right, rect_path)
    print(f"    Saved -> {rect_path}")

    # Convert to grayscale float32 for disparity computation
    rect_left_gray = img_as_float32(
        cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY))
    rect_right_gray = img_as_float32(
        cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY))

    # Compute disparity map (Task 5)
    print(f"    Computing disparity (win={win_size}, max_d={max_disparity})...")
    disparity = compute_disparity_map(
        rect_left_gray, rect_right_gray,
        win_size=win_size, max_disparity=max_disparity)

    # Post-processing: mask out black border pixels from rectification
    # and smooth salt-and-pepper noise with a median filter
    from scipy.ndimage import median_filter
    disparity[rect_left_gray < 0.01] = np.nan
    disparity[rect_right_gray < 0.01] = np.nan
    disparity = median_filter(disparity, size=max(5, win_size // 2 | 1))

    n_valid = np.sum(np.isfinite(disparity))
    print(f"    Disparity: {n_valid:,} valid pixels "
          f"({100*n_valid/(h*w):.1f}% of image)")

    # Save disparity visualization
    rect_left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
    disp_path = os.path.join(OUTPUT_DIR, f'{label}_disparity.png')
    save_disparity_visualization(disparity, rect_left_rgb, disp_path)
    print(f"    Saved -> {disp_path}")

    return rect_left, rect_right, disparity


def main():
    args = parse_args()

    print("  Part B: Uncalibrated Stereo")

    calibrated = args.dataset in CALIBRATED_DATASETS
    image_dir = os.path.join(DATA_DIR, DATASET_DIRS[args.dataset])

    # 1. Load images (and compute M matrices for calibrated datasets)

    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpeg', '.jpg'))
    ])
    print(f"\n  Loading {len(image_paths)} {args.dataset} images...")

    if calibrated:
        markers = get_markers(os.path.join(DATA_DIR, 'markers.txt'))

    images = []
    Ms = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            continue

        name = os.path.basename(path)
        if calibrated:
            points2d, points3d = detect_aruco_points(img, markers)
            M, res = estimate_camera_matrix(points2d, points3d)
            Ms.append(M)
            print(f"    {name}: residual={res:.1f}")
        else:
            print(f"    {name}")
        images.append(img)

    # Resize uncalibrated images for tractable disparity computation
    if not calibrated:
        scale = 0.5
        images = [cv2.resize(img, None, fx=scale, fy=scale) for img in images]
        print(f"    Resized to {images[0].shape[1]}x{images[0].shape[0]}")

    # 2. SIFT matching + RANSAC (Tasks 3-4)

    import student

    print(f"\n  --- {args.dataset} pair: SIFT + RANSAC ---")
    img1, img2 = images[0], images[1]

    points1, points2 = get_matches(img1, img2, args.num_keypoints)
    print(f"    SIFT matches: {len(points1)}")

    if not args.no_intermediate_vis:
        show_matches(img1, img2, points1, points2)

    F_est, inliers1, inliers2, residual = \
        ransac_fundamental_matrix(points1, points2, args.ransac_iters)
    print(f"    RANSAC inliers: {len(inliers1)} / {len(points1)}  "
          f"residual={residual:.2f}")

    if not args.no_intermediate_vis:
        show_matches(img1, img2, inliers1, inliers2)

    if args.visualize_ransac:
        print(f'    Visualizing RANSAC')
        student.visualize_ransac()
        student.inlier_counts = []
        student.inlier_residuals = []

    # Compare estimated F against ground-truth F from calibration
    if calibrated:
        M1, M2 = Ms[0], Ms[1]
        F_gt = compute_fundamental_from_cameras(M1, M2)
        f_dist = compare_fundamental_matrices(F_est, F_gt)
        print(f"    F vs F_gt (from M): {f_dist:.4f}  "
              f"(0 = identical, sqrt(2) = orthogonal)")

    # 3. Rectification + Disparity (Task 5)

    win_size = 31 if not calibrated else 11
    print(f"\n  --- {args.dataset} pair: Rectification + Disparity ---")
    rectify_and_compute_disparity(
        img1, img2, F_est, inliers1, inliers2,
        args.max_disparity, args.dataset, win_size=win_size)

    # Done

    print(f"\n  Output directory: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
