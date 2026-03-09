"""
Helpers for Camera Geometry Pipeline
======================================

Please DO NOT MODIFY this! We use it to build
our pipelines.

Sections:
  1. Data Loading (markers, images)
  2. Feature Matching (SIFT + Lowe's ratio test)
  3. ArUco Detection (pipeline plumbing)
  4. Geometry Helpers (fundamental matrix from cameras)
  5. Depth Range and View Selection
  6. Visualization (matches, rectification, disparity, depth, point clouds)
"""

import os
import numpy as np
import cv2
from skimage import img_as_float32
import matplotlib.pyplot as plt
import random


# =============================================================================
#  1. Data Loading
# =============================================================================


def get_markers(markers_path):
    """
    Returns a dictionary mapping a marker ID to a 4x3 array
    containing the 3d points for each of the 4 corners of the
    marker in our scanning setup.
    """
    markers = {}
    with open(markers_path) as f:
        first_dim = 0
        second_dim = 0
        for i, line in enumerate(f.readlines()):
            if i == 0:
                first_dim, second_dim = [float(x) for x in line.split()]
            else:
                info = [float(x) for x in line.split()]
                markers[i] = [
                    [info[0], info[1], info[2]],
                    [info[0] + first_dim * info[3], info[1] + first_dim * info[4], info[2] + first_dim * info[5]],
                    [info[0] + first_dim * info[3] + second_dim * info[6], info[1] + first_dim * info[4] + second_dim * info[7], info[2] + first_dim * info[5] + second_dim * info[8]],
                    [info[0] + second_dim * info[6], info[1] + second_dim * info[7], info[2] + second_dim * info[8]],
                ]
    return markers


# =============================================================================
#  2. Feature Matching
# =============================================================================


def get_matches(image1, image2, num_keypoints=5000):
    """
    Wraps OpenCV's SIFT function and feature matcher.
    Returns two N x 2 numpy arrays, 2d points in image1 and image2
    that are proposed matches.
    """
    # Find keypoints and descriptors with SIFT
    sift = cv2.SIFT_create(nfeatures=num_keypoints)
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    # Match descriptors using 2NN + Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test to filter good matches
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]

    # Extract matched keypoints
    points1 = np.array([kp1[m.queryIdx].pt for m in good])
    points2 = np.array([kp2[m.trainIdx].pt for m in good])

    return points1, points2


# =============================================================================
#  3. ArUco Detection
# =============================================================================


def detect_aruco_points(image, markers):
    """
    Detect ArUco markers in an image and return 2D-3D point correspondences.

    This function handles the OpenCV ArUco detection boilerplate, extracting
    the 2D corner positions in the image and matching them to the known 3D
    positions from the markers dictionary.

    :param image: a single image (numpy array)
    :param markers: dictionary mapping marker ID -> 4x3 array of 3D points
    :return: (points2d, points3d)
             points2d: N x 2 array of detected 2D image coordinates
             points3d: N x 3 array of corresponding 3D world coordinates
    """
    # Try new-style API first (OpenCV >= 4.7), fall back to legacy
    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        markerCorners, markerIds, _ = detector.detectMarkers(image)
    except AttributeError:
        dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
        parameters = cv2.aruco.DetectorParameters_create()
        markerCorners, markerIds, _ = cv2.aruco.detectMarkers(
            image, dictionary, parameters=parameters)

    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    return np.array(points2d), np.array(points3d)


# =============================================================================
#  4. Geometry Helpers
# =============================================================================


def _skew(a):
    """3x3 skew-symmetric matrix [a]x."""
    return np.array([[    0, -a[2],  a[1]],
                     [ a[2],     0, -a[0]],
                     [-a[1],  a[0],     0]])


def compute_fundamental_from_cameras(M1, M2):
    """
    Compute the fundamental matrix directly from two projection matrices.

    :param M1: 3x4 projection matrix of camera 1
    :param M2: 3x4 projection matrix of camera 2
    :return: F, 3x3 fundamental matrix (rank 2)
    """
    A1 = M1[:, :3]
    A2 = M2[:, :3]
    t2 = M2[:, 3]
    C1 = -np.linalg.solve(A1, M1[:, 3])
    B = A2 @ np.linalg.inv(A1)
    a = A2 @ C1 + t2
    return _skew(a) @ B


def compare_fundamental_matrices(F_est, F_gt):
    """
    Compare an estimated F against ground-truth F.

    Since F is defined only up to scale and sign, we normalize both
    to unit Frobenius norm and report the minimum distance (accounting
    for sign ambiguity).

    :param F_est: 3x3 estimated fundamental matrix
    :param F_gt:  3x3 ground-truth fundamental matrix
    :return: distance in [0, sqrt(2)], where 0 = identical
    """
    Fe = F_est / np.linalg.norm(F_est)
    Fg = F_gt / np.linalg.norm(F_gt)
    return min(np.linalg.norm(Fe - Fg), np.linalg.norm(Fe + Fg))


# =============================================================================
#  5. Depth Range and View Selection
# =============================================================================


def compute_lambda_range(M_ref, markers):
    """
    Determine the range of the depth parameter `lam` from known
    3D marker positions.

    For each known 3D point P, we find the lam such that
    C_ref + lam * ray = P, where ray = A^{-1} @ projected_pixel.

    Returns (lam_lo, lam_hi) with 15% margins.
    """
    A = M_ref[:, :3]
    A_inv = np.linalg.inv(A)
    C = -np.linalg.solve(A, M_ref[:, 3])

    lam_vals = []
    for mid in markers:
        for p3d in markers[mid]:
            p3d = np.array(p3d)
            p2d_h = M_ref @ np.append(p3d, 1.0)
            if abs(p2d_h[2]) < 1e-8:
                continue
            ray = A_inv @ (p2d_h / p2d_h[2])
            dp = p3d - C
            j = np.argmax(np.abs(ray))
            lam_vals.append(dp[j] / ray[j])

    lo, hi = min(lam_vals), max(lam_vals)
    margin = (hi - lo) * 0.15
    return lo - margin, hi + margin


def select_nearest_views(ref_idx, Ms, max_views=4):
    """
    Select the `max_views` nearest cameras by Euclidean distance
    to the reference camera.

    :param ref_idx: index of the reference view
    :param Ms: list of 3x4 projection matrices
    :param max_views: number of neighbors to return
    :return: list of (view_index, distance) tuples, sorted by distance
    """
    def _null(M):
        C = np.linalg.svd(M)[2][-1]
        return C[:3] / C[3]
    centers = [_null(M) for M in Ms]
    C_ref = centers[ref_idx]
    dists = []
    for j in range(len(centers)):
        if j == ref_idx:
            continue
        d = np.linalg.norm(centers[j] - C_ref)
        dists.append((j, d))
    dists.sort(key=lambda x: x[1])
    return dists[:max_views]


# =============================================================================
#  6. Visualization
# =============================================================================


def save_reprojections(images, Ms, markers, output_path):
    """
    Reprojection check: overlay detected ArUco corners (green crosses) and
    reprojected 3D marker points (blue dots) for each image.
    Uses the student's project() function for the reprojection.
    """
    from student import project  # deferred so helpers.py has no top-level student dep

    pts3d = []
    for mid in markers:
        pts3d.extend(markers[mid])
    pts3d = np.array(pts3d)

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i in range(n):
        rgb = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
        axes[i].imshow(rgb)

        det2d, _ = detect_aruco_points(images[i], markers)
        axes[i].scatter(det2d[:, 0], det2d[:, 1],
                        marker='+', c='lime', s=60, linewidths=1.5,
                        label='Detected corners', zorder=3)

        reproj = project(Ms[i], pts3d)
        axes[i].scatter(reproj[:, 0], reproj[:, 1],
                        marker='o', c='dodgerblue', s=20, alpha=0.8,
                        label='Reprojected', zorder=4)

        axes[i].set_title(f'Image {i}', fontsize=10)
        axes[i].axis('off')
        if i == 0:
            axes[i].legend(fontsize=7, loc='upper right')

    fig.suptitle('ArUco Reprojection Check (blue should align with green)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def show_matches(image1, image2, points1, points2):
    """
    Shows matches from image1 to image2, represented by Nx2 arrays
    points1 and points2.
    """
    image1 = img_as_float32(image1)
    image2 = img_as_float32(image2)

    # Ensure both images are 3-channel for display
    if image1.ndim == 2:
        image1 = np.stack([image1] * 3, axis=-1)
    if image2.ndim == 2:
        image2 = np.stack([image2] * 3, axis=-1)

    # Pad the shorter image so heights match for hstack
    h1, h2 = image1.shape[0], image2.shape[0]
    if h1 < h2:
        pad = np.zeros((h2 - h1, image1.shape[1], image1.shape[2]),
                        dtype=image1.dtype)
        image1 = np.vstack([image1, pad])
    elif h2 < h1:
        pad = np.zeros((h1 - h2, image2.shape[1], image2.shape[2]),
                        dtype=image2.dtype)
        image2 = np.vstack([image2, pad])

    fig = plt.figure()
    fig.canvas.manager.set_window_title("Matches between image pair.")
    plt.axis('off')

    matches_image = np.hstack([image1, image2])
    plt.imshow(matches_image)

    shift = image1.shape[1]
    for i in range(0, points1.shape[0]):
        random_color = lambda: random.randint(0, 255)
        cur_color = ('#%02X%02X%02X' % (random_color(), random_color(), random_color()))

        x1 = points1[i, 1]
        y1 = points1[i, 0]
        x2 = points2[i, 1]
        y2 = points2[i, 0]

        x = np.array([x1, x2])
        y = np.array([y1, y2 + shift])
        plt.plot(y, x, c=cur_color, linewidth=0.5)

    plt.show()


def show_point_cloud(points3d, colors):
    """
    Show 3D points with their corresponding colors.
    Marker size adapts to point count for readable visualizations.
    """
    n = len(points3d)
    if n > 10000:
        s = 1.0
    elif n > 3000:
        s = 1.0
    else:
        s = 4.0

    fig = plt.figure(figsize=(9, 9))
    fig.canvas.manager.set_window_title("Recovered 3D points.")
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(points3d[:, 0], points3d[:, 1], points3d[:, 2],
               c=colors, s=s, alpha=0.8, edgecolors='none')

    # Equal aspect ratio for all axes
    mid = points3d.mean(axis=0)
    span = (points3d.max(axis=0) - points3d.min(axis=0)).max() / 2 * 1.1
    ax.set_xlim(mid[0] - span, mid[0] + span)
    ax.set_ylim(mid[1] - span, mid[1] + span)
    ax.set_zlim(mid[2] - span, mid[2] + span)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Reconstructed 3D points ({n:,})")

    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('lightgray')
    ax.yaxis.pane.set_edgecolor('lightgray')
    ax.zaxis.pane.set_edgecolor('lightgray')

    plt.tight_layout()
    plt.show()


def save_rectified_pair(rect_left, rect_right, output_path):
    """
    Side-by-side rectified pair with horizontal guide lines.
    Images should be BGR or RGB uint8.
    """
    h1, w1 = rect_left.shape[:2]
    h2, w2 = rect_right.shape[:2]
    h = max(h1, h2)

    # Convert to RGB if needed (BGR → RGB for matplotlib)
    if rect_left.ndim == 3 and rect_left.shape[2] == 3:
        left_rgb = cv2.cvtColor(rect_left, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(rect_right, cv2.COLOR_BGR2RGB)
    else:
        left_rgb = rect_left
        right_rgb = rect_right

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].imshow(left_rgb)
    axes[0].set_title('Rectified Left')
    axes[0].axis('off')
    axes[1].imshow(right_rgb)
    axes[1].set_title('Rectified Right')
    axes[1].axis('off')

    # Draw horizontal guide lines across both images
    for y in np.linspace(0, h - 1, 20, dtype=int):
        for ax in axes:
            ax.axhline(y=y, color='lime', linewidth=0.5, alpha=0.5)

    fig.suptitle('Epipolar Rectification (matching rows = same epipolar line)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_disparity_visualization(disparity, rect_left_rgb, output_path):
    """
    3-panel visualization: reference | disparity map (magma) | overlay.
    disparity: H x W float32, rect_left_rgb: H x W x 3 uint8 (RGB).
    """
    valid = np.isfinite(disparity)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(rect_left_rgb)
    axes[0].set_title('Rectified Left Image')
    axes[0].axis('off')

    disp_show = disparity.copy().astype(float)
    disp_show[~valid] = np.nan
    im1 = axes[1].imshow(disp_show, cmap='magma')
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label='disparity (pixels)')
    n_valid = int(np.sum(valid))
    axes[1].set_title(f'Disparity Map ({n_valid:,} pixels)')
    axes[1].axis('off')

    if np.any(valid):
        d_min, d_max = np.nanpercentile(disp_show[valid], [2, 98])
        d_norm = np.clip((disparity - d_min) / (d_max - d_min + 1e-8), 0, 1)
        d_norm[~valid] = 0
        disp_rgb = (plt.cm.magma(d_norm)[:, :, :3] * 255).astype(np.uint8)
        overlay = rect_left_rgb.copy().astype(float)
        alpha = 0.6
        for c in range(3):
            overlay[:, :, c][valid] = (
                alpha * disp_rgb[:, :, c][valid] +
                (1 - alpha) * rect_left_rgb[:, :, c][valid])
        overlay[~valid] *= 0.3
        axes[2].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[2].set_title('Disparity Overlay (bright = close)')
    axes[2].axis('off')

    fig.suptitle('Uncalibrated Stereo: Disparity from F',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_depth_visualization(lambda_map, ref_rgb, output_path):
    """
    3-panel visualization: reference | depth map | depth overlay.
    Valid pixels are those where lambda_map is finite (not NaN).
    """
    valid = np.isfinite(lambda_map)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(ref_rgb)
    axes[0].set_title('Reference Image')
    axes[0].axis('off')

    depth_show = lambda_map.copy()
    depth_show[~valid] = np.nan
    im1 = axes[1].imshow(depth_show, cmap='turbo')
    plt.colorbar(im1, ax=axes[1], shrink=0.7, label='lambda (depth)')
    n_valid = int(np.sum(valid))
    axes[1].set_title(f'Depth Map ({n_valid:,} pixels)')
    axes[1].axis('off')

    if np.any(valid):
        d_min, d_max = np.nanpercentile(depth_show[valid], [2, 98])
        d_norm = np.clip((lambda_map - d_min) / (d_max - d_min + 1e-8), 0, 1)
        d_norm[~valid] = 0
        depth_rgb = (plt.cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
        overlay = ref_rgb.copy().astype(float)
        alpha = 0.6
        for c in range(3):
            overlay[:, :, c][valid] = (
                alpha * depth_rgb[:, :, c][valid] +
                (1 - alpha) * ref_rgb[:, :, c][valid])
        overlay[~valid] *= 0.3
        axes[2].imshow(np.clip(overlay, 0, 255).astype(np.uint8))
    axes[2].set_title('Depth Overlay (warm = close)')
    axes[2].axis('off')

    fig.suptitle('Part A: Plane-Sweep Dense Stereo', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_dense_cloud(pts3d, colors, Ms, output_path,
                     max_pts=120000):
    """3-view point cloud visualization with camera positions from Ms."""
    if len(pts3d) == 0:
        print("  No points for cloud")
        return

    if len(pts3d) > max_pts:
        idx = np.random.choice(len(pts3d), max_pts, replace=False)
        pts3d = pts3d[idx]
        colors = colors[idx]

    fig = plt.figure(figsize=(18, 8))
    views = [(30, -60, 'Oblique'), (90, -90, 'Top-Down'), (0, -90, 'Front')]

    for panel, (elev, azim, subtitle) in enumerate(views):
        ax = fig.add_subplot(1, 3, panel + 1, projection='3d')
        ax.scatter(pts3d[:, 0], pts3d[:, 1], pts3d[:, 2],
                   c=colors, s=0.3, alpha=0.6, edgecolors='none')
        if Ms is not None:
            def _null(M):
                C = np.linalg.svd(M)[2][-1]
                return C[:3] / C[3]
            ccs = np.array([_null(M) for M in Ms])
            ax.scatter(ccs[:, 0], ccs[:, 1], ccs[:, 2],
                       c='red', s=50, marker='^', label='Cameras')
        mid = pts3d.mean(axis=0)
        span = (pts3d.max(axis=0) - pts3d.min(axis=0)).max() / 2 * 1.1
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'{subtitle} ({len(pts3d):,} pts)', fontsize=10)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    fig.suptitle('3D Point Cloud', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_point_cloud_ply(pts3d, colors, output_path):
    """
    Save a colored point cloud as binary PLY (drag-drop into the course
    viewer at https://browncsci1430.github.io/resources/pointcloud_viewer/).

    pts3d:  N x 3 float  (x, y, z)
    colors: N x 3 float  in [0, 1] (r, g, b)
    """
    n = len(pts3d)
    if n == 0:
        print("  No points for PLY export")
        return

    pts = np.asarray(pts3d, dtype=np.float32)
    rgb = np.clip(np.asarray(colors) * 255, 0, 255).astype(np.uint8)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )

    vertex_dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('r', 'u1'), ('g', 'u1'), ('b', 'u1'),
    ])
    vertices = np.empty(n, dtype=vertex_dtype)
    vertices['x'] = pts[:, 0]
    vertices['y'] = pts[:, 1]
    vertices['z'] = pts[:, 2]
    vertices['r'] = rgb[:, 0]
    vertices['g'] = rgb[:, 1]
    vertices['b'] = rgb[:, 2]

    with open(output_path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(vertices.tobytes())
