"""
=====================================
CSCI 1430 - Brown University
Camera Geometry - student.py
=====================================

  Part A -- Calibrated Geometry:
    Task 1: Camera Projection  (4 functions)
      camera_center(M)                              (~1 line)
      project(M, points3d)                          (~3 lines)
      reprojection_error(M, points3d, points2d)     (~1 line)
      estimate_camera_matrix(points2d, points3d)    (~15 lines)

    Task 2: Dense Stereo via Plane Sweeping  (4 functions)
      back_project(M, points2d, lambdas)            (~6 lines)
      compute_depth_homography(M_ref, M_other, lam) (~10 lines)
      compute_ncc(ref_gray, warped_gray, win_size)  (~12 lines)
      plane_sweep_stereo(...)                       (~30 lines)

  Part B -- Uncalibrated Geometry:
    Task 3: Fundamental Matrix  (1 function)
      estimate_fundamental_matrix(points1, points2) (~20 lines)

    Task 4: RANSAC  (1 function)
      ransac_fundamental_matrix(matches1, matches2, num_iters)  (~30 lines)

    Task 5: Uncalibrated Stereo Disparity  (1 function)
      compute_disparity_map(rect_left_gray, rect_right_gray, win_size, max_disparity)  (~30 lines)

  Extra Credit:
    normalize_coordinates(points)
    compute_sampson_distance(F, points1, points2)
    decompose_projection_matrix(M)
    estimate_relative_pose(F, K1, K2, inliers1, inliers2)
    compute_plane_homography(M_ref, M_other, n, d)
"""

import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


# =============================================================================
#  Part A — Calibrated Geometry
# =============================================================================

# Task 1: Camera Projection
#
# Four functions that implement the projection matrix M: 
# - camera_center(): extract the camera center 
# - project(): forward-project 3D→2D 
# - reprojection_error(): measure reprojection error
# - estimate_camera_matrix(): estimate M from 2D-3D correspondences via the Direct Linear Transform (DLT).

def camera_center(M):
    """
    Extract the camera center in world coordinates from a 3x4 projection
    matrix.  Recall M = [A | m4], so:

        C = -A^{-1} m4

    :param M: 3x4 projection matrix
    :return: length-3 numpy array, the camera center in world coordinates
    """
    A = M[:, :3]
    m4 = M[:, 3]
    return -np.linalg.inv(A) @ m4


def project(M, points3d):
    """
    Forward-project 3D world points to 2D image coordinates using M.

        [su, sv, s]^T = M @ [X, Y, Z, 1]^T
        (u, v) = (su / s, sv / s)

    :param M: 3x4 projection matrix
    :param points3d: N x 3 array of 3D world coordinates
    :return: N x 2 array of 2D image coordinates (u, v)
    """
    points3d_h = np.hstack([points3d, np.ones((points3d.shape[0], 1))])
    proj_h = (M @ points3d_h.T).T
    return proj_h[:, :2] / proj_h[:, 2:3]


def reprojection_error(M, points3d, points2d):
    """
    Per-point L2 reprojection error: how far each projected 3D point
    lands from its observed 2D position.

        error_i = || project(M, P_i) - p_i ||_2

    Use our project() function.

    :param M: 3x4 projection matrix
    :param points3d: N x 3 array of 3D world coordinates
    :param points2d: N x 2 array of observed 2D image coordinates
    :return: length-N array of L2 reprojection errors
    """
    projected = project(M, points3d)
    return np.linalg.norm(projected - points2d, axis=1)



def estimate_camera_matrix(points2d, points3d):
    """
    Estimate the 3x4 camera matrix M from 2D-3D correspondences via 
    the direct linear transform (DLT).

    Build the 2N x 12 matrix A from the homogeneous system Am = 0.  Each
    correspondence (u,v) <-> (X,Y,Z) gives two rows:

        [X Y Z 1  0 0 0 0  -uX -uY -uZ -u]
        [0 0 0 0  X Y Z 1  -vX -vY -vZ -v]

    Solve via SVD.

    After solving, compute and return the residual as the sum of squared reprojection
    errors using our reprojection_error() function.

    For extra credit: apply coordinate normalization to both points2d and
    points3d before building A, then un-normalize M afterward.
    See normalize_coordinates().

    :param points2d: N x 2 array of 2D image coordinates
    :param points3d: N x 3 array of corresponding 3D world coordinates
    :return: M, the 3x4 camera matrix
             residual, the sum of squared reprojection error (scalar)
    """
    n = points2d.shape[0]
    A = np.zeros((2 * n, 12), dtype=np.float64)

    for i, ((u, v), (X, Y, Z)) in enumerate(zip(points2d, points3d)):
        A[2 * i] = [X, Y, Z, 1, 0, 0, 0, 0, -u * X, -u * Y, -u * Z, -u]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, 1, -v * X, -v * Y, -v * Z, -v]

    _, _, Vt = np.linalg.svd(A)
    M = Vt[-1].reshape(3, 4)

    if np.abs(M[-1, -1]) > 1e-12:
        M = M / M[-1, -1]

    errors = reprojection_error(M, points3d, points2d)
    residual = np.sum(errors ** 2)
    return M, residual


# Task 2: Dense Stereo via Plane Sweeping
#
# Four functions that turn calibrated cameras into a dense depth map.
# - back_project(): inverts the projection to create 3D points.  
# - compute_depth_homography(): builds the depth-dependent homography H(lambda).  
# - compute_ncc(): measures local similarity.
# - plane_sweep_stereo(): sweeps candidate depths and picks the best NCC.


def back_project(M, points2d, lambdas):
    """
    Back-project 2D image points to 3D world coordinates at given depths (lambdas).
    This is the INVERSE of project().

        C   = camera_center(M)
        ray = A^{-1} @ [u, v, 1]^T    (direction in world coordinates)
        P   = C + lam * ray

    :param M: 3x4 projection matrix
    :param points2d: N x 2 array of 2D image coordinates
    :param lambdas: length-N array of lambda (depth) parameters
    :return: N x 3 array of 3D world coordinates
    """
    A = M[:, :3]
    C = camera_center(M)

    points2d_h = np.hstack([points2d, np.ones((points2d.shape[0], 1))])
    rays = np.linalg.solve(A, points2d_h.T).T
    points3d = C[None, :] + lambdas[:, None] * rays
    return points3d


def compute_depth_homography(M_ref, M_other, lam):
    """
    Compute the depth-dependent homography between two views at given depth (lambda).

        B = A_other @ A_ref^{-1}
        a = A_other @ C_ref + t_other
        e3 = [0, 0, 1]^T
        H(lam) = lam * B + np.outer(a, e3)

    :param M_ref:   3x4 projection matrix of reference camera
    :param M_other: 3x4 projection matrix of other camera
    :param lam:     scalar lambda (depth) parameter
    :return: H, 3x3 homography matrix
    """
    A_ref = M_ref[:, :3]
    A_other = M_other[:, :3]
    t_other = M_other[:, 3]
    C_ref = camera_center(M_ref)

    B = A_other @ np.linalg.inv(A_ref)
    a = A_other @ C_ref + t_other
    e3 = np.array([0.0, 0.0, 1.0])

    H = lam * B + np.outer(a, e3)
    return H


def compute_ncc(ref_gray, warped_gray, win_size):
    """
    Compute per-pixel windowed normalized cross correlation (NCC) between 
    reference and warped image. NCC is exactly as it is named: we subtract 
    the mean of each patch and divide through by the standard deviation to
    remove brightness variations before correlating. E is the correlation 
    surface over a local window.

        NCC = (E[ref*warped] - E[ref]*E[warped]) / (std[ref]*std[warped] + eps)

    Implementation tips:
    - We can use cv2.boxFilter(img, -1, (win_size, win_size)) to compute a local mean.
    - Variance: var = E[x^2] - E[x]^2
    - Std:      std = sqrt(max(var, 0) + eps).

    :param ref_gray:    H x W float32, reference image
    :param warped_gray: H x W float32, warped image
    :param win_size:    int, window size
    :return: H x W float32, NCC scores in [-1, 1]
    """
    ref_gray = ref_gray.astype(np.float32)
    warped_gray = warped_gray.astype(np.float32)
    ksize = (win_size, win_size)
    eps = 1e-8

    mean_ref = cv2.boxFilter(ref_gray, -1, ksize)
    mean_warp = cv2.boxFilter(warped_gray, -1, ksize)

    mean_ref2 = cv2.boxFilter(ref_gray * ref_gray, -1, ksize)
    mean_warp2 = cv2.boxFilter(warped_gray * warped_gray, -1, ksize)
    mean_refwarp = cv2.boxFilter(ref_gray * warped_gray, -1, ksize)

    var_ref = mean_ref2 - mean_ref * mean_ref
    var_warp = mean_warp2 - mean_warp * mean_warp
    cov = mean_refwarp - mean_ref * mean_warp

    std_ref = np.sqrt(np.maximum(var_ref, 0.0) + eps)
    std_warp = np.sqrt(np.maximum(var_warp, 0.0) + eps)

    ncc = cov / (std_ref * std_warp + eps)
    return np.clip(ncc, -1.0, 1.0).astype(np.float32)


def plane_sweep_stereo(ref_gray, other_grays, M_ref, Ms_other, lambdas,
                       win_size=11, ncc_threshold=0.4):
    """
    Multi-view plane-sweep stereo.

    For each candidate depth lam in lambdas:
      1. For the other view, compute H(lam) and warp with
         cv2.warpPerspective(img, H, (w,h), flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
      2. Compute NCC between ref and warped view
      3. If the NCC beats the current best at a pixel, update.
      
      4. [Better] Warp not just the first but _all_ images in other_grays
         Then, compute the average NCC across these views.

    Tips: 
    - We recommend _visualizing the warped image_ to make sure it makes sense.
    Can we (as a human) visually match them? What will NCC do?
    - NCC will always return some number; make sure we guard against incorrect values.
    
    :param ref_gray:    H x W float32, reference grayscale image
    :param other_grays: list of H x W float32, other-view images
    :param M_ref:       3x4 projection matrix of reference camera
    :param Ms_other:    list of 3x4 projection matrices
    :param lambdas:     1D array of lambda (depth) candidates
    :param win_size:    NCC window size
    :param ncc_threshold: minimum NCC to accept a lambda (depth) (default 0.4)
    :return: lambda_map, H x W float32 lambda (depth) map (NaN where unreliable)
    """
    h, w = ref_gray.shape
    best_ncc = np.full((h, w), -np.inf, dtype=np.float32)
    lambda_map = np.full((h, w), np.nan, dtype=np.float32)

    for lam in lambdas:
        nccs = []

        for other_gray, M_other in zip(other_grays, Ms_other):
            H = compute_depth_homography(M_ref, M_other, lam)
            warped = cv2.warpPerspective(
                other_gray,
                H,
                (w, h),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            ncc = compute_ncc(ref_gray, warped, win_size)
            nccs.append(ncc)

        avg_ncc = np.mean(nccs, axis=0)
        improve = avg_ncc > best_ncc
        best_ncc[improve] = avg_ncc[improve]
        lambda_map[improve] = lam

    lambda_map[best_ncc < ncc_threshold] = np.nan
    return lambda_map


# =============================================================================
#  Part B — Uncalibrated Geometry
# =============================================================================

# Task 3: Fundamental Matrix Estimation

def estimate_fundamental_matrix(points1, points2):
    """
    Estimate the fundamental matrix F from point correspondences using
    the 8-point algorithm with SVD and rank-2 enforcement.

    Steps:
      1. Build the N x 9 data matrix A where each row is
         [u'*u, u'*v, u', v'*u, v'*v, v', u, v, 1]
      2. Solve Af = 0 via SVD: f is the last row of V^T.  Reshape to 3x3.
      3. Enforce rank 2: decompose F with SVD, zero out the smallest
         singular value, and reconstruct.

    The residual is: sum_i (x_i'^T F x_i)^2

    For extra credit: apply coordinate normalization before step 1 and
    un-normalize F afterward.  See normalize_coordinates().

    :param points1: N x 2 array of 2D points in image 1
    :param points2: N x 2 array of 2D points in image 2
    :return: F_matrix, the 3x3 fundamental matrix
             residual, the sum of squared algebraic error
    """
    u1 = points1[:, 0]
    v1 = points1[:, 1]
    u2 = points2[:, 0]
    v2 = points2[:, 1]

    A = np.column_stack([
        u2 * u1,
        u2 * v1,
        u2,
        v2 * u1,
        v2 * v1,
        v2,
        u1,
        v1,
        np.ones_like(u1),
    ])

    _, _, Vt = np.linalg.svd(A)
    F = Vt[-1].reshape(3, 3)

    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0.0
    F = U @ np.diag(S) @ Vt

    norm = np.linalg.norm(F)
    if norm > 0:
        F = F / norm

    points1_h = np.hstack([points1, np.ones((points1.shape[0], 1))])
    points2_h = np.hstack([points2, np.ones((points2.shape[0], 1))])
    errs = np.sum(points2_h * (F @ points1_h.T).T, axis=1)
    residual = np.sum(errs ** 2)

    return F, residual


# Task 4: RANSAC 
# Robust estimation: find F despite noisy feature matches.

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Find the best fundamental matrix using RANSAC.

    For each iteration:
      1. Randomly sample 8 correspondences
      2. Estimate F using estimate_fundamental_matrix() on the sample
      3. Compute algebraic error |x'^T F x| for ALL correspondences
         (or Sampson distance if we implemented compute_sampson_distance() extra credit)
      4. Count inliers (error < threshold; start around 0.005 for algebraic error)
      5. Keep the F with the most inliers

    After the loop, re-estimate F from ALL inliers of the best model.

    For visualization, don't forget to append to inlier_counts and inlier_residuals each
    iteration (for the RANSAC convergence visualization).

    :param matches1: N x 2 array of 2D points in image 1
    :param matches2: N x 2 array of 2D points in image 2
    :param num_iters: number of RANSAC iterations
    :return: best_Fmatrix, best_inliers1, best_inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)

    inlier_counts.clear()
    inlier_residuals.clear()

    n = matches1.shape[0]
    threshold = 0.005

    matches1_h = np.hstack([matches1, np.ones((n, 1))])
    matches2_h = np.hstack([matches2, np.ones((n, 1))])

    best_F = None
    best_mask = None
    best_count = -1
    best_residual = np.inf

    for _ in range(num_iters):
        sample_idx = np.random.choice(n, 8, replace=False)
        F_candidate, _ = estimate_fundamental_matrix(
            matches1[sample_idx], matches2[sample_idx]
        )

        errors = np.abs(np.sum(matches2_h * (F_candidate @ matches1_h.T).T, axis=1))
        mask = errors < threshold
        count = np.sum(mask)

        residual = np.sum(errors[mask] ** 2) if count > 0 else np.inf

        inlier_counts.append(int(count))
        inlier_residuals.append(residual)

        if count > best_count or (count == best_count and residual < best_residual):
            best_F = F_candidate
            best_mask = mask
            best_count = count
            best_residual = residual

    if best_mask is None or np.sum(best_mask) < 8:
        best_mask = np.ones(n, dtype=bool)

    best_inliers1 = matches1[best_mask]
    best_inliers2 = matches2[best_mask]

    best_F, _ = estimate_fundamental_matrix(best_inliers1, best_inliers2)

    best_inliers1_h = np.hstack([best_inliers1, np.ones((best_inliers1.shape[0], 1))])
    best_inliers2_h = np.hstack([best_inliers2, np.ones((best_inliers2.shape[0], 1))])
    final_errors = np.abs(np.sum(best_inliers2_h * (best_F @ best_inliers1_h.T).T, axis=1))
    best_inlier_residual = np.sum(final_errors ** 2)

    return best_F, best_inliers1, best_inliers2, best_inlier_residual

# Task 5: Uncalibrated Stereo Disparity

def compute_disparity_map(rect_left_gray, rect_right_gray, win_size=11,
                          max_disparity=64):
    """
    Compute a disparity map from a rectified stereo pair using NCC.
    The stencil code will perform the rectification once F is computed.
    
    After epipolar rectification the epipolar lines are horizontal, so the
    matching point in the right image is always on the same row, shifted
    by some disparity d. For each candidate d in (-max_disparity,
    +max_disparity), shift the right image by d pixels and compute
    per-pixel NCC against the left image using compute_ncc(). Track the
    disparity with the best (highest) NCC at each pixel.

    This is structurally almost identical to plane_sweep_stereo -- a loop over
    candidates, computing NCC at each one -- but here we shift the image
    horizontally instead of warping with a homography.

    :param rect_left_gray:  H x W float32, rectified left image (grayscale)
    :param rect_right_gray: H x W float32, rectified right image (grayscale)
    :param win_size:        NCC window size (default 11)
    :param max_disparity:   search range is (-max_disparity, +max_disparity)
    :return: H x W float32 disparity map (NaN where no valid match)
    """
    h, w = rect_left_gray.shape
    best_ncc = np.full((h, w), -np.inf, dtype=np.float32)
    disparity_map = np.full((h, w), np.nan, dtype=np.float32)

    for d in range(-max_disparity, max_disparity + 1):
        shifted = np.zeros_like(rect_right_gray, dtype=np.float32)
        valid = np.zeros((h, w), dtype=bool)

        if d > 0:
            shifted[:, d:] = rect_right_gray[:, :-d]
            valid[:, d:] = True
        elif d < 0:
            shifted[:, :d] = rect_right_gray[:, -d:]
            valid[:, :d] = True
        else:
            shifted[:] = rect_right_gray
            valid[:] = True

        ncc = compute_ncc(rect_left_gray, shifted, win_size)
        ncc[~valid] = -np.inf

        improve = ncc > best_ncc
        best_ncc[improve] = ncc[improve]
        disparity_map[improve] = d

    disparity_map[best_ncc < 0.2] = np.nan
    return disparity_map


###############################################################################
# EXTRA CREDIT SECTION

# Below this line are functions for extra credit 
# that we can evaluate in the autograder.

###############################################################################
# Extra Credit: Coordinate Normalization

def normalize_coordinates(points):
    """
    EXTRA CREDIT: Hartley normalization — zero mean, average distance sqrt(D)
    from the centroid, where D is the dimensionality (2 or 3).

    This improves numerical conditioning of any DLT-style estimation
    (camera matrix M via estimate_camera_matrix, or fundamental matrix F
    via estimate_fundamental_matrix).

    Build T = T_scale @ T_offset where:
      T_offset translates the centroid to the origin
      T_scale  scales so the average distance from the origin is sqrt(D)
              (sqrt(2) for 2D points, sqrt(3) for 3D points)
      s = sqrt(D) / mean_distance, where
          mean_distance = mean(||p_i - centroid||)

    For 2D input (N x 2): returns (normalized_points [N x 2], T [3x3])
    For 3D input (N x 3): returns (normalized_points [N x 3], T [4x4])

    :param points: N x D array of points (D = 2 or 3)
    :return: (normalized_points, T)
    """
    raise NotImplementedError("TODO (Extra Credit): implement normalize_coordinates")


# Extra Credit: Sampson Distance

def compute_sampson_distance(F, points1, points2):
    """
    EXTRA CREDIT: Per-point Sampson distance — the first-order approximation
    to geometric reprojection error for the fundamental matrix.

    Steps:
      1. Build homogeneous coordinates: x = [u, v, 1]^T, x' = [u', v', 1]^T
      2. Compute Fx, F^T x'
      3. Compute numerator: (x'^T F x)^2
      4. Compute denominator: (Fx)[0]^2 + (Fx)[1]^2 + (F^T x')[0]^2 + (F^T x')[1]^2
      5. Return numerator / denominator

    If we implement this, we can use it as the distance metric in RANSAC
    instead of the algebraic error |x'^T F x|, for a more principled
    inlier/outlier threshold in pixel^2 units rather than algebraic error units.

    :param F: 3x3 fundamental matrix
    :param points1: N x 2 array of 2D points in image 1
    :param points2: N x 2 array of 2D points in image 2
    :return: length-N array of Sampson distances
    """
    raise NotImplementedError("TODO (Extra Credit): implement compute_sampson_distance")


# Extra Credit: Decompose M into K[R|t]

def decompose_projection_matrix(M):
    """
    EXTRA CREDIT: Factor M into K, R, t via RQ decomposition of M[:,:3].

    Use cv2.RQDecomp3x3(M[:,:3]), and normalize so that K[2,2] = 1.
    Compute t = K^{-1} @ M[:,3].

    :param M: 3x4 projection matrix
    :return: K (3x3 upper-triangular), R (3x3 rotation), t (length-3)
    """
    raise NotImplementedError("TODO (Extra Credit): implement decompose_projection_matrix")


# Extra Credit: Essential Matrix and Relative Pose

def estimate_relative_pose(F, K1, K2, inliers1, inliers2):
    """
    EXTRA CREDIT: From F and intrinsic matrices, compute E and decompose
    to (R, t).

    Steps:
      1. E = K2^T @ F @ K1
      2. Enforce essential matrix constraint: SVD, set singular values
         to [s, s, 0] where s = (S[0]+S[1])/2
      3. Decompose E into 4 candidate (R, t) pairs using the W matrix
      4. Cheirality check: pick the (R, t) where most points are in
         front of both cameras

    :param F: 3x3 fundamental matrix
    :param K1: 3x3 intrinsic matrix of camera 1
    :param K2: 3x3 intrinsic matrix of camera 2
    :param inliers1: N x 2 inlier points in image 1
    :param inliers2: N x 2 inlier points in image 2
    :return: R (3x3), t (length-3)
    """
    raise NotImplementedError("TODO (Extra Credit): implement estimate_relative_pose")


# Extra Credit: Oriented Plane Homography (toward multi-view stereo)

def compute_plane_homography(M_ref, M_other, n, d):
    """
    EXTRA CREDIT: Compute the homography induced by a general oriented plane.
    We can then use this in our plane sweep algorithm. But, beware - we now
    have to search over a larger space of parameters, which will take a long time.

        s = d - n^T C_ref
        m = A_ref^{-T} n      (hint: use np.linalg.solve(A_ref.T, n))
        H(n, d) = s * B + np.outer(a, m)

    where B = A_other @ A_ref^{-1} and a = A_other @ C_ref + t_other.

    When the plane is fronto-parallel (n parallel to A_ref^T @ e3),
    this reduces to the H(lam) from compute_depth_homography().

    :param M_ref:   3x4 projection matrix of the reference camera
    :param M_other: 3x4 projection matrix of the other camera
    :param n:       length-3 array, plane normal (world coordinates)
    :param d:       scalar, plane offset (n^T P = d)
    :return: H, a 3x3 numpy array
    """
    raise NotImplementedError("TODO (Extra Credit): implement compute_plane_homography")


# /////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    """Two-panel RANSAC diagnostic:
    1. Inlier count vs. iteration
    2. Residual vs. iteration
    """
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    fig = plt.figure(figsize=(8, 6))
    fig.canvas.manager.set_window_title("RANSAC Convergence")

    plt.subplot(2, 1, 1)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()

    plt.tight_layout()
    plt.show()
