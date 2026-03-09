"""
Live Camera Calibration Demo — Direct Linear Transform (DLT)
CSCI 1430 - Brown University

Interactive demonstration of the DLT algorithm for camera matrix estimation.

Scene: Synthetic 3D markers in world space with a known true camera matrix M = K[R|t].
Demo: Add Gaussian pixel noise to the true 2D projections, then recover M from
      noisy 2D-3D correspondences using DLT least-squares (the A matrix).

Left panel:  3D overhead of the scene showing marker positions and camera frustum.
             Markers are colour-coded by depth (orange=near, blue=far).
Right panel: Camera projection view.
  - Green filled circles  = true (noiseless) 2D projections
  - Red filled circles    = noisy observations (fed to DLT)
  - Blue hollow circles   = DLT-estimated reprojections
  - Grey lines            = residual vectors (noisy -> estimated)

Controls:
  - Noise (px)    : std-dev of Gaussian pixel noise added to 2D points
  - N points      : number of 3D-2D correspondences used for DLT
  - Hartley Norm  : toggle Hartley normalization before DLT
  - Show A matrix : display the upper rows of the 2Nx12 DLT matrix

Status bar displays two distinct metrics:
  cond(A)  = s[0]/s[-2]  -- conditioning of the constraint matrix
             (LARGE without Hartley; much smaller with Hartley ON)
  fit resid = s[-1]       -- how well Ah=0 is satisfied; ~0 for a perfect fit
             (small at low noise, grows with noise regardless of Hartley)
"""

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, make_lookat_Rt,
    fov_to_focal, render_scene,
    make_frustum_mesh, make_axis_mesh,
    make_sphere, make_octahedron,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    setup_viewport, make_state_updater, make_reset_callback,
    create_parameter_table, add_parameter_row,
    load_fonts, bind_mono_font,
    add_global_controls, control_panel,
    poll_collapsible_panels,
)


# =============================================================================
# Constants / Defaults
# =============================================================================

DEFAULTS = {
    "noise_px": 0.0,
    "n_points": 12,
    "use_normal_eqs": False,
    "offset_exp": 0,
    "scale_exp": 0,
    "ui_scale": 1.5,
}

GUIDE_CALIBRATION = [
    {"title": "Camera calibration via DLT",
     "body": "Given N 3D-to-2D correspondences, estimate the 3\u00d74 projection "
             "matrix M by solving Ah = 0 where h = vec(M). Each correspondence "
             "contributes 2 rows to A, so N points give a 2N\u00d712 system."},
    {"title": "SVD gives the null vector",
     "body": "SVD(A) = U\u03a3V\u1d40. The last row of V\u1d40 minimises ||Ah|| subject to "
             "||h||=1. Reshape to 3\u00d74 to get M. The smallest singular value "
             "(\u03c3\u2081\u2082) measures fit quality \u2014 it should be near zero."},
    {"title": "Hartley normalization",
     "body": "Toggle Hartley to see the condition number improve dramatically. "
             "Hartley normalization translates points to zero mean and scales so "
             "the mean distance from the origin is sqrt(d) before building A. This makes the system "
             "numerically stable without changing the geometric solution."},
    {"title": "Coordinate distortion",
     "body": "Use the Origin Shift and World Scale sliders to simulate distant "
             "coordinate origins or unit mismatches (e.g., millimetres vs metres). "
             "Without Hartley, conditioning explodes. With Hartley, it stays stable."},
    {"title": "Degeneracies",
     "body": "Toggle 'Coplanar' mode: all training points lie on a plane. "
             "Training error stays small but the off-plane probe points (white "
             "octahedrons) show large reprojection error. DLT needs non-degenerate "
             "3D point configurations for a reliable estimate."},
]

IMG_W, IMG_H   = 480, 480
OVERVIEW_SIZE  = 500

# Fixed true camera intrinsics (known to the demo; not given to DLT)
TRUE_K = build_intrinsic(
    fx=380.0, fy=380.0, skew=0.0,
    cx=IMG_W / 2.0, cy=IMG_H / 2.0,
)

# Fixed true camera: wide-angle view so the whole marker cloud is visible
_TRUE_Rt = make_lookat_Rt(
    eye=np.array([2.5, 2.0, 6.5]),
    target=np.array([0.0, 0.0, 1.5]),
)
TRUE_M = TRUE_K @ _TRUE_Rt  # 3x4

# Overview camera intrinsics (fixed)
_OV_K = build_intrinsic(
    fov_to_focal(48, OVERVIEW_SIZE), fov_to_focal(48, OVERVIEW_SIZE),
    0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2,
)

# Orbit camera initial position — spherical coords around a target point
_OV_TARGET  = np.array([0.0, 0.5, 1.5])
_OV_EYE0    = np.array([9.0, 7.0, 4.0])
_d0         = _OV_EYE0 - _OV_TARGET
_OV_R0      = float(np.linalg.norm(_d0))
_OV_EL0     = float(np.arcsin(np.clip(_d0[1] / _OV_R0, -1.0, 1.0)))
_OV_AZ0     = float(np.arctan2(_d0[0], _d0[2]))

# 5x5x5 = 125 candidate 3D markers spread across X, Y, Z
_ALL_PTS3D = np.array(
    [[x, y, z]
     for x in np.linspace(-1.8, 1.8, 5)
     for y in np.linspace(-1.5, 1.5, 5)
     for z in np.linspace(0.0, 3.0, 5)],
    dtype=float,
)  # shape (125, 3)


_SUPERSCRIPTS = str.maketrans("0123456789", "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079")
def _sup(n):
    """Convert an integer to Unicode superscript digits (e.g. 3 → '³')."""
    return str(int(n)).translate(_SUPERSCRIPTS)


def _project(M, pts3d):
    ph = np.column_stack([pts3d, np.ones(len(pts3d))])
    p = (M @ ph.T).T
    w = p[:, 2:3]
    return p[:, :2] / (w + 1e-10)


# Keep only points that project inside the image (with 15px margin)
_true_proj = _project(TRUE_M, _ALL_PTS3D)
_depth     = (_TRUE_Rt[2, :3] @ _ALL_PTS3D.T + _TRUE_Rt[2, 3])
_in_image  = (
    (_true_proj[:, 0] >= 15) & (_true_proj[:, 0] < IMG_W - 15) &
    (_true_proj[:, 1] >= 15) & (_true_proj[:, 1] < IMG_H - 15) &
    (_depth > 0.1)
)
VISIBLE_PTS3D = _ALL_PTS3D[_in_image]
N_AVAILABLE   = len(VISIBLE_PTS3D)

# Fixed noise pattern — scale by noise_px each frame (no flickering)
_rng         = np.random.default_rng(1430)
_FIXED_NOISE = _rng.standard_normal((N_AVAILABLE, 2))
_PERM        = _rng.permutation(N_AVAILABLE)   # fixed random order for point sampling

# Off-plane "probe" points — NOT used in DLT, only used to reveal degeneracy.
# When Coplanar is ON, M_est reprojects training pts perfectly but predicts
# WRONG locations for these off-plane probes.
_PROBE_CANDS = np.array([
    [ 0.0,  0.0, 0.0],    [ 0.8,  0.4, 0.3],   [-0.8, -0.4, 0.3],
    [ 0.5, -0.3, 0.8],    [-0.5,  0.3, 2.2],
    [ 0.8, -0.4, 2.7],    [-0.8,  0.4, 2.7],    [ 0.0,  0.0, 3.0],
], dtype=float)
_pc_proj  = _project(TRUE_M, _PROBE_CANDS)
_pc_depth = _TRUE_Rt[2, :3] @ _PROBE_CANDS.T + _TRUE_Rt[2, 3]
_pc_vis   = (
    (_pc_proj[:, 0] >= 25) & (_pc_proj[:, 0] < IMG_W - 25) &
    (_pc_proj[:, 1] >= 25) & (_pc_proj[:, 1] < IMG_H - 25) &
    (_pc_depth > 0.1)
)
PROBE_PTS3D = _PROBE_CANDS[_pc_vis]   # shape (K, 3)

# Z range for colour coding
_Z_MIN   = VISIBLE_PTS3D[:, 2].min()
_Z_MAX   = VISIBLE_PTS3D[:, 2].max() + 1e-6
_Z_PLANE = float(VISIBLE_PTS3D[:, 2].mean())  # fixed z for coplanar mode

# Coplanar mode needs unique (x,y) representatives — the 5×5×5 grid has multiple
# entries sharing the same (x,y) but different z, which all collapse to the same 3D
# point when z is forced to _Z_PLANE.  Pick, for each unique (x,y), the one whose
# original z is closest to _Z_PLANE so the "true" 3D position is most representative.
_cop_dict: dict = {}
for _i, _pt in enumerate(VISIBLE_PTS3D):
    _key = (round(float(_pt[0]), 5), round(float(_pt[1]), 5))
    if _key not in _cop_dict or abs(_pt[2] - _Z_PLANE) < abs(_cop_dict[_key][1] - _Z_PLANE):
        _cop_dict[_key] = (_i, float(_pt[2]))
_COPLANAR_IDXS = np.array([v[0] for v in _cop_dict.values()], dtype=int)
_N_COPLANAR    = len(_COPLANAR_IDXS)
_COPLANAR_PERM = _rng.permutation(_N_COPLANAR)


def _depth_color(z):
    """BGR colour by world-z: orange (low-z/far) -> green (mid) -> blue (high-z/near)."""
    t = float(np.clip((z - _Z_MIN) / (_Z_MAX - _Z_MIN), 0, 1))
    if t < 0.5:
        # orange (40,160,220) -> green (50,200,80)
        s = t * 2
        return (int(40 + s * 10), int(160 + s * 40), int(220 - s * 140))
    else:
        # green (50,200,80) -> blue (200,80,40)
        s = (t - 0.5) * 2
        return (int(50 + s * 150), int(200 - s * 120), int(80 - s * 40))


# =============================================================================
# State
# =============================================================================

class State:
    noise_px    = DEFAULTS["noise_px"]
    n_points    = DEFAULTS["n_points"]
    offset_exp  = DEFAULTS["offset_exp"]
    scale_exp   = DEFAULTS["scale_exp"]
    show_A_matrix  = False
    use_hartley    = False
    use_coplanar   = False
    use_normal_eqs = DEFAULTS["use_normal_eqs"]
    M_est    = None
    residual = float("inf")
    cond_A   = float("inf")
    fit_resid = float("inf")


state = State()


# =============================================================================
# Orbit Camera State
# =============================================================================

class OvCam:
    """Spherical orbit camera for the 3D overview panel.

    Controls:
      Left-drag   : orbit (azimuth / elevation)
      Scroll wheel: zoom in / out
    """
    az     = _OV_AZ0
    el     = _OV_EL0
    radius = _OV_R0
    target = _OV_TARGET.copy()
    _prev  = None   # previous mouse position during a drag

    @classmethod
    def reset(cls):
        cls.az     = _OV_AZ0
        cls.el     = _OV_EL0
        cls.radius = _OV_R0
        cls._prev  = None

    @classmethod
    def make_Rt(cls):
        eye = cls.target + cls.radius * np.array([
            np.cos(cls.el) * np.sin(cls.az),
            np.sin(cls.el),
            np.cos(cls.el) * np.cos(cls.az),
        ])
        return make_lookat_Rt(eye, cls.target)


# =============================================================================
# DLT Core Math
# =============================================================================

def _normalize_2d(pts):
    """Hartley normalization for 2D. Returns (pts_norm, T 3x3)."""
    c = pts.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(pts - c, axis=1)) + 1e-10
    s = np.sqrt(2.0) / mean_dist
    T = np.array([[s, 0, -s*c[0]], [0, s, -s*c[1]], [0, 0, 1.0]])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    return (T @ pts_h.T).T[:, :2], T


def _normalize_3d(pts):
    """Hartley normalization for 3D. Returns (pts_norm, T 4x4)."""
    c = pts.mean(axis=0)
    mean_dist = np.mean(np.linalg.norm(pts - c, axis=1)) + 1e-10
    s = np.sqrt(3.0) / mean_dist
    T = np.array([
        [s, 0, 0, -s*c[0]],
        [0, s, 0, -s*c[1]],
        [0, 0, s, -s*c[2]],
        [0, 0, 0,  1.0   ],
    ])
    pts_h = np.column_stack([pts, np.ones(len(pts))])
    return (T @ pts_h.T).T[:, :3], T


def build_A_matrix(pts2d, pts3d):
    """Build the 2Nx12 DLT system.

    Each correspondence (u,v) <-> (X,Y,Z) contributes two rows:
      row 2i  : [X Y Z 1  0 0 0 0  -uX -uY -uZ -u]
      row 2i+1: [0 0 0 0  X Y Z 1  -vX -vY -vZ -v]
    """
    N = len(pts2d)
    A = np.zeros((2 * N, 12))
    for i in range(N):
        X, Y, Z = pts3d[i]
        u, v    = pts2d[i]
        A[2*i]   = [X, Y, Z, 1,  0, 0, 0, 0,  -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0,  X, Y, Z, 1,  -v*X, -v*Y, -v*Z, -v]
    return A


def estimate_M_dlt(pts2d, pts3d, use_hartley=False, use_normal_eqs=False,
                   origin_offset=0.0, world_scale=1.0):
    """Estimate 3x4 M via DLT.

    Parameters:
        use_hartley    -- centre + scale points before solving (improves conditioning)
        use_normal_eqs -- solve via eigenvalue of A^T A instead of SVD.
                          This SQUARES the condition number, making normalization
                          visibly critical for accuracy.
        origin_offset  -- add this constant to 3D coordinates before solving,
                          then undo the transform on the result.  Simulates a
                          world origin far from the scene, which worsens
                          conditioning.  Normalization centres it away.
        world_scale    -- multiply 3D coordinates by this factor (2D stays in
                          pixels).  Simulates a unit mismatch (e.g. mm vs m).
                          Normalization rescales it away.

    Returns:
        M_est     -- 3x4 camera matrix (scaled so M[2,3]=1), in original coords
        A         -- 2Nx12 DLT matrix (as seen by the solver, for display)
        cond_A    -- conditioning metric (large = poorly conditioned)
        fit_resid -- fit quality (~0 for a perfect fit; grows with noise)
    """
    assert len(pts2d) >= 6, "DLT needs at least 6 correspondences."

    # Scale 3D coordinates to simulate a unit mismatch.
    # Hartley normalization rescales both 2D and 3D to unit std; without it
    # the A matrix columns for 3D vs 2D span different magnitudes.
    S = float(world_scale)
    if S != 1.0:
        pts3d = pts3d * S

    # Shift 3D coordinates to simulate a distant world origin.
    # Hartley normalization centres both sets; without it the
    # A matrix columns span wildly different scales.
    O = float(origin_offset)
    if O != 0.0:
        pts3d = pts3d + O

    T2, T3 = np.eye(3), np.eye(4)
    if use_hartley:
        pts2d, T2 = _normalize_2d(pts2d)
        pts3d, T3 = _normalize_3d(pts3d)

    A = build_A_matrix(pts2d, pts3d)

    if use_normal_eqs:
        # Normal equations: eigenvalue of A^T A.
        # This squares the condition number, making normalization critical.
        ATA = A.T @ A
        eigvals, eigvecs = np.linalg.eigh(ATA)
        # eigvals sorted ascending; eigvecs[:,0] = smallest eigenvalue direction
        m = eigvecs[:, 0]
        # Condition: skip eigvals[0] (the null-space ~0 eigenvalue)
        cond_A    = float(eigvals[-1] / (eigvals[1] + 1e-12))
        fit_resid = float(np.sqrt(max(eigvals[0], 0.0)))
    else:
        # Standard SVD (numerically robust)
        _, s, Vt = np.linalg.svd(A)
        m = Vt[-1]          # null-vector (smallest singular value direction)
        # Condition number EXCLUDING the null space:
        # s[-1] is ~0 (the solution direction), s[-2] is the next smallest.
        cond_A    = float(s[0]  / (s[-2] + 1e-12))
        fit_resid = float(s[-1])

    M_est = m.reshape(3, 4)
    if use_hartley:
        M_est = np.linalg.inv(T2) @ M_est @ T3

    # Undo the 3D shift: M currently maps (X+O,Y+O,Z+O,1) → (u,v,1).
    # We want M that maps (X,Y,Z,1) → (u,v,1), so M = M_shifted @ T3_fwd.
    if O != 0.0:
        T3_fwd = np.array([[1,0,0,O], [0,1,0,O], [0,0,1,O], [0,0,0,1]])
        M_est = M_est @ T3_fwd

    # Undo the world scale: M currently maps (S*X,S*Y,S*Z,1) → (u,v,1).
    # We want M that maps (X,Y,Z,1) → (u,v,1), so M = M_s @ diag(S,S,S,1).
    if S != 1.0:
        M_est = M_est @ np.diag([S, S, S, 1.0])

    denom = M_est[2, 3]
    if abs(denom) > 1e-10:
        M_est = M_est / denom
    else:
        M_est = M_est / (np.linalg.norm(M_est) + 1e-10)

    return M_est, A, cond_A, fit_resid


# =============================================================================
# Rendering Helpers
# =============================================================================

def make_ground_grid(y_level=-1.8, n_lines=9, color=(55, 55, 55)):
    """Build a ground-plane grid as a line-segment mesh for render_scene.

    Returns a mesh dict with 2-vertex faces (line segments) so the grid
    participates in the painter's-algorithm depth sort instead of being
    painted on top of everything.
    """
    xs = np.linspace(-2.5, 2.5, n_lines)
    zs = np.linspace(-0.5, 4.5, n_lines)

    verts = []
    faces = []
    idx = 0
    for x in xs:
        verts.append([x, y_level, zs[0]])
        verts.append([x, y_level, zs[-1]])
        faces.append([idx, idx + 1])
        idx += 2
    for z in zs:
        verts.append([xs[0], y_level, z])
        verts.append([xs[-1], y_level, z])
        faces.append([idx, idx + 1])
        idx += 2

    return {
        "vertices": np.array(verts, dtype=np.float64),
        "faces": faces,
        "color": color,
    }


def _draw_diamond(img, center, size, color, thickness=2):
    """Draw a hollow diamond (rotated square) marker."""
    cx, cy = int(center[0]), int(center[1])
    s = size // 2
    pts = np.array(
        [[cx, cy - s], [cx + s, cy], [cx, cy + s], [cx - s, cy]], dtype=np.int32
    )
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, color, thickness, cv2.LINE_AA)


def draw_projection_canvas(pts2d_true, pts2d_noisy, pts2d_est, img_w, img_h,
                            pts2d_probe_true=None, pts2d_probe_est=None):
    """Draw the 2D projection view.

    Colour coding (BGR):
      Green (0,200,60)    -- true (noiseless) projections (training pts, circle)
      Red   (50,50,200)   -- noisy observations fed to DLT
      Cyan  (200,200,40)  -- DLT-estimated reprojections (hollow ring, training)
      White diamond       -- true projection of off-plane probe point
      Magenta diamond     -- DLT-estimated projection of off-plane probe
      Red/grey line       -- error vector between probe true and estimated
    """
    canvas = np.full((img_h, img_w, 3), 22, dtype=np.uint8)

    # Faint grid
    for x in range(0, img_w, 40):
        cv2.line(canvas, (x, 0), (x, img_h-1), (42, 42, 42), 1)
    for y in range(0, img_h, 40):
        cv2.line(canvas, (0, y), (img_w-1, y), (42, 42, 42), 1)

    # Principal-point cross-hair
    cx, cy = img_w // 2, img_h // 2
    cv2.line(canvas, (cx-15, cy), (cx+15, cy), (60, 60, 60), 1)
    cv2.line(canvas, (cx, cy-15), (cx, cy+15), (60, 60, 60), 1)

    def ipt(p):
        return (int(np.clip(round(p[0]), 0, img_w-1)),
                int(np.clip(round(p[1]), 0, img_h-1)))

    # Off-plane probe error lines (drawn first, so markers go on top)
    if pts2d_probe_true is not None and pts2d_probe_est is not None:
        for pt, pe in zip(pts2d_probe_true, pts2d_probe_est):
            err = float(np.linalg.norm(np.array(pt) - np.array(pe)))
            col = (50, 50, 220) if err > 5.0 else (70, 70, 70)
            cv2.line(canvas, ipt(pt), ipt(pe), col, 1, cv2.LINE_AA)

    # Training residual lines
    for pn, pe in zip(pts2d_noisy, pts2d_est):
        cv2.line(canvas, ipt(pn), ipt(pe), (80, 80, 80), 1)

    # True projections — green filled
    for p in pts2d_true:
        cv2.circle(canvas, ipt(p), 7, (0, 200, 60), -1)
        cv2.circle(canvas, ipt(p), 7, (0, 240, 90), 1)

    # Noisy observations — red
    for p in pts2d_noisy:
        cv2.circle(canvas, ipt(p), 4, (50, 50, 200), -1)

    # DLT estimated — cyan hollow ring
    for p in pts2d_est:
        cv2.circle(canvas, ipt(p), 10, (200, 200, 40), 2)

    # Off-plane probe points
    if pts2d_probe_true is not None and pts2d_probe_est is not None:
        for pt in pts2d_probe_true:
            _draw_diamond(canvas, ipt(pt), 16, (220, 220, 220), 2)   # white = truth
        for pe in pts2d_probe_est:
            _draw_diamond(canvas, ipt(pe), 16, (200, 80, 240), 2)    # magenta = DLT

    # Legend (3 entries always; 2 probe entries only in coplanar mode)
    show_probe = pts2d_probe_true is not None
    n_legend   = 5 if show_probe else 3
    lx, ly     = 12, img_h - 22 - (n_legend - 1) * 22 - 14
    step       = 22
    cv2.circle(canvas, (lx+7, ly+2), 7, (0, 200, 60), -1)
    cv2.putText(canvas, "True projection (training)", (lx+18, ly+7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    ly += step
    cv2.circle(canvas, (lx+4, ly+2), 4, (50, 50, 200), -1)
    cv2.putText(canvas, "Noisy input to DLT", (lx+18, ly+7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    ly += step
    cv2.circle(canvas, (lx+7, ly+2), 10, (200, 200, 40), 2)
    cv2.putText(canvas, "DLT estimated (training)", (lx+18, ly+7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
    if show_probe:
        ly += step
        _draw_diamond(canvas, (lx+7, ly+2), 14, (220, 220, 220), 2)
        cv2.putText(canvas, "True position (off-plane probe)", (lx+18, ly+7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)
        ly += step
        _draw_diamond(canvas, (lx+7, ly+2), 14, (200, 80, 240), 2)
        cv2.putText(canvas, "DLT predicted (off-plane probe)", (lx+18, ly+7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1, cv2.LINE_AA)

    return canvas  # BGR uint8


# =============================================================================
# Callbacks
# =============================================================================

def on_show_A(sender, value):
    state.show_A_matrix = value

def on_hartley(sender, value):
    state.use_hartley = value

def on_coplanar(sender, value):
    state.use_coplanar = value

def on_normal_eqs(sender, value):
    state.use_normal_eqs = value

def on_mouse_wheel(sender, app_data):
    """Zoom the overview orbit camera on scroll wheel."""
    if dpg.is_item_hovered("overview_img"):
        OvCam.radius = float(np.clip(OvCam.radius * (0.85 ** app_data), 2.0, 60.0))

def _calib_extra_reset():
    """Extra reset logic beyond DEFAULTS iteration."""
    state.show_A_matrix = False
    state.use_hartley = False
    state.use_coplanar = False
    OvCam.reset()
    # Non-DEFAULTS checkboxes (not auto-reset by convention)
    for tag, val in [
        ("show_A_check",   False),
        ("hartley_check",  False),
        ("coplanar_check", False),
    ]:
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)


# =============================================================================
# Main
# =============================================================================

def main():
    n_avail = N_AVAILABLE
    if n_avail < 6:
        raise RuntimeError(f"Only {n_avail} visible points — need at least 6.")

    npts_max = min(40, n_avail)
    print(f"[liveCalibration] {n_avail} visible 3D markers; slider up to {npts_max}")

    dpg.create_context()

    load_fonts()

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)

    with dpg.texture_registry(tag="texture_registry"):
        blank_ov   = [0.0] * (OVERVIEW_SIZE * OVERVIEW_SIZE * 4)
        blank_proj = [0.0] * (IMG_W * IMG_H * 4)
        dpg.add_raw_texture(OVERVIEW_SIZE, OVERVIEW_SIZE, blank_ov,
                            format=dpg.mvFormat_Float_rgba, tag="overview_tex")
        dpg.add_raw_texture(IMG_W, IMG_H, blank_proj,
                            format=dpg.mvFormat_Float_rgba, tag="proj_tex")

    with dpg.window(label="Camera Calibration Demo (DLT)", tag="main_window"):

        # ── Global controls ──────────────────────────────────────────────────
        add_global_controls(
            DEFAULTS, state,
            reset_extra=_calib_extra_reset,
            guide=GUIDE_CALIBRATION, guide_title="Camera Calibration (DLT)",
        )

        # ── Control panels ───────────────────────────────────────────────────
        with dpg.group(horizontal=True):

            # Block 1: Input Data ─────────────────────────────────────────────
            with control_panel("Input Data", width=290, height=170,
                               color=(255, 200, 100)):
                with create_parameter_table():
                    add_parameter_row(
                        "Noise (px)", "noise_px_slider",
                        DEFAULTS["noise_px"], 0.0, 20.0,
                        make_state_updater(state, "noise_px"),
                        make_reset_callback(state, "noise_px", "noise_px_slider", DEFAULTS["noise_px"]),
                        format_str="%.1f",
                    )
                    add_parameter_row(
                        "N points", "n_points_slider",
                        DEFAULTS["n_points"], 6, npts_max,
                        make_state_updater(state, "n_points"),
                        make_reset_callback(state, "n_points", "n_points_slider", DEFAULTS["n_points"]),
                        slider_type="int",
                    )
                dpg.add_checkbox(
                    label="Coplanar (z=const)",
                    default_value=False, callback=on_coplanar, tag="coplanar_check",
                )

            dpg.add_spacer(width=8)

            # Block 2: Solver ─────────────────────────────────────────────────
            with control_panel("Solver", width=220, height=170,
                               color=(150, 200, 255)):
                dpg.add_checkbox(
                    label="Hartley Normalize",
                    default_value=False, callback=on_hartley, tag="hartley_check",
                )
                dpg.add_spacer(height=4)
                dpg.add_checkbox(
                    label="Solve via A\u1d40A",
                    default_value=DEFAULTS["use_normal_eqs"],
                    callback=on_normal_eqs, tag="use_normal_eqs_checkbox",
                )

            dpg.add_spacer(width=8)

            # Block 3: Coordinate Scale/Shift ─────────────────────────────────
            with control_panel("3D Coordinate Shift/Scale", width=330, height=170,
                               color=(220, 180, 100)):
                with create_parameter_table():
                    add_parameter_row(
                        "Shift 10\u207f", "offset_exp_slider",
                        DEFAULTS["offset_exp"], 0, 8,
                        make_state_updater(state, "offset_exp"),
                        make_reset_callback(state, "offset_exp", "offset_exp_slider", DEFAULTS["offset_exp"]),
                        slider_type="int",
                    )
                    add_parameter_row(
                        "Scale 10\u207f", "scale_exp_slider",
                        DEFAULTS["scale_exp"], 0, 6,
                        make_state_updater(state, "scale_exp"),
                        make_reset_callback(state, "scale_exp", "scale_exp_slider", DEFAULTS["scale_exp"]),
                        slider_type="int",
                    )
                dpg.add_text(
                    "\u2191 Increase these, then normalize!",
                    color=(255, 200, 100),
                )

            dpg.add_spacer(width=8)

            # Block 4: Inspect ────────────────────────────────────────────────
            with control_panel("Inspect", width=180, height=170,
                               color=(150, 255, 150)):
                dpg.add_checkbox(
                    label="Show A matrix",
                    default_value=False, callback=on_show_A, tag="show_A_check",
                )

        dpg.add_separator()

        # ── Status bar ───────────────────────────────────────────────────────
        dpg.add_text("", tag="solver_label", color=(160, 255, 160))
        dpg.add_text("", tag="status_text",  color=(255, 220, 100))
        dpg.add_text("", tag="status_text2", color=(180, 220, 255))
        dpg.add_text("", tag="status_text3", color=(180, 220, 255))

        dpg.add_separator()

        # ── A matrix display ─────────────────────────────────────────────────
        dpg.add_text("", tag="a_matrix_text", color=(160, 200, 255))

        # ── Image panels ─────────────────────────────────────────────────────
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("3D Scene Overview  (blue=near, orange=far)",
                             color=(150, 255, 150))
                dpg.add_image("overview_tex", tag="overview_img",
                              width=OVERVIEW_SIZE, height=OVERVIEW_SIZE)
                dpg.add_text("[drag: orbit  |  scroll: zoom]",
                             color=(110, 160, 110))
            dpg.add_spacer(width=2)
            with dpg.group():
                dpg.add_text("Camera Projection View", color=(150, 200, 255))
                dpg.add_image("proj_tex", tag="proj_img",
                              width=IMG_W, height=IMG_H)
            dpg.add_spacer(width=2)
            with dpg.group():
                dpg.add_text("Camera Matrix M  [3×4]", color=(200, 180, 255))
                dpg.add_text("", tag="m_text", color=(220, 220, 220))

    bind_mono_font("m_text", "a_matrix_text")

    setup_viewport(
        "Camera Calibration Demo (DLT)", 1480, 880,
        "main_window", lambda: None, DEFAULTS["ui_scale"],
    )

    # ── Main render loop ──────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        # ── Point selection ──────────────────────────────────────────────────
        # Coplanar mode needs unique-(x,y) representatives so that collapsing z
        # doesn't create duplicate 3D points (which would make N look wrong).
        N = int(np.clip(state.n_points, 6, n_avail))
        if state.use_coplanar:
            N   = min(N, _N_COPLANAR)
            sel = _COPLANAR_IDXS[_COPLANAR_PERM[:N]]
        else:
            sel = _PERM[:N]

        pts3d_orig = VISIBLE_PTS3D[sel].copy()   # real (non-collapsed) 3D positions
        pts3d      = pts3d_orig.copy()
        if state.use_coplanar:
            pts3d[:, 2] = _Z_PLANE               # collapse onto a fixed z plane

        # Ground-truth 2D projections (no noise)
        pts2d_true = _project(TRUE_M, pts3d)

        # Noisy observations: fixed noise pattern scaled by noise_px
        pts2d_noisy = pts2d_true + _FIXED_NOISE[sel] * state.noise_px

        # ── DLT ─────────────────────────────────────────────────────────────
        offset = 10.0 ** state.offset_exp if state.offset_exp > 0 else 0.0
        wscale = 10.0 ** state.scale_exp  if state.scale_exp  > 0 else 1.0
        try:
            M_est, A, cond_A, fit_resid = estimate_M_dlt(
                pts2d_noisy, pts3d, state.use_hartley, state.use_normal_eqs,
                origin_offset=offset, world_scale=wscale)
            pts2d_est = _project(M_est, pts3d)
            diff      = pts2d_est - pts2d_true
            residual  = float(np.sqrt((diff**2).sum(axis=1)).mean())
        except Exception as exc:
            print(f"DLT failed: {exc}")
            M_est     = TRUE_M.copy()
            A         = np.zeros((2*N, 12))
            cond_A    = float("inf")
            fit_resid = float("inf")
            residual  = float("inf")
            pts2d_est = pts2d_true.copy()

        state.M_est     = M_est
        state.residual  = residual
        state.cond_A    = cond_A
        state.fit_resid = fit_resid

        # Reference M — N points from the general permutation (with full z-range)
        # so the DLT is well-conditioned.  Only needed in coplanar mode for the
        # side-by-side matrix comparison.
        if state.use_coplanar:
            ref_sel = _PERM[:N]
            pts3d_ref       = VISIBLE_PTS3D[ref_sel]
            pts2d_ref_true  = _project(TRUE_M, pts3d_ref)
            pts2d_ref_noisy = pts2d_ref_true + _FIXED_NOISE[ref_sel] * state.noise_px
            try:
                M_ref, _, _, _ = estimate_M_dlt(
                    pts2d_ref_noisy, pts3d_ref, state.use_hartley,
                    state.use_normal_eqs, origin_offset=offset,
                    world_scale=wscale)
            except Exception:
                M_ref = TRUE_M.copy()
        else:
            M_ref = None

        # ── Orbit camera mouse controls ──────────────────────────────────────
        if dpg.is_item_hovered("overview_img"):
            mx, my = dpg.get_mouse_pos()
            if dpg.is_mouse_button_down(0):
                if OvCam._prev is not None:
                    dx = mx - OvCam._prev[0]
                    dy = my - OvCam._prev[1]
                    OvCam.az += dx * 0.008
                    OvCam.el  = float(np.clip(OvCam.el - dy * 0.008, -0.2, 1.4))
                OvCam._prev = (mx, my)
            else:
                OvCam._prev = None
        else:
            OvCam._prev = None

        ov_Rt = OvCam.make_Rt()

        # ── 3D Overview ──────────────────────────────────────────────────────
        # Depth-coded marker spheres — colour by ORIGINAL z so coplanar spheres
        # (all at same z plane) still show visible depth variation.
        marker_meshes = [
            make_sphere(center=tuple(pts3d[i]), radius=0.13,
                        color=_depth_color(pts3d_orig[i, 2]))
            for i in range(N)
        ]
        # Off-plane probe octahedra (diamond shape) — white, only in coplanar mode
        if state.use_coplanar:
            marker_meshes += [
                make_octahedron(center=tuple(pt), radius=0.225, color=(210, 210, 210))
                for pt in PROBE_PTS3D
            ]
        frustum = make_frustum_mesh(TRUE_K, _TRUE_Rt, IMG_W, IMG_H, near=0.5, far=9.0)
        axes    = make_axis_mesh(origin=(0, 0, 0), length=1.5)
        grid    = [make_ground_grid()]

        overview_img = render_scene(
            marker_meshes + frustum + axes + grid, _OV_K, ov_Rt,
            OVERVIEW_SIZE, OVERVIEW_SIZE,
        )

        # ── Off-plane probe projections (only shown in coplanar mode) ───────────
        if state.use_coplanar and len(PROBE_PTS3D) > 0:
            pts2d_probe_true = _project(TRUE_M, PROBE_PTS3D)
            pts2d_probe_est  = _project(M_est,  PROBE_PTS3D)
            probe_diff    = pts2d_probe_est - pts2d_probe_true
            probe_residual = float(np.sqrt((probe_diff**2).sum(axis=1)).mean())
        else:
            pts2d_probe_true = None
            pts2d_probe_est  = None
            probe_residual   = 0.0

        # ── Projection canvas ────────────────────────────────────────────────
        proj_img = draw_projection_canvas(
            pts2d_true, pts2d_noisy, pts2d_est, IMG_W, IMG_H,
            pts2d_probe_true=pts2d_probe_true,
            pts2d_probe_est=pts2d_probe_est,
        )

        # ── Update DPG textures ──────────────────────────────────────────────
        dpg.set_value("overview_tex", convert_cv_to_dpg(overview_img))
        dpg.set_value("proj_tex",     convert_cv_to_dpg(proj_img))

        # ── Status ───────────────────────────────────────────────────────────
        cond_str  = f"{cond_A:.1f}"    if np.isfinite(cond_A)    else "inf"
        res_str   = f"{residual:.4f}" if np.isfinite(residual)  else "inf"
        fit_str   = f"{fit_resid:.4f}" if np.isfinite(fit_resid) else "inf"
        coplanar_warn = "  *** COPLANAR ***" if state.use_coplanar else ""
        if state.use_coplanar:
            probe_str = f"{probe_residual:.1f}" if np.isfinite(probe_residual) else "inf"
            probe_part = f"  |  Off-plane probe: {probe_str} px"
        else:
            probe_part = ""
        if state.use_normal_eqs:
            dpg.set_value("solver_label",
                          "Solver: A\u1d40A eigenvalue  (squares the condition number)")
        else:
            dpg.set_value("solver_label",
                          "Solver: SVD  (numerically robust)")
        dpg.set_value(
            "status_text",
            f"N={N} pts  |  Noise={state.noise_px:.1f} px  |  "
            f"Train reproj: {res_str} px{probe_part}"
            f"{coplanar_warn}",
        )
        if state.use_normal_eqs:
            cond_label = f"\u03bb\u2099/\u03bb\u2082 = {cond_str}"
            fit_label  = f"\u221a\u03bb\u2081 = {fit_str}"
        else:
            cond_label = f"\u03c3\u2081/\u03c3\u2099\u208b\u2081 = {cond_str}"
            fit_label  = f"\u03c3\u2099 = {fit_str}"
        dpg.set_value(
            "status_text2",
            f"fit residual ({fit_label})  "
            f"(~0 = perfect fit; grows with noise)",
        )
        # Build conditioning hint based on current settings
        has_distortion = state.offset_exp > 0 or state.scale_exp > 0
        if has_distortion and not state.use_hartley:
            cond_hint = "Toggle Normalization ON to fix this!"
        elif has_distortion and state.use_hartley:
            cond_hint = "Normalization centred + rescaled \u2014 distortion neutralised"
        elif state.use_normal_eqs:
            cond_hint = "A\u1d40A squares it \u2014 try Shift/Scale + Normalization"
        else:
            cond_hint = "Try Shift or Scale, then toggle Normalization"
        distort_parts = []
        if state.offset_exp > 0:
            distort_parts.append(f"Shift=10{_sup(state.offset_exp)}")
        if state.scale_exp > 0:
            distort_parts.append(f"Scale=10{_sup(state.scale_exp)}")
        distort_str = ("  |  " + ", ".join(distort_parts)) if distort_parts else ""
        dpg.set_value(
            "status_text3",
            f"cond ({cond_label}){distort_str}  ({cond_hint})",
        )

        # ── M matrix display ─────────────────────────────────────────────────
        def _fmt_M(M, label):
            lines = [label]
            for r in range(3):
                vals = "  ".join(f"{M[r, c]:9.3f}" for c in range(4))
                lines.append(f"  {vals}")
            return "\n".join(lines)

        if state.use_coplanar:
            m_lines  = _fmt_M(M_ref, f"Reference M  (N={N}, non-coplanar):")
            m_lines += "\n\n"
            m_lines += _fmt_M(M_est, f"Degenerate M  (N={N}, coplanar):")
        else:
            m_lines = _fmt_M(M_est, f"Estimated M  (N={N} pts):")
        dpg.set_value("m_text", m_lines)

        # ── A matrix display ─────────────────────────────────────────────────
        if state.show_A_matrix:
            n_show = min(8, A.shape[0])
            lines  = [f"A  ({A.shape[0]}x12, first {n_show} rows shown):"]
            for r in range(n_show):
                vals = "  ".join(f"{v:9.3f}" for v in A[r])
                lines.append(f"  row {r}: [{vals}]")
            if A.shape[0] > n_show:
                lines.append(f"  ... {A.shape[0]-n_show} more rows")
            dpg.set_value("a_matrix_text", "\n".join(lines))
        else:
            dpg.set_value("a_matrix_text", "")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
