"""
Live Plane Sweep Stereo Demo — Depth-Dependent Homography H(λ)
CSCI 1430 - Brown University

Interactive demonstration of plane sweep stereo using two calibrated cameras.

Core math (HW3 Task 2):
  B = A_other @ inv(A_ref)          [homography at infinity]
  a = A_other @ C_ref + t_other     [epipole in other camera]
  H(λ) = λ·B + outer(a, [0,0,1])   [depth-dependent homography]

Scene: a synthetic checkerboard plane at depth λ_true is rendered from two
camera viewpoints.  H(λ) maps reference pixels → other pixels assuming scene
depth λ.  When λ = λ_true, the warp perfectly aligns the two views (NCC ≈ 1).

Panels:
  Left   — Reference image (fixed)
  Centre — Other image warped by H(λ) (changes with slider)
  Right  — NCC vs. λ curve (pre-computed at startup, current λ highlighted)

Status bar: current NCC and residual between warped and reference.
"""

import numpy as np
import cv2
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, make_lookat_Rt,
    fov_to_focal, render_scene,
    make_frustum_mesh, make_axis_mesh,
    make_sphere, make_cube, create_default_scene,
    make_ground_grid, raycast_scene,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    load_fonts, setup_viewport, make_state_updater, make_reset_callback,
    create_parameter_table, add_parameter_row,
    add_global_controls, bind_mono_font, control_panel,
    poll_collapsible_panels,
)


# =============================================================================
# Constants / Defaults
# =============================================================================

DEFAULTS = {
    "lam": 4.5,                # current depth hypothesis \u03bb (slider)
    "cam_baseline": 0.8,       # horizontal separation between cameras
    "cam_convergence": 0.15,   # 0 = fronto-parallel, 1 = max toe-in (25°)
    "cam_distance": 5.0,       # other camera z (ref stays at z=5)
    "patch_size": 7,           # NCC patch width in pixels (squared for area)
    "epipole_weight": 1.0,        # α: weight of epipole term in H(λ)
    "mode": "Two-View Geometry",  # "Two-View Geometry" or "Plane Sweep"
    "ui_scale": 1.5,
}

MODE_ITEMS = ["Two-View Geometry", "Plane Sweep"]

GUIDE_PLANE_SWEEP = [
    {"title": "Two modes",
     "body": "This demo has two modes.\n\n"
             "Two-View Geometry: explore how two cameras relate \u2014 "
             "epipoles, epipolar lines, and the homography warp between views. "
             "Toggle the Homography warp checkbox to see the other image in its "
             "own coordinate frame vs. warped into the reference frame.\n\n"
             "Plane Sweep: evaluate depth hypotheses by sweeping \u03bb. "
             "The NCC curve and \u03bb map show where the warp best aligns the views."},
    {"title": "H(\u03bb) = \u03bb\u00b7B + \u03b1\u00b7a e\u2083\u1d40",
     "body": "A family of homographies parameterized by depth \u03bb.\n\n"
             "B = K\u2082 R_rel K\u2081\u207b\u00b9 is the homography at infinity: "
             "unproject a ref pixel to a ray, rotate into the other camera\u2019s "
             "frame, reproject. This handles rotation and intrinsics only.\n\n"
             "a = K\u2082 c\u2081 + t\u2082 is the epipole \u2014 the projection of "
             "the reference camera\u2019s centre into the other view. The term "
             "a e\u2083\u1d40 puts a into column 3 only (a constant homogeneous "
             "translation applied to every pixel, independent of depth).\n\n"
             "Sweeping \u03bb scales the rotation term while the epipole term "
             "stays fixed."},
    {"title": "Epipole \u03b1 slider",
     "body": "Scales the epipole term: H = \u03bb\u00b7B + \u03b1\u00b7a e\u2083\u1d40.\n\n"
             "\u03b1 = 1: full correct homography (default).\n"
             "\u03b1 = 0: rotation-only warp \u2014 no baseline correction. "
             "Close objects visibly misalign because the translation between "
             "cameras is ignored.\n\n"
             "Slide \u03b1 to see how the epipole term gradually corrects for "
             "camera separation. With Show matrices on, watch column 3 of H "
             "scale with \u03b1."},
    {"title": "Epipoles & epipolar lines",
     "body": "The epipole is where one camera\u2019s centre projects into the "
             "other\u2019s image. Epipolar lines through any pixel converge at "
             "the epipole.\n\n"
             "Homography warp ON: both images share the reference frame, so "
             "the same epipolar lines appear on both.\n"
             "Homography warp OFF: each image has its own epipole (e on the "
             "reference, e\u2032 on the other)."},
    {"title": "NCC & plane sweep",
     "body": "Normalized Cross-Correlation (NCC) measures patch alignment: "
             "\u22121 (anti-correlated) to +1 (perfect match).\n\n"
             "In Plane Sweep mode, click a pixel to select a patch. The NCC-vs-\u03bb "
             "curve shows how well the warp aligns that patch at each depth. "
             "The peak corresponds to the surface\u2019s true \u03bb.\n\n"
             "The \u03bb map buttons compute per-pixel depth: GT from the rendered "
             "scene, NCC via exhaustive plane sweep."},
    {"title": "Connection to calibration",
     "body": "H(\u03bb) is built directly from the camera projection matrices M "
             "that DLT estimates. B comes from the intrinsics and relative "
             "rotation; a encodes the camera separation.\n\n"
             "Accurate calibration is critical \u2014 errors in M propagate "
             "directly into depth estimates. Try adjusting the camera sliders "
             "to see how baseline and convergence affect the geometry."},
]

IMG_W, IMG_H = 400, 400
OVERVIEW_SIZE = 400
LAM_MIN, LAM_MAX = 0.5, 8.0
LAM_TRUE = 4.5          # true reference distance (matches liveCamera setup)
N_SWEEP = 80            # number of depth hypotheses in the pre-computed curve

# Camera setup
K_SHARED = build_intrinsic(
    fx=350.0, fy=350.0, skew=0.0,
    cx=IMG_W / 2.0, cy=IMG_H / 2.0,
)

# ── Camera presets (bookmarks into baseline × convergence × distance space) ──
# Each entry: (name, baseline, convergence, distance)
CAMERA_PRESETS = [
    ("Fronto-parallel (narrow)", 0.3, 0.0, 5.0),
    ("Fronto-parallel (wide)",   1.0, 0.0, 5.0),
    ("Converging (mild)",        0.8, 0.15, 5.0),
    ("Converging (strong)",      1.2, 0.5, 5.0),
    ("Epipole visible",          1.5, 0.0, 2.0),
]
PRESET_NAMES = [p[0] for p in CAMERA_PRESETS]

# Camera geometry constants
_CAM_Y = 0.5
_REF_Z = 5.0          # reference camera always at z=5
_MAX_TOE_IN_DEG = 25  # max convergence angle in degrees
_MAX_TOE_IN = np.radians(_MAX_TOE_IN_DEG)


class Cam:
    """Mutable camera state — updated when baseline/convergence changes."""
    Rt_ref = None
    Rt_other = None
    M_ref = None
    M_other = None
    ref_img = None
    other_img = None
    gt_depth: np.ndarray | None = None

cam = Cam()


def apply_camera_params(baseline, convergence, distance):
    """Compute cameras from (baseline, convergence, distance) and re-render.

    baseline:     horizontal separation between cameras
    convergence:  0 = fronto-parallel, 1 = max toe-in (25°)
                  Toe-in angle = convergence * 25°, independent of distance.
    distance:     z-position of the OTHER camera only (ref stays at z=_REF_Z).
                  distance=5 → same z as ref (symmetric).
                  distance<5 → other camera closer to scene → epipole enters ref view.
    """
    ref_eye   = np.array([-baseline / 2, _CAM_Y, _REF_Z])
    other_eye = np.array([ baseline / 2, _CAM_Y, distance])

    # Toe-in angle: each camera toes in by the same angle
    toe_in = convergence * _MAX_TOE_IN
    ref_shift   = _REF_Z   * np.tan(toe_in) if convergence > 1e-6 else 0.0
    other_shift = distance * np.tan(toe_in) if convergence > 1e-6 else 0.0

    # Each camera toes inward (ref looks rightward, other looks leftward)
    ref_target   = np.array([-baseline / 2 + ref_shift,   _CAM_Y, 0.0])
    other_target = np.array([ baseline / 2 - other_shift, _CAM_Y, 0.0])

    cam.Rt_ref   = make_lookat_Rt(ref_eye, ref_target)
    cam.Rt_other = make_lookat_Rt(other_eye, other_target)
    cam.M_ref    = K_SHARED @ cam.Rt_ref
    cam.M_other  = K_SHARED @ cam.Rt_other
    cam.ref_img, z_buf = render_scene(_SCENE, K_SHARED, cam.Rt_ref,
                                      IMG_W, IMG_H, flip_y=True,
                                      return_zbuf=True)
    # Convert z-buffer to depth map: inf (background) → NaN
    gt = z_buf.copy()
    gt[gt == np.inf] = np.nan
    cam.gt_depth = gt

    cam.other_img = render_scene(_SCENE, K_SHARED, cam.Rt_other,
                                 IMG_W, IMG_H, flip_y=True)


# Create scene and initialize default cameras
_SCENE = create_default_scene()
apply_camera_params(DEFAULTS["cam_baseline"], DEFAULTS["cam_convergence"], DEFAULTS["cam_distance"])

# Overview camera (orbit camera for 3D visualization)
_OV_TARGET = np.array([0.0, 0.5, 0.0])
_OV_EYE0 = np.array([0.0, 8.0, 12.0])
_d0 = _OV_EYE0 - _OV_TARGET
_OV_R0 = float(np.linalg.norm(_d0))
_OV_EL0 = float(np.arcsin(np.clip(_d0[1] / _OV_R0, -1.0, 1.0)))
_OV_AZ0 = float(np.arctan2(_d0[0], _d0[2]))

_OV_K  = build_intrinsic(fov_to_focal(50, OVERVIEW_SIZE), fov_to_focal(50, OVERVIEW_SIZE),
                          0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2)


class OvCam:
    """Spherical orbit camera for the 3D overview."""
    az = _OV_AZ0
    el = _OV_EL0
    radius = _OV_R0
    target = _OV_TARGET.copy()
    _prev = None

    @classmethod
    def reset(cls):
        cls.az = _OV_AZ0
        cls.el = _OV_EL0
        cls.radius = _OV_R0
        cls._prev = None

    @classmethod
    def make_Rt(cls):
        eye = cls.target + cls.radius * np.array([
            np.cos(cls.el) * np.sin(cls.az),
            np.sin(cls.el),
            np.cos(cls.el) * np.cos(cls.az),
        ])
        return make_lookat_Rt(eye, cls.target)


# =============================================================================
# Plane Sweep Math
# =============================================================================

def decompose_H(M_ref, M_other):
    """Decompose into B and a:  H(λ) = λ·B + outer(a, e₃ᵀ)."""
    A_ref   = M_ref[:, :3]
    t_ref   = M_ref[:, 3]
    A_other = M_other[:, :3]
    t_other = M_other[:, 3]

    C_ref = np.linalg.solve(A_ref, -t_ref)      # (3,) camera center
    B     = A_other @ np.linalg.inv(A_ref)       # (3×3) homography at infinity
    a     = A_other @ C_ref + t_other            # (3,) epipole-like term
    return B, a


def compute_epipole_ref(M_ref, M_other, img_h):
    """Compute the epipole in the reference image (display coords with flip_y).

    This is the projection of the other camera center into the reference view.
    Returns (x, y) in display coords, or None if at infinity.
    Also returns the epipolar direction (dx, dy) for line drawing.
    """
    A_ref   = M_ref[:, :3]
    t_ref   = M_ref[:, 3]
    A_other = M_other[:, :3]
    t_other = M_other[:, 3]

    C_other = np.linalg.solve(A_other, -t_other)   # other camera center in world
    e_h = A_ref @ C_other + t_ref                   # homogeneous coords in ref

    if abs(e_h[2]) > 1e-8:
        ex = e_h[0] / e_h[2]
        ey = img_h - 1 - e_h[1] / e_h[2]           # flip_y for display
        return (ex, ey)
    else:
        # Epipole at infinity — return a point very far in the right direction
        ex = e_h[0] * 1e6
        ey = -(e_h[1]) * 1e6                        # flip_y for direction
        return (ex, ey)


def compute_H_lam(M_ref, M_other, lam):
    """H(λ) = λ·B + outer(a, e₃ᵀ)  [HW3 Task 2 — compute_depth_homography]."""
    B, a = decompose_H(M_ref, M_other)
    e3 = np.array([0.0, 0.0, 1.0])
    H = lam * B + np.outer(a, e3)               # (3×3)
    return H


def warp_other_to_ref(other_img, H, img_w, img_h):
    """Warp other_img by H(λ) into reference coordinates (WARP_INVERSE_MAP).

    H is derived from unflipped camera matrices, but images use flip_y=True.
    Conjugate with the vertical flip matrix F so the warp is correct in
    display coordinates: H_display = F · H · F.
    """
    F = np.array([[1, 0, 0],
                  [0, -1, img_h - 1],
                  [0, 0, 1]], dtype=np.float64)
    H_display = F @ H @ F
    return cv2.warpPerspective(
        other_img, H_display, (img_w, img_h),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def compute_ncc_score(ref, warped, center=None, patch_size=32):
    """Compute NCC between ref and warped.

    If center is None: compute global NCC over all valid pixels.
    If center is (cx, cy): compute local NCC in a patch around that point.
    """
    if center is not None:
        # Local NCC around a specific point
        cx, cy = center
        h = patch_size // 2
        y1 = max(0, cy - h)
        y2 = min(warped.shape[0], cy + h + 1)
        x1 = max(0, cx - h)
        x2 = min(warped.shape[1], cx + h + 1)

        ref_patch = ref[y1:y2, x1:x2].astype(float)
        warped_patch = warped[y1:y2, x1:x2].astype(float)

        # Check for valid pixels in warped patch
        valid = (warped_patch.sum(axis=2) > 0)
        if valid.sum() < 1:
            return -1.0

        r = ref_patch[valid]
        w = warped_patch[valid]
    else:
        # Global NCC over all valid pixels
        valid = (warped.sum(axis=2) > 0)   # pixels that were filled
        if valid.sum() < 100:
            return -1.0
        r = ref[valid].astype(float)
        w = warped[valid].astype(float)

    r = r.ravel().astype(float)
    w = w.ravel().astype(float)
    if r.size == 3:  # 1x1 patch: color similarity, 1=identical, -1=max difference
        return float(1.0 - 2.0 * np.mean(np.abs(r - w)) / 255.0)
    r -= r.mean()
    w -= w.mean()
    denom = (np.linalg.norm(r) * np.linalg.norm(w))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(r, w) / denom)



# NCC curve will be recomputed when camera params, patch size, or point change
_LAM_CURVE = np.linspace(LAM_MIN, LAM_MAX, N_SWEEP)


# =============================================================================
# Matrix / Depth-map helpers
# =============================================================================

def _fmt_mat(mat):
    """Format matrix with square brackets for display."""
    rows, cols = mat.shape
    lines = []
    for r in range(rows):
        cells = "  ".join(f"{mat[r, c]:7.2f}" for c in range(cols))
        lines.append(f"[ {cells} ]")
    return "\n".join(lines)


def compute_ncc_map(ref, warped, patch_size):
    """Per-pixel NCC between ref and warped using box-filter convolutions."""
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY).astype(np.float64)
    w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float64)
    valid = (warped.sum(axis=2) > 0).astype(np.float64)
    w *= valid

    k = (patch_size, patch_size)
    n = cv2.boxFilter(valid, -1, k, normalize=False)

    r_sum = cv2.boxFilter(r, -1, k, normalize=False)
    w_sum = cv2.boxFilter(w, -1, k, normalize=False)

    area = float(patch_size * patch_size)
    r_mean = r_sum / area
    w_mean = np.divide(w_sum, n, out=np.zeros_like(w_sum), where=n > 0)

    r_c = r - r_mean
    w_c = (w - w_mean) * valid

    rw = cv2.boxFilter(r_c * w_c, -1, k, normalize=False)
    rr = cv2.boxFilter(r_c * r_c, -1, k, normalize=False)
    ww = cv2.boxFilter(w_c * w_c, -1, k, normalize=False)

    denom = np.sqrt(np.maximum(rr * ww, 0.0)) + 1e-10
    ncc = rw / denom
    ncc[n < patch_size * 0.5] = -1.0
    return ncc



def compute_ncc_depth_image(ref, other, M_ref, M_other, img_w, img_h,
                             lam_range, patch_size, valid_mask=None):
    """Per-pixel plane sweep: find depth lambda that maximises NCC at each pixel.

    valid_mask: optional (H,W) bool array.  Only pixels where valid_mask is True
                will be swept; the rest stay NaN.  When None, all non-black
                reference pixels are treated as valid.
    """
    if valid_mask is None:
        valid_mask = ref.sum(axis=2) > 0

    best_ncc = np.full((img_h, img_w), -2.0, dtype=np.float64)
    best_lam = np.full((img_h, img_w), np.nan, dtype=np.float64)

    for lam in lam_range:
        H = compute_H_lam(M_ref, M_other, lam)
        w = warp_other_to_ref(other, H, img_w, img_h)
        ncc_map = compute_ncc_map(ref, w, patch_size)
        better = valid_mask & (ncc_map > best_ncc)
        best_ncc[better] = ncc_map[better]
        best_lam[better] = lam

    return best_lam


def depth_to_colormap(depth, lam_min, lam_max):
    """Colorise a depth map using TURBO.  Close (small lambda) = warm, far = cool."""
    valid = ~np.isnan(depth)
    norm = np.zeros_like(depth)
    norm[valid] = 1.0 - np.clip((depth[valid] - lam_min) / (lam_max - lam_min), 0, 1)
    gray = (norm * 255).astype(np.uint8)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


# =============================================================================
# NCC Curve Canvas
# =============================================================================

def draw_ncc_curve(lam_current, ncc_current, ncc_curve, lam_true=LAM_TRUE, width=400, height=300):
    """Draw the NCC vs. λ curve with the current λ highlighted."""
    canvas = np.full((height, width, 3), 20, dtype=np.uint8)

    # Axes padding
    px, py = 50, 20
    pw = width - px - 10
    ph = height - py - 40

    # Draw axes
    cv2.line(canvas, (px, py), (px, py + ph), (150, 150, 150), 1)
    cv2.line(canvas, (px, py + ph), (px + pw, py + ph), (150, 150, 150), 1)

    # Axis labels
    cv2.putText(canvas, "NCC", (2, py + ph // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"lam (true={lam_true:.1f})", (px + pw // 2 - 40, height - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

    # Y-axis ticks: -1.0 to 1.0
    ncc_lo, ncc_hi = -1.0, 1.0
    for tick in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        ty = int(py + ph - (tick - ncc_lo) / (ncc_hi - ncc_lo) * ph)
        cv2.line(canvas, (px - 4, ty), (px, ty), (100, 100, 100), 1)
        cv2.putText(canvas, f"{tick:.1f}", (2, ty + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    # X-axis ticks
    for tick in [1, 2, 3, 4, 5, 6, 7]:
        tx = int(px + (tick - LAM_MIN) / (LAM_MAX - LAM_MIN) * pw)
        cv2.line(canvas, (tx, py + ph), (tx, py + ph + 4), (100, 100, 100), 1)
        cv2.putText(canvas, str(tick), (tx - 4, height - 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (140, 140, 140), 1)

    def to_canvas(lam, ncc):
        cx = int(px + (lam - LAM_MIN) / (LAM_MAX - LAM_MIN) * pw)
        cy = int(py + ph - (ncc - ncc_lo) / (ncc_hi - ncc_lo) * ph)
        return cx, cy

    # Draw curve
    pts_curve = [to_canvas(l, n) for l, n in zip(_LAM_CURVE, ncc_curve)]
    for i in range(len(pts_curve) - 1):
        cv2.line(canvas, pts_curve[i], pts_curve[i + 1], (100, 220, 100), 2)

    # True depth vertical line
    tx_true, _ = to_canvas(lam_true, 0)
    cv2.line(canvas, (tx_true, py), (tx_true, py + ph), (80, 80, 180), 1)
    cv2.putText(canvas, f"true", (tx_true + 2, py + 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 220), 1)

    # Current λ vertical line + prominent circle marker on the curve
    cx_cur, cy_cur = to_canvas(lam_current, max(ncc_lo, min(ncc_hi, ncc_current)))
    cv2.line(canvas, (cx_cur, py), (cx_cur, py + ph), (60, 200, 220), 2)

    # Large filled circle with white outline to highlight current point
    cv2.circle(canvas, (cx_cur, cy_cur), 8, (60, 200, 220), -1)      # Filled cyan
    cv2.circle(canvas, (cx_cur, cy_cur), 8, (255, 255, 255), 2)      # White outline

    # Label with both lam and NCC value (use 'lam' instead of λ for cv2.putText ASCII compatibility)
    label_text = f"lam={lam_current:.2f}  NCC={ncc_current:.3f}" if ncc_current >= -0.5 else f"lam={lam_current:.2f}"
    label_x = cx_cur - 60 if cx_cur > width - 140 else cx_cur + 4
    label_y = py + 26 if cy_cur > py + 40 else py + 40
    cv2.putText(canvas, label_text, (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 200, 220), 1, cv2.LINE_AA)

    return canvas  # BGR


# =============================================================================
# State
# =============================================================================

class State:
    lam = DEFAULTS["lam"]
    cam_baseline = DEFAULTS["cam_baseline"]
    cam_convergence = DEFAULTS["cam_convergence"]
    cam_distance = DEFAULTS["cam_distance"]
    patch_size = DEFAULTS["patch_size"]
    point_x = IMG_W // 2
    point_y = IMG_H // 2
    lam_true = LAM_TRUE
    show_epipoles = False
    show_epi_lines = False
    show_matrices = False
    show_rectify = True
    epipole_weight = DEFAULTS["epipole_weight"]
    mode = DEFAULTS["mode"]
    _prev_cam_baseline = DEFAULTS["cam_baseline"]
    _prev_cam_convergence = DEFAULTS["cam_convergence"]
    _prev_cam_distance = DEFAULTS["cam_distance"]
    _prev_patch_size = DEFAULTS["patch_size"]
    _prev_point = (IMG_W // 2, IMG_H // 2)
    _ncc_curve_cached = None
    _depth_img = None       # cached depth-map visualisation (BGR)
    _depth_label = ""       # "GT" or "NCC"
    _depth_stale = False    # True when camera params changed since last compute

state = State()

# Initialize lam_true via raycast at the default point
_init_hit = raycast_scene(_SCENE, K_SHARED, cam.Rt_ref, IMG_W // 2, IMG_H // 2, IMG_H)
if _init_hit is not None:
    state.lam_true = _init_hit



# ── Mouse wheel callback for zoom ────────────────────────────────────────────
def on_mouse_wheel(sender, app_data):
    """Zoom orbit camera on scroll wheel."""
    if (dpg.is_item_hovered("gm_overview_img")
            or dpg.is_item_hovered("sw_overview_img")):
        OvCam.radius = np.clip(OvCam.radius - app_data * 0.5, 3.0, 30.0)


# ── Mouse click callback for point selection ────────────────────────────────────
def on_mouse_click(sender, app_data):
    """Select NCC evaluation point by clicking on either image."""
    # Accept clicks on ref or warped image in either mode
    for img_tag in ("gm_ref_img", "gm_warped_img",
                    "sw_ref_img", "sw_warped_img"):
        if dpg.is_item_hovered(img_tag):
            mx, my = dpg.get_mouse_pos(local=False)
            img_pos = dpg.get_item_rect_min(img_tag)
            img_size = dpg.get_item_rect_size(img_tag)

            local_x = mx - img_pos[0]
            local_y = my - img_pos[1]

            if img_size[0] > 0 and img_size[1] > 0:
                px = int(local_x / img_size[0] * IMG_W)
                py = int(local_y / img_size[1] * IMG_H)

                state.point_x = np.clip(px, 0, IMG_W - 1)
                state.point_y = np.clip(py, 0, IMG_H - 1)

                # Raycast to get ground-truth depth at this pixel
                hit = raycast_scene(_SCENE, K_SHARED, cam.Rt_ref,
                                     int(state.point_x), int(state.point_y), IMG_H)
                state.lam_true = hit if hit is not None else LAM_TRUE
            break


# =============================================================================
# Main
# =============================================================================

def main():
    dpg.create_context()
    load_fonts()

    with dpg.handler_registry():
        dpg.add_mouse_wheel_handler(callback=on_mouse_wheel)
        dpg.add_mouse_click_handler(button=0, callback=on_mouse_click)

    # ── Textures ──────────────────────────────────────────────────────────────
    with dpg.texture_registry(tag="texture_registry"):
        blank = [0.0] * (IMG_W * IMG_H * 4)
        for t in ["ref_tex", "warped_tex", "depth_tex"]:
            dpg.add_raw_texture(IMG_W, IMG_H, list(blank),
                                format=dpg.mvFormat_Float_rgba, tag=t)
        dpg.add_raw_texture(400, 300, [0.0] * (400 * 300 * 4),
                            format=dpg.mvFormat_Float_rgba, tag="curve_tex")
        dpg.add_raw_texture(OVERVIEW_SIZE, OVERVIEW_SIZE,
                            [0.0] * (OVERVIEW_SIZE * OVERVIEW_SIZE * 4),
                            format=dpg.mvFormat_Float_rgba, tag="overview_tex")

    # ── Callbacks (defined before UI) ─────────────────────────────────────────
    def on_preset_selected(sender, value):
        for name, bl, conv, dist in CAMERA_PRESETS:
            if name == value:
                state.cam_baseline = bl
                state.cam_convergence = conv
                state.cam_distance = dist
                dpg.set_value("cam_baseline_slider", bl)
                dpg.set_value("cam_convergence_slider", conv)
                dpg.set_value("cam_distance_slider", dist)
                break

    def update_patch_size(sender, width):
        odd = int(width) | 1
        state.patch_size = odd
        if dpg.does_item_exist("patch_size_display"):
            dpg.set_value("patch_size_display", f"{odd} px")
        # Patch size only affects NCC depth, not GT
        if state._depth_label == "NCC \u03bb (plane sweep)":
            state._depth_stale = True

    def on_gt_depth(sender=None, app_data=None):
        if state._depth_label == "GT \u03bb (rendered)" and not state._depth_stale:
            return
        state._depth_img = depth_to_colormap(cam.gt_depth, LAM_MIN, LAM_MAX)
        state._depth_label = "GT \u03bb (rendered)"
        state._depth_stale = False

    def on_ncc_depth(sender=None, app_data=None):
        if state._depth_label == "NCC \u03bb (plane sweep)" and not state._depth_stale:
            return
        state._depth_label = "Computing NCC \u03bb map..."
        assert cam.gt_depth is not None
        valid_mask = ~np.isnan(cam.gt_depth)
        lam_range = np.linspace(LAM_MIN, LAM_MAX, 60)
        depth = compute_ncc_depth_image(
            cam.ref_img, cam.other_img, cam.M_ref, cam.M_other,
            IMG_W, IMG_H, lam_range, int(state.patch_size),
            valid_mask=valid_mask)
        state._depth_img = depth_to_colormap(depth, LAM_MIN, LAM_MAX)
        state._depth_label = "NCC \u03bb (plane sweep)"
        state._depth_stale = False

    # ── Window ────────────────────────────────────────────────────────────────
    with dpg.window(label="Plane Sweep Stereo Demo", tag="main_window"):

        def _extra_reset():
            if dpg.does_item_exist("mode_radio"):
                dpg.set_value("mode_radio", DEFAULTS["mode"])

        add_global_controls(DEFAULTS, state,
                            reset_extra=_extra_reset,
                            guide=GUIDE_PLANE_SWEEP, guide_title="Plane Sweep Stereo")
        dpg.add_separator()

        # ── Control panels (row) ──────────────────────────────────────────
        with dpg.group(horizontal=True):

            # ── Mode selector (leftmost) ──────────────────────────────────
            with control_panel("Mode", width=200, height=180,
                               color=(255, 220, 100)):
                dpg.add_radio_button(
                    MODE_ITEMS,
                    default_value=DEFAULTS["mode"],
                    callback=make_state_updater(state, "mode"),
                    tag="mode_radio")

            dpg.add_spacer(width=8)

            # ── Camera (shared, always visible) ───────────────────────────
            with control_panel("Camera", width=440, height=180,
                               color=(150, 200, 255)):
                with dpg.group(horizontal=True):
                    dpg.add_text("Preset")
                    dpg.add_combo(PRESET_NAMES, default_value=PRESET_NAMES[2],
                                  callback=on_preset_selected,
                                  tag="preset_combo", width=230)
                with create_parameter_table():
                    add_parameter_row(
                        "Baseline", "cam_baseline_slider",
                        DEFAULTS["cam_baseline"], 0.0, 5.0,
                        make_state_updater(state, "cam_baseline"),
                        make_reset_callback(state, "cam_baseline",
                                            "cam_baseline_slider",
                                            DEFAULTS["cam_baseline"]),
                        format_str="%.2f")
                    add_parameter_row(
                        "Convergence", "cam_convergence_slider",
                        DEFAULTS["cam_convergence"], 0.0, 1.0,
                        make_state_updater(state, "cam_convergence"),
                        make_reset_callback(state, "cam_convergence",
                                            "cam_convergence_slider",
                                            DEFAULTS["cam_convergence"]),
                        format_str="%.2f")
                    add_parameter_row(
                        "Other cam Z", "cam_distance_slider",
                        DEFAULTS["cam_distance"], 1.0, 8.0,
                        make_state_updater(state, "cam_distance"),
                        make_reset_callback(state, "cam_distance",
                                            "cam_distance_slider",
                                            DEFAULTS["cam_distance"]),
                        format_str="%.1f")

            dpg.add_spacer(width=8)

            # ── Plane Sweep (Mode 2 only) ─────────────────────────────────
            with control_panel("Plane sweep (depth \u03bb)", width=360, height=180,
                               color=(220, 180, 100), tag="sweep_panel",
                               default_open=True):
                with create_parameter_table():
                    add_parameter_row(
                        "\u03bb", "lam_slider",
                        DEFAULTS["lam"], LAM_MIN, LAM_MAX,
                        make_state_updater(state, "lam"),
                        make_reset_callback(state, "lam", "lam_slider",
                                            DEFAULTS["lam"]),
                        format_str="%.2f")
                with dpg.group(horizontal=True):
                    dpg.add_text("NCC patch")
                    dpg.add_slider_int(
                        tag="patch_size_slider",
                        default_value=DEFAULTS["patch_size"],
                        min_value=1, max_value=31,
                        callback=update_patch_size, width=160, clamped=True)
                    dpg.add_text(f"{DEFAULTS['patch_size']} px",
                                 tag="patch_size_display")
                dpg.add_separator()
                with dpg.group(horizontal=True):
                    dpg.add_button(label="GT \u03bb map", callback=on_gt_depth)
                    dpg.add_button(label="NCC \u03bb map",
                                   callback=on_ncc_depth)

            dpg.add_spacer(width=8)

            # ── Visualisation (always visible, far right) ────────────────
            with control_panel("Visualisation", width=300, height=180,
                               color=(150, 255, 150)):
                dpg.add_checkbox(
                    label="Epipoles",
                    callback=lambda s, v: setattr(state, 'show_epipoles', v))
                dpg.add_checkbox(
                    label="Epipolar lines",
                    callback=lambda s, v: setattr(state, 'show_epi_lines', v))
                dpg.add_checkbox(
                    label="Show matrices",
                    callback=lambda s, v: setattr(state, 'show_matrices', v))
                dpg.add_checkbox(
                    label="Homography warp", default_value=True,
                    callback=lambda s, v: setattr(state, 'show_rectify', v))
                with create_parameter_table():
                    add_parameter_row(
                        "Epipole \u03b1", "epipole_weight_slider",
                        DEFAULTS["epipole_weight"], 0.0, 1.0,
                        make_state_updater(state, "epipole_weight"),
                        make_reset_callback(state, "epipole_weight",
                                            "epipole_weight_slider",
                                            DEFAULTS["epipole_weight"]),
                        format_str="%.2f")

        dpg.add_separator()

        # ── Status bar ───────────────────────────────────────────────────────
        dpg.add_text("", tag="status_text", color=(255, 220, 100))
        dpg.add_separator()

        # ── Matrix display (shared, toggled by checkbox) ──────────────────
        with dpg.group(tag="matrix_group", show=False):
            dpg.add_text(
                "H(\u03bb) = \u03bb \u00b7 B  +  \u03b1 \u00b7 a e\u2083\u1d40"
                "     where  B = K' R_rel K\u207b\u00b9,"
                "   a = K' C + t'  (epipole in other view)",
                tag="matrix_formula", color=(200, 200, 120))
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text(
                        "\u03bb \u00b7 B  (depth-scaled rotation):",
                        color=(150, 200, 255))
                    dpg.add_text("", tag="lam_b_text")
                dpg.add_spacer(width=20)
                with dpg.group():
                    dpg.add_text(
                        "+   \u03b1 \u00b7 a e\u2083\u1d40  (epipole \u2014 only col 3):",
                        tag="outer_label", color=(150, 255, 150))
                    dpg.add_text("", tag="outer_text")
                dpg.add_spacer(width=20)
                with dpg.group():
                    dpg.add_text("=   H(\u03bb):",
                                 color=(255, 200, 150))
                    dpg.add_text("", tag="h_text")
            dpg.add_text("", tag="epipole_text", color=(150, 255, 150))
            dpg.add_separator()

        # ══════════════════════════════════════════════════════════════════════
        # MODE 1: Two-View Geometry
        # ══════════════════════════════════════════════════════════════════════
        with dpg.group(tag="geom_images"):

            # Images: ref + warped + 3D overview
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Reference image", color=(150, 255, 150))
                    dpg.add_image("ref_tex", tag="gm_ref_img",
                                  width=IMG_W, height=IMG_H)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("Warped other  [H(\u03bb)]",
                                 tag="gm_warped_label",
                                 color=(150, 200, 255))
                    dpg.add_image("warped_tex", tag="gm_warped_img",
                                  width=IMG_W, height=IMG_H)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("3D Overview  (drag/scroll)",
                                 color=(150, 255, 150))
                    dpg.add_image("overview_tex", tag="gm_overview_img",
                                  width=OVERVIEW_SIZE,
                                  height=OVERVIEW_SIZE)

        # ══════════════════════════════════════════════════════════════════════
        # MODE 2: Plane Sweep
        # ══════════════════════════════════════════════════════════════════════
        with dpg.group(tag="sweep_images", show=False):

            # Images: ref + warped + NCC curve + depth map + 3D overview
            with dpg.group(horizontal=True):
                with dpg.group():
                    dpg.add_text("Reference image", color=(150, 255, 150))
                    dpg.add_image("ref_tex", tag="sw_ref_img",
                                  width=IMG_W, height=IMG_H)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("Warped other  [H(\u03bb)]",
                                 tag="sw_warped_label",
                                 color=(150, 200, 255))
                    dpg.add_image("warped_tex", tag="sw_warped_img",
                                  width=IMG_W, height=IMG_H)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("NCC vs. \u03bb", color=(220, 180, 100))
                    dpg.add_image("curve_tex", tag="sw_curve_img",
                                  width=400, height=300)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("\u03bb map", tag="sw_depth_label",
                                 color=(180, 150, 255))
                    dpg.add_image("depth_tex", tag="sw_depth_img",
                                  width=IMG_W, height=IMG_H)
                dpg.add_spacer(width=10)
                with dpg.group():
                    dpg.add_text("3D Overview  (drag/scroll)",
                                 color=(150, 255, 150))
                    dpg.add_image("overview_tex", tag="sw_overview_img",
                                  width=OVERVIEW_SIZE,
                                  height=OVERVIEW_SIZE)

    setup_viewport("Plane Sweep Stereo Demo \u2014 H(\u03bb)", 1700, 920,
                   "main_window", lambda: None, DEFAULTS["ui_scale"])

    bind_mono_font("lam_b_text", "outer_text", "h_text")
    dpg.set_value("ref_tex", convert_cv_to_dpg(cam.ref_img))

    # ── Main loop ─────────────────────────────────────────────────────────────
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()

        # ── Camera param change ──────────────────────────────────────────────
        cam_changed = (
            state.cam_baseline != state._prev_cam_baseline or
            state.cam_convergence != state._prev_cam_convergence or
            state.cam_distance != state._prev_cam_distance
        )
        if cam_changed:
            apply_camera_params(state.cam_baseline, state.cam_convergence,
                                state.cam_distance)
            state._ncc_curve_cached = None
            if state._depth_img is not None:
                state._depth_stale = True
            hit = raycast_scene(_SCENE, K_SHARED, cam.Rt_ref,
                                 int(state.point_x), int(state.point_y), IMG_H)
            state.lam_true = hit if hit is not None else LAM_TRUE
            state._prev_cam_baseline = state.cam_baseline
            state._prev_cam_convergence = state.cam_convergence
            state._prev_cam_distance = state.cam_distance

        # ── Mode toggling ──────────────────────────────────────────────────────
        is_geom = state.mode == "Two-View Geometry"
        dpg.configure_item("sweep_panel", show=not is_geom)
        dpg.configure_item("geom_images", show=is_geom)
        dpg.configure_item("sweep_images", show=not is_geom)

        # ── Orbit camera mouse handling ──────────────────────────────────────
        ov_hovered = (dpg.is_item_hovered("gm_overview_img")
                      or dpg.is_item_hovered("sw_overview_img"))
        if ov_hovered:
            mx, my = dpg.get_mouse_pos()
            if dpg.is_mouse_button_down(0):
                if OvCam._prev is not None:
                    dx = mx - OvCam._prev[0]
                    dy = my - OvCam._prev[1]
                    OvCam.az += dx * 0.01
                    OvCam.el = np.clip(OvCam.el - dy * 0.01, -1.4, 1.4)
                OvCam._prev = (mx, my)
            else:
                OvCam._prev = None
        else:
            OvCam._prev = None

        lam = float(state.lam)
        patch_size = int(state.patch_size)
        alpha = float(state.epipole_weight)

        # ── Compute H(lambda) and warp ───────────────────────────────────────
        B, a = decompose_H(cam.M_ref, cam.M_other)
        e3 = np.array([0.0, 0.0, 1.0])
        outer_ae3 = np.outer(a, e3)
        H = lam * B + alpha * outer_ae3
        warped = warp_other_to_ref(cam.other_img, H, IMG_W, IMG_H)

        # Choose what to display in the second panel
        if state.show_rectify:
            display_img = warped
        else:
            display_img = cam.other_img

        # NCC at selected point (always against warped for curve consistency)
        selected_point = (int(state.point_x), int(state.point_y))
        ncc = compute_ncc_score(cam.ref_img, warped,
                                center=selected_point, patch_size=patch_size)

        # ── NCC curve recompute (only in sweep mode) ─────────────────────────
        if not is_geom:
            curve_needs_recompute = (
                state._ncc_curve_cached is None or
                patch_size != state._prev_patch_size or
                selected_point != state._prev_point or
                alpha != getattr(state, '_prev_alpha', None)
            )
            if curve_needs_recompute:
                state._ncc_curve_cached = np.array([
                    compute_ncc_score(
                        cam.ref_img,
                        warp_other_to_ref(cam.other_img,
                                          l * B + alpha * outer_ae3,
                                          IMG_W, IMG_H),
                        center=selected_point, patch_size=patch_size)
                    for l in _LAM_CURVE
                ])
                state._prev_patch_size = patch_size
                state._prev_point = selected_point
                state._prev_alpha = alpha
        ncc_curve = state._ncc_curve_cached

        # ── 3D overview ──────────────────────────────────────────────────────
        ov_Rt = OvCam.make_Rt()
        frustum_ref = make_frustum_mesh(K_SHARED, cam.Rt_ref,
                                         IMG_W, IMG_H, near=0.3, far=6.0)
        frustum_other = make_frustum_mesh(K_SHARED, cam.Rt_other,
                                           IMG_W, IMG_H, near=0.3, far=6.0)
        axes = make_axis_mesh(origin=(0, 0, 0), length=1.0)
        ov_img = render_scene(_SCENE + frustum_ref + frustum_other + axes,
                              _OV_K, ov_Rt, OVERVIEW_SIZE, OVERVIEW_SIZE)

        # ── Image overlays ───────────────────────────────────────────────────
        if not is_geom and ncc_curve is not None:
            curve = draw_ncc_curve(lam, ncc, ncc_curve, lam_true=state.lam_true)
            dpg.set_value("curve_tex", convert_cv_to_dpg(curve))

        cx, cy = int(state.point_x), int(state.point_y)
        mk_col = (60, 200, 220)
        hh = max(1, state.patch_size // 2)

        display_marked = display_img.copy()
        cv2.rectangle(display_marked, (cx - hh, cy - hh),
                      (cx + hh, cy + hh), mk_col, 1)
        if not is_geom and state.show_rectify:
            cv2.putText(display_marked, f"lam={lam:.2f}", (cx + 15, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, mk_col, 1, cv2.LINE_AA)

        ref_marked = cam.ref_img.copy()
        cv2.rectangle(ref_marked, (cx - hh, cy - hh),
                      (cx + hh, cy + hh), mk_col, 1)

        # ── Epipoles / epipolar lines ────────────────────────────────────────
        if (state.show_epipoles or state.show_epi_lines) \
                and state.cam_baseline > 1e-4:
            ex_ref, ey_ref = compute_epipole_ref(cam.M_ref, cam.M_other,
                                                 IMG_H)
            # Other image: when warp is on both views share the ref frame,
            # so the same epipole applies.  When warp is off the other
            # image is in its own coordinate frame → use e' (projection of
            # the ref camera centre into the other view).
            if state.show_rectify:
                ex_disp, ey_disp = ex_ref, ey_ref
            else:
                ex_disp, ey_disp = compute_epipole_ref(
                    cam.M_other, cam.M_ref, IMG_H)

            if state.show_epi_lines:
                grid_color = (30, 120, 120)
                step = 50
                for gx in range(step, IMG_W, step):
                    for gy in range(step, IMG_H, step):
                        # Ref image — lines converge to e_ref
                        ddx, ddy = ex_ref - gx, ey_ref - gy
                        ln = np.sqrt(ddx * ddx + ddy * ddy)
                        if ln > 1e-3:
                            ddx, ddy = ddx / ln, ddy / ln
                            g1 = (int(gx - 2000 * ddx),
                                  int(gy - 2000 * ddy))
                            g2 = (int(gx + 2000 * ddx),
                                  int(gy + 2000 * ddy))
                            cv2.line(ref_marked, g1, g2,
                                     grid_color, 1, cv2.LINE_AA)
                        # Other image — lines converge to e_disp
                        ddx, ddy = ex_disp - gx, ey_disp - gy
                        ln = np.sqrt(ddx * ddx + ddy * ddy)
                        if ln > 1e-3:
                            ddx, ddy = ddx / ln, ddy / ln
                            g1 = (int(gx - 2000 * ddx),
                                  int(gy - 2000 * ddy))
                            g2 = (int(gx + 2000 * ddx),
                                  int(gy + 2000 * ddy))
                            cv2.line(display_marked, g1, g2,
                                     grid_color, 1, cv2.LINE_AA)

                # Highlighted epipolar line — ref image
                ddx, ddy = ex_ref - cx, ey_ref - cy
                length = np.sqrt(ddx * ddx + ddy * ddy)
                if length > 1e-3:
                    ddx, ddy = ddx / length, ddy / length
                    p1 = (int(cx - 2000 * ddx), int(cy - 2000 * ddy))
                    p2 = (int(cx + 2000 * ddx), int(cy + 2000 * ddy))
                    epi_color = (50, 255, 255)
                    cv2.line(ref_marked, p1, p2, epi_color, 2, cv2.LINE_AA)

                # Highlighted epipolar line — other image
                if state.show_rectify:
                    # Same frame as ref → identical line
                    ddx, ddy = ex_ref - cx, ey_ref - cy
                    length = np.sqrt(ddx * ddx + ddy * ddy)
                    if length > 1e-3:
                        ddx, ddy = ddx / length, ddy / length
                        p1 = (int(cx - 2000 * ddx),
                              int(cy - 2000 * ddy))
                        p2 = (int(cx + 2000 * ddx),
                              int(cy + 2000 * ddy))
                        epi_color = (50, 255, 255)
                        cv2.line(display_marked, p1, p2,
                                 epi_color, 2, cv2.LINE_AA)
                else:
                    # Own frame: epipolar line passes through e' and
                    # B @ p_ref (infinite-depth projection of clicked pt).
                    B, _ = decompose_H(cam.M_ref, cam.M_other)
                    p_uf = np.array([cx, IMG_H - 1 - cy, 1.0])
                    q = B @ p_uf
                    if abs(q[2]) > 1e-8:
                        qx = q[0] / q[2]
                        qy = IMG_H - 1 - q[1] / q[2]
                    else:
                        qx, qy = q[0] * 1e6, -(q[1]) * 1e6
                    ddx, ddy = ex_disp - qx, ey_disp - qy
                    length = np.sqrt(ddx * ddx + ddy * ddy)
                    if length > 1e-3:
                        ddx, ddy = ddx / length, ddy / length
                        p1 = (int(qx - 2000 * ddx),
                              int(qy - 2000 * ddy))
                        p2 = (int(qx + 2000 * ddx),
                              int(qy + 2000 * ddy))
                        epi_color = (50, 255, 255)
                        cv2.line(display_marked, p1, p2,
                                 epi_color, 2, cv2.LINE_AA)

            if state.show_epipoles:
                epi_dot = (255, 50, 200)
                # Ref epipole
                epi_pt = (int(round(ex_ref)), int(round(ey_ref)))
                if (-500 < epi_pt[0] < IMG_W + 500
                        and -500 < epi_pt[1] < IMG_H + 500):
                    cv2.drawMarker(ref_marked, epi_pt, epi_dot,
                                   cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
                    cv2.putText(ref_marked, "e",
                                (epi_pt[0] + 8, epi_pt[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                epi_dot, 1, cv2.LINE_AA)
                # Other image epipole
                epi_pt2 = (int(round(ex_disp)), int(round(ey_disp)))
                if (-500 < epi_pt2[0] < IMG_W + 500
                        and -500 < epi_pt2[1] < IMG_H + 500):
                    cv2.drawMarker(display_marked, epi_pt2, epi_dot,
                                   cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)
                    lbl = "e" if state.show_rectify else "e'"
                    cv2.putText(display_marked, lbl,
                                (epi_pt2[0] + 8, epi_pt2[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                epi_dot, 1, cv2.LINE_AA)

        # ── Update textures ──────────────────────────────────────────────────
        dpg.set_value("ref_tex", convert_cv_to_dpg(ref_marked))
        dpg.set_value("warped_tex", convert_cv_to_dpg(display_marked))
        dpg.set_value("overview_tex", convert_cv_to_dpg(ov_img))

        if not is_geom:
            if state._depth_img is not None:
                dpg.set_value("depth_tex",
                              convert_cv_to_dpg(state._depth_img))
            if state._depth_label:
                lbl = state._depth_label
                if state._depth_stale:
                    lbl += "  (stale)"
                dpg.set_value("sw_depth_label", lbl)
            else:
                dpg.set_value("sw_depth_label",
                              "\u03bb map \u2014 click GT or NCC")

        # Warped panel labels (mode-specific wording)
        if state.show_rectify:
            dpg.set_value("gm_warped_label",
                          "Other \u2192 ref (homography warp)")
            dpg.set_value("sw_warped_label",
                          "Other \u2192 ref via H(\u03bb)  (same coord frame)")
        else:
            wlbl = "Other camera  (own coord frame)"
            dpg.set_value("gm_warped_label", wlbl)
            dpg.set_value("sw_warped_label", wlbl)

        # ── Matrix display toggle ─────────────────────────────────────────
        dpg.configure_item("matrix_group", show=state.show_matrices)
        if state.show_matrices:
            dpg.set_value("lam_b_text", _fmt_mat(lam * B))
            dpg.set_value("outer_text", _fmt_mat(alpha * outer_ae3))
            dpg.set_value("h_text", _fmt_mat(H))
            if alpha < 1.0 - 1e-6:
                dpg.set_value("outer_label",
                              f"+   \u03b1\u00b7a e\u2083\u1d40"
                              f"  (epipole, \u03b1={alpha:.2f}):")
            else:
                dpg.set_value("outer_label",
                              "+   \u03b1 \u00b7 a e\u2083\u1d40"
                              "  (epipole \u2014 only col 3):")
            dpg.set_value("epipole_text",
                           f"a = ({a[0]:.2f}, {a[1]:.2f}, {a[2]:.2f})  "
                           f"\u2014 epipole e' in other image;  "
                           f"\u03b1\u00b7a e\u2083\u1d40 puts \u03b1\u00b7a"
                           f" into column 3 only"
                           f" (homogeneous translation)")

        if dpg.does_item_exist("patch_size_display"):
            dpg.set_value("patch_size_display", f"{patch_size} px")

        # ── Status bar ───────────────────────────────────────────────────────
        toe_in_deg = state.cam_convergence * _MAX_TOE_IN_DEG
        conv_label = ("parallel" if toe_in_deg < 0.5
                      else f"toe-in={toe_in_deg:.1f}\u00b0")
        cam_status = (f"baseline={state.cam_baseline:.2f}  {conv_label}"
                      f"  cam dist={state.cam_distance:.1f}")
        if is_geom:
            dpg.set_value(
                "status_text",
                f"{cam_status}  |  "
                f"Point: ({state.point_x}, {state.point_y})")
        else:
            ncc_str = f"{ncc:.4f}" if ncc >= -0.5 else "invalid"
            dpg.set_value(
                "status_text",
                f"{cam_status}  |  "
                f"\u03bb = {lam:.3f}  |  NCC = {ncc_str}  |  "
                f"True \u03bb = {state.lam_true:.2f}  |  "
                f"Point: ({state.point_x}, {state.point_y})  |  "
                f"{'>>> PEAK NEAR HERE <<<' if abs(lam - state.lam_true) < 0.3 else ''}")

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


if __name__ == "__main__":
    main()
