"""
Live 3D Camera Demo with Dear PyGui controls
CSCI 1430 - Brown University

Interactive exploration of intrinsic (K) and extrinsic [R|t] camera matrices.
Two synchronized panels: camera view and overhead overview with frustum wireframe.
Toggle between world and camera reference frame interpretations.
"""

import numpy as np
import dearpygui.dearpygui as dpg

from utils.demo_3d import (
    build_intrinsic, build_rotation, build_extrinsic, euler_from_rotation,
    fov_to_focal, make_lookat_Rt,
    render_scene, create_default_scene,
    make_frustum_mesh, make_axis_mesh, make_camera_axes_mesh,
    format_matrix,
)
from utils.demo_utils import convert_cv_to_dpg
from utils.demo_ui import (
    load_fonts, bind_mono_font,
    setup_viewport, make_state_updater, make_reset_callback,
    create_parameter_table, add_parameter_row,
    add_global_controls, control_panel,
    poll_collapsible_panels,
)


# =============================================================================
# Defaults
# =============================================================================

DEFAULTS = {
    "rot_x": 0.0,
    "rot_y": 0.0,
    "rot_z": 0.0,
    "trans_x": 0.0,
    "trans_y": 0.0,
    "trans_z": -5.0,
    "focal_length": 35.0,
    "sensor_w": 36.0,
    "sensor_h": 36.0,
    "skew": 0.0,
    "img_w": 400,
    "img_h": 400,
    "cx_offset": 0.0,
    "cy_offset": 0.0,
    "ui_scale": 1.5,
}

OVERVIEW_SIZE = 400

GUIDE_CAMERA = [
    {"title": "The projection pipeline",
     "body": "A 3D world point is transformed to pixel coordinates through:\n"
             "  World \u2192 [R|t] \u2192 Camera coords \u2192 K \u2192 Pixel\n"
             "M = K[R|t] is the full 3\u00d74 projection matrix. "
             "Adjust sliders to see each component's effect on the rendered scene."},
    {"title": "Intrinsic matrix K",
     "body": "Focal length controls zoom (field of view).\n"
             "Principal point (cx, cy) shifts where the optical axis hits the sensor.\n"
             "Skew (usually 0) shears the pixel grid.\n"
             "K converts 3D camera coordinates into 2D pixel coordinates."},
    {"title": "Extrinsic matrix [R|t]",
     "body": "R (3\u00d73 rotation): camera orientation via Euler angles (rx, ry, rz).\n"
             "t (3\u00d71 translation): camera position displacement.\n"
             "Together they place the camera in the world coordinate system."},
    {"title": "Reference frames",
     "body": "World frame: sliders control camera pose relative to the world.\n"
             "Camera frame: sliders control where the world appears relative to "
             "the camera. Toggle between them to see the dual interpretation. "
             "The rendered images are always identical \u2014 only the slider meanings change."},
    {"title": "The 3\u00d74 matrix M",
     "body": "M = K[R|t] combines intrinsic and extrinsic parameters. "
             "Toggle 'Show matrices algebraically' to see how the symbolic "
             "parameters compose into M. Each slider change updates M in real time."},
]


# =============================================================================
# State
# =============================================================================

class State:
    # Extrinsic
    rot_x = DEFAULTS["rot_x"]
    rot_y = DEFAULTS["rot_y"]
    rot_z = DEFAULTS["rot_z"]
    trans_x = DEFAULTS["trans_x"]
    trans_y = DEFAULTS["trans_y"]
    trans_z = DEFAULTS["trans_z"]

    # Intrinsic
    focal_length = DEFAULTS["focal_length"]
    sensor_w = DEFAULTS["sensor_w"]
    sensor_h = DEFAULTS["sensor_h"]
    img_w = DEFAULTS["img_w"]
    img_h = DEFAULTS["img_h"]
    skew = DEFAULTS["skew"]
    cx_offset = DEFAULTS["cx_offset"]
    cy_offset = DEFAULTS["cy_offset"]

    # Resolution tracking (for texture recreation)
    _prev_img_w = DEFAULTS["img_w"]
    _prev_img_h = DEFAULTS["img_h"]
    _texture_counter = 0
    _cam_texture_tag = "camera_texture_0"

    # Mode
    camera_frame = True  # False = world frame, True = camera frame
    show_symbolic = False  # False = numeric values, True = algebraic symbols

    # Cached matrices
    K = None
    Rt = None
    M = None


state = State()

# Scene (created once)
scene_meshes = create_default_scene()

# Overview camera (fixed)
overview_Rt = make_lookat_Rt(
    eye=np.array([8.0, 6.0, 8.0]),
    target=np.array([0.0, 0.5, 0.0]),
)
overview_K = build_intrinsic(
    fov_to_focal(50, OVERVIEW_SIZE), fov_to_focal(50, OVERVIEW_SIZE),
    0, OVERVIEW_SIZE / 2, OVERVIEW_SIZE / 2,
)


# =============================================================================
# Callbacks
# =============================================================================

def on_reference_frame_change(sender, value):
    """Handle world/camera frame toggle, converting slider values to keep the view stable."""
    new_camera_frame = (value == "Camera")

    if new_camera_frame == state.camera_frame:
        return

    # Get current extrinsic to preserve the view
    R_curr = state.Rt[:, :3] if state.Rt is not None else np.eye(3)
    t_curr = state.Rt[:, 3] if state.Rt is not None else np.array([0, 0, 5.0])

    if new_camera_frame:
        # Switching TO camera frame: extract camera pose from current [R|t]
        # Camera position = -R^T @ t
        cam_pos = -R_curr.T @ t_curr
        # Camera orientation = R^T
        R_cam = R_curr.T
        alpha, beta, gamma = euler_from_rotation(R_cam)

        state.rot_x = alpha
        state.rot_y = beta
        state.rot_z = gamma
        state.trans_x = cam_pos[0]
        state.trans_y = cam_pos[1]
        state.trans_z = cam_pos[2]
    else:
        # Switching TO world frame: use current [R|t] directly
        alpha, beta, gamma = euler_from_rotation(R_curr)

        state.rot_x = alpha
        state.rot_y = beta
        state.rot_z = gamma
        state.trans_x = t_curr[0]
        state.trans_y = t_curr[1]
        state.trans_z = t_curr[2]

    state.camera_frame = new_camera_frame

    # Update slider UI to match new values
    _sync_sliders_to_state()


def _sync_sliders_to_state():
    """Update all slider widgets to match current state values."""
    slider_map = {
        "rot_x_slider": state.rot_x,
        "rot_y_slider": state.rot_y,
        "rot_z_slider": state.rot_z,
        "trans_x_slider": state.trans_x,
        "trans_y_slider": state.trans_y,
        "trans_z_slider": state.trans_z,
        "focal_length_slider": state.focal_length,
        "sensor_w_slider": state.sensor_w,
        "sensor_h_slider": state.sensor_h,
        "img_w_slider": state.img_w,
        "img_h_slider": state.img_h,
        "skew_slider": state.skew,
        "cx_offset_slider": state.cx_offset,
        "cy_offset_slider": state.cy_offset,
    }
    for tag, val in slider_map.items():
        if dpg.does_item_exist(tag):
            dpg.set_value(tag, val)


def _camera_extra_reset():
    """Extra reset logic beyond DEFAULTS iteration."""
    state.camera_frame = True
    state.show_symbolic = False
    if dpg.does_item_exist("ref_frame_combo"):
        dpg.set_value("ref_frame_combo", "Camera")
    if dpg.does_item_exist("symbolic_toggle"):
        dpg.set_value("symbolic_toggle", False)
    _sync_sliders_to_state()


def _recreate_camera_texture(w, h):
    """Recreate the camera texture at a new resolution using a unique tag."""
    old_tag = state._cam_texture_tag
    state._texture_counter += 1
    new_tag = f"camera_texture_{state._texture_counter}"
    blank = [0.0] * (w * h * 4)
    dpg.add_raw_texture(w, h, blank,
                        format=dpg.mvFormat_Float_rgba, tag=new_tag,
                        parent="texture_registry")
    if dpg.does_item_exist("camera_image"):
        dpg.configure_item("camera_image", texture_tag=new_tag)
    state._cam_texture_tag = new_tag
    if dpg.does_item_exist(old_tag):
        dpg.delete_item(old_tag)


def update_image_sizes():
    """Adjust image display sizes to fit viewport."""
    vp_width = dpg.get_viewport_client_width()
    vp_height = dpg.get_viewport_client_height()

    # Available space for images (below controls)
    available_width = vp_width - 60
    available_height = vp_height - 480

    img_size = min(available_width // 2, available_height)
    img_size = max(img_size, 100)

    for tag in ["camera_image", "overview_image"]:
        if dpg.does_item_exist(tag):
            dpg.configure_item(tag, width=img_size, height=img_size)


def on_viewport_resize():
    update_image_sizes()


# =============================================================================
# Computation (pure NumPy/OpenCV, no DPG dependency)
# =============================================================================

def compute_camera_frame(
    focal_length, sensor_w, sensor_h, img_w, img_h,
    skew, cx_offset, cy_offset,
    rot_x, rot_y, rot_z, trans_x, trans_y, trans_z,
    camera_frame,
):
    """Build K, [R|t], M matrices and render camera + overview images.

    Returns (K, Rt, M, camera_img, overview_img).
    """
    iw, ih = int(img_w), int(img_h)
    fx = focal_length * iw / sensor_w
    fy = focal_length * ih / sensor_h
    cx = iw / 2.0 + cx_offset
    cy = ih / 2.0 + cy_offset
    K = build_intrinsic(fx, fy, skew, cx, cy)

    Rt, R = build_extrinsic(
        rot_x, rot_y, rot_z,
        trans_x, trans_y, trans_z,
        1.0, camera_frame=camera_frame,
    )

    M = K @ Rt
    camera_img = render_scene(scene_meshes, K, Rt, iw, ih)

    frustum_meshes = make_frustum_mesh(K, Rt, iw, ih, near=0.3, far=5.0)
    world_axes = make_axis_mesh(origin=(0, 0, 0), length=1.5)
    cam_axes = make_camera_axes_mesh(Rt, length=0.8)
    overview_meshes = scene_meshes + frustum_meshes + world_axes + cam_axes
    overview_img = render_scene(overview_meshes, overview_K, overview_Rt,
                                OVERVIEW_SIZE, OVERVIEW_SIZE)

    return K, Rt, M, camera_img, overview_img


# =============================================================================
# Main
# =============================================================================

def main():
    # Dear PyGui setup
    dpg.create_context()

    load_fonts()

    with dpg.texture_registry(tag="texture_registry"):
        cam_w, cam_h = DEFAULTS["img_w"], DEFAULTS["img_h"]
        blank_cam = [0.0] * (cam_w * cam_h * 4)
        dpg.add_raw_texture(cam_w, cam_h, blank_cam,
                            format=dpg.mvFormat_Float_rgba, tag=state._cam_texture_tag)
        blank_ov = [0.0] * (OVERVIEW_SIZE * OVERVIEW_SIZE * 4)
        dpg.add_raw_texture(OVERVIEW_SIZE, OVERVIEW_SIZE, blank_ov,
                            format=dpg.mvFormat_Float_rgba, tag="overview_texture")

    with dpg.window(label="3D Camera Demo", tag="main_window"):
        add_global_controls(
            DEFAULTS, state,
            reset_extra=_camera_extra_reset,
            guide=GUIDE_CAMERA, guide_title="3D Camera",
        )

        dpg.add_separator()

        # --- Parameter controls in columns ---
        with dpg.group(horizontal=True):
            # View Options
            with control_panel("View Options", width=220, height=280,
                               color=(255, 220, 100)):
                dpg.add_text("Reference Frame:")
                dpg.add_combo(
                    items=["World", "Camera"],
                    default_value="Camera",
                    callback=on_reference_frame_change,
                    tag="ref_frame_combo",
                    width=100,
                )
                dpg.add_spacer(height=5)
                dpg.add_checkbox(
                    label="Show matrices\nalgebraically",
                    default_value=False,
                    callback=_on_symbolic_toggle,
                    tag="symbolic_toggle",
                )

            dpg.add_spacer(width=10)

            # Column 1: Pixel-space projection
            with control_panel("Projection to Pixels", width=360, height=280,
                               color=(150, 200, 255)):
                with create_parameter_table():
                        add_parameter_row("img_w (px)", "img_w_slider", DEFAULTS["img_w"],
                                          100, 800,
                                          make_state_updater(state, "img_w"),
                                          make_reset_callback(state, "img_w", "img_w_slider", DEFAULTS["img_w"]),
                                          slider_type="int")
                        add_parameter_row("img_h (px)", "img_h_slider", DEFAULTS["img_h"],
                                          100, 800,
                                          make_state_updater(state, "img_h"),
                                          make_reset_callback(state, "img_h", "img_h_slider", DEFAULTS["img_h"]),
                                          slider_type="int")
                        add_parameter_row("Skew s (px)", "skew_slider", DEFAULTS["skew"],
                                          -500.0, 500.0,
                                          make_state_updater(state, "skew"),
                                          make_reset_callback(state, "skew", "skew_slider", DEFAULTS["skew"]),
                                          format_str="%.1f")
                        add_parameter_row("Center_x offset (px)", "cx_offset_slider", DEFAULTS["cx_offset"],
                                          -200.0, 200.0,
                                          make_state_updater(state, "cx_offset"),
                                          make_reset_callback(state, "cx_offset", "cx_offset_slider", DEFAULTS["cx_offset"]),
                                          format_str="%.1f")
                        add_parameter_row("Center_y offset (px)", "cy_offset_slider", DEFAULTS["cy_offset"],
                                          -200.0, 200.0,
                                          make_state_updater(state, "cy_offset"),
                                          make_reset_callback(state, "cy_offset", "cy_offset_slider", DEFAULTS["cy_offset"]),
                                          format_str="%.1f")

            dpg.add_spacer(width=10)

            # Column 2: Physical camera
            with control_panel("Physical Camera", width=360, height=310,
                               color=(220, 180, 100)):
                with create_parameter_table():
                        add_parameter_row("f (mm)", "focal_length_slider", DEFAULTS["focal_length"],
                                          5.0, 200.0,
                                          make_state_updater(state, "focal_length"),
                                          make_reset_callback(state, "focal_length", "focal_length_slider", DEFAULTS["focal_length"]),
                                          format_str="%.1f")
                        add_parameter_row("Sensor W (mm)", "sensor_w_slider", DEFAULTS["sensor_w"],
                                          5.0, 100.0,
                                          make_state_updater(state, "sensor_w"),
                                          make_reset_callback(state, "sensor_w", "sensor_w_slider", DEFAULTS["sensor_w"]),
                                          format_str="%.1f")
                        add_parameter_row("Sensor H (mm)", "sensor_h_slider", DEFAULTS["sensor_h"],
                                          5.0, 100.0,
                                          make_state_updater(state, "sensor_h"),
                                          make_reset_callback(state, "sensor_h", "sensor_h_slider", DEFAULTS["sensor_h"]),
                                          format_str="%.1f")

                dpg.add_spacer(height=5)
                dpg.add_text("Physical to Pixel:", color=(200, 200, 150))
                dpg.add_text("", tag="conversion_text")

            dpg.add_spacer(width=10)

            # Column 3: Extrinsic
            with control_panel("Extrinsic [R|t]", width=360, height=280,
                               color=(150, 255, 150)):
                with create_parameter_table():
                        add_parameter_row("Rot X (rad)", "rot_x_slider", DEFAULTS["rot_x"],
                                          -np.pi, np.pi,
                                          make_state_updater(state, "rot_x"),
                                          make_reset_callback(state, "rot_x", "rot_x_slider", DEFAULTS["rot_x"]),
                                          format_str="%.2f")
                        add_parameter_row("Rot Y (rad)", "rot_y_slider", DEFAULTS["rot_y"],
                                          -np.pi, np.pi,
                                          make_state_updater(state, "rot_y"),
                                          make_reset_callback(state, "rot_y", "rot_y_slider", DEFAULTS["rot_y"]),
                                          format_str="%.2f")
                        add_parameter_row("Rot Z (rad)", "rot_z_slider", DEFAULTS["rot_z"],
                                          -np.pi, np.pi,
                                          make_state_updater(state, "rot_z"),
                                          make_reset_callback(state, "rot_z", "rot_z_slider", DEFAULTS["rot_z"]),
                                          format_str="%.2f")
                        add_parameter_row("Tx (m)", "trans_x_slider", DEFAULTS["trans_x"],
                                          -10.0, 10.0,
                                          make_state_updater(state, "trans_x"),
                                          make_reset_callback(state, "trans_x", "trans_x_slider", DEFAULTS["trans_x"]),
                                          format_str="%.2f")
                        add_parameter_row("Ty (m)", "trans_y_slider", DEFAULTS["trans_y"],
                                          -10.0, 10.0,
                                          make_state_updater(state, "trans_y"),
                                          make_reset_callback(state, "trans_y", "trans_y_slider", DEFAULTS["trans_y"]),
                                          format_str="%.2f")
                        add_parameter_row("Tz (m)", "trans_z_slider", DEFAULTS["trans_z"],
                                          -10.0, 10.0,
                                          make_state_updater(state, "trans_z"),
                                          make_reset_callback(state, "trans_z", "trans_z_slider", DEFAULTS["trans_z"]),
                                          format_str="%.2f")

        dpg.add_separator()

        # --- Matrix display: 3 panels side by side ---
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("K (Intrinsic 3\u00d73):", color=(150, 200, 255))
                dpg.add_text("", tag="k_text")
            dpg.add_spacer(width=30)
            with dpg.group():
                dpg.add_text("[R|t] (Extrinsic 3\u00d74):", color=(150, 255, 150))
                dpg.add_text("", tag="rt_text")
            dpg.add_spacer(width=30)
            with dpg.group():
                dpg.add_text("M = K[R|t] (Full 3\u00d74):", color=(255, 200, 150))
                dpg.add_text("", tag="m_text")

        dpg.add_separator()

        # --- Image panels ---
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_text("", tag="camera_view_label", color=(150, 200, 255))
                dpg.add_image(state._cam_texture_tag, tag="camera_image")
            dpg.add_spacer(width=10)
            with dpg.group():
                dpg.add_text("Overview (with frustum)", color=(150, 255, 150))
                dpg.add_image("overview_texture", tag="overview_image")

    bind_mono_font("k_text", "rt_text", "m_text")

    # Viewport setup
    setup_viewport("3D Camera Demo", 1400, 850,
                    "main_window", on_viewport_resize, DEFAULTS["ui_scale"])
    update_image_sizes()

    # === Main loop ===
    while dpg.is_dearpygui_running():
        poll_collapsible_panels()
        # Dynamic image resolution
        iw, ih = int(state.img_w), int(state.img_h)

        # Recreate camera texture if resolution changed
        if iw != state._prev_img_w or ih != state._prev_img_h:
            _recreate_camera_texture(iw, ih)
            state._prev_img_w, state._prev_img_h = iw, ih

        K, Rt, M, camera_img, overview_img = compute_camera_frame(
            state.focal_length, state.sensor_w, state.sensor_h,
            state.img_w, state.img_h,
            state.skew, state.cx_offset, state.cy_offset,
            state.rot_x, state.rot_y, state.rot_z,
            state.trans_x, state.trans_y, state.trans_z,
            state.camera_frame,
        )
        state.K, state.Rt, state.M = K, Rt, M
        fx = state.focal_length * iw / state.sensor_w
        fy = state.focal_length * ih / state.sensor_h

        # Update textures
        dpg.set_value(state._cam_texture_tag, convert_cv_to_dpg(camera_img))
        dpg.set_value("overview_texture", convert_cv_to_dpg(overview_img))

        # Update camera view label
        dpg.set_value("camera_view_label", f"Camera View ({iw}x{ih})")

        # Update conversion display
        conv = (
            f"f_x = f \u00b7 img_w / sensor_w\n"
            f"   = {state.focal_length:.1f} \u00b7 {iw} / {state.sensor_w:.1f}\n"
            f"   = {fx:.2f} px\n"
            f"\n"
            f"f_y = f \u00b7 img_h / sensor_h\n"
            f"   = {state.focal_length:.1f} \u00b7 {ih} / {state.sensor_h:.1f}\n"
            f"   = {fy:.2f} px"
        )
        dpg.set_value("conversion_text", conv)

        # Update matrix display
        if state.show_symbolic:
            dpg.set_value("k_text", _K_SYMBOLIC)
            dpg.set_value("rt_text", _RT_SYMBOLIC)
            dpg.set_value("m_text", _M_SYMBOLIC)
        else:
            dpg.set_value("k_text", _fmt_mat(K))
            dpg.set_value("rt_text", _fmt_mat(Rt))
            dpg.set_value("m_text", _fmt_mat(M))

        dpg.render_dearpygui_frame()

    dpg.destroy_context()


def _fmt_mat(mat):
    """Format matrix for display as aligned text."""
    rows, cols = mat.shape
    lines = []
    for r in range(rows):
        cells = " ".join(f"{mat[r, c]:8.2f}" for c in range(cols))
        lines.append(f"[ {cells} ]")
    return "\n".join(lines)


# Symbolic (algebraic) matrix templates
_K_SYMBOLIC = (
    "[  f_x    s    c_x ]\n"
    "[   0    f_y   c_y ]\n"
    "[   0     0     1  ]"
)

_RT_SYMBOLIC = (
    "[ r11  r12  r13 | t_x ]\n"
    "[ r21  r22  r23 | t_y ]\n"
    "[ r31  r32  r33 | t_z ]"
)

_M_SYMBOLIC = (
    "[ m11  m12  m13  m14 ]\n"
    "[ m21  m22  m23  m24 ]\n"
    "[ m31  m32  m33  m34 ]"
)


def _on_symbolic_toggle(sender, value):
    state.show_symbolic = value


if __name__ == "__main__":
    main()
