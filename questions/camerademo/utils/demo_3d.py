"""
3D rendering utilities for CSCI 1430 camera demo.
Software renderer using NumPy + OpenCV with painter's algorithm.
"""

import numpy as np
import cv2


# =============================================================================
# Mesh Generation
# =============================================================================

def make_cube(center, size, color):
    """Create a cube mesh.

    Args:
        center: (x, y, z) center position
        size: edge length
        color: (B, G, R) base color

    Returns:
        Mesh dict with vertices (8,3), faces (6 quads), color
    """
    cx, cy, cz = center
    h = size / 2.0
    vertices = np.array([
        [cx - h, cy - h, cz - h],
        [cx + h, cy - h, cz - h],
        [cx + h, cy + h, cz - h],
        [cx - h, cy + h, cz - h],
        [cx - h, cy - h, cz + h],
        [cx + h, cy - h, cz + h],
        [cx + h, cy + h, cz + h],
        [cx - h, cy + h, cz + h],
    ])
    # Quads with outward-facing normals (CCW when viewed from outside)
    faces = [
        [0, 3, 2, 1],  # back  (-Z)
        [4, 5, 6, 7],  # front (+Z)
        [0, 1, 5, 4],  # bottom (-Y)
        [2, 3, 7, 6],  # top (+Y)
        [0, 4, 7, 3],  # left (-X)
        [1, 2, 6, 5],  # right (+X)
    ]
    return {"vertices": vertices, "faces": faces, "color": color}


def make_sphere(center, radius, color, n_lat=10, n_lon=16):
    """Create a UV sphere mesh.

    Args:
        center: (x, y, z) center position
        radius: sphere radius
        color: (B, G, R) base color
        n_lat: number of latitude divisions
        n_lon: number of longitude divisions

    Returns:
        Mesh dict
    """
    cx, cy, cz = center
    vertices = []
    faces = []

    # Top pole
    vertices.append([cx, cy + radius, cz])

    # Latitude rings
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            x = cx + radius * np.sin(phi) * np.cos(theta)
            y = cy + radius * np.cos(phi)
            z = cz + radius * np.sin(phi) * np.sin(theta)
            vertices.append([x, y, z])

    # Bottom pole
    vertices.append([cx, cy - radius, cz])
    vertices = np.array(vertices)

    # Top cap triangles (outward normal = away from center, i.e. +Y at pole)
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j_next, 1 + j])

    # Quad strips between latitude rings
    for i in range(n_lat - 2):
        ring_start = 1 + i * n_lon
        next_ring_start = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            faces.append([
                ring_start + j,
                ring_start + j_next,
                next_ring_start + j_next,
                next_ring_start + j,
            ])

    # Bottom cap triangles
    bottom = len(vertices) - 1
    last_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([bottom, last_ring + j, last_ring + j_next])

    return {"vertices": vertices, "faces": faces, "color": color}


def make_octahedron(center, radius, color):
    """Create a regular octahedron mesh — looks like a 3D diamond.

    Args:
        center: (x, y, z) center position
        radius: distance from center to each vertex
        color: (B, G, R) base color

    Returns:
        Mesh dict with 6 vertices and 8 triangular faces
    """
    cx, cy, cz = center
    r = radius
    vertices = np.array([
        [cx,   cy+r, cz  ],  # 0: top
        [cx+r, cy,   cz  ],  # 1: right
        [cx,   cy,   cz+r],  # 2: front
        [cx-r, cy,   cz  ],  # 3: left
        [cx,   cy,   cz-r],  # 4: back
        [cx,   cy-r, cz  ],  # 5: bottom
    ], dtype=np.float64)
    # CCW winding for outward normals (verified via cross-product)
    faces = [
        [0, 2, 1],  # top-front-right
        [0, 3, 2],  # top-left-front
        [0, 4, 3],  # top-back-left
        [0, 1, 4],  # top-right-back
        [5, 1, 2],  # bottom-right-front
        [5, 2, 3],  # bottom-front-left
        [5, 3, 4],  # bottom-left-back
        [5, 4, 1],  # bottom-back-right
    ]
    return {"vertices": vertices, "faces": faces, "color": color}


def make_cylinder(base_center, radius, height, color, n_seg=16):
    """Create a cylinder mesh (Y-axis aligned).

    Args:
        base_center: (x, y, z) center of the bottom circle
        radius: cylinder radius
        height: cylinder height
        color: (B, G, R) base color
        n_seg: number of circumference segments

    Returns:
        Mesh dict
    """
    cx, cy, cz = base_center
    vertices = []
    faces = []

    # Bottom circle center
    vertices.append([cx, cy, cz])
    # Top circle center
    vertices.append([cx, cy + height, cz])

    # Bottom ring vertices (indices 2 .. 2+n_seg-1)
    for i in range(n_seg):
        theta = 2.0 * np.pi * i / n_seg
        x = cx + radius * np.cos(theta)
        z = cz + radius * np.sin(theta)
        vertices.append([x, cy, z])

    # Top ring vertices (indices 2+n_seg .. 2+2*n_seg-1)
    for i in range(n_seg):
        theta = 2.0 * np.pi * i / n_seg
        x = cx + radius * np.cos(theta)
        z = cz + radius * np.sin(theta)
        vertices.append([x, cy + height, z])

    vertices = np.array(vertices)

    bot_start = 2
    top_start = 2 + n_seg

    # Bottom cap (outward normal = -Y)
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([0, bot_start + i, bot_start + i_next])

    # Top cap (outward normal = +Y)
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([1, top_start + i_next, top_start + i])

    # Side quads (outward normal = radially outward)
    for i in range(n_seg):
        i_next = (i + 1) % n_seg
        faces.append([
            bot_start + i,
            top_start + i,
            top_start + i_next,
            bot_start + i_next,
        ])

    return {"vertices": vertices, "faces": faces, "color": color}


def make_ground_grid(y, extent, spacing, color):
    """Create a wireframe ground grid.

    Args:
        y: height of the grid plane
        extent: half-extent of the grid
        spacing: distance between grid lines
        color: (B, G, R) line color

    Returns:
        Mesh dict with 2-vertex faces (line segments)
    """
    vertices = []
    faces = []
    idx = 0
    vals = np.arange(-extent, extent + spacing * 0.5, spacing)

    # Lines parallel to Z
    for x in vals:
        vertices.append([x, y, -extent])
        vertices.append([x, y, extent])
        faces.append([idx, idx + 1])
        idx += 2

    # Lines parallel to X
    for z in vals:
        vertices.append([-extent, y, z])
        vertices.append([extent, y, z])
        faces.append([idx, idx + 1])
        idx += 2

    return {"vertices": np.array(vertices), "faces": faces, "color": color}


# =============================================================================
# Camera Math
# =============================================================================

def build_intrinsic(fx, fy, skew, cx, cy):
    """Build the 3x3 intrinsic matrix K.

    Args:
        fx, fy: focal lengths (pixels)
        skew: skew parameter
        cx, cy: principal point (pixels)

    Returns:
        K as (3, 3) ndarray
    """
    return np.array([
        [fx, skew, cx],
        [0,  fy,   cy],
        [0,  0,    1],
    ], dtype=np.float64)


def build_rotation(alpha, beta, gamma):
    """Build rotation matrix from Euler angles. Order: Rz @ Ry @ Rx.

    Args:
        alpha: rotation about X axis (radians)
        beta: rotation about Y axis (radians)
        gamma: rotation about Z axis (radians)

    Returns:
        R as (3, 3) ndarray
    """
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    cg, sg = np.cos(gamma), np.sin(gamma)

    Rx = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])
    Ry = np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])
    Rz = np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]])

    return Rz @ Ry @ Rx


def euler_from_rotation(R):
    """Extract Euler angles (alpha, beta, gamma) from a rotation matrix.
    Assumes Rz @ Ry @ Rx order. Handles gimbal lock approximately.

    Returns:
        (alpha, beta, gamma) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        alpha = np.arctan2(R[2, 1], R[2, 2])
        beta = np.arctan2(-R[2, 0], sy)
        gamma = np.arctan2(R[1, 0], R[0, 0])
    else:
        alpha = np.arctan2(-R[1, 2], R[1, 1])
        beta = np.arctan2(-R[2, 0], sy)
        gamma = 0.0
    return alpha, beta, gamma


def build_extrinsic(rx, ry, rz, tx, ty, tz, scale, camera_frame=False):
    """Build the 3x4 extrinsic matrix [R|t].

    Args:
        rx, ry, rz: Euler angles (radians)
        tx, ty, tz: translation
        scale: scale factor
        camera_frame: if True, parameters describe camera pose in world;
                      if False, parameters are the world-to-camera transform directly

    Returns:
        Tuple of (Rt (3,4), R (3,3))
    """
    R = build_rotation(rx, ry, rz)

    if camera_frame:
        # Sliders describe camera pose in world
        # Camera orientation in world = R
        # Camera position in world = scale * [tx, ty, tz]
        cam_pos = scale * np.array([tx, ty, tz])
        R_ext = R.T
        t_ext = -R.T @ cam_pos
        Rt = np.hstack([R_ext, t_ext.reshape(3, 1)])
        return Rt, R_ext
    else:
        # Sliders directly specify the extrinsic
        R_scaled = scale * R
        t = np.array([tx, ty, tz])
        Rt = np.hstack([R_scaled, t.reshape(3, 1)])
        return Rt, R_scaled


def fov_to_focal(fov_degrees, image_dim_px):
    """Convert field of view angle to focal length in pixels.

    Args:
        fov_degrees: full angle FOV in degrees (clamped to [1, 179])
        image_dim_px: image dimension (width or height) in pixels

    Returns:
        Focal length in pixels
    """
    fov_degrees = np.clip(fov_degrees, 1.0, 179.0)
    fov_rad = np.radians(fov_degrees)
    return (image_dim_px / 2.0) / np.tan(fov_rad / 2.0)


def make_lookat_Rt(eye, target, up=None):
    """Build extrinsic matrix from look-at parameters.

    Args:
        eye: (3,) camera position in world
        target: (3,) point the camera looks at
        up: (3,) world up vector (default [0, 1, 0])

    Returns:
        Rt as (3, 4) ndarray
    """
    if up is None:
        up = np.array([0.0, 1.0, 0.0])
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    forward = target - eye
    forward = forward / (np.linalg.norm(forward) + 1e-12)
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-12)
    new_up = np.cross(right, forward)

    # CV convention: camera looks along +Z, Y=up, X=right
    # (matches homework where identity R + positive tz = looking down +Z)
    R = np.stack([right, new_up, forward], axis=0)
    t = -R @ eye
    return np.hstack([R, t.reshape(3, 1)])


# =============================================================================
# Software Renderer
# =============================================================================

def render_scene(meshes, K, Rt, img_w, img_h, light_dir=None, bg_color=(40, 40, 40),
                 flip_y=True, return_zbuf=False):
    """Render meshes to a BGR uint8 image.

    Polygons use painter's algorithm (far-to-near) for high-quality
    anti-aliased fills and outlines via OpenCV.  A z-buffer is populated
    alongside so that wireframe lines (ground grid, axes, frustums) are
    correctly occluded by — or drawn in front of — solid geometry.

    Args:
        meshes: list of mesh dicts
        K: (3, 3) intrinsic matrix
        Rt: (3, 4) extrinsic matrix
        img_w, img_h: output image dimensions
        light_dir: (3,) light direction vector (points toward light). Default: (0.3, -0.8, 0.5)
        bg_color: (B, G, R) background color
        flip_y: if True, flip image vertically for upright display
                (compensates for pinhole inversion, matching standard camera display)
        return_zbuf: if True, also return the z-buffer (camera-space depth per pixel).
                     Background pixels have value np.inf.

    Returns:
        BGR uint8 image of shape (img_h, img_w, 3)
        — or, if return_zbuf is True, a tuple (image, z_buf) where z_buf
          is float64 shape (img_h, img_w) with np.inf for background.
    """
    if light_dir is None:
        light_dir = np.array([0.3, -0.8, 0.5])
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-12)

    M = K @ Rt
    R = Rt[:, :3]
    t_vec = Rt[:, 3]
    cam_pos = -R.T @ t_vec
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    img = np.full((img_h, img_w, 3), bg_color, dtype=np.uint8)
    z_buf = np.full((img_h, img_w), np.inf, dtype=np.float64)

    # Separate polygons and lines
    poly_list = []  # (mean_depth, pts_2d, shaded_color, normal_cam, d_plane)
    line_list = []  # (mean_depth, pts_2d, color, z1, z2)

    for mesh in meshes:
        verts = mesh["vertices"]
        base_color = np.array(mesh["color"], dtype=np.float64)
        face_colors = mesh.get("face_colors")  # optional per-face colors

        N = len(verts)
        verts_h = np.hstack([verts, np.ones((N, 1))])
        projected = M @ verts_h.T  # (3, N)
        depths = projected[2]

        valid_mask = depths > 0.01
        uv = np.zeros((2, N))
        uv[:, valid_mask] = projected[:2, valid_mask] / depths[valid_mask]

        for fi, face_idx in enumerate(mesh["faces"]):
            face_idx = list(face_idx)
            n_verts = len(face_idx)

            fc = np.array(face_colors[fi], dtype=np.float64) \
                if face_colors is not None else base_color

            face_depths = depths[face_idx]
            if np.any(face_depths <= 0.01):
                continue

            mean_depth = np.mean(face_depths)
            pts_2d = uv[:, face_idx].T.astype(np.int32)

            if n_verts == 2:
                line_list.append((mean_depth, pts_2d, fc,
                                  float(face_depths[0]), float(face_depths[1])))
            else:
                v0 = verts[face_idx[0]]
                v1 = verts[face_idx[1]]
                v2 = verts[face_idx[2]]
                normal_world = np.cross(v1 - v0, v2 - v0)
                norm_len = np.linalg.norm(normal_world)
                if norm_len < 1e-12:
                    continue
                normal_world = normal_world / norm_len

                face_center = np.mean(verts[face_idx], axis=0)
                if np.dot(normal_world, face_center - cam_pos) > 0:
                    continue

                intensity = np.clip(np.dot(normal_world, light_dir), 0.0, 1.0)
                intensity = 0.35 + 0.65 * intensity
                shaded = np.clip(fc * intensity, 0, 255).astype(np.uint8)

                n_cam = R @ normal_world
                v0_cam = R @ v0 + t_vec
                d_plane = float(np.dot(n_cam, v0_cam))

                poly_list.append((mean_depth, pts_2d, shaded, n_cam, d_plane))

    # --- Phase 1: draw polygons (painter's far-to-near) and populate z-buffer ---
    poly_list.sort(key=lambda f: -f[0])
    for _, pts, color, n_cam, d_plane in poly_list:
        fill_color = tuple(int(v) for v in color)
        outline_color = tuple(max(0, int(v * 0.5)) for v in color)
        cv2.fillPoly(img, [pts], fill_color)
        cv2.polylines(img, [pts], True, outline_color, 1, cv2.LINE_AA)
        _zbuf_update_poly(z_buf, pts, n_cam, d_plane, fx, fy, cx, cy,
                          img_w, img_h)

    # --- Phase 2: draw lines, z-tested against polygon depths ---
    line_list.sort(key=lambda f: -f[0])
    for _, pts, color, z1, z2 in line_list:
        _zbuf_draw_line(img, z_buf, pts, z1, z2, color, img_w, img_h)

    if flip_y:
        img = cv2.flip(img, 0)
        if return_zbuf:
            z_buf = cv2.flip(z_buf, 0)

    if return_zbuf:
        return img, z_buf
    return img


def _zbuf_update_poly(z_buf, pts, n_cam, d_plane, fx, fy, cx, cy,
                      img_w, img_h):
    """Populate z-buffer for a polygon using its plane equation (no drawing)."""
    x0 = max(0, int(pts[:, 0].min()))
    x1 = min(img_w - 1, int(pts[:, 0].max()))
    y0 = max(0, int(pts[:, 1].min()))
    y1 = min(img_h - 1, int(pts[:, 1].max()))
    if x0 > x1 or y0 > y1:
        return

    h, w = y1 - y0 + 1, x1 - x0 + 1
    pts_s = pts.copy()
    pts_s[:, 0] -= x0
    pts_s[:, 1] -= y0

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts_s], 255)

    ys_l, xs_l = np.where(mask > 0)
    if len(ys_l) == 0:
        return

    xs = (xs_l + x0).astype(np.float64)
    ys = (ys_l + y0).astype(np.float64)
    a, b, c = float(n_cam[0]), float(n_cam[1]), float(n_cam[2])
    denom = a * (xs - cx) / fx + b * (ys - cy) / fy + c
    ok = np.abs(denom) > 1e-12
    z = np.full(len(xs), np.inf)
    z[ok] = d_plane / denom[ok]
    z[z <= 0] = np.inf

    xi = xs_l + x0
    yi = ys_l + y0
    closer = z < z_buf[yi, xi]
    if np.any(closer):
        z_buf[yi[closer], xi[closer]] = z[closer]


def _zbuf_draw_line(img, z_buf, pts, z1, z2, color, img_w, img_h):
    """Rasterize a line segment through the z-buffer with AA blending."""
    p1 = pts[0].astype(np.float64)
    p2 = pts[1].astype(np.float64)
    color_tuple = tuple(int(v) for v in color)

    x0 = max(0, int(min(p1[0], p2[0])) - 1)
    x1 = min(img_w - 1, int(max(p1[0], p2[0])) + 1)
    y0 = max(0, int(min(p1[1], p2[1])) - 1)
    y1 = min(img_h - 1, int(max(p1[1], p2[1])) + 1)
    if x0 > x1 or y0 > y1:
        return

    h, w = y1 - y0 + 1, x1 - x0 + 1
    mask = np.zeros((h, w), dtype=np.uint8)
    p1_l = (int(p1[0]) - x0, int(p1[1]) - y0)
    p2_l = (int(p2[0]) - x0, int(p2[1]) - y0)
    cv2.line(mask, p1_l, p2_l, 255, 1, cv2.LINE_AA)

    ys_l, xs_l = np.where(mask > 0)
    if len(ys_l) == 0:
        return

    xs = xs_l + x0
    ys = ys_l + y0

    line_vec = p2 - p1
    line_len_sq = np.dot(line_vec, line_vec)
    if line_len_sq < 0.5:
        return
    pixel_xy = np.column_stack([xs, ys]).astype(np.float64)
    t = np.clip(np.dot(pixel_xy - p1, line_vec) / line_len_sq, 0.0, 1.0)

    # Perspective-correct depth: 1/z interpolates linearly in screen space
    inv_z1 = 1.0 / max(z1, 1e-10)
    inv_z2 = 1.0 / max(z2, 1e-10)
    z_at = 1.0 / np.maximum((1.0 - t) * inv_z1 + t * inv_z2, 1e-10)

    closer = z_at < z_buf[ys, xs]
    if not np.any(closer):
        return

    ys_c, xs_c = ys[closer], xs[closer]
    alpha = mask[ys_l[closer], xs_l[closer]].astype(np.float64) / 255.0
    new_c = np.array(color_tuple, dtype=np.float64)
    old_c = img[ys_c, xs_c].astype(np.float64)
    blended = old_c * (1.0 - alpha[:, None]) + new_c * alpha[:, None]
    img[ys_c, xs_c] = np.clip(blended, 0, 255).astype(np.uint8)


# =============================================================================
# Frustum Visualization
# =============================================================================

def compute_frustum_corners(K, Rt, img_w, img_h, near=0.3, far=5.0):
    """Compute the 8 world-space corners of the camera frustum.

    Args:
        K: (3, 3) intrinsic matrix
        Rt: (3, 4) extrinsic matrix
        img_w, img_h: image dimensions
        near, far: near and far plane distances

    Returns:
        (cam_pos (3,), near_corners (4,3), far_corners (4,3))
    """
    K_inv = np.linalg.inv(K)
    R = Rt[:, :3]
    t = Rt[:, 3]

    # Camera position in world
    cam_pos = -R.T @ t

    # Image corners in pixel coordinates (homogeneous)
    corners_px = np.array([
        [0,     0,     1],  # top-left
        [img_w, 0,     1],  # top-right
        [img_w, img_h, 1],  # bottom-right
        [0,     img_h, 1],  # bottom-left
    ], dtype=np.float64).T  # (3, 4)

    # Ray directions in camera space
    rays_cam = K_inv @ corners_px  # (3, 4)

    # Scale to near and far depths
    near_pts_cam = rays_cam * (near / rays_cam[2:3])
    far_pts_cam = rays_cam * (far / rays_cam[2:3])

    # Transform to world space: p_world = R^T @ (p_cam - t)
    near_pts_world = (R.T @ (near_pts_cam - t.reshape(3, 1))).T  # (4, 3)
    far_pts_world = (R.T @ (far_pts_cam - t.reshape(3, 1))).T    # (4, 3)

    return cam_pos, near_pts_world, far_pts_world


def make_frustum_mesh(K, Rt, img_w, img_h, near=0.3, far=5.0, color=(0, 200, 255)):
    """Create a wireframe mesh for the camera frustum.

    Returns a list of mesh dicts (frustum lines + camera origin marker).
    """
    cam_pos, near_corners, far_corners = compute_frustum_corners(K, Rt, img_w, img_h, near, far)

    # Combine all vertices
    # 0: cam_pos, 1-4: near corners, 5-8: far corners
    vertices = np.vstack([
        cam_pos.reshape(1, 3),
        near_corners,
        far_corners,
    ])

    faces = []

    # Edges from camera origin to near corners
    for i in range(4):
        faces.append([0, 1 + i])

    # Near plane edges
    for i in range(4):
        faces.append([1 + i, 1 + (i + 1) % 4])

    # Far plane edges
    for i in range(4):
        faces.append([5 + i, 5 + (i + 1) % 4])

    # Connecting edges (near to far)
    for i in range(4):
        faces.append([1 + i, 5 + i])

    frustum_mesh = {
        "vertices": vertices,
        "faces": faces,
        "color": color,
    }

    return [frustum_mesh]


def make_axis_mesh(origin, length=1.0):
    """Create axis indicator meshes (3 colored line segments).

    Args:
        origin: (3,) origin point
        length: axis length

    Returns:
        List of 3 mesh dicts (X=red, Y=green, Z=blue in BGR)
    """
    o = np.asarray(origin, dtype=np.float64)
    meshes = []
    # X axis (red in BGR = (0, 0, 255))
    meshes.append({
        "vertices": np.array([o, o + [length, 0, 0]]),
        "faces": [[0, 1]],
        "color": (0, 0, 255),
    })
    # Y axis (green = (0, 255, 0))
    meshes.append({
        "vertices": np.array([o, o + [0, length, 0]]),
        "faces": [[0, 1]],
        "color": (0, 255, 0),
    })
    # Z axis (blue = (255, 0, 0))
    meshes.append({
        "vertices": np.array([o, o + [0, 0, length]]),
        "faces": [[0, 1]],
        "color": (255, 0, 0),
    })
    return meshes


def make_camera_axes_mesh(Rt, length=0.8):
    """Create axis indicators at the camera position, aligned to camera axes.

    Args:
        Rt: (3, 4) extrinsic matrix
        length: axis length

    Returns:
        List of 3 mesh dicts for camera X/Y/Z axes in world space
    """
    R = Rt[:, :3]
    t = Rt[:, 3]
    cam_pos = -R.T @ t

    # Camera axes in world space
    cam_right = R.T @ np.array([1, 0, 0]) * length
    cam_up = R.T @ np.array([0, 1, 0]) * length
    cam_forward = R.T @ np.array([0, 0, 1]) * length

    meshes = []
    # X axis (red)
    meshes.append({
        "vertices": np.array([cam_pos, cam_pos + cam_right]),
        "faces": [[0, 1]],
        "color": (0, 0, 255),
    })
    # Y axis (green)
    meshes.append({
        "vertices": np.array([cam_pos, cam_pos + cam_up]),
        "faces": [[0, 1]],
        "color": (0, 255, 0),
    })
    # Z axis (blue) - note: camera looks down -Z, so this points backward
    meshes.append({
        "vertices": np.array([cam_pos, cam_pos + cam_forward]),
        "faces": [[0, 1]],
        "color": (255, 0, 0),
    })
    return meshes


# =============================================================================
# Default Scene
# =============================================================================

def create_default_scene():
    """Create the default 3D scene with basic primitives.

    Returns:
        List of mesh dicts
    """
    return [
        make_cube(center=(-1.5, 0.5, 0), size=1.0, color=(80, 80, 220)),
        make_cube(center=(1.5, 0.5, -1), size=0.7, color=(220, 80, 80)),
        make_sphere(center=(0, 0.8, -2), radius=0.6, color=(80, 220, 80)),
        make_cylinder(base_center=(2, 0, 1), radius=0.3, height=1.5, color=(80, 220, 220)),
        make_ground_grid(y=0.0, extent=5.0, spacing=1.0, color=(100, 100, 100)),
    ]


def make_checker_cube(center, size, color_a, color_b, n_div=4):
    """Cube with checkerboard pattern on each face (n_div x n_div grid).

    Uses per-face colors via the ``face_colors`` mesh field.
    """
    cx, cy, cz = center
    h = size / 2.0

    # 6 face definitions: (origin corner, u_axis, v_axis)
    # Each spans a unit square in [0,1]^2 mapped to the face.
    # Winding: cross(u, v) must point OUTWARD from the cube so that
    # the renderer's back-face cull (dot(normal, face→cam) < 0) keeps them.
    corners = [
        (np.array([cx-h, cy-h, cz-h]), np.array([0,size,0]), np.array([size,0,0])),  # back  -Z  cross→(0,0,-1)
        (np.array([cx-h, cy-h, cz+h]), np.array([size,0,0]), np.array([0,size,0])),  # front +Z  cross→(0,0,+1)
        (np.array([cx-h, cy-h, cz-h]), np.array([size,0,0]), np.array([0,0,size])),  # bottom -Y cross→(0,-1,0)
        (np.array([cx-h, cy+h, cz-h]), np.array([0,0,size]), np.array([size,0,0])),  # top +Y    cross→(0,+1,0)
        (np.array([cx-h, cy-h, cz-h]), np.array([0,0,size]), np.array([0,size,0])),  # left -X   cross→(-1,0,0)
        (np.array([cx+h, cy-h, cz-h]), np.array([0,size,0]), np.array([0,0,size])),  # right +X  cross→(+1,0,0)
    ]

    vertices = []
    faces = []
    face_colors = []
    idx = 0

    for origin, u_ax, v_ax in corners:
        for i in range(n_div):
            for j in range(n_div):
                s0, s1 = i / n_div, (i + 1) / n_div
                t0, t1 = j / n_div, (j + 1) / n_div
                v0 = origin + s0 * u_ax + t0 * v_ax
                v1 = origin + s1 * u_ax + t0 * v_ax
                v2 = origin + s1 * u_ax + t1 * v_ax
                v3 = origin + s0 * u_ax + t1 * v_ax
                vertices.extend([v0, v1, v2, v3])
                faces.append([idx, idx+1, idx+2, idx+3])
                face_colors.append(color_a if (i + j) % 2 == 0 else color_b)
                idx += 4

    return {"vertices": np.array(vertices), "faces": faces,
            "color": color_a, "face_colors": face_colors}


def make_checker_sphere(center, radius, color_a, color_b, n_lat=12, n_lon=20):
    """Sphere with checkerboard pattern (alternating lat/lon colors).

    Uses per-face colors via the ``face_colors`` mesh field.
    """
    cx, cy, cz = center
    vertices = []
    faces = []
    face_colors = []

    # Top pole
    vertices.append([cx, cy + radius, cz])

    # Latitude rings
    for i in range(1, n_lat):
        phi = np.pi * i / n_lat
        for j in range(n_lon):
            theta = 2.0 * np.pi * j / n_lon
            x = cx + radius * np.sin(phi) * np.cos(theta)
            y = cy + radius * np.cos(phi)
            z = cz + radius * np.sin(phi) * np.sin(theta)
            vertices.append([x, y, z])

    # Bottom pole
    vertices.append([cx, cy - radius, cz])
    vertices = np.array(vertices)

    # Top cap triangles
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([0, 1 + j_next, 1 + j])
        face_colors.append(color_a if j % 2 == 0 else color_b)

    # Quad strips
    for i in range(n_lat - 2):
        ring = 1 + i * n_lon
        next_ring = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            j_next = (j + 1) % n_lon
            faces.append([ring + j, ring + j_next,
                          next_ring + j_next, next_ring + j])
            face_colors.append(color_a if (i + j) % 2 == 0 else color_b)

    # Bottom cap
    bottom = len(vertices) - 1
    last_ring = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        j_next = (j + 1) % n_lon
        faces.append([bottom, last_ring + j, last_ring + j_next])
        face_colors.append(color_a if j % 2 == 0 else color_b)

    return {"vertices": vertices, "faces": faces,
            "color": color_a, "face_colors": face_colors}


def make_checker_cylinder(base_center, radius, height, color_a, color_b,
                          n_seg=16, n_rings=6):
    """Cylinder with checkerboard pattern on its side and caps.

    Uses per-face colors via the ``face_colors`` mesh field.
    """
    cx, cy, cz = base_center
    vertices = []
    faces = []
    face_colors = []

    # Bottom center (0), top center (1)
    vertices.append([cx, cy, cz])
    vertices.append([cx, cy + height, cz])

    # Rings along the height (n_rings + 1 rings of n_seg vertices each)
    ring_start = 2
    for r in range(n_rings + 1):
        frac = r / n_rings
        y_r = cy + frac * height
        for s in range(n_seg):
            theta = 2.0 * np.pi * s / n_seg
            vertices.append([cx + radius * np.cos(theta), y_r,
                             cz + radius * np.sin(theta)])

    vertices = np.array(vertices)

    # Bottom cap
    bot_ring = ring_start
    for s in range(n_seg):
        s_next = (s + 1) % n_seg
        faces.append([0, bot_ring + s, bot_ring + s_next])
        face_colors.append(color_a if s % 2 == 0 else color_b)

    # Top cap
    top_ring = ring_start + n_rings * n_seg
    for s in range(n_seg):
        s_next = (s + 1) % n_seg
        faces.append([1, top_ring + s_next, top_ring + s])
        face_colors.append(color_a if s % 2 == 0 else color_b)

    # Side quads
    for r in range(n_rings):
        r0 = ring_start + r * n_seg
        r1 = ring_start + (r + 1) * n_seg
        for s in range(n_seg):
            s_next = (s + 1) % n_seg
            faces.append([r0 + s, r1 + s, r1 + s_next, r0 + s_next])
            face_colors.append(color_a if (r + s) % 2 == 0 else color_b)

    return {"vertices": vertices, "faces": faces,
            "color": color_a, "face_colors": face_colors}


def make_checker_ground(y, extent, spacing, color_a, color_b):
    """Checkerboard ground plane (filled quads instead of wireframe grid).

    Winding gives upward (+Y) normals so the plane faces the camera.
    """
    vertices = []
    faces = []
    face_colors = []
    idx = 0
    vals = np.arange(-extent, extent, spacing)
    for i, x in enumerate(vals):
        for j, z in enumerate(vals):
            v0 = [x, y, z]
            v1 = [x, y, z + spacing]
            v2 = [x + spacing, y, z + spacing]
            v3 = [x + spacing, y, z]
            vertices.extend([v0, v1, v2, v3])
            faces.append([idx, idx+1, idx+2, idx+3])
            face_colors.append(color_a if (i + j) % 2 == 0 else color_b)
            idx += 4
    return {"vertices": np.array(vertices), "faces": faces,
            "color": color_a, "face_colors": face_colors}


def create_textured_scene():
    """Scene variant with checkerboard textures — same object layout as default.

    Returns:
        List of mesh dicts
    """
    return [
        make_checker_cube(center=(-1.5, 0.5, 0), size=1.0,
                          color_a=(60, 60, 200), color_b=(180, 180, 255), n_div=4),
        make_checker_cube(center=(1.5, 0.5, -1), size=0.7,
                          color_a=(200, 60, 60), color_b=(255, 180, 180), n_div=4),
        make_checker_sphere(center=(0, 0.8, -2), radius=0.6,
                            color_a=(60, 200, 60), color_b=(180, 255, 180)),
        make_checker_cylinder(base_center=(2, 0, 1), radius=0.3, height=1.5,
                              color_a=(60, 200, 200), color_b=(180, 255, 255)),
        make_checker_ground(y=0.0, extent=5.0, spacing=1.0,
                            color_a=(70, 70, 70), color_b=(130, 130, 130)),
    ]


# =============================================================================
# Ray Casting
# =============================================================================

def raycast_scene(scene, K, Rt, px, py, img_h, flip_y=True):
    """Ray-cast from a camera through pixel (px, py) into a mesh scene.

    Args:
        scene: list of mesh dicts with 'vertices' and 'faces'
        K: (3,3) intrinsic matrix
        Rt: (3,4) extrinsic matrix [R | t]
        px, py: pixel coordinates (image space, origin top-left)
        img_h: image height in pixels
        flip_y: if True, invert py before unprojecting (matches render_scene flip_y=True)

    Returns:
        z-depth (camera-space z) of nearest intersection, or None if no hit
    """
    py_cam = (img_h - 1 - py) if flip_y else py
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    ray_cam = np.array([(px - cx) / fx, (py_cam - cy) / fy, 1.0])

    R = Rt[:, :3]
    t = Rt[:, 3]
    cam_pos = -R.T @ t
    ray_world = R.T @ ray_cam
    ray_world /= np.linalg.norm(ray_world)

    best_t = np.inf
    for mesh in scene:
        verts = mesh["vertices"]
        for face in mesh["faces"]:
            for i in range(1, len(face) - 1):
                v0 = verts[face[0]]
                v1 = verts[face[i]]
                v2 = verts[face[i + 1]]
                e1 = v1 - v0
                e2 = v2 - v0
                h = np.cross(ray_world, e2)
                a = e1 @ h
                if abs(a) < 1e-8:
                    continue
                f = 1.0 / a
                s = cam_pos - v0
                u = f * (s @ h)
                if u < 0 or u > 1:
                    continue
                q = np.cross(s, e1)
                v = f * (ray_world @ q)
                if v < 0 or u + v > 1:
                    continue
                t_hit = f * (e2 @ q)
                if 1e-4 < t_hit < best_t:
                    best_t = t_hit

    if best_t == np.inf:
        return None
    hit_world = cam_pos + best_t * ray_world
    hit_cam = R @ hit_world + t
    return float(hit_cam[2])


# =============================================================================
# Matrix Formatting
# =============================================================================

def format_matrix(mat, label=""):
    """Format a matrix as a string for display.

    Args:
        mat: 2D ndarray
        label: optional label prefix

    Returns:
        Formatted string
    """
    rows, cols = mat.shape
    lines = []
    if label:
        lines.append(label)
    for r in range(rows):
        row_str = "  ".join(f"{mat[r, c]:7.2f}" for c in range(cols))
        bracket = "|" if 0 < r < rows - 1 else ("/" if r == 0 else "\\")
        lines.append(f"  {bracket} {row_str} {bracket}")
    return "\n".join(lines)
