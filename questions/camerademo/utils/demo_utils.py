"""
Shared utilities for CSCI 1430 computer vision demos.
Contains image conversion, camera initialization, and frame acquisition functions.
"""

import os
import sys
import cv2
import numpy as np

# Data directory (sibling to utils/)
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(_UTILS_DIR), "data")


def convert_cv_to_dpg(image, clip=False):
    """Convert OpenCV image to Dear PyGui texture format.

    Args:
        image: OpenCV image (BGR or grayscale), uint8 or float
        clip: If True, clip float values to [0, 1] range before conversion

    Returns:
        Flattened float32 RGBA array suitable for dpg.set_value()
    """
    if clip:
        image = np.clip(image, 0, 1)

    if len(image.shape) == 2:  # Grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)

    image = image.astype(np.float32) / 255.0
    return image.flatten()


def resize_with_letterbox(img, target_width, target_height):
    """Resize image to fit within target dimensions, maintaining aspect ratio with black bars.

    Args:
        img: OpenCV image (BGR)
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Resized image with letterboxing/pillarboxing as needed
    """
    h, w = img.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))

    # Create black canvas and center the image
    canvas = np.zeros((target_height, target_width, 3), dtype=img.dtype)
    x_offset = (target_width - new_w) // 2
    y_offset = (target_height - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas


def init_camera(camera_id=0, width=None, height=None):
    """Initialize camera with platform-specific backend.

    Args:
        camera_id: Camera device ID (default 0)
        width: Requested frame width (None for default)
        height: Requested frame height (None for default)

    Returns:
        Tuple of (cap, frame_width, frame_height, use_camera)
        - cap: cv2.VideoCapture object or None
        - frame_width: Width of camera frames (0 if no camera)
        - frame_height: Height of camera frames (0 if no camera)
        - use_camera: True if camera is available and working
    """
    if os.name == 'nt':
        cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(camera_id)

    if cap.isOpened():
        # Set resolution if specified
        if width is not None:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height is not None:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        ret, frame = cap.read()
        if ret:
            # Return actual resolution obtained (may differ from requested)
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Log resolution info
            if width is not None and height is not None:
                if actual_width != width or actual_height != height:
                    print(f"Camera: Requested {width}x{height}, got {actual_width}x{actual_height}")
                else:
                    print(f"Camera: Using {actual_width}x{actual_height}")
            else:
                print(f"Camera: Default resolution {actual_width}x{actual_height}")

            return cap, actual_width, actual_height, True
        cap.release()

    return None, 0, 0, False


def load_fallback_image(data_dir=None, filename="cat.jpg"):
    """Load fallback image for cat mode.

    Args:
        data_dir: Directory containing the fallback image (defaults to DATA_DIR)
        filename: Name of the fallback image file

    Returns:
        Loaded image as numpy array

    Exits:
        If image cannot be loaded
    """
    if data_dir is None:
        data_dir = DATA_DIR
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        print(f"Error: Fallback image not found at {path}", file=sys.stderr)
        sys.exit(1)

    img = cv2.imread(path)
    if img is None:
        print("Error: Could not load fallback image", file=sys.stderr)
        sys.exit(1)

    return img


def get_frame(cap, fallback_image, use_camera, cat_mode, target_size=None, letterbox=True):
    """Get frame from camera or fallback image.

    Args:
        cap: cv2.VideoCapture object
        fallback_image: Fallback image to use when camera unavailable
        use_camera: Whether camera is available
        cat_mode: Whether to use fallback image instead of camera
        target_size: Optional (width, height) tuple to resize frame
        letterbox: If True, use letterboxing for fallback; if False, stretch

    Returns:
        Frame as numpy array, or None if camera read failed
    """
    if use_camera and not cat_mode:
        ret, frame = cap.read()
        if not ret:
            return None
        if target_size:
            frame = cv2.resize(frame, target_size)
    else:
        frame = fallback_image.copy()
        if target_size:
            if letterbox:
                frame = resize_with_letterbox(frame, target_size[0], target_size[1])
            else:
                frame = cv2.resize(frame, target_size)
    return frame


def convert_cv_to_dpg_float(image):
    """Convert float [0,1] image to Dear PyGui texture format.

    Unlike convert_cv_to_dpg which expects uint8, this handles float images
    directly without the /255 normalization.

    Args:
        image: Float image with values in [0, 1] range (grayscale or BGR)

    Returns:
        Flattened float32 RGBA array suitable for dpg.set_value()
    """
    image = np.clip(image, 0, 1)

    if len(image.shape) == 2:  # Grayscale
        rgba = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.float32)
        rgba[:, :, 0] = image
        rgba[:, :, 1] = image
        rgba[:, :, 2] = image
        rgba[:, :, 3] = 1.0
        return rgba.flatten()
    else:
        image = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_BGR2RGBA)
        return (image.astype(np.float32) / 255.0).flatten()


def crop_to_square(image, target_size=None):
    """Crop image to square from center, optionally resize.

    Args:
        image: Input image (any shape)
        target_size: Optional int to resize square to

    Returns:
        Square image
    """
    h, w = image.shape[:2]

    if w > h:
        crop = (w - h) // 2
        image = image[:, crop:crop + h]
    elif h > w:
        crop = (h - w) // 2
        image = image[crop:crop + w, :]

    if target_size is not None:
        image = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_AREA)

    return image


def apply_affine_transform(image, rotation=0, scale=1.0, translate_x=0, translate_y=0,
                           border_mode=cv2.BORDER_REFLECT):
    """Apply rotation, scale, and translation to image.

    Args:
        image: Input image
        rotation: Rotation angle in degrees
        scale: Scale factor
        translate_x: X translation as percentage of width (-100 to 100)
        translate_y: Y translation as percentage of height (-100 to 100)
        border_mode: OpenCV border mode for out-of-bounds pixels

    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, rotation, scale)
    M[0, 2] += translate_x * w / 100.0
    M[1, 2] += translate_y * h / 100.0
    return cv2.warpAffine(image, M, (w, h), borderMode=border_mode)


def apply_brightness(image, scale=1.0, shift=0.0):
    """Apply brightness adjustment to image.

    Args:
        image: Input image (uint8)
        scale: Multiplicative factor
        shift: Additive offset

    Returns:
        Adjusted image (uint8)
    """
    result = image.astype(np.float32) * scale + shift
    return np.clip(result, 0, 255).astype(np.uint8)
