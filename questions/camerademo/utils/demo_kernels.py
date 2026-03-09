"""
Shared kernel utilities for CSCI 1430 computer vision demos.
Contains kernel creation and manipulation functions.
"""

import cv2
import numpy as np


# Available kernel presets
KERNEL_PRESETS = [
    "Identity",
    "Box Blur",
    "Gaussian",
    "Sharpen",
    "Edge (Sobel X)",
    "Edge (Sobel Y)",
    "Laplacian",
    "Laplacian of Gaussian",
    "Emboss",
]

# Kernels that use the sigma parameter
SIGMA_KERNELS = ["Gaussian", "Laplacian of Gaussian"]

# Kernels with zero DC response that should not be normalized
ZERO_DC_KERNELS = ["Edge (Sobel X)", "Edge (Sobel Y)", "Laplacian", "Laplacian of Gaussian"]


def create_kernel(kernel_type, size, sigma=1.0):
    """Create a kernel of the specified type.

    Args:
        kernel_type: Name of kernel preset
        size: Kernel size (will be forced to odd)
        sigma: Sigma for Gaussian-based kernels

    Returns:
        Kernel as numpy float64 array
    """
    size = size if size % 2 == 1 else size + 1

    if kernel_type == "Identity":
        kernel = np.zeros((size, size), dtype=np.float64)
        kernel[size // 2, size // 2] = 1.0

    elif kernel_type == "Box Blur" or kernel_type == "Box":
        kernel = np.ones((size, size), dtype=np.float64) / (size * size)

    elif kernel_type == "Gaussian":
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
        kernel = np.outer(gauss, gauss)
        kernel = kernel / kernel.sum()

    elif kernel_type == "Sharpen":
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float64)
        if size != 3:
            kernel = resize_kernel(kernel, size)

    elif kernel_type == "Edge (Sobel X)" or kernel_type == "Edge Vertical":
        kernel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float64)
        if size != 3:
            kernel = resize_kernel(kernel, size)

    elif kernel_type == "Edge (Sobel Y)" or kernel_type == "Edge Horizontal":
        kernel = np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=np.float64)
        if size != 3:
            kernel = resize_kernel(kernel, size)

    elif kernel_type == "Laplacian":
        kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float64)
        if size != 3:
            kernel = resize_kernel(kernel, size)

    elif kernel_type == "Laplacian of Gaussian":
        ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
        xx, yy = np.meshgrid(ax, ax)
        r2 = xx**2 + yy**2
        sigma2 = sigma**2
        kernel = (r2 - 2 * sigma2) / (sigma2**2) * np.exp(-r2 / (2 * sigma2))
        kernel = kernel - kernel.mean()  # Zero DC response

    elif kernel_type == "Emboss":
        kernel = np.array([
            [-2, -1, 0],
            [-1, 1, 1],
            [0, 1, 2]
        ], dtype=np.float64)
        if size != 3:
            kernel = resize_kernel(kernel, size)

    elif kernel_type == "Random":
        kernel = np.random.randn(size, size).astype(np.float64)
        if kernel.sum() != 0:
            kernel = kernel / kernel.sum()
        else:
            kernel = kernel / (size * size)

    else:
        # Default to box blur
        kernel = np.ones((size, size), dtype=np.float64) / (size * size)

    return kernel


def resize_kernel(kernel, target_size):
    """Resize a kernel to target size by padding with zeros or cropping center.

    Args:
        kernel: Input kernel array
        target_size: Target size (will be square)

    Returns:
        Resized kernel
    """
    src_size = kernel.shape[0]
    if target_size == src_size:
        return kernel

    result = np.zeros((target_size, target_size), dtype=np.float64)
    offset = (target_size - src_size) // 2

    if target_size > src_size:
        result[offset:offset + src_size, offset:offset + src_size] = kernel
    else:
        src_offset = (src_size - target_size) // 2
        result = kernel[src_offset:src_offset + target_size,
                        src_offset:src_offset + target_size].copy()

    return result


def pad_kernel_to_image_size(kernel, image_shape):
    """Pad kernel to match image size for FFT convolution.

    Places kernel center at (0,0) for proper FFT convolution.

    Args:
        kernel: Input kernel
        image_shape: Target (height, width) tuple

    Returns:
        Padded kernel with center at (0,0)
    """
    kh, kw = kernel.shape
    ih, iw = image_shape

    padded = np.zeros((ih, iw), dtype=np.float64)
    padded[:kh, :kw] = kernel
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)

    return padded


def create_gaussian_kernel_fft(image_shape, sigma):
    """Create Gaussian kernel directly at image size, centered at (0,0) for FFT.

    This is more accurate than padding a small kernel for large sigma values.

    Args:
        image_shape: Target (height, width) tuple
        sigma: Gaussian sigma

    Returns:
        Gaussian kernel at image size, normalized
    """
    ih, iw = image_shape

    y = np.arange(ih, dtype=np.float64)
    x = np.arange(iw, dtype=np.float64)

    y = np.where(y > ih / 2, y - ih, y)
    x = np.where(x > iw / 2, x - iw, x)

    yy, xx = np.meshgrid(y, x, indexing='ij')

    kernel = np.exp(-0.5 * (xx**2 + yy**2) / (sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


def visualize_kernel(kernel, target_size):
    """Visualize a kernel as a normalized grayscale image.

    Args:
        kernel: Kernel array
        target_size: Size to resize visualization to

    Returns:
        Normalized [0,1] float image
    """
    k_min, k_max = kernel.min(), kernel.max()
    if k_max - k_min > 0:
        kernel_vis = (kernel - k_min) / (k_max - k_min)
    else:
        kernel_vis = np.ones_like(kernel) * 0.5

    kernel_vis = cv2.resize(kernel_vis, (target_size, target_size),
                            interpolation=cv2.INTER_NEAREST)
    return kernel_vis
