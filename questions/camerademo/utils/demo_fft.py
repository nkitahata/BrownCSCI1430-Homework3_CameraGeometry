"""
Shared FFT utilities for CSCI 1430 computer vision demos.
Contains FFT processing, visualization, and convolution functions.
"""

import numpy as np


def visualize_fft_amplitude(fft_data, image_shape, use_log=True):
    """Convert FFT to amplitude visualization.

    Args:
        fft_data: Complex FFT array
        image_shape: Original image shape (height, width) for normalization
        use_log: If True, use log scaling for better visualization

    Returns:
        Normalized [0,1] amplitude image (shifted so DC is at center)
    """
    amplitude = np.abs(fft_data)
    amplitude_shifted = np.fft.fftshift(amplitude)

    if use_log:
        # Avoid log(0) by replacing zeros with epsilon
        amplitude_shifted[amplitude_shifted == 0] = np.finfo(float).eps
        amplitude_vis = np.log(amplitude_shifted)

        # Normalize based on expected range
        height, width = image_shape
        min_log_amplitude = 0.0
        max_log_amplitude = 0.9 * np.log(height * width)
        amplitude_vis = (amplitude_vis - min_log_amplitude) / (max_log_amplitude - min_log_amplitude)
    else:
        # Linear normalization
        amp_max = amplitude_shifted.max()
        if amp_max > 0:
            amplitude_vis = amplitude_shifted / amp_max
        else:
            amplitude_vis = np.zeros_like(amplitude_shifted)

    return np.clip(amplitude_vis, 0, 1)


def process_convolution(image, kernel_padded):
    """Perform convolution using FFT multiplication.

    Args:
        image: Input image (2D array)
        kernel_padded: Kernel padded to image size (from pad_kernel_to_image_size)

    Returns:
        Tuple of (result, image_fft, kernel_fft, product_fft)
    """
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel_padded)
    product_fft = image_fft * kernel_fft
    result = np.fft.ifft2(product_fft).real

    return result, image_fft, kernel_fft, product_fft


def process_deconvolution(blurred_image, kernel_padded, regularization=0.01):
    """Perform deconvolution using FFT division with Wiener-like regularization.

    Uses the formula: F(result) = F(blurred) * conj(F(kernel)) / (|F(kernel)|^2 + reg)

    Args:
        blurred_image: Blurred input image (2D array)
        kernel_padded: Kernel padded to image size
        regularization: Regularization parameter (higher = more stable, less sharp)

    Returns:
        Tuple of (result, blurred_fft, kernel_fft, result_fft)
    """
    blurred_fft = np.fft.fft2(blurred_image)
    kernel_fft = np.fft.fft2(kernel_padded)

    kernel_conj = np.conj(kernel_fft)
    kernel_power = np.abs(kernel_fft) ** 2

    result_fft = blurred_fft * kernel_conj / (kernel_power + regularization)
    result = np.fft.ifft2(result_fft).real

    return result, blurred_fft, kernel_fft, result_fft
