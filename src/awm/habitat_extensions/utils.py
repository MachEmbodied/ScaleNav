"""
Utility functions for AWM habitat evaluation.

Coordinate transforms and observation processing helpers.
"""

import numpy as np




def habitat_obs_to_rgb(observation: dict) -> np.ndarray:
    """
    Extract RGB image from habitat observation dict.

    Args:
        observation: Habitat observation with "rgb" key.
            Shape: (H, W, 4) RGBA uint8 or (H, W, 3) RGB uint8.

    Returns:
        (H, W, 3) uint8 RGB array.
    """
    rgb = observation["rgb"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]
    return rgb


def resize_rgb(rgb: np.ndarray, target_size: tuple) -> np.ndarray:
    """
    Resize RGB image to target_size (H, W).

    Uses PIL for quality interpolation.
    """
    from PIL import Image
    h, w = target_size
    img = Image.fromarray(rgb)
    img = img.resize((w, h), Image.BILINEAR)
    return np.array(img)
