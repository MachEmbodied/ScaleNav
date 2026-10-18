"""Symlog normalization for body-frame action displacements."""
from typing import Union
import numpy as np
import torch


class SymlogNormalizer:
    """
    Symlog normalizer for handling long-tailed trajectory distributions.

    Uses symmetric logarithmic transformation to compress large values
    while maintaining sensitivity around zero.

    Formula:
        normalize: y = sign(x) * ln(1 + |x| / c)
        denormalize: x = sign(y) * c * (exp(|y|) - 1)

    Args:
        c: Scale factor. Should be set to the core distribution size of data.
            For trajectory data with values around 0.01, use c=0.01.
    """

    def __init__(self, c: float = 0.01):
        self.c = c

    def normalize(self, x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Forward transformation: compress long-tailed data.

        Args:
            x: Input tensor or array with trajectory values.

        Returns:
            Normalized values with compressed dynamic range.
        """
        if isinstance(x, np.ndarray):
            return np.sign(x) * np.log(1 + np.abs(x) / self.c)
        return torch.sign(x) * torch.log(1 + torch.abs(x) / self.c)

    def denormalize(self, y: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Inverse transformation: recover original scale.

        Args:
            y: Normalized tensor or array.

        Returns:
            Denormalized values in original physical scale.
        """
        if isinstance(y, np.ndarray):
            return np.sign(y) * self.c * (np.exp(np.abs(y)) - 1)
        return torch.sign(y) * self.c * (torch.exp(torch.abs(y)) - 1)



def normalize_trajectory_data(traj, mode="symlog", symlog_c=0.01):
    if mode != "symlog":
        raise ValueError("ScaleNav uses symlog action normalization")
    return SymlogNormalizer(c=symlog_c).normalize(traj)


def denormalize_trajectory_data(traj, mode="symlog", symlog_c=0.01):
    if mode != "symlog":
        raise ValueError("ScaleNav uses symlog action normalization")
    return SymlogNormalizer(c=symlog_c).denormalize(traj)
