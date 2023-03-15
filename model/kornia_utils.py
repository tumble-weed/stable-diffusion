import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.core import Tensor
from kornia.filters import filter2d, gaussian_blur2d
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_SHAPE
import kornia.geometry.transform.pyramid as K_pyramid


def build_pyramid(
    input: Tensor, max_level: int, border_type: str = 'reflect', downscale = 2.,align_corners: bool = False
) -> List[Tensor]:
    r"""Construct the Gaussian pyramid for a tensor image.

    .. image:: _static/img/build_pyramid.png

    The function constructs a vector of images and builds the Gaussian pyramid
    by recursively applying pyrDown to the previously built pyramid layers.

    Args:
        input : the tensor to be used to construct the pyramid.
        max_level: 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) or max_level < 0,
        f"Invalid max_level, it must be a positive integer. Got: {max_level}",
    )

    # create empty list and append the original image
    pyramid: List[Tensor] = []
    pyramid.append(input)

    # iterate and downsample
    for _ in range(max_level - 1):
        img_curr: Tensor = pyramid[-1]
        img_down: Tensor = K_pyramid.pyrdown(img_curr, border_type, align_corners,factor=downscale)
        pyramid.append(img_down)

    return pyramid


