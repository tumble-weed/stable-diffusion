import math
from typing import List, Tuple

import torch
import torch.nn.functional as F

from kornia.core import Module, Tensor, pad, stack, tensor
from kornia.filters import filter2d, gaussian_blur2d
from kornia.testing import KORNIA_CHECK, KORNIA_CHECK_IS_TENSOR, KORNIA_CHECK_SHAPE


from kornia.filters.kernels import normalize_kernel2d
import kornia.filters.filter
from termcolor import colored
def masked_filter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    mask:torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Convolve a tensor with a 2d kernel.

    The function applies a given kernel to a tensor. The kernel is applied
    independently at each depth channel of the tensor. Before applying the
    kernel, the function applies padding according to the specified mode so
    that the output remains in the same shape.

    Args:
        input: the input tensor with shape of
          :math:`(B, C, H, W)`.
        kernel: the kernel to be convolved with the input
          tensor. The kernel shape must be :math:`(1, kH, kW)` or :math:`(B, kH, kW)`.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        normalized: If True, kernel will be L1 normalized.
        padding: This defines the type of padding.
          2 modes available ``'same'`` or ``'valid'``.

    Return:
        torch.Tensor: the convolved tensor of same size and numbers of channels
        as the input with shape :math:`(B, C, H, W)`.

    Example:
        >>> input = torch.tensor([[[
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 5., 0., 0.],
        ...    [0., 0., 0., 0., 0.],
        ...    [0., 0., 0., 0., 0.],]]])
        >>> kernel = torch.ones(1, 3, 3)
        >>> filter2d(input, kernel, padding='same')
        tensor([[[[0., 0., 0., 0., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 5., 5., 5., 0.],
                  [0., 0., 0., 0., 0.]]]])
    """
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)
    


    if normalized:
        tmp_kernel = normalize_kernel2d(tmp_kernel)

    tmp_kernel = tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = kornia.filters.filter._compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)
        mask = F.pad(mask, padding_shape, mode=border_type)
    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.

    def masked_conv2d(input, tmp_kernel,mask, padding=0, stride=1):
         if False:
            mask = torch.ones_like(mask)
            print(colored('setting mask to ones in masked_conv2d','red'))        
        #  inputu = torch.nn.functional.unfold(input, kernel_size=tmp_kernel.shape[-1], dilation=(1, 1), stride=stride, padding=padding)
         inputu = input.unfold(2,tmp_kernel.shape[-2],1).unfold(3,tmp_kernel.shape[-1],1)
        #  masku = torch.nn.functional.unfold(mask, kernel_size=tmp_kernel.shape[-1], dilation=(1, 1), stride=stride, padding=padding)
         masku = mask.unfold(2,tmp_kernel.shape[-2],1).unfold(3,tmp_kernel.shape[-1],1)
         tmp_kernel = tmp_kernel.permute(1,0,2,3)
        #  import ipdb;ipdb.set_trace()
         input_dot_kernel = inputu * tmp_kernel[:,:,None,None,:,:]
         mask_dot_kernel =  masku * tmp_kernel[:,:,None,None,:,:]
         masked_input_dot_kernel = (input_dot_kernel * masku)
         denom = mask_dot_kernel.sum(dim=(-1,-2))
         denom = denom + (denom == 0).float()
         out = masked_input_dot_kernel.sum(dim=(-1,-2))/denom
         assert out.shape == input.shape[:2] + ( input.shape[-2] - 2*(tmp_kernel.shape[2]//2),input.shape[-1] - 2*(tmp_kernel.shape[2]//2))
         return out
    '''
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)
    '''
    output = masked_conv2d(input, tmp_kernel, mask, padding=0, stride=1)

    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out



__all__ = [
    "PyrDown",
    "PyrUp",
    "ScalePyramid",
    "pyrdown",
    "pyrup",
    "build_pyramid",
    "build_laplacian_pyramid",
    "upscale_double",
]

import torch
import math

def gaussian_kernel(size, sigma):
    """
    Function to generate a 2D Gaussian kernel
    
    Args:
    - size (int): Kernel size
    - sigma (float): Standard deviation of Gaussian distribution
    
    Returns:
    - kernel (Tensor): 2D Gaussian kernel of size (size x size)
    """
    # Create a one-dimensional Gaussian distribution
    gaussian = torch.Tensor([math.exp(-(x - size//2)**2/float(2*sigma**2)) for x in range(size)])
    
    # Outer product of the Gaussian distribution with itself to get a 2D Gaussian kernel
    kernel = torch.outer(gaussian, gaussian)
    
    # Normalize the kernel so that the sum of all elements equals 1
    kernel = kernel / torch.sum(kernel)
    kernel = kernel.unsqueeze(0)
    return kernel


def _get_pyramid_gaussian_kernel() -> Tensor:
    """Utility function that return a pre-computed gaussian kernel."""
    return (
        tensor(
            [
                [
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [6.0, 24.0, 36.0, 24.0, 6.0],
                    [4.0, 16.0, 24.0, 16.0, 4.0],
                    [1.0, 4.0, 6.0, 4.0, 1.0],
                ]
            ]
        )
        / 256.0
    )


class PyrDown(Module):
    r"""Blur a tensor and downsamples it.

    Args:
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H / 2, W / 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrDown()(input)  # 1x2x2x2
    """

    def __init__(self, border_type: str = 'reflect', align_corners: bool = False, factor: float = 2.0) -> None:
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners
        self.factor: float = factor

    def forward(self, input: Tensor) -> Tensor:
        return pyrdown(input, self.border_type, self.align_corners, self.factor)


class PyrUp(Module):
    r"""Upsample a tensor and then blurs it.

    Args:
        borde_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the upsampled tensor.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H * 2, W * 2)`

    Examples:
        >>> input = torch.rand(1, 2, 4, 4)
        >>> output = PyrUp()(input)  # 1x2x8x8
    """

    def __init__(self, border_type: str = 'reflect', align_corners: bool = False):
        super().__init__()
        self.border_type: str = border_type
        self.align_corners: bool = align_corners

    def forward(self, input: Tensor) -> Tensor:
        return pyrup(input, self.border_type, self.align_corners)


class ScalePyramid(Module):
    r"""Create an scale pyramid of image, usually used for local feature detection.

    Images are consequently smoothed with Gaussian blur and downscaled.

    Args:
        n_levels: number of the levels in octave.
        init_sigma: initial blur level.
        min_size: the minimum size of the octave in pixels.
        double_image: add 2x upscaled image as 1st level of pyramid. OpenCV SIFT does this.

    Returns:
        1st output: images
        2nd output: sigmas (coefficients for scale conversion)
        3rd output: pixelDists (coefficients for coordinate conversion)

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output 1st: :math:`[(B, C, NL, H, W), (B, C, NL, H/2, W/2), ...]`
        - Output 2nd: :math:`[(B, NL), (B, NL), (B, NL), ...]`
        - Output 3rd: :math:`[(B, NL), (B, NL), (B, NL), ...]`

    Examples:
        >>> input = torch.rand(2, 4, 100, 100)
        >>> sp, sigmas, pds = ScalePyramid(3, 15)(input)
    """

    def __init__(self, n_levels: int = 3, init_sigma: float = 1.6, min_size: int = 15, double_image: bool = False):
        super().__init__()
        # 3 extra levels are needed for DoG nms.
        self.n_levels = n_levels
        self.extra_levels: int = 3
        self.init_sigma = init_sigma
        self.min_size = min_size
        self.border = min_size // 2 - 1
        self.sigma_step = 2 ** (1.0 / float(self.n_levels))
        self.double_image = double_image

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + '(n_levels='
            + str(self.n_levels)
            + ', '
            + 'init_sigma='
            + str(self.init_sigma)
            + ', '
            + 'min_size='
            + str(self.min_size)
            + ', '
            + 'extra_levels='
            + str(self.extra_levels)
            + ', '
            + 'border='
            + str(self.border)
            + ', '
            + 'sigma_step='
            + str(self.sigma_step)
            + ', '
            + 'double_image='
            + str(self.double_image)
            + ')'
        )

    def get_kernel_size(self, sigma: float):
        ksize = int(2.0 * 4.0 * sigma + 1.0)

        #  matches OpenCV, but may cause padding problem for small images
        #  PyTorch does not allow to pad more than original size.
        #  Therefore there is a hack in forward function

        if ksize % 2 == 0:
            ksize += 1
        return ksize

    def get_first_level(self, input: Tensor) -> Tuple[Tensor, float, float]:
        pixel_distance = 1.0
        cur_sigma = 0.5
        # Same as in OpenCV up to interpolation difference
        if self.double_image:
            x = upscale_double(input)
            pixel_distance = 0.5
            cur_sigma *= 2.0
        else:
            x = input

        if self.init_sigma > cur_sigma:
            sigma = max(math.sqrt(self.init_sigma**2 - cur_sigma**2), 0.01)
            ksize = self.get_kernel_size(sigma)
            cur_level = gaussian_blur2d(x, (ksize, ksize), (sigma, sigma))
            cur_sigma = self.init_sigma
        else:
            cur_level = x
        return cur_level, cur_sigma, pixel_distance

    def forward(self, x: Tensor) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        bs, _, _, _ = x.size()
        cur_level, cur_sigma, pixel_distance = self.get_first_level(x)

        sigmas = [cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pixel_dists = [pixel_distance * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device).to(x.dtype)]
        pyr = [[cur_level]]
        oct_idx = 0
        while True:
            cur_level = pyr[-1][0]
            for level_idx in range(1, self.n_levels + self.extra_levels):
                sigma = cur_sigma * math.sqrt(self.sigma_step**2 - 1.0)
                ksize = self.get_kernel_size(sigma)

                # Hack, because PyTorch does not allow to pad more than original size.
                # But for the huge sigmas, one needs huge kernel and padding...

                ksize = min(ksize, min(cur_level.size(2), cur_level.size(3)))
                if ksize % 2 == 0:
                    ksize += 1

                cur_level = gaussian_blur2d(cur_level, (ksize, ksize), (sigma, sigma))
                cur_sigma *= self.sigma_step
                pyr[-1].append(cur_level)
                sigmas[-1][:, level_idx] = cur_sigma
                pixel_dists[-1][:, level_idx] = pixel_distance
            _pyr = pyr[-1][-self.extra_levels]
            nextOctaveFirstLevel = _pyr[:, :, ::2, ::2]

            pixel_distance *= 2.0
            cur_sigma = self.init_sigma
            if min(nextOctaveFirstLevel.size(2), nextOctaveFirstLevel.size(3)) <= self.min_size:
                break
            pyr.append([nextOctaveFirstLevel])
            sigmas.append(cur_sigma * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device))
            pixel_dists.append(pixel_distance * torch.ones(bs, self.n_levels + self.extra_levels).to(x.device))
            oct_idx += 1

        output_pyr = [stack(i, 2) for i in pyr]

        return output_pyr, sigmas, pixel_dists


def pyrdown(input: Tensor, mask: Tensor, border_type: str = 'reflect', align_corners: bool = False, factor: float = 2.0) -> Tensor:
    r"""Blur a tensor and downsamples it.

    .. image:: _static/img/pyrdown.png

    Args:
        input: the tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.
        factor: the downsampling factor

    Return:
        the downsampled tensor.

    Examples:
        >>> input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        >>> pyrdown(input, align_corners=True)
        tensor([[[[ 3.7500,  5.2500],
                  [ 9.7500, 11.2500]]]])
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: Tensor = _get_pyramid_gaussian_kernel()
    _, _, height, width = input.shape
    # blur image
    x_blur: Tensor = masked_filter2d(input, kernel,mask, border_type)

    # TODO: use kornia.geometry.resize/rescale
    # downsample.
    out: Tensor = F.interpolate(
        x_blur,
        size=(int(float(height) / factor), int(float(width) // factor)),
        mode='bilinear',
        align_corners=align_corners,
    )
    return out


def pyrup(input: Tensor, border_type: str = 'reflect', align_corners: bool = False) -> Tensor:
    assert False, 'not implemented for masked blur'
    r"""Upsample a tensor and then blurs it.

    .. image:: _static/img/pyrup.png

    Args:
        input: the tensor to be downsampled.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``, ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        the downsampled tensor.

    Examples:
        >>> input = torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2)
        >>> pyrup(input, align_corners=True)
        tensor([[[[0.7500, 0.8750, 1.1250, 1.2500],
                  [1.0000, 1.1250, 1.3750, 1.5000],
                  [1.5000, 1.6250, 1.8750, 2.0000],
                  [1.7500, 1.8750, 2.1250, 2.2500]]]])
    """
    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])

    kernel: Tensor = _get_pyramid_gaussian_kernel()
    # upsample tensor
    _, _, height, width = input.shape
    # TODO: use kornia.geometry.resize/rescale
    x_up: Tensor = F.interpolate(input, size=(height * 2, width * 2), mode='bilinear', align_corners=align_corners)

    # blurs upsampled tensor
    x_blur: Tensor = filter2d(x_up, kernel, border_type)
    return x_blur


def build_pyramid(
    input: Tensor, max_level: int, border_type: str = 'reflect', align_corners: bool = False
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
        img_down: Tensor = pyrdown(img_curr, border_type, align_corners)
        pyramid.append(img_down)

    return pyramid


def is_powerof_two(x):
    # check if number x is a power of two
    return x and (not (x & (x - 1)))


def find_next_powerof_two(x):
    # return the nearest power of 2
    n = math.ceil(math.log(x) / math.log(2))
    return 2**n


def build_laplacian_pyramid(
    input: Tensor, max_level: int, border_type: str = 'reflect', align_corners: bool = False
) -> List[Tensor]:
    r"""Construct the Laplacian pyramid for a tensor image.

    The function constructs a vector of images and builds the Laplacian pyramid
    by recursively computing the difference after applying
    pyrUp to the adjacent layer in it's Gaussian pyramid.

    See :cite:`burt1987laplacian` for more details.

    Args:
        input : the tensor to be used to construct the pyramid with shape :math:`(B, C, H, W)`.
        max_level: 0-based index of the last (the smallest) pyramid layer.
          It must be non-negative.
        border_type: the padding mode to be applied before convolving.
          The expected modes are: ``'constant'``, ``'reflect'``,
          ``'replicate'`` or ``'circular'``.
        align_corners: interpolation flag.

    Return:
        Output: :math:`[(B, C, H, W), (B, C, H/2, W/2), ...]`
    """

    KORNIA_CHECK_SHAPE(input, ["B", "C", "H", "W"])
    KORNIA_CHECK(
        isinstance(max_level, int) or max_level < 0,
        f"Invalid max_level, it must be a positive integer. Got: {max_level}",
    )

    h = input.size()[2]
    w = input.size()[3]
    require_padding = not (is_powerof_two(w) or is_powerof_two(h))

    if require_padding:
        # in case of arbitrary shape tensor image need to be padded.
        # Reference: https://stackoverflow.com/a/29967555
        padding = (0, find_next_powerof_two(w) - w, 0, find_next_powerof_two(h) - h)
        input = pad(input, padding, "reflect")

    # create gaussian pyramid
    gaussian_pyramid: List[Tensor] = build_pyramid(input, max_level, border_type, align_corners)
    # create empty list
    laplacian_pyramid: List[Tensor] = []

    # iterate and compute difference of adjacent layers in a gaussian pyramid
    for i in range(max_level - 1):
        img_expand: Tensor = pyrup(gaussian_pyramid[i + 1], border_type, align_corners)
        laplacian: Tensor = gaussian_pyramid[i] - img_expand
        laplacian_pyramid.append(laplacian)
    laplacian_pyramid.append(gaussian_pyramid[-1])
    return laplacian_pyramid


def upscale_double(x: Tensor) -> Tensor:
    r"""Upscale image by the factor of 2, even indices maps to original indices.

    Odd indices are linearly interpolated from the even ones.

    Args:
        x: input image.

    Shape:
        - Input: :math:`(*, H, W)`
        - Output :math:`(*, H, W)`
    """
    KORNIA_CHECK_IS_TENSOR(x)
    KORNIA_CHECK_SHAPE(x, ["*", "H", "W"])
    double_shape = x.shape[:-2] + (x.shape[-2] * 2, x.shape[-1] * 2)
    upscaled = torch.zeros(double_shape, device=x.device, dtype=x.dtype)
    upscaled[..., ::2, ::2] = x
    upscaled[..., ::2, 1::2][..., :-1] = (upscaled[..., ::2, ::2][..., :-1] + upscaled[..., ::2, 2::2]) / 2
    upscaled[..., ::2, -1] = upscaled[..., ::2, -2]
    upscaled[..., 1::2, :][..., :-1, :] = (upscaled[..., ::2, :][..., :-1, :] + upscaled[..., 2::2, :]) / 2
    upscaled[..., -1, :] = upscaled[..., -2, :]
    return upscaled