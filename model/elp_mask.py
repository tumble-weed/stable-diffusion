#========================================================
import torch
import math
import torch.nn.functional as F
#========================================================
class MaskGenerator:
    r"""Mask generator.
    The class takes as input the mask parameters and returns
    as output a mask.
    Args:
        shape (tuple of int): output shape.
        step (int): parameterization step in pixels.
        sigma (float): kernel size.
        clamp (bool, optional): whether to clamp the mask to [0,1]. Defaults to True.
        pooling_mehtod (str, optional): `'softmax'` (default),  `'sum'`, '`sigmoid`'.
    Attributes:
        shape (tuple): the same as the specified :attr:`shape` parameter.
        shape_in (tuple): spatial size of the parameter tensor.
        shape_out (tuple): spatial size of the output mask including margin.
    """

    def __init__(self, shape, step, sigma, clamp=True, pooling_method='softmax'):
        self.shape = shape
        self.step = step
        self.sigma = sigma
        self.coldness = 20
        self.clamp = clamp
        self.pooling_method = pooling_method

        assert int(step) == step

        # self.kernel = lambda z: (z < 1).float()
        self.kernel = lambda z: torch.exp(-2 * ((z - .5).clamp(min=0)**2))

        self.margin = self.sigma
        # self.margin = 0
        self.padding = 1 + math.ceil((self.margin + sigma) / step)
        self.radius = 1 + math.ceil(sigma / step)
        self.shape_in = [math.ceil(z / step) for z in self.shape]
        self.shape_mid = [
            z + 2 * self.padding - (2 * self.radius + 1) + 1
            for z in self.shape_in
        ]
        self.shape_up = [self.step * z for z in self.shape_mid]
        self.shape_out = [z - step + 1 for z in self.shape_up]

        self.weight = torch.zeros((
            1,
            (2 * self.radius + 1)**2,
            self.shape_out[0],
            self.shape_out[1]
        ))

        step_inv = [
            torch.tensor(zm, dtype=torch.float32) /
            torch.tensor(zo, dtype=torch.float32)
            for zm, zo in zip(self.shape_mid, self.shape_up)
        ]

        for ky in range(2 * self.radius + 1):
            for kx in range(2 * self.radius + 1):
                uy, ux = torch.meshgrid(
                    torch.arange(self.shape_out[0], dtype=torch.float32),
                    torch.arange(self.shape_out[1], dtype=torch.float32)
                )
                iy = torch.floor(step_inv[0] * uy) + ky - self.padding
                ix = torch.floor(step_inv[1] * ux) + kx - self.padding

                delta = torch.sqrt(
                    (uy - (self.margin + self.step * iy))**2 +
                    (ux - (self.margin + self.step * ix))**2
                )

                k = ky * (2 * self.radius + 1) + kx

                self.weight[0, k] = self.kernel(delta / sigma)

    def generate(self, mask_in):
        r"""Generate a mask.
        The function takes as input a parameter tensor :math:`\bar m` for
        :math:`K` masks, which is a :math:`K\times 1\times H_i\times W_i`
        tensor where `H_i\times W_i` are given by :attr:`shape_in`.
        Args:
            mask_in (:class:`torch.Tensor`): mask parameters.
        Returns:
            tuple: a pair of mask, cropped and full. The cropped mask is a
            :class:`torch.Tensor` with the same spatial shape :attr:`shape`
            as specfied upon creating this object. The second mask is the same,
            but with an additional margin and shape :attr:`shape_out`.
        """
        mask = F.unfold(mask_in,
                        (2 * self.radius + 1,) * 2,
                        padding=(self.padding,) * 2)
        mask = mask.reshape(
            len(mask_in), -1, self.shape_mid[0], self.shape_mid[1])
        mask = F.interpolate(mask, size=self.shape_up, mode='nearest')
        mask = F.pad(mask, (0, -self.step + 1, 0, -self.step + 1))
        mask = self.weight * mask

        if self.pooling_method == 'sigmoid':
            if self.coldness == float('+Inf'):
                mask = (mask.sum(dim=1, keepdim=True) - 5 > 0).float()
            else:
                mask = torch.sigmoid(
                    self.coldness * mask.sum(dim=1, keepdim=True) - 3
                )
        elif self.pooling_method == 'softmax':
            if self.coldness == float('+Inf'):
                mask = mask.max(dim=1, keepdim=True)[0]
            else:
                mask = (
                    mask * F.softmax(self.coldness * mask, dim=1)
                ).sum(dim=1, keepdim=True)

        elif self.pooling_method == 'sum':
            mask = mask.sum(dim=1, keepdim=True)
        else:
            assert False, f"Unknown pooling method {self.pooling_method}"
        m = round(self.margin)
        if self.clamp:
            mask = mask.clamp(min=0, max=1)
        cropped = mask[:, :, m:m + self.shape[0], m:m + self.shape[1]]
        return cropped, mask

    def to(self, dev):
        """Switch to another device.
        Args:
            dev: PyTorch device.
        Returns:
            MaskGenerator: self.
        """
        self.weight = self.weight.to(dev)
        return self