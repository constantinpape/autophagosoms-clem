import torch.nn as nn
import torch.nn.functional as F


class Upsample(nn.Module):
    """ Upsample the input and change the number of channels
    via 1x1 Convolution if a different number of input/output channels is specified.
    """

    def __init__(self, scale_factor, mode='nearest',
                 in_channels=None, out_channels=None, align_corners=False,
                 ndim=3):
        super().__init__()
        self.mode = mode
        self.scale_factor = scale_factor
        self.align_corners = align_corners
        if in_channels != out_channels:
            if ndim == 2:
                self.conv = nn.Conv2d(in_channels, out_channels, 1)
            elif ndim == 3:
                self.conv = nn.Conv3d(in_channels, out_channels, 1)
            else:
                raise ValueError("Only 2d and 3d supported")
        else:
            self.conv = None

    def forward(self, input):
        x = F.interpolate(input, scale_factor=self.scale_factor,
                          mode=self.mode, align_corners=self.align_corners)
        if self.conv is not None:
            return self.conv(x)
        else:
            return x


def get_activation(activation, **kwargs):
    """ Get activation from str or nn.Module
    """
    if activation is None:
        return None
    elif isinstance(activation, str):
        activation_cls = getattr(nn, activation, None)
        activation = activation_cls(**kwargs)
        assert isinstance(activation, nn.Module)
    else:
        activation = activation(**kwargs)
        assert isinstance(activation, nn.Module)
    return activation
