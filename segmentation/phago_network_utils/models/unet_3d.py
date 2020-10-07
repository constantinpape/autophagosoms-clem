import torch.nn as nn
from .base import UNetBase
from .layers import Upsample, get_activation


#
# 3D U-Net implementations
#


class UNet3d(UNetBase):
    """ 3d U-Net for segmentation as described in
    https://arxiv.org/abs/1606.06650
    """
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 3d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part):
        # padding = kernel_size // 2 (= 1 for kernel size hard-coded to 3)
        padding = 1 if self.pad_convs else 0

        # p_dropout can be single value or can depend on the part
        if isinstance(self.p_dropout, float):
            p_dropout = self.p_dropout
        else:
            p_dropout = self.p_dropout.get(part, 0.)

        # single unit in conv block
        def single_conv(inc, outc):
            convs = []

            # check if we have a normalization layer and put it as first in the conv block
            # see this for a long discussion where to put it ....
            # https://www.reddit.com/r/MachineLearning/comments/67gonq/d_batch_normalization_before_or_after_relu/
            if self.norm is not None:
                if self.norm == 'BatchNorm':
                    norm = nn.BatchNorm3d(outc)
                elif self.norm == 'GroupNorm':
                    n_groups = min(32, inc)
                    norm = nn.GroupNorm(n_groups, inc)
                elif self.norm == 'InstanceNorm':
                    norm = nn.InstanceNorm3d(outc)
                convs.append(norm)

            # add the convolution and the activation
            convs.extend([nn.Conv3d(inc, outc, kernel_size=3, padding=padding),
                          nn.ReLU(inplace=True) if getattr(self,
                                                           'inner_activation',
                                                           'ReLU') == 'ReLU'
                          else get_activation(self.inner_activation)])

            # add dropout if specified
            if p_dropout > 0:
                convs.append(nn.Dropout3d(p=p_dropout))
            return nn.Sequential(*convs)

        # full conv-block
        return nn.Sequential(single_conv(in_channels, out_channels),
                             single_conv(out_channels, out_channels))

    def _upsampler(self, in_channels, out_channels, level):
        # upsample via trilinear interpolation + 1x1 conv
        return Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=2,
                        mode='trilinear')

    # pooling via maxpool3d
    def _pooler(self, level):
        return nn.MaxPool3d(2)

    def _out_conv(self, in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, 1)


class AnisotropicUNet(UNet3d):
    """ 3D U-Net with anisotropic scaling

    Arguments:
      scale_factors: list of scale factors
      in_channels: number of input channels
      out_channels: number of output channels
      initial_features: number of features after first convolution
      gain: growth factor of features
      pad_convs: whether to use padded convolutions
      final_activation: activation applied to the network output
    """
    @staticmethod
    def _validate_scale_factors(scale_factors):
        assert isinstance(scale_factors, (list, tuple))
        for sf in scale_factors:
            assert isinstance(sf, (int, tuple, list))
            if not isinstance(sf, int):
                assert len(sf) == 3
                assert all(isinstance(sff, int) for sff in sf)

    def __init__(self, scale_factors, in_channels=1,
                 out_channels=1, initial_features=64,
                 gain=2, pad_convs=False, p_dropout=0.,
                 final_activation=None, return_side_outputs=False,
                 norm=None, activation='ReLU'):
        self._validate_scale_factors(scale_factors)
        self.scale_factors = scale_factors
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         initial_features=initial_features, gain=gain,
                         depth=len(self.scale_factors), pad_convs=pad_convs,
                         final_activation=final_activation, p_dropout=p_dropout,
                         return_side_outputs=return_side_outputs,
                         norm=norm, activation=activation)

    # we use trilinear and 1d convolutions instead of transpsosed conv
    def _upsampler(self, in_channels, out_channels, level):
        scale_factor = self.scale_factors[level]
        return Upsample(in_channels=in_channels,
                        out_channels=out_channels,
                        scale_factor=scale_factor,
                        mode='trilinear')

    def _pooler(self, level):
        return nn.MaxPool3d(self.scale_factors[level])
