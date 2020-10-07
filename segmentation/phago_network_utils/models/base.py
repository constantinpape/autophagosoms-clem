import torch
import torch.nn as nn
from .layers import get_activation


class UNetBase(nn.Module):
    """ UNet Base class implementation

    Deriving classes must implement
    - _conv_block(in_channels, out_channels, level, part)
        return conv block for a U-Net level
    - _pooler(level)
        return pooling operation used for downsampling in-between encoders
    - _upsampler(in_channels, out_channels, level)
        return upsampling operation used for upsampling in-between decoders
    - _out_conv(in_channels, out_channels)
        return output conv layer

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      depth: depth of the network
      initial_features: number of features after first convolution
      gain: growth factor of features
      pad_convs: whether to use padded convolutions
      norm: whether to use batch-norm, group-norm or None
      p_dropout: dropout probability
      final_activation: activation applied to the network output
      return_side_outputs: whether to return side outputs derived from decoders
    """
    norms = ('BatchNorm', 'GroupNorm', 'InstanceNorm')

    def __init__(self, in_channels, out_channels, depth=4,
                 initial_features=64, gain=2, pad_convs=False,
                 norm=None, p_dropout=0.0, final_activation=None,
                 return_side_outputs=False, activation='ReLU'):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pad_convs = pad_convs
        if norm is not None:
            assert norm in self.norms
        self.norm = norm
        assert isinstance(p_dropout, (float, dict))
        self.p_dropout = p_dropout
        self.inner_activation = activation

        # modules of the encoder path
        n_features = [in_channels] + [initial_features * gain ** level
                                      for level in range(self.depth)]
        self.encoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='encoder')
                                      for level in range(self.depth)])

        # the base convolution block
        self.base = self._conv_block(n_features[-1], gain * n_features[-1], part='base', level=0)

        # modules of the decoder path
        n_features = [initial_features * gain ** level
                      for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        self.decoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       self.depth - level - 1, part='decoder')
                                      for level in range(self.depth)])

        # the pooling layers;
        self.poolers = nn.ModuleList([self._pooler(level) for level in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1],
                                                         self.depth - level - 1)
                                         for level in range(self.depth)])

        # do we return the decoder side outputs?
        self.return_side_outputs = return_side_outputs and (self.out_channels is not None)

        if self.return_side_outputs:
            # we need to slice away the features of the base
            n_features = n_features[1:]
            self.out_conv = nn.ModuleList([self._out_conv(n_features[level], out_channels)
                                           for level in range(self.depth)])
        elif self.out_channels is not None:
            self.out_conv = self._out_conv(n_features[-1], out_channels)
        else:
            self.out_conv = None
            self.out_channels = n_features[-1]

        # activation applied to the outputs
        self.activation = get_activation(final_activation)

    @staticmethod
    def _crop_tensor(input_, shape_to_crop):
        input_shape = input_.shape
        # get the difference between the shapes
        shape_diff = tuple((ish - csh) // 2
                           for ish, csh in zip(input_shape, shape_to_crop))
        if all(sd == 0 for sd in shape_diff):
            return input_
        # calculate the crop
        crop = tuple(slice(sd, sh - sd)
                     for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def _apply_default(self, input_):
        x = input_

        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x,
                                                          encoder_out[level]))

        # apply output conv and activation (if given)
        if self.out_conv is not None:
            x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

    def _apply_with_side_outputs(self, input_):
        x = input_

        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)
        outputs = []

        # apply decoder path and collect decoder outputs
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x, encoder_out[level]))
            outputs.append(x)

        # apply output convolutions and activations (if given)
        outputs = [out_conv(out) for out, out_conv in zip(outputs, self.out_conv)]
        if self.activation is not None:
            outputs = [self.activation(out) for out in outputs]

        # we invert the outputs so that we get the full scale output first
        return outputs[::-1]

    def forward(self, input):
        # if return side outputs is true, we collect the decoder outputs,
        # apply the out convolution and (if given) the activation
        return_side_outputs = getattr(self, 'return_side_outputs', False)

        if return_side_outputs:
            return self._apply_with_side_outputs(input)
        else:
            return self._apply_default(input)
