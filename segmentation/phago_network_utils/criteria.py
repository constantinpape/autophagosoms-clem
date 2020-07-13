import torch
import torch.nn as nn
from inferno.extensions.criteria import SorensenDiceLoss
from inferno.utils.torch_utils import flatten_samples


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=1., beta=1.):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.bce = nn.BCELoss()
        self.dice = SorensenDiceLoss()

    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + self.beta * self.dice(input, target)


class RobustDiceLoss(nn.Module):
    """ Computes loss scalar between input and target based on the Sorensen-Dice similarity.

    This implementation is adapted s.t. also batches for which the target is all background
    yield a meaningful loss. For these batches, the relevant term of the binary cross entropy
    is computed.
    For both inputs and targets it must be the case that `input_or_target.size(1) = num_channels`.
    """

    def __init__(self, sigma_weight_zero=1e-4, channelwise=True, weight=None, eps=1e-6):
        super().__init__()
        self.register_buffer('weight', weight)

        self.channelwise = channelwise
        self.eps = eps
        self.sigma_weight_zero = sigma_weight_zero

    def forward(self, input_, target):
        """
        input_:      torch.FloatTensor or torch.cuda.FloatTensor
        target:     torch.FloatTensor or torch.cuda.FloatTensor

        Expected shape of the inputs: (batch_size, nb_channels, ...)
        """
        assert input_.shape == target.shape

        # compute the score individual for the channels
        if self.channelwise:

            # transform input and target to (C, N*H*W...)
            input_ = flatten_samples(input_)
            target = flatten_samples(target)

            # compute the weight between dice term and cross entropy term
            # as a function of the foreground ratio
            foreground_ratio = target.sum(1) / target.shape[1]
            # we use a normal function with very small sigma to compute the weight
            # between bce and dice term. this approximates a delta peak around
            # foreground_ratio = 0
            weight = torch.exp(- .5 * (foreground_ratio / self.sigma_weight_zero) ** 2)

            # compute the dice term of the loss
            numerator = (input_ * target).sum(1)
            denominator = (input_ * input_).sum(1) + (target * target).sum(1)
            dice_term = 1. - 2 * (numerator / denominator.clamp(min=self.eps))

            # NOTE computing the bce term like this is not stable, so we use the torch implementation
            # and use the full bce (where 1 term will not contribute) instead
            # compute the cross entropy term (we only need the part that is non-zero for target == 0)
            # bce_term = (-1. * (1. - target) * torch.log(1 - input_)).mean(1)
            bce_term = nn.functional.binary_cross_entropy(input_, target,
                                                          reduction='none').mean(1)

            # compute the combined loss
            loss = (1. - weight) * dice_term + weight * bce_term

            # apply a class weight to the channels if specified
            if self.weight is not None:
                assert self.weight.shape == loss.shape
                loss = self.weight * loss

            # sum over the channels to compute the total loss
            loss = loss.sum()

        # compute the score for all channels
        else:
            numerator = (input_ * target).sum()
            denominator = (input_ * input_).sum() + (target * target).sum()
            dice_term = 1 - 2. * (numerator / denominator.clamp(min=self.eps))

            foreground_ratio = target.sum(1) / target.shape[1]
            weight = torch.exp(- .5 * (foreground_ratio / self.sigma_weight_zero) ** 2)

            bce_term = nn.functional.binary_cross_entropy(input_, target, reduction='none')

            loss = (1. - weight) * dice_term + weight * bce_term

        return loss
