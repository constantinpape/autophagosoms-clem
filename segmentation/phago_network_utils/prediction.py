from concurrent import futures
from itertools import product

import numpy as np
import torch

from tqdm import tqdm
from inferno.trainers import Trainer


class Blocking:
    def make_blocking(self, shape, block_shape):
        ranges = [range(sha // bsha if sha % bsha == 0 else sha // bsha + 1)
                  for sha, bsha in zip(shape, block_shape)]
        start_points = product(*ranges)

        min_coords = [0] * len(shape)
        max_coords = shape

        blocks = [tuple(slice(max(sp * bsha, minc),
                              min((sp + 1) * bsha, maxc))
                        for sp, bsha, minc, maxc in zip(start_point,
                                                        block_shape,
                                                        min_coords,
                                                        max_coords))
                  for start_point in start_points]
        return blocks

    def __init__(self, shape, block_shape):
        self.shape = shape
        self.block_shape = block_shape
        self._blocks = self.make_blocking(shape, block_shape)
        self.n_blocks = len(self._blocks)

    def __getitem__(self, block_id):
        return self._blocks[block_id]

    def __len__(self):
        return len(self._blocks)


# default zero mean unit variance normalization
def normalize(input_, mean=None, std=None, eps=1e-6):
    input_ = input_.astype('float32')
    mean = input_.mean() if mean is None else mean
    std = input_.std() if std is None else std
    return (input_ - mean) / (std + eps)


def to_uint8(data, float_range=(0., 1.), safe_scale=True):
    if safe_scale:
        mult = np.floor(255. / (float_range[1] - float_range[0]))
    else:
        mult = np.ceil(255. / (float_range[1] - float_range[0]))
    add = 255 - mult * float_range[1]
    return np.clip((data * mult + add).round(), 0, 255).astype('uint8')


def _load_block(input_, offset, block_shape, halo,
                padding_mode='reflect'):
    shape = input_.shape

    starts = [off - ha for off, ha in zip(offset, halo)]
    stops = [off + bs + ha for off, bs, ha in zip(offset, block_shape, halo)]

    # we pad the input volume if necessary
    pad_left = None
    pad_right = None

    # check for padding to the left
    if any(start < 0 for start in starts):
        pad_left = tuple(abs(start) if start < 0 else 0 for start in starts)
        starts = [max(0, start) for start in starts]

    # check for padding to the right
    if any(stop > shape[i] for i, stop in enumerate(stops)):
        pad_right = tuple(stop - shape[i] if stop > shape[i] else 0 for i, stop in enumerate(stops))
        stops = [min(shape[i], stop) for i, stop in enumerate(stops)]

    bb = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    data = input_[bb]

    # pad if necessary
    if pad_left is not None or pad_right is not None:
        pad_left = (0, 0, 0) if pad_left is None else pad_left
        pad_right = (0, 0, 0) if pad_right is None else pad_right
        pad_width = tuple((pl, pr) for pl, pr in zip(pad_left, pad_right))
        data = np.pad(data, pad_width, mode=padding_mode)

    return data


def _load_model(checkpoint_path, device, use_best=True, out_channels=1):
    model = Trainer().load(checkpoint_path, best=use_best,
                           map_location=torch.device(device)).model

    # monkey patch the model if it does not have out_channels attribute
    if not hasattr(model, 'out_channels'):
        model.out_channels = out_channels

    model.eval()
    return model


def predict_with_halo(input_, checkpoint_path, gpus,
                      block_shape, halo, use_best=True,
                      output=None, preprocess=None, postprocess=None):
    """ Run block-wise network prediction with halo.

    Arguments:
        input_ [arraylike] - input data
        checkpoint_path [str] - path to network checkpoint
        gpus [list[int]] - list of gpus ids used for prediction
        inner_block_shape [tuple] - shape of inner block used for prediction
        outer_block_shape [tuple] - shape of outer block used for prediction
        use_best [bool] - whether to use the best checkpoint for prediction (default: True)
        output [arraylike] - output data, will be allocated if None (default: None)
        preprocess [callable] - function to preprocess input data
            before passing it to the network (default: None)
        postprocess [callable] - function to postprocess
            the network predictions (default: None)
        model_is_inferno [bool] - is this an inferno checkpoint or a pure pytorch model? (default:True)
    """

    thread_data = [(_load_model(checkpoint_path, gpu, use_best),
                    torch.device(gpu)) for gpu in gpus]

    shape = input_.shape
    blocking = Blocking(shape, block_shape)

    if output is None:
        n_out = thread_data[0][0].out_channels
        output = np.zeros((n_out,) + shape, dtype='float32')

    n_workers = len(gpus)

    def predict_block(block_id):
        worker_id = block_id % n_workers
        net, device = thread_data[worker_id]

        with torch.no_grad():
            block = blocking[block_id]
            offset = [bb.start for bb in block]
            inp = _load_block(input_, offset, block_shape, halo)
            if preprocess is not None:
                inp = preprocess(inp)

            inp = torch.from_numpy(inp[None, None]).to(device)
            out = net(inp).cpu().numpy().squeeze(0)

            this_block_shape = tuple(bb.stop - bb.start for bb in block)
            inner_bb = tuple(slice(ha, ha + bs) for ha, bs in zip(halo, this_block_shape))
            out = out[(slice(None),) + inner_bb]
            if postprocess is not None:
                out = postprocess(out)

            output[(slice(None),) + block] = out

    n_blocks = len(blocking)
    with futures.ThreadPoolExecutor(n_workers) as tp:
        list(tqdm(tp.map(predict_block, range(n_blocks)), total=n_blocks))

    return output
