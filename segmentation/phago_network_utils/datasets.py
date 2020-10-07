import json
import numpy as np
import z5py

from inferno.utils.io_utils import yaml2dict
from inferno.io.core import Concatenate
from torch.utils.data import Dataset, DataLoader
from inferno.io.transform import Compose
from inferno.io.transform.base import Transform
from inferno.io.transform.generic import AsTorchBatch, Cast, Normalize
from inferno.io.transform.volume import AdditiveNoise
from inferno.io.transform.volume import RandomFlip3D
from inferno.io.transform.image import RandomRotate, ElasticTransform

from neurofire.transform.segmentation import ConnectedComponents3D
from neurofire.transform.affinities import affinity_config_to_transform


class Semantics(Transform):
    n_classes = 3

    def __init__(self, rg_as_class=True, **super_kwargs):
        self.rg_as_class = rg_as_class
        super().__init__(**super_kwargs)

    def one_hot(self, seg):
        shape = seg.shape
        output = np.zeros((self.n_classes,) + shape, dtype='float32')
        assert seg.max() <= self.n_classes
        # zero is not treated as an independent class here
        for c in range(1, self.n_classes + 1):
            output[c - 1] = seg == c
        return output

    def two_labels(self, seg):
        shape = seg.shape
        output = np.zeros((2,) + shape, dtype='uint8')
        assert seg.max() <= self.n_classes

        # first channel: r (label id 1) and rg (label id 2)
        # second channel: g (label id 3) and rg (label id 2)
        output[0, :] = seg == 1
        output[1, :] = seg == 3
        output[:] += seg == 2

        output = (output > 0).astype('float32')
        return output

    def volume_function(self, seg):

        # the logic of the labels is as follows:
        # 1 = objects have red marker
        # 2 = objects have red and green marker
        # 3 = objects have green marker

        # first channel = foreground channel
        output = [(seg > 0).astype('float32')[None]]

        if self.rg_as_class:
            # if we treat the rg channel as separate class, we
            # can use normal one hot encoding
            output.append(self.one_hot(seg))
        else:
            # if we don't treat the rg channel as separate class, we
            # need to do a more custim encoding
            output.append(self.two_labels(seg))

        # output must be inverted to be compatible with the affinity
        # conventions
        output = np.concatenate(output, axis=0)
        return output


# TODO implement different strategies for the one-hot encoding
# depending on what makes sense biologically
class SemanticsAndAffinities(Transform):
    def __init__(self, affinity_config, **super_kwargs):
        self.aff_trafo = affinity_config_to_transform(**affinity_config)
        self.semantic_trafo = Semantics()
        self.cc_trafo = ConnectedComponents3D()
        super().__init__(**super_kwargs)

    def volume_function(self, seg):
        # cast segmentation to labels for semantic and affinity training
        # compute one hot encoding for the semantic channels
        # we must invert the semantic channel, to have the same fg/bg convention
        # as the affniities
        semantic_channels = 1. - self.semantic_trafo(seg)

        # apply connected components before computing affinities
        seg = self.cc_trafo(seg)

        # apply the affinity transformation
        affs = self.aff_trafo(seg)

        # splice the affinities into input and mask channels
        split_channel = affs.shape[0] // 2
        affs, aff_mask = affs[:split_channel], affs[split_channel:]

        # make dummy mask channels for the semantic channels
        semantic_mask = np.ones_like(semantic_channels)

        # merge semantic and affinity channels as well als masks
        return np.concatenate([semantic_channels, affs, semantic_mask, aff_mask],
                              axis=0)


class PadTo(Transform):
    def __init__(self, shape, **super_kwargs):
        self.shape = tuple(shape)
        super().__init__(**super_kwargs)

    def volume_function(self, x):
        shape = tuple(x.shape)
        if self.shape == shape:
            return x
        pad_width = [sh1 - sh2 for sh1, sh2 in zip(self.shape, shape)]
        pad_width = [[0, pw] for pw in pad_width]
        x = np.pad(x, pad_width=pad_width, mode='reflect')
        return x


class AutophagosomDataset(Dataset):
    def __init__(self, name, volume_config, slicing_config, master_config):
        assert 'raw' in volume_config
        assert 'labels' in volume_config

        raw_volume_kwargs = volume_config.get('raw')[name]
        label_volume_kwargs = volume_config.get('labels')[name]

        self.raw_volume = self.load_volume(raw_volume_kwargs)
        self.label_volume = self.load_volume(label_volume_kwargs)
        self.shape = self.raw_volume.shape

        slice_conf = slicing_config[name]
        self.window_size = slice_conf['window_size']
        self.anchor_list = slice_conf['anchor_list']

        # acnchor_list can be json or list
        if isinstance(self.anchor_list, str):
            with open(self.anchor_list, 'r') as f:
                self.anchor_list = json.load(f)
        assert isinstance(self.anchor_list, list)
        self.sample_size = slice_conf.get('sampling_size', None)
        if isinstance(self.sample_size, int):
            self.sample_size = 3 * [self.sample_size]

        self.master_config = master_config
        self.make_transforms()

    # only works for z5py!
    def load_volume(self, volume_kwargs):
        path = volume_kwargs['data_path']
        key = volume_kwargs['data_key']
        return z5py.File(path)[key]

    def sample_location(self, index):
        anchor = self.anchor_list[index]
        if self.sample_size is not None:
            sample = [np.random.randint(-ssize, ssize + 1)
                      for ssize in self.sample_size]
            anchor = [anc + sa for anc, sa in zip(anchor, sample)]
        slice_ = tuple(slice(max(0, anc - wsize // 2),
                             min(sh, anc + wsize // 2))
                       for anc, wsize, sh in zip(anchor, self.window_size, self.shape))
        return slice_

    def __getitem__(self, index):
        slice_ = self.sample_location(index)

        raw = self.raw_volume[slice_]
        if self.raw_transforms is not None:
            raw = self.raw_transforms(raw)

        labels = self.label_volume[slice_]
        if self.label_transforms is not None:
            labels = self.label_transforms(labels)

        tensors = [raw, labels]
        if self.transforms is not None:
            tensors = self.transforms(tensors)
        return tensors

    def __len__(self):
        return len(self.anchor_list)

    def make_transforms(self):
        transforms = Compose(PadTo(self.window_size), RandomFlip3D(), RandomRotate())
        if self.master_config.get('elastic_transform'):
            elastic_transform_config = self.master_config.get('elastic_transform')
            transforms.add(ElasticTransform(alpha=elastic_transform_config.get('alpha', 2000.),
                                            sigma=elastic_transform_config.get('sigma', 50.),
                                            order=elastic_transform_config.get('order', 0)))

        # affinity transforms for affinity targets
        # we apply the affinity target calculation only to the segmentation (1)
        affinity_config = self.master_config.get('affinity_config', None)

        # Do we also train with semantic labels ?
        train_semantic = self.master_config.get('train_semantic', False)

        if affinity_config is None:
            if train_semantic:
                transforms.add(Semantics(apply_to=[1]))
                self.label_transforms = None
            else:
                self.label_transforms = Cast('float32')
        elif affinity_config == 'distances':
            # TODO read the bandwidths from the config
            self.label_transforms = Compose(Cast('int64'), ConnectedComponents3D())
            from ..transforms.distance_transform import SignedDistanceTransform
            transforms.add(SignedDistanceTransform(fg_bandwidth=8,
                                                   bg_bandwidth=32,
                                                   apply_to=[1]))
        else:
            if train_semantic:
                # we can't apply connected components yet if we train semantics and affinities
                self.label_transforms = Cast('int64')
                transforms.add(SemanticsAndAffinities(affinity_config, apply_to=[1]))
            else:
                self.label_transforms = Compose(Cast('int64'), ConnectedComponents3D())
                transforms.add(affinity_config_to_transform(apply_to=[1], **affinity_config))

        self.transforms = transforms
        sigma = 0.025
        self.raw_transforms = Compose(Cast('float32'), Normalize(), AdditiveNoise(sigma=sigma))


class AutophagosomDatasets(Concatenate):
    def __init__(self, names, volume_config, slicing_config, master_config):
        datasets = [AutophagosomDataset(name=name,
                                        volume_config=volume_config,
                                        slicing_config=slicing_config,
                                        master_config=master_config)
                    for name in names]

        super().__init__(*datasets)
        self.transforms = self.get_transforms()

    def get_transforms(self):
        transforms = AsTorchBatch(3)
        return transforms

    @classmethod
    def from_config(cls, config):
        config = yaml2dict(config)
        names = config.get('names')
        volume_config = config.get('volume_config')
        slicing_config = config.get('slicing_config')
        master_config = config.get('master_config')
        return cls(names=names, volume_config=volume_config,
                   slicing_config=slicing_config, master_config=master_config)


def get_autophagosom_loader(config):
    config = yaml2dict(config)
    loader_config = config.pop('loader_config')
    datasets = AutophagosomDatasets.from_config(config)
    loader = DataLoader(datasets, **loader_config)
    return loader
