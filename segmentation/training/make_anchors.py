import os
import json
from random import random, shuffle

import numpy as np
import z5py

# from vigra.analysis import labelVolumeWithBackground, extractRegionFeatures
from scipy.ndimage import label
from skimage.measure import regionprops


def compute_centers(name, out_path, scale, scale_factor=None, sample_empty=0., validation_fraction=.15):
    """ Compute center coordinats for the training batches by sampling coordiantes in
    the center of each label component as well as off center - and random locations.
    """
    path = f'../../data/{name}/images/local/fibsem-labels.n5'
    f = z5py.File(path, 'r')
    ds_labels = f[f'setup0/timepoint0/s{scale}']
    ds_labels.n_threads = 8
    labels = ds_labels[:]
    labels_cc, _ = label(labels)

    ids = np.unique(labels_cc)[1:]
    centers = regionprops(labels_cc)
    centers = np.array([[0, 0, 0]] + [prop.centroid for prop in centers])

    # add one center and one off center coordinate
    offset = 16

    center_results = []
    for sid in ids:
        center = centers[sid]
        if scale_factor is None:
            center = [int(ce) for ce in center]
        else:
            center = [int(ce * scale_factor) for ce in center]
        center_results.append(center)

        center = [ce + (2 * int(random()) - 1) * offset for ce in center]
        center = [max(ce, 0) for ce in center]
        center_results.append(center)

    # sample some random centers
    if sample_empty > 0.:
        shape = labels.shape
        shape = [int(sh * scale_factor) for sh in shape]
        margin = [64, 128, 128]
        n_empty = int(sample_empty * len(center_results))
        print("Adding center coordinates without foreground target:")
        print(n_empty, '/', len(center_results))
        empty_centers = [np.random.randint(marge, sh - marge, size=(n_empty, 1))
                         for marge, sh in zip(margin, shape)]
        empty_centers = np.concatenate(empty_centers, axis=1)
        center_results.extend(empty_centers.tolist())

    with open(out_path, 'w') as f:
        json.dump(center_results, f)

    shuffle(center_results)

    val_idx = int((1. - validation_fraction) * len(center_results))
    centers_train, centers_val = center_results[:val_idx], center_results[val_idx:]
    out_prefix = os.path.splitext(out_path)[0]

    train_out_path = out_prefix + '_train.json'
    with open(train_out_path, 'w') as f:
        json.dump(centers_train, f)

    val_out_path = out_prefix + '_val.json'
    with open(val_out_path, 'w') as f:
        json.dump(centers_val, f)


def compute_all_centers():
    names = ['1spd', '1spdbaf', 'baf']
    scale = 1
    scale_factor = 2
    data_scale = 2
    os.makedirs('anchors', exist_ok=True)
    for name in names:
        print("Creating training anchor points for", name)
        out_path = os.path.join('anchors', '%s_s%i.json' % (name, scale))
        compute_centers(name, out_path, scale=data_scale,
                        scale_factor=scale_factor, sample_empty=.25)


if __name__ == '__main__':
    # view_sample(name='1spd')
    compute_all_centers()
