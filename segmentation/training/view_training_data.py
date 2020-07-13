import argparse
import os

import napari
import z5py

from vigra import labelVolumeWithBackground


def view_sample(name, scale=2, with_ccs=False):
    path_raw = f'../../data/{name}/images/local/fibsem-{name}-raw.n5'
    f = z5py.File(path_raw, 'r')
    ds_raw = f[f'setup0/timepoint0/s{scale}']
    print(ds_raw.shape)
    ds_raw.n_threads = 8
    raw = ds_raw[:]

    path_labels = f'../../data/{name}/images/local/fibsem-{name}-labels.n5'
    have_labels = os.path.exists(path_labels)
    if have_labels:
        f = z5py.File(path_labels, 'r')
        ds_labels = f[f'setup0/timepoint0/s{scale}']
        print(ds_labels.shape)
        ds_labels.n_threads = 8
        labels = ds_labels[:]

    # print("Run opening and connected components ...")
    # labels_pp = binary_opening(labels, iterations=4).astype('uint32')
    # labels_pp = labelVolumeWithBackground(labels_pp)

    if have_labels and with_ccs:
        print("Run connected components ...")
        labels_cc = labelVolumeWithBackground(labels)

    with napari.gui_qt():
        viewer = napari.Viewer()
        viewer.add_image(raw)
        if have_labels:
            viewer.add_labels(labels)
        if have_labels and with_ccs:
            viewer.add_labels(labels_cc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('name')
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--with_ccs', type=int, default=0)
    args = parser.parse_args()
    view_sample(args.name, args.scale, args.with_ccs)
