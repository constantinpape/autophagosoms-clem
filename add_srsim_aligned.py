import argparse
import os

import z5py
from mobie import add_image_data
# from init_dataset import add_xml_for_s3

ROOT = './data'
DEFAULT_CHUNKS = [1, 512, 512]


def add_srsim(dataset, target='local', max_jobs=32):

    path = f'./alignment/{dataset}.n5'
    assert os.path.exists(path), f"Could not find {path}; make sure to run the corresponding script /alignment"
    with z5py.File(path, 'r') as f:
        attrs = f.attrs
        resolution = attrs['resolution']
        scale_factors = attrs['scale_factors']
        channel_names = list(f.keys())

    for channel in channel_names:

        image_name = f'srsim-{channel}'
        add_image_data(path, channel,
                       root=ROOT,
                       dataset_name=dataset,
                       image_name=image_name,
                       resolution=resolution,
                       scale_factors=scale_factors,
                       chunks=DEFAULT_CHUNKS,
                       target=target, max_jobs=max_jobs)

    # we don't add this to s3 yet
    # dataset_folder = os.path.join(ROOT, dataset)
    # out_path = os.path.join(dataset_folder, 'images', 'local', f'{seg_name}.n5')
    # xml_path = os.path.splitext(out_path)[0] + '.xml'
    # add_xml_for_s3(xml_path, out_path)
    # print("You also need to add the files in", dataset_folder, "to git")


if __name__ == '__main__':
    desc = 'Add aligned srsim to an existing dataset'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset', type=str, help='Name of the dataset to be added')

    args = parser.parse_args()
    add_srsim(args.dataset)
