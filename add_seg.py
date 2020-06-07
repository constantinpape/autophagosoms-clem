import argparse
import os

from mobie import add_segmentation
from init_dataset import add_xml_for_s3

ROOT = './data'
DEFAULT_RESOLUTION = [0.005, 0.005, 0.005]
DEFAULT_CHUNKS = [64, 64, 64]


def add_seg_to_dataset(dataset, path, seg_name, key, resolution,
                       target='local', max_jobs=32):
    scale_factors = 6 * [[2, 2, 2]]

    add_segmentation(path, key,
                     root=ROOT,
                     dataset_name=dataset,
                     segmentation_name=seg_name,
                     resolution=resolution,
                     scale_factors=scale_factors,
                     chunks=DEFAULT_CHUNKS,
                     target=target, max_jobs=max_jobs,
                     add_default_table=True)

    # convert tif stack to bdv.n5 format
    dataset_folder = os.path.join(ROOT, dataset)
    out_path = os.path.join(dataset_folder, 'images', 'local', f'{seg_name}.n5')
    xml_path = os.path.splitext(out_path)[0] + '.xml'
    add_xml_for_s3(xml_path, out_path)
    print("You also need to add the files in", dataset_folder, "to git")


if __name__ == '__main__':
    desc = 'Add segmentation to an existing dataset'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset', type=str, help='Name of the dataset to be added')
    parser.add_argument('path', type=str, help='Path to the raw data for this dataset')
    parser.add_argument('name', type=str, help='Name of the segmentation')
    parser.add_argument('key', type=str, help='Key to the input data (only necessary for n5/hd5 input data)')

    parser.add_argument('--resolution', type=float, nargs=3, default=DEFAULT_RESOLUTION)

    args = parser.parse_args()
    add_seg_to_dataset(args.dataset, args.path, args.name, args.key, args.resolution)
