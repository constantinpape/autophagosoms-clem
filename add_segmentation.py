import argparse
import os
# NOTE I will refactor this into https://github.com/platybrowser/mmb-python eventually
from mmb_utils import add_segmentation_to_image_dict, import_segmentation, have_dataset
from mmpb.attributes.base_attributes import base_attributes
from initialize_dataset import add_xml_for_s3

ROOT = './data'
DEFAULT_RESOLUTION = [0.005, 0.005, 0.005]


def add_table(path, table_folder, resolution, tmp_folder, target, max_jobs):
    out_path = os.path.join(table_folder, 'default.csv')
    base_attributes(path, 'setup0/timepoint0/s0', out_path, resolution,
                    tmp_folder, target, max_jobs, correct_anchors=False)


def add_segmentation(dataset, path, seg_name, key, resolution,
                     target='local', max_jobs=32):
    if not have_dataset(ROOT, dataset):
        raise ValueError(f"Expect to have the dataset {dataset} to be initialized already!.")

    # TODO check that the shape and resolution match the raw data

    folder = os.path.join(ROOT, dataset)
    out_path = os.path.join(folder, 'images', 'local', f'fibsem-{dataset}-{seg_name}.n5')
    tmp_folder = f'tmp_{dataset}_{seg_name}'
    import_segmentation(path, key, out_path, resolution, tmp_folder)

    table_folder = os.path.join(folder, 'tables', seg_name)
    os.makedirs(table_folder, exist_ok=True)
    add_table(out_path, table_folder, resolution, tmp_folder, target, max_jobs)

    xml_path = os.path.splitext(out_path)[0] + '.xml'
    add_xml_for_s3(xml_path, out_path)
    add_segmentation_to_image_dict(folder, xml_path, table_folder)


if __name__ == '__main__':
    desc = 'Add segmentation to an existing dataset'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset', type=str, help='Name of the dataset to be added')
    parser.add_argument('path', type=str, help='Path to the raw data for this dataset')
    parser.add_argument('name', type=str, help='Name of the segmentation')
    parser.add_argument('--key', type=str, default='',
                        help='Key to the input data (only necessary for n5/hd5 input data)')

    parser.add_argument('--resolution', type=float, nargs=3, default=DEFAULT_RESOLUTION)

    args = parser.parse_args()
    add_segmentation(args.dataset, args.path, args.name, args.key, args.resolution)
