import argparse
import os

# NOTE: I will refactor this into https://github.com/platybrowser/mmb-python eventually
from mmpb.files.xml_utils import write_s3_xml
from mmpb.release_helper import make_folder_structure
from mmb_utils import (add_dataset, check_dataset,
                       import_raw_volume,
                       initialize_bookmarks, initialize_image_dict)

ROOT = './data'
# TODO
DEFAULT_RESOLUTION = [1., 1., 1.]


def add_xml_for_s3(xml_path, data_path):
    # TODO would need a bucket for this
    bucket_name = ''
    xml_out_path = xml_path.replace('local', 'remote')

    path_in_bucket = os.path.relpath(data_path, start=ROOT)
    write_s3_xml(xml_path, xml_out_path, path_in_bucket,
                 bucket_name=bucket_name)

    print("In order to add the data to the EMBL S3, please run the following command:")
    full_s3_path = f'embl/{bucket_name}/{path_in_bucket}'
    mc_command = f"mc cp -r {os.path.relpath(data_path)}/ {full_s3_path}/"
    print(mc_command)


def initialize_dataset(dataset, path, resolution,
                       overwrite, upload,
                       target='local', max_jobs=32):
    check_dataset(ROOT, dataset, overwrite, upload)
    assert os.path.exists(path), path

    tmp_folder = './tmp_%s' % dataset

    # create output folder structure
    output_folder = os.path.join(ROOT, dataset)
    make_folder_structure(output_folder)

    # TODO naming schemen?
    out_path = os.path.join(output_folder, 'images', 'local', 'fibsem-raw.n5')
    import_raw_volume(path, out_path, tmp_folder, resolution,
                      target=target, max_jobs=max_jobs)
    # TODO make mask?

    xml_path = os.path.splitext(out_path)[0] + '.xml'
    if upload:
        add_xml_for_s3(xml_path, out_path)

    # initialize the image dict and bookmarks
    initialize_image_dict(output_folder, xml_path)
    initialize_bookmarks(output_folder)

    # register this stack in datasets.json
    add_dataset(ROOT, dataset)
    print("You need to add the files in", output_folder, "to git")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('dataset', type=str, help='Name of the dataset to be added')
    # TODO figure out which formats to support
    parser.add_argument('path', type=str, help='Path to the raw data for this dataset')
    parser.add_argument('--resolution', type=float, nargs=3, default=DEFAULT_RESOLUTION)
    parser.add_argument('--upload', type=int, default=0,
                        help='Whether to upload the data to s3')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='Whether to over-write an existing dataset')

    args = parser.parse_args()

    # TODO add support for these options
    overwrite, upload = bool(args.overwrite), bool(args.upload)
    assert not upload
    assert not overwrite

    initialize_dataset(args.dataset, args.path, overwrite, upload)
