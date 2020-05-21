import argparse
import os

# NOTE I will refactor this into https://github.com/platybrowser/mmb-python eventually
from mmpb.files.xml_utils import write_s3_xml
from mmpb.release_helper import make_folder_structure
from mmb_utils import (add_dataset, have_dataset,
                       import_raw_volume,
                       initialize_bookmarks, initialize_image_dict)

ROOT = './data'
# we have an isotropic resolution of 5 nm
DEFAULT_RESOLUTION = [0.005, 0.005, 0.005]


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


def initialize_dataset(dataset, path, in_key, resolution,
                       overwrite, upload, is_default,
                       target='local', max_jobs=32):
    assert os.path.exists(path), path
    # TODO check the proper combinations of overwrite / upload
    if have_dataset(ROOT, dataset):
        raise ValueError(f"Dataset name {dataset} exists already")

    tmp_folder = f'./tmp_{dataset}'
    print("Temporary files will be written to", tmp_folder,
          "this folder can be savely removed after the computation is done")

    # create output folder structure
    output_folder = os.path.join(ROOT, dataset)
    make_folder_structure(output_folder)

    raw_name = f'fibsem-{dataset}-raw'
    out_path = os.path.join(output_folder, 'images', 'local', f'{raw_name}.n5')
    import_raw_volume(path, in_key, out_path, resolution, tmp_folder,
                      target=target, max_jobs=max_jobs)

    # TODO make/add mask?

    xml_path = os.path.splitext(out_path)[0] + '.xml'
    if upload:
        add_xml_for_s3(xml_path, out_path)

    # initialize the image dict and bookmarks
    initialize_image_dict(output_folder, xml_path)
    initialize_bookmarks(output_folder, out_path, raw_name)

    # register this stack in datasets.json
    add_dataset(ROOT, dataset, is_default_dataset=is_default)
    print("You need to add the files in", output_folder, "to git")


if __name__ == '__main__':
    desc = 'Initialize dataset by adding raw data and creating folder structure'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset', type=str, help='Name of the dataset to be added')
    parser.add_argument('path', type=str, help='Path to the raw data for this dataset')
    parser.add_argument('--key', type=str, default='',
                        help='Key to the input data (only necessary for n5/hd5 input data)')

    parser.add_argument('--resolution', type=float, nargs=3, default=DEFAULT_RESOLUTION)
    parser.add_argument('--upload', type=int, default=0,
                        help='Whether to upload the data to s3')
    parser.add_argument('--overwrite', type=int, default=0,
                        help='Whether to over-write an existing dataset')
    parser.add_argument('--is_defult', type=int, default=0,
                        help='Is this the default dataset for the viewer?')

    args = parser.parse_args()

    # TODO add support for these options
    overwrite, upload = bool(args.overwrite), bool(args.upload)
    assert not upload
    assert not overwrite

    initialize_dataset(args.dataset, args.path, args.key, args.resolution,
                       overwrite, upload, bool(args.is_default))
