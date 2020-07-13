import argparse
import os

from mobie import initialize_dataset
from mobie.xml_utils import copy_xml_as_n5_s3

ROOT = './data'
# we have an isotropic resolution of 5 nm
DEFAULT_RESOLUTION = [0.005, 0.005, 0.005]
DEFAULT_CHUNKS = [64, 64, 64]


def add_xml_for_s3(xml_path, data_path):
    bucket_name = 'autophagosomes'
    xml_out_path = xml_path.replace('local', 'remote')

    path_in_bucket = os.path.relpath(data_path, start=ROOT)
    copy_xml_as_n5_s3(xml_path, xml_out_path,
                      service_endpoint='https://s3.embl.de',
                      bucket_name=bucket_name,
                      path_in_bucket=path_in_bucket,
                      authentication='Protected')

    print("In order to add the data to the EMBL S3, please run the following command:")
    full_s3_path = f'embl/{bucket_name}/{path_in_bucket}'
    mc_command = f"mc cp -r {os.path.relpath(data_path)}/ {full_s3_path}/"
    print(mc_command)


def init_dataset(dataset, path, in_key, resolution,
                 is_default, target='local', max_jobs=32,
                 time_limit=None):
    assert os.path.exists(path), path

    raw_name = 'fibsem-raw'
    scale_factors = 6 * [[2, 2, 2]]

    initialize_dataset(path, in_key, ROOT,
                       dataset, raw_name,
                       resolution, DEFAULT_CHUNKS, scale_factors,
                       is_default=is_default, target=target,
                       max_jobs=max_jobs, time_limit=time_limit)

    dataset_folder = os.path.join(ROOT, dataset)
    out_path = os.path.join(dataset_folder, 'images', 'local', f'{raw_name}.n5')
    xml_path = os.path.splitext(out_path)[0] + '.xml'
    add_xml_for_s3(xml_path, out_path)
    print("You also need to add the files in", dataset_folder, "to git")


if __name__ == '__main__':
    desc = 'Initialize dataset by adding raw data and creating folder structure'
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument('dataset', type=str, help='Name of the dataset to be added')
    parser.add_argument('path', type=str, help='Path to the raw data for this dataset')
    parser.add_argument('key', type=str, help='Key to the input data (only necessary for n5/hd5 input data)')

    parser.add_argument('--resolution', type=float, nargs=3, default=DEFAULT_RESOLUTION)
    parser.add_argument('--is_default', type=int, default=0,
                        help='Is this the default dataset for the viewer?')

    args = parser.parse_args()

    init_dataset(args.dataset, args.path, args.key, args.resolution, bool(args.is_default))
