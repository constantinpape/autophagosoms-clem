import json
import os


def _load_datasets(dset_file):
    try:
        with open(dset_file) as f:
            datasets = json.load(f)
    except Exception:
        datasets = {}
    return datasets


def add_dataset(root, dataset, is_default_dataset=False):
    dset_file = os.path.join(root, 'datasets.json')

    dataset_dict = _load_datasets(dset_file)
    datasets = dataset_dict['datasets']

    if dataset not in datasets:
        datasets.append(dataset)
        dataset_dict['datasets'] = datasets

    if is_default_dataset:
        dataset_dict['default'] = dataset

    with open(dset_file, 'w') as f:
        json.dump(dataset_dict, f, indent=2, sort_keys=True)


def have_dataset(root, dataset):
    dset_file = os.path.join(root, 'datasets.json')
    datasets = _load_datasets(dset_file)['datasets']
    return dataset in datasets
