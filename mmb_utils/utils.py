import json
import os


def _load_datasets(dset_file):
    try:
        with open(dset_file) as f:
            datasets = json.load(f)
    except Exception:
        datasets = []
    return datasets


def add_dataset(root, dataset):
    dset_file = os.path.join(root, 'datasets.json')
    datasets = _load_datasets(dset_file)
    if dataset not in datasets:
        datasets.append(dataset)
        with open(dset_file, 'w') as f:
            json.dump(datasets, f)


def have_dataset(root, dataset):
    dset_file = os.path.join(root, 'datasets.json')
    datasets = _load_datasets(dset_file)
    return dataset in datasets
