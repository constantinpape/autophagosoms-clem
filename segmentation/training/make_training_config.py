import os
import yaml


# only the names of datasets for wihich we have labels available
# ALL_NAMES = ['10spd', '10spdbaf', '1spd', '1spdbaf', 'baf', 'control']
ALL_NAMES = ['1spd', 'baf', '1spdbaf']


def make_template_config(sample_names, config_name, window_size, sample_size):
    train_config = {}
    val_config = {}

    train_config['names'] = sample_names
    val_config['names'] = sample_names

    elastic_config = {'alpha': 1000., 'sigma': 50., 'order': 0}
    aff_config = {'offsets': []}
    master_config = {'affinity_config': aff_config,
                     'elastic_transformation_config': elastic_config}
    train_config['master_config'] = master_config
    val_config['master_config'] = master_config

    loader_config = {'batch_size': 1, 'num_workers': 6,
                     'drop_last': True, 'pin_memory': False, 'shuffle': True}
    train_config['loader_config'] = loader_config
    val_config['loader_config'] = loader_config

    slicing_config_train = {}
    slicing_config_val = {}
    raw_config = {}
    labels_config = {}

    scale = 1
    for name in sample_names:

        raw_path = os.path.abspath(f'../../data/{name}/images/local/fibsem-raw.n5')
        raw_key = f'setup0/timepoint0/s{scale}'
        raw_conf = {'data_path': raw_path, 'data_key': raw_key, 'dtype': 'float32'}

        label_path = os.path.abspath(f'../../data/{name}/images/local/fibsem-labels.n5')
        label_key = f'setup0/timepoint0/s{scale}'
        label_conf = {'data_path': label_path, 'data_key': label_key, 'dtype': 'int64'}

        raw_config[name] = raw_conf
        labels_config[name] = label_conf

        anchor_path = './anchors/%s_s%i_train.json' % (name, scale)
        slicing_conf = {'window_size': window_size, 'sampling_size': sample_size,
                        'anchor_list': os.path.abspath(anchor_path)}
        slicing_config_train[name] = slicing_conf

        anchor_path = './anchors/%s_s%i_val.json' % (name, scale)
        slicing_conf = {'window_size': window_size, 'sampling_size': sample_size,
                        'anchor_list': os.path.abspath(anchor_path)}
        slicing_config_val[name] = slicing_conf

    train_config['volume_config'] = {'raw': raw_config, 'labels': labels_config}
    val_config['volume_config'] = {'raw': raw_config, 'labels': labels_config}

    train_config['slicing_config'] = slicing_config_train
    val_config['slicing_config'] = slicing_config_val

    train_config_path = os.path.join('./template_config', f'train_{config_name}.yaml')
    with open(train_config_path, 'w') as f:
        yaml.dump(train_config, f)

    val_config_path = os.path.join('./template_config', f'validation_{config_name}.yaml')
    with open(val_config_path, 'w') as f:
        yaml.dump(val_config, f)


def make_full_config():
    size = [32, 256, 256]
    sampling_size = [8, 32, 32]
    names = ALL_NAMES
    config_name = 'fullV1'
    os.makedirs('template_config', exist_ok=True)
    print("Make template config for", names)
    make_template_config(names, config_name, size, sampling_size)


# TODO implement different configs for validation experiments
if __name__ == '__main__':
    make_full_config()
