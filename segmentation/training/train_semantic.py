#! /g/kreshuk/pape/Work/software/conda/miniconda3/envs/torch13/bin/python

import os
import sys
import logging
import argparse
import yaml

from inferno.trainers.basic import Trainer
from inferno.trainers.callbacks.scheduling import AutoLR
from inferno.utils.io_utils import yaml2dict
from inferno.trainers.callbacks.essentials import SaveAtBestValidationScore
# from inferno.extensions.criteria import SorensenDiceLoss

from inferno.trainers.callbacks.logging.tensorboard import TensorboardLogger

import phago_network_utils.models as models
from phago_network_utils.criteria import RobustDiceLoss
from phago_network_utils.datasets import get_autophagosom_loader


logging.basicConfig(format='[+][%(asctime)-15s][%(name)s %(levelname)s]'
                           ' %(message)s',
                    stream=sys.stdout,
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def set_up_training(project_directory, config, data_config):

    # Get model
    model_name = config.get('model_name')
    model = getattr(models, model_name)(**config.get('model_kwargs'))

    loss = RobustDiceLoss()
    loss_train = loss_val = metric = loss

    # Build trainer and validation metric
    logger.info("Building trainer.")
    smoothness = 0.9

    trainer = Trainer(model)\
        .save_every((1000, 'iterations'),
                    to_directory=os.path.join(project_directory, 'Weights'))\
        .build_criterion(loss_train)\
        .build_validation_criterion(loss_val)\
        .build_optimizer(**config.get('training_optimizer_kwargs'))\
        .evaluate_metric_every('never')\
        .validate_every((100, 'iterations'), for_num_iterations=1)\
        .register_callback(SaveAtBestValidationScore(smoothness=smoothness,
                                                     verbose=True))\
        .build_metric(metric)\
        .register_callback(AutoLR(factor=0.99,
                                  patience='100 iterations',
                                  monitor_while='validating',
                                  monitor_momentum=smoothness,
                                  consider_improvement_with_respect_to='previous'))\

    logger.info("Building logger.")
    # Build logger
    tensorboard = TensorboardLogger(log_scalars_every=(1, 'iteration'),
                                    log_images_every=(100, 'iterations'),
                                    log_histograms_every='never').observe_states(
        ['validation_input', 'validation_prediction, validation_target'],
        observe_while='validating'
    )

    trainer.build_logger(tensorboard,
                         log_directory=os.path.join(project_directory, 'Logs'))
    return trainer


def load_checkpoint(project_directory):
    logger.info("Loading trainer from directory %s" % project_directory)
    trainer = Trainer().load(from_directory=project_directory,
                             filename='Weights/checkpoint.pytorch')
    return trainer


def training(project_directory,
             train_configuration_file,
             data_configuration_file,
             validation_configuration_file,
             max_training_iters=int(1e5),
             from_checkpoint=False):

    logger.info("Loading config from {}.".format(train_configuration_file))
    config = yaml2dict(train_configuration_file)

    logger.info("Loading training data loader from %s." % data_configuration_file)
    train_loader = get_autophagosom_loader(data_configuration_file)
    data_config = yaml2dict(data_configuration_file)

    logger.info("Loading validation data loader from %s." % validation_configuration_file)
    validation_loader = get_autophagosom_loader(validation_configuration_file)

    if from_checkpoint:
        trainer = load_checkpoint(project_directory)
    else:
        trainer = set_up_training(project_directory, config, data_config)
    trainer.set_max_num_iterations(max_training_iters)

    # Bind loader
    logger.info("Binding loaders to trainer.")
    trainer.bind_loader('train',
                        train_loader).bind_loader('validate',
                                                  validation_loader)

    # Set devices
    if config.get('devices'):
        logger.info("Using devices {}".format(config.get('devices')))
        trainer.cuda(config.get('devices'))
        # mixed precision loss scaling has issues with semantic training
        trainer.mixed_precision = True

    # Go!
    logger.info("Lift off!")
    trainer.fit()


# configuration for the network
def make_model_config(train_config_file, gpus):
    template = './template_config/model_config.yaml'

    # we predict 4 semantic channels:
    # foreground, r, g and r + g
    n_out = 4

    template = yaml2dict(template)
    template['model_kwargs']['out_channels'] = n_out
    template['devices'] = gpus
    with open(train_config_file, 'w') as f:
        yaml.dump(template, f)


# configuration for training data
def make_train_config(data_config_file, n_batches, name, only_new):
    if only_new:
        template = yaml2dict('./template_config/train_%s_only_new_labels.yaml' % name)
    else:
        template = yaml2dict('./template_config/train_%s.yaml' % name)
    template['master_config']['affinity_config'] = None
    template['master_config']['train_semantic'] = True
    template['loader_config']['batch_size'] = n_batches
    template['loader_config']['num_workers'] = 8 * n_batches
    with open(data_config_file, 'w') as f:
        yaml.dump(template, f)


# configuration for validation data
def make_validation_config(validation_config_file, name, only_new):
    if only_new:
        template = yaml2dict('./template_config/validation_%s_only_new_labels.yaml' % name)
    else:
        template = yaml2dict('./template_config/validation_%s.yaml' % name)
    template['master_config']['affinity_config'] = None
    template['master_config']['train_semantic'] = True
    with open(validation_config_file, 'w') as f:
        yaml.dump(template, f)


def copy_train_file(project_directory):
    from shutil import copyfile
    file_path = os.path.abspath(__file__)
    dst = os.path.join(project_directory, 'train.py')
    copyfile(file_path, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('project_directory', type=str)
    parser.add_argument('name', type=str)
    parser.add_argument('--only_new', type=int, default=1)
    parser.add_argument('--gpus', nargs='+', default=[0], type=int)
    parser.add_argument('--max_train_iters', type=int, default=int(1e5))
    parser.add_argument('--from_checkpoint', type=int, default=0)

    args = parser.parse_args()
    name = args.name
    only_new = bool(args.only_new)
    assert only_new

    project_directory = args.project_directory + "_%s" % name
    from_checkpoint = bool(args.from_checkpoint)
    os.makedirs(project_directory, exist_ok=True)

    gpus = list(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
    gpus = list(range(len(gpus)))

    model_config = os.path.join(project_directory, 'model_config.yml')
    train_config = os.path.join(project_directory, 'train.yml')
    validation_config = os.path.join(project_directory, 'validation.yml')

    batch_size = 1
    # only copy files if we DON'T load from checkponit
    if not from_checkpoint:
        n_batches = len(gpus) * batch_size
        make_model_config(model_config, gpus)
        make_train_config(train_config, n_batches, name, only_new)
        make_validation_config(validation_config, name, only_new)
        copy_train_file(project_directory)

    training(project_directory,
             model_config,
             train_config,
             validation_config,
             max_training_iters=args.max_train_iters,
             from_checkpoint=from_checkpoint)


if __name__ == '__main__':
    main()
