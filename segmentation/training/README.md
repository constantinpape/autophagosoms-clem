# Network Tranining

Train a U-Net for segmentation of autophagosomes.

## Usage


1. Check the training data visually by running `view_training_data.py` (needs graphical access)
2. In order to train a network based on the relatively sparse labels in the volume, we first find anchorpoints for extracting the training batches. For this, run the script `make_anchors.py`
3. Make the training config files by running `make_training_config.py` 

To train a network for semantic segmentation (i.e. to predict the different types of autophagosomes, but not individual autophagosomes) you can
then run the scropt `train_semantic.py`.
There are different options to run this script:

New training run:
```
python train_semantic.py /path/to/checkpoint config_name
```
At `/path/to/checkpoint` the script will create a folder that will hold the model and optimizer parameters as well as the tensorboard logs.
From this checkpoint the training can be restarted or the trained model can be loaded for predictions.
Note that if this checkpoint already exists it will be overridden!
The `config_name` denotes the name of the training / validation config that is loaded from the template_config` folder.
For example if the `config_name` is `fullV1`, the configs `template_config/train_fullV1.yaml` and `template_config/validation_fullV1.yaml` are used (these are the configs created by `make_training_config.py`, if nothing is changed in there).

Training from an exising checkpoint:
```
python train_semantic.py /path/to/checkpoint config_name --from_checkpoint 1
```


TODO explain how to start up tensorboard.
