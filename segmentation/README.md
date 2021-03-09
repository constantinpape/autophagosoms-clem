# Segmentation

Code for U-Net `training` for EM autophagosom segementation and `prediction` for large volumes.


## Installation

Set up the training conda environment and activate it via
```shell
conda env create -f environment.yaml
conda activate autophago-seg-env
pip install -e .
```

Note: you will need to choose a cuda version that is compatible with your GPU driver.
You can change the cuda version that is being installed via [this line](https://github.com/mobie/autophagosomes-clem-datasets/blob/master/network_training/environment.yaml#L8).
