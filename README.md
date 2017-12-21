# keras-conv1d

## Description

Code for training and evaluating 1D convolutional neural network with Keras.  Use multiple channels and filters to explore conv1d options for [HLS4ML project](https://github.com/hls-fpga-machine-learning/keras-training).

The example is for the raw, inertial signals of [this dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip).

## Setup

Install all dependencies using miniconda (from [HLS4ML project](https://github.com/hls-fpga-machine-learning/keras-training))):

Install `miniconda2` by sourcing `install_miniconda.sh` in your home directory. Log out and log back in after this.
```bash
cp install_miniconda.sh ~/
cd ~
source install_miniconda.sh
```

Install the rest of the dependencies:
```bash
cd ~/keras-conv1d
source install.sh
```

Each time you log in set things up:
```bash
source setup.sh
```

## Run
```
cd train
python keras_conv1d.py
```

