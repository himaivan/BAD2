Copyright (c) 2018 Queensland University of Technology, written by Ivan Himawan <i.himawan@qut.edu.au>

# This repository contains 3D-CNN+RNN implementation for the second edition of bird audio detection challenge 2.

Detail of the environment is in req.txt (generated using 'pip freeze')

Quick readme:

This version uses Matlab for feature extraction. The csv files are slightly modified (i.e., removing the header) to simplify the training/testing process.

./extracting_features.sh

Tensorflow for training 3D-CNN+RNN models.

Train with different initialization.

python cnn_3d_rnn.py 777
results would be in eval_final.csv

python cnn_3d_rnn.py 888

python cnn_3d_rnn.py 999


Average all results.

Readme:

Setting the Python environment

1. Download and installing conda.
download url: https://repo.continuum.io/miniconda/
```bash
bash Miniconda2-latest-Linux-x86_64.sh
```
2. Create virtual environment for python using conda.
```
./miniconda2/bin/conda create -n bad2
source activate bad2
pip install tensorflow-gpu==1.4.1
```

Feature extraction process

This version of feature extraction process use Matlab scripts to compute spectrogram.
```bash
./extracting_features.sh
```
Information regarding model training

Tensorflow is used to implement the deep architectures. Require tensorflow (>=1.4.0) with GPU support.
``bash
python cnn_3d_rnn.py 777
```
Model ensemble can be performed by running several instances of model training (i.e., using different seed), and average the predictions.
