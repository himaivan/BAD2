# BAD2
bird audio detection challenge 2

Basic usage:

This version uses Matlab for feature extraction.

./extracting_features.sh

Tensorflow for training 3D-CNN+RNN models.

Train with different initialization.

python cnn_3d_rnn.py 777
results would be in eval_final.csv

python cnn_3d_rnn.py 888

python cnn_3d_rnn.py 999


Average all results.
