Copyright (c) 2018 Queensland University of Technology, written by Ivan Himawan <i.himawan@qut.edu.au>

# This repository contains 3D-CNN+RNN implementation for the second edition of bird audio detection challenge 2.

Detail of the environment is in req.txt (generated using 'pip freeze')

Quick readme:

This version uses Matlab for feature extraction. For this work, we used log Mel-spectrogram as features. Alternatively, you may used other python libraries, i.e., librosa for feature extraction if you have no access to Matlab.
For example,
```
import librosa
y,nsr = librosa.load(filename, sr=44100)
C = librosa.feature.melspectrogram(y, sr=nsr, None, n_fft=1024, hop_length=512, power=1.0)
C = librosa.core.amplitude_to_db(C)
```
The csv files are slightly modified (i.e., removing the header) to simplify the training/testing process.

Running feature extraction:
```
./extracting_features.sh
```

Tensorflow for training 3D-CNN+RNN models.
Training with different initialization:
```
python cnn_3d_rnn.py 777
```
results would be in eval_final.csv
You may run with different seed that give random weights initialization,
```
python cnn_3d_rnn.py 888
python cnn_3d_rnn.py 999
```
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
```bash
python cnn_3d_rnn.py 777
```
Model ensemble can be performed by running several instances of model training (i.e., using different seed), and average the predictions.

# Citation
If you used this code please kindly cite the following paper:
```
@InProceedings{himawan2018,
  Title                    = {3D convolution recurrent neural networks for bird sound detection},
  Author                   = {Himawan, Ivan and Towsey, Michael and Roe, Pau},
  Booktitle                = {Workshop on on Detection and Classification of Acoustic Scenes and Events},
  Year                     = {2018},
}
```
# License
To apply the Apache License to your work, attach the following boilerplate notice, with the fields enclosed by brackets "[]" replaced with your own identifying information. (Don't include the brackets!) The text should be enclosed in the appropriate comment syntax for the file format. We also recommend that a file or class name and description of purpose be included on the same "printed page" as the copyright notice for easier identification within third-party archives.
```
Copyright [2018] [Ivan Himawan, Queensland University of Technology]

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
# Contribution
We appreciate your kind feedback. Please try our code, and help us with code inspections to improve our work.
