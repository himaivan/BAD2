import numpy as np
import cPickle as pickle
import sys
import math
import os
import h5py
import argparse
from keras.utils import np_utils
from random import randint
from scipy.io import loadmat
from sklearn import preprocessing

from numpy.random import seed
seed(1)

nFrames = 5
nFeatures = 80
nTime = 200

from bad_3d_2sec import prep_data1, prep_data2, prep_data3, prep_cv1, prep_cv2, prep_cv3

def data_preparation(batch_size):
   # training data
   featname = 'melspec'
   x_t1, tn1, Y_t1 = prep_data1(featname)
   x_t2, tn2, Y_t2 = prep_data2(featname)
   x_t3, tn3, Y_t3 = prep_data3(featname)

   # randomized the order
   x_train = np.vstack([x_t1,x_t2,x_t3])
   Y_train = np.vstack([Y_t1,Y_t2,Y_t3])
   np.random.seed(1)

   mytrain = np.asarray(x_train).reshape(-1, nFrames, nFeatures, nTime, 1)

   ord = np.random.permutation(len(mytrain))
   XX = [mytrain[i] for i in ord]
   YY = [Y_train[i] for i in ord]

   num = int(round(len(mytrain)*0.97)) # use approx. 97% of the training data.
   trX = np.asarray(XX[0:num])
   trY = YY[0:num]
   teX = np.asarray(XX[num:])
   teY = YY[num:]

   training_batch = zip(range(0, len(trX), batch_size), range(batch_size, len(trX) + 1, batch_size))
   ntotal = len(training_batch)
   
   datadir = 'train_b'+str(batch_size)

   if not os.path.exists(datadir):                                                                             
      os.makedirs(datadir)

   my_c_t = 0
   for start, end in training_batch:
      np.save(datadir+'/'+'mydata.'+str(my_c_t),trX[start:end])
      np.save(datadir+'/'+'mylabel.'+str(my_c_t),trY[start:end])
      my_c_t = my_c_t + 1

   np.save('trX',trX)
   np.save('trY',trY)
   np.save('teX',teX)
   np.save('teY',teY)
   
   return ntotal
