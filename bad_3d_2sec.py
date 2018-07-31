import numpy as np
import cPickle as pickle
import sys
import math
import os
import h5py
import numpy as np
import argparse
from keras.utils import np_utils
from random import randint
from scipy.io import loadmat
from sklearn import preprocessing

from numpy.random import seed
seed(1)

nFeatures= 80
nTime = 200

def prep_data1(featname):
    samples = []

    f = h5py.File('data/ff1010bird_metadata.csv.mat')
    num_obj = len(f['myfeat'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['myfeat'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    #datatr1 = np.array([s.reshape(nFeatures,-1).reshape(-1) for s in samples])
    datatr1 = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])
    fnames1 = [x.split(',')[0].rstrip().split('/')[-1] for x in open('ff1010bird_metadata.csv').readlines()]
    classit = [x.split(',')[1].rstrip() for x in open('ff1010bird_metadata.csv').readlines()]
    # correct labels
    classes = np.array(classit)

    labeltr1 = np_utils.to_categorical(classes)
    print(labeltr1.shape)

    return datatr1, fnames1, labeltr1

def prep_data2(featname):
    samples = []

    f = h5py.File('data/warblrb10k_public_metadata.csv.mat')
    num_obj = len(f['myfeat'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['myfeat'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    #datatr2 = np.array([s.reshape(nFeatures,-1).reshape(-1) for s in samples])
    datatr2 = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])
    fnames2 = [x.split(',')[0].rstrip().split('/')[-1] for x in open('warblrb10k_public_metadata.csv').readlines()]
    classit = [x.split(',')[1].rstrip() for x in open('warblrb10k_public_metadata.csv').readlines()]

    # correct labels
    classes = np.array(classit)

    labeltr2 = np_utils.to_categorical(classes)
    print(labeltr2.shape)

    return datatr2, fnames2, labeltr2

def prep_data3(featname):
    samples = []

    f = h5py.File('BirdVox_label.csv.mat')
    num_obj = len(f['myfeat'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['myfeat'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    #datatr2 = np.array([s.reshape(nFeatures,-1).reshape(-1) for s in samples])
    datatr3 = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])
    fnames3 = [x.split(',')[0].rstrip().split('/')[-1] for x in open('BirdVox_label.csv').readlines()]
    classit = [x.split(',')[1].rstrip() for x in open('BirdVox_label.csv').readlines()]

    # correct labels
    classes = np.array(classit)

    labeltr3 = np_utils.to_categorical(classes)
    print(labeltr3.shape)

    return datatr3, fnames3, labeltr3

def prep_cv1(featname):
    samples = []

    f = h5py.File('data/wabrlrb10k.list.csv.mat')
    num_obj = len(f['myfeat'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['myfeat'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    #datacv = np.array([s.reshape(nFeatures,-1).reshape(-1) for s in samples])
    datacv = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])
    fnamescv = [x.split(',')[0].rstrip().split('/')[-1] for x in open('wabrlrb10k.list.csv').readlines()]

    return datacv, fnamescv

def prep_cv2(featname):
    samples = []

    f = h5py.File('data/poland.list.csv.mat')
    num_obj = len(f['myfeat'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['myfeat'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    #datacv = np.array([s.reshape(nFeatures,-1).reshape(-1) for s in samples])
    datacv = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])
    fnamescv = [x.split(',')[0].rstrip().split('/')[-1] for x in open('poland.list.csv').readlines()]

    return datacv, fnamescv

def prep_cv3(featname):
    samples = []

    f = h5py.File('data/chern.list.csv.mat')
    num_obj = len(f['myfeat'])
    #print num_obj
    for p in range(0,num_obj):
       #print  p
       mat = f['myfeat'][p][0]
       samples.append(np.array(f[mat]).T)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])
    filter_x = (shapes < nTime/2)
    for i in range(len(samples)):
       if filter_x[i]:
          # replicate the array x times first and then trim it
          rr = int(math.ceil(float(nTime)/shapes[i]))
          temp = samples[i].reshape(nFeatures,-1)
          tttt = np.tile(temp,rr)
          samples[i] = tttt.reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    shapes = np.array([s.shape[1] / nFeatures for s in samples])

    # pad the shortest
    filter_s = (shapes < nTime)
    for i in range(len(samples)):
       if filter_s[i]:
          temp = samples[i].reshape(nFeatures,-1)
          temo = samples[i].reshape(nFeatures,-1)[:,0:nTime-shapes[i]]
          samples[i] = np.hstack([temp,temo]).reshape(1,-1)

    filter_l = (shapes > nTime)
    for j in range(len(samples)):
       if filter_l[j]:
          samples[j] = samples[j].reshape(nFeatures,-1)[:,0:nTime].reshape(1,-1)

    #datacv = np.array([s.reshape(nFeatures,-1).reshape(-1) for s in samples])
    datacv = np.array([preprocessing.scale(s.reshape(nFeatures,-1)).reshape(-1)  for s in samples])
    fnamescv = [x.split(',')[0].rstrip().split('/')[-1] for x in open('chern.list.csv').readlines()]

    return datacv, fnamescv

