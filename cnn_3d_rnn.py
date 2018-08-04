#(c) 2018 Ivan Himawan, Queensland University of Technology

import tensorflow as tf
import numpy as np
import cPickle as pickle
import sys
import math
import os
import h5py

from scipy.io import loadmat
from sklearn import preprocessing

batch_size = 8
nFeatures = 80
nTime = 200
nFrames = 5

flags=False
training=True
testing=True

init = sys.argv[1]
seed = int(init)
print seed

tf.reset_default_graph()
tf.set_random_seed(seed)

featname = 'melspec'
directory = featname+'_mrnn_gpu'

from bad_3d_2sec import prep_data1, prep_data2, prep_data3, prep_cv1, prep_cv2, prep_cv3
from prepare_train import data_preparation

# Conv Nets
def weight_variable_conv(shape):
    initial = tf.contrib.layers.xavier_initializer_conv2d(seed=seed)
    return tf.Variable(initial(shape))

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.001))

def bias_variable(shape):
    initial = tf.constant_initializer()
    return tf.Variable(initial(shape))

gru_size = 32

def RNN(X_in, _name):

    _H = X_in
    batch_size_T = tf.shape(_H)[0]
    with tf.variable_scope(_name):
       cell = tf.contrib.rnn.GRUCell(gru_size)
       _GRU_O, _GRU_S = tf.nn.dynamic_rnn(cell, _H, dtype=tf.float32)

    return _GRU_O[:,-1,:]

def model(x, w_fc1, b_fc1, keep_prob_conv, keep_prob_fc, isx):

    h_conv1 = tf.layers.conv3d(inputs=x, filters=8, kernel_size=[3,3,3], padding='SAME', use_bias=False, activation=None)
    h_conv1 = tf.layers.batch_normalization(inputs=h_conv1, training=isx)
    h_conv1 = tf.nn.relu(h_conv1)
    h_conv1_pool = tf.layers.max_pooling3d(inputs=h_conv1, pool_size=[1,2,2], strides=[1,2,2], padding='SAME')
    h_conv1_drop = tf.layers.dropout(inputs=h_conv1_pool, rate=keep_prob_fc, training=isx)
    print(h_conv1_drop.get_shape())
    h_conv2 = tf.layers.conv3d(inputs=h_conv1_drop, filters=32, kernel_size=[3,3,3], padding='SAME', use_bias=False, activation=None)
    h_conv2 = tf.layers.batch_normalization(inputs=h_conv2, training=isx)
    h_conv2 = tf.nn.relu(h_conv2)
    h_conv2_pool = tf.layers.max_pooling3d(inputs=h_conv2, pool_size=[2,2,2], strides=2, padding='SAME')
    h_conv2_drop = tf.layers.dropout(inputs=h_conv2_pool, rate=keep_prob_fc, training=isx)
    print(h_conv2_drop.get_shape())
    h_conv3 = tf.layers.conv3d(inputs=h_conv2_drop, filters=16, kernel_size=[3,3,3], padding='SAME', use_bias=False, activation=None)
    h_conv3 = tf.layers.batch_normalization(inputs=h_conv3, training=isx)
    h_conv3 = tf.nn.relu(h_conv3)
    h_conv3_pool = tf.layers.max_pooling3d(inputs=h_conv3, pool_size=[3,2,2], strides=[3,2,2], padding='SAME')
    h_conv3_drop = tf.squeeze(tf.layers.dropout(inputs=h_conv3_pool, rate=keep_prob_fc, training=isx),1)
    print(h_conv3_drop)
    h_conv3_features = tf.unstack(h_conv3_drop, axis=3)

    print(h_conv3_drop.get_shape().as_list()[3])
    print(h_conv3_features[0])

    rnn_output = []
    for channel_index in range(h_conv3_drop.get_shape().as_list()[3]):
       name = "gru_"+str(channel_index)
       hf = RNN(tf.transpose(h_conv3_features[channel_index], [0, 2, 1]), name)

       rnn_output.append(hf)

    total_rnn = tf.concat(rnn_output,1)

    print(total_rnn)
    print(total_rnn.get_shape())

    h_fc1 = total_rnn

    return tf.matmul(h_fc1, w_fc1) + b_fc1

if __name__ == "__main__":

    X = tf.placeholder("float", [None, nFrames, nFeatures, nTime, 1])
    Y = tf.placeholder("float", [None, 2])
    isTrain = tf.placeholder(tf.bool, shape=())

    w_fc1 = weight_variable(shape=[gru_size * 16, 2])
    b_fc1 = bias_variable([2])

    p_keep_conv = tf.placeholder("float")
    p_keep_hidden = tf.placeholder("float")

    py_x = model(X, w_fc1, b_fc1, p_keep_conv, p_keep_hidden, isTrain)
    cost_softmax = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))

    cost = cost_softmax

    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
       train_op = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)
    predict_op = tf.argmax(py_x, 1)
    auc = tf.metrics.auc(tf.argmax(Y, 1), tf.nn.softmax(py_x)[:,1])

    if training:
        if not os.path.exists(directory):
              os.makedirs(directory)
        lfile = open(directory+'/log'+'_'+str(init)+'.txt',"w")
        cv_accs = []
        loss_func = []
        cv_loss = []
        saver = tf.train.Saver(max_to_keep=1000)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
      
        ntotal = data_preparation(batch_size)
        
        teX = np.load('teX.npy')
        teY = np.load('teY.npy')

        from random import shuffle
        lnum = range(ntotal)
        shuffle(lnum)
        datadir = 'train_b'+str(batch_size)

        saved_model_counter = 0
        for i in range(150):
            print("preparing batch")
            sess.run(tf.local_variables_initializer())
            loss_epoch = []
            cc = 0
            while cc < ntotal:
                #if cc % 100 == 0:
                #   print(cc)
                trainX = np.load(datadir+'/'+'mydata.'+str(lnum[cc])+'.npy')
                trainY = np.load(datadir+'/'+'mylabel.'+str(lnum[cc])+'.npy')
                _, loss_iter = sess.run([train_op, cost_softmax],
                                        feed_dict={X: trainX, Y: trainY,
                                                   p_keep_conv: 0.5, p_keep_hidden: 0.5, isTrain: True})
                loss_epoch.append(loss_iter)
                cc = cc + 1
            loss_func.append(np.mean(loss_epoch))
            pred = sess.run(predict_op, feed_dict={X: teX, p_keep_conv: 1.0, p_keep_hidden: 1.0, isTrain: False})
            test_accuracy = np.mean(np.argmax(teY, axis=1) == pred)
            test_auc, test_loss = sess.run([auc, cost_softmax], feed_dict={X: teX, Y: teY, p_keep_conv: 1.0, p_keep_hidden: 1.0, isTrain: False})

            print(i, loss_func[-1], test_accuracy, test_auc[1], test_loss)
            lfile.write("%s , %s , %s , %s, %s\n" % (str(i), str(loss_func[-1]), str(test_accuracy), str(test_auc[1]), str(test_loss)))
            saver.save(sess, directory+'/model1'+'_'+str(init), global_step=saved_model_counter)
            saved_model_counter += 1
            cv_accs.append(test_accuracy)
            cv_loss.append(test_loss)
        if not flags:
           saver.save(sess, directory+'/model1'+'_'+str(init), global_step=saved_model_counter)
           print("saved model %d: %f" % (saved_model_counter, test_accuracy))
           saved_model_counter += 1
        lfile.close()
    if testing:
        testidx = np.asarray(cv_accs).argsort()[-5:][::-1]
        testXx1, fnames1 = prep_cv1(featname)
        testXx2, fnames2 = prep_cv2(featname)
        testXx3, fnames3 = prep_cv3(featname)
        testXx = np.vstack((testXx1,testXx2,testXx3))
        fnames = fnames1 + fnames2 + fnames3

        testX = testXx.reshape(-1, nFrames, nFeatures, nTime, 1)

        total = np.zeros(len(testX))

        for i in range(0, len(testidx)):
           myindex = testidx[i]
           model_fname = directory+'/model1'+'_'+str(init)+'-'+str(myindex)
           print(model_fname)
           sess = tf.Session()
           new_saver = tf.train.Saver()
           new_saver.restore(sess, model_fname)

           logits = np.asarray([sess.run(py_x,
                                      feed_dict={X: testX[i, ][None, ], p_keep_conv: 1.0,
                                                 p_keep_hidden: 1.0, isTrain: False})
                             for i in range(len(testX))]).squeeze()
           probs = tf.nn.softmax(logits)
           test_probs = sess.run(probs)
           test_probs = test_probs[:, 1]
           total = total + test_probs
           
        for i, f in enumerate(fnames):
           final_fnames.append(f)

        res = np.divide(total, 5)
        with open(directory+'/eval_final.csv','w') as myfile:
           for i in range(len(final_fnames)):
              myfile.write("%s,%f\n"%(final_fnames[i].split('.wav')[0], res[i]))

        myfile.close()
