# -*- coding: utf-8 -*-
"""
Created on Wed May 15 14:54:39 2019

@author: gsoum
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 10 20:21:02 2019

@author: gsoum
"""

import logging
import sys
import os.path
from collections import OrderedDict
import numpy as np

from braindecode.datasets.bbci import  BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.torch_ext.util import set_random_seeds
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet
from braindecode.models.util import to_dense_prediction_model
from braindecode.experiments.experiment import Experiment
from braindecode.torch_ext.util import np_to_var
from braindecode.datautil.iterators import CropsFromTrialsIterator,ClassBalancedBatchSizeIterator
from braindecode.experiments.stopcriteria import MaxEpochs, NoDecrease, Or
from braindecode.torch_ext.constraints import MaxNormDefaultConstraint
from braindecode.experiments.monitors import LossMonitor, MisclassMonitor, \
    RuntimeMonitor, CroppedTrialMisclassMonitor

from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize

import tensorflow as tf
import eager_STFT as eagerSTFT
from scipy import signal
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel('DEBUG')


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results independent of sensor selection
    # There is an inbuilt heuristic that tries to use only EEG channels and that definitely
    # works for datasets in our paper
    loader = BBCIDataset(filename, load_sensor_names=load_sensor_names)

    log.info("Loading data...")
    cnt = loader.load()

    # Cleaning: First find all trials that have absolute microvolt values
    # larger than +- 800 inside them and remember them for removal later
    log.info("Cutting trials...")

    marker_def = OrderedDict([('Right Hand', [1]), ('Left Hand', [2],),
                              ('Rest', [3]), ('Feet', [4])])
    clean_ival = [0, 4000]

    set_for_cleaning = create_signal_target_from_raw_mne(cnt, marker_def,
                                                  clean_ival)

    clean_trial_mask = np.max(np.abs(set_for_cleaning.X), axis=(1, 2)) < 800

    log.info("Clean trials: {:3d}  of {:3d} ({:5.1f}%)".format(
        np.sum(clean_trial_mask),
        len(set_for_cleaning.X),
        np.mean(clean_trial_mask) * 100))

    # now pick only sensors with C in their name
    # as they cover motor cortex
    C_sensors = ['FC5', 'FC1', 'FC2', 'FC6', 'C3', 'C4', 'CP5',
                 'CP1', 'CP2', 'CP6', 'FC3', 'FCz', 'FC4', 'C5', 'C1', 'C2',
                 'C6',
                 'CP3', 'CPz', 'CP4', 'FFC5h', 'FFC3h', 'FFC4h', 'FFC6h',
                 'FCC5h',
                 'FCC3h', 'FCC4h', 'FCC6h', 'CCP5h', 'CCP3h', 'CCP4h', 'CCP6h',
                 'CPP5h',
                 'CPP3h', 'CPP4h', 'CPP6h', 'FFC1h', 'FFC2h', 'FCC1h', 'FCC2h',
                 'CCP1h',
                 'CCP2h', 'CPP1h', 'CPP2h']
    if debug:
        C_sensors = load_sensor_names
    cnt = cnt.pick_channels(C_sensors)

    # Further preprocessings as descibed in paper
    log.info("Resampling...")
    cnt = resample_cnt(cnt, 250.0)
    log.info("Highpassing...")
    cnt = mne_apply(
        lambda a: highpass_cnt(
            a, low_cut_hz, cnt.info['sfreq'], filt_order=3, axis=1),
        cnt)
    log.info("Standardizing...")
    cnt = mne_apply(
        lambda a: exponential_running_standardize(a.T, factor_new=1e-3,
                                                  init_block_size=1000,
                                                  eps=1e-4).T,
        cnt)

    # Trial interval, start at -500 already, since improved decoding for networks
    ival = [-500, 4000]

    dataset = create_signal_target_from_raw_mne(cnt, marker_def, ival)
    dataset.X = dataset.X[clean_trial_mask]
    dataset.y = dataset.y[clean_trial_mask]
    return dataset


def load_train_valid_test(
        train_filename, test_filename, low_cut_hz, debug=False):
    log.info("Loading train...")
    full_train_set = load_bbci_data(
        train_filename, low_cut_hz=low_cut_hz, debug=debug)

    log.info("Loading test...")
    test_set = load_bbci_data(
        test_filename, low_cut_hz=low_cut_hz, debug=debug)
    valid_set_fraction = 0.8
    train_set, valid_set = split_into_two_sets(full_train_set,
                                               valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set

if __name__ == "__main__":
    
    input_time_length = 1125
    low_cut_hz = 0
    
    subject_id = 1
    
    pd = {}
    pd["window_type"] = "hann"
    pd["window_size"] = 64
    pd["overlap"] = 56
    
    #Conv Layer
    pd["filters"] = [4]
    pd["kernel"] = [3]
    
    pd["magnitude_only"] = True
    
    #Optimizer Parameters
    pd["learning_rate"] = 0.001
    weight_decay = 0
    #Learning Parameters
    pd["iterations"] = 250
    pd["batch_size"] = 256
    pd["test_batch_size"] = 126
    
    #SLiding window parameters for train
    pd["sliding_window_size"] = 1000
    pd["shift"] = 1
    
    pd["repeats"] = 126
    
    pd["device_name"] = "/gpu:0"

    data_folder = './data/'
    train_filename =  os.path.join(data_folder, 'train/{:d}.mat'.format(subject_id))
    test_filename =  os.path.join(data_folder, 'test/{:d}.mat'.format(subject_id))
    
    brain_eeg = tf.Graph()
    with brain_eeg.as_default():
        
        with tf.device(pd['device_name']):

            #Dataset preparation
            train_placeholder_x = tf.placeholder(tf.float32 , shape = [None,44,1000])
            train_placeholder_y = tf.placeholder(tf.int32 , shape = [None])
            
            test_placeholder_x = tf.placeholder(tf.float32,shape = [None,44,1000])
            test_placeholder_y = tf.placeholder(tf.int32,shape = [None])
            
            train_dataset = tf.data.Dataset.from_tensor_slices((train_placeholder_x, train_placeholder_y)).shuffle(500).batch(pd["batch_size"])
            test_dataset = tf.data.Dataset.from_tensor_slices((test_placeholder_x, test_placeholder_y)).batch(pd["test_batch_size"])

            train_iter = train_dataset.make_initializable_iterator()
            test_iter = test_dataset.make_initializable_iterator()
            
            train_x, train_y = train_iter.get_next()
            test_x , test_y = test_iter.get_next()
            
            train_x_shape = tf.shape(train_x)
            test_x_shape = tf.shape(test_x)
                        
            #Train the network
            window = tf.constant(signal.get_window(pd["window_type"], pd["window_size"]),dtype=tf.float32)
            train_set_x = eagerSTFT.stft(train_x , window , pd["window_size"] , pd["overlap"])
            train_set_x = tf.transpose(train_set_x , perm = [0,3,2,1])
            
            if(pd["magnitude_only"] == True):
                train_set_x = tf.abs(train_set_x)
            else:
                train_set_x = tf.concat([ tf.abs(train_set_x) , tf.angle(train_set_x) ] , axis = 3)
            print(train_set_x.shape)
            
            #Adding noise to train set
            noise1 = tf.keras.layers.GaussianNoise(stddev=0.3)
            #train_set_x_noise1 = noise1(train_set_x)
            #train_set_x_noise2 = noise1(train_set_x)
            #train_set_x_noise3 = noise1(train_set_x)
            #train_set_x = tf.concat([train_set_x,train_set_x_noise1,train_set_x_noise2,train_set_x_noise3],axis=0)
            
            with tf.variable_scope("eeg_network" , reuse=tf.AUTO_REUSE):
                
                conv1 = tf.keras.layers.Conv2D(pd["filters"][0] , pd["kernel"][0],kernel_regularizer=tf.keras.regularizers.l2(l=0.001))
                dropout = tf.keras.layers.Dropout(rate=0.5)
                batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)
                elu = tf.keras.layers.Activation("elu")
                maxpool1 = tf.keras.layers.MaxPool2D(2)
                
                #conv2 = tf.keras.layers.Conv2D(pd["filters"][1] , pd["kernel"][1] , kernel_regularizer = tf.keras.regularizers.l2(l=0.01))
                #batch_norm2 = tf.keras.layers.BatchNormalization(axis=-1)
                #maxpool2 = tf.keras.layers.MaxPool2D(2)
                
                flatten = tf.keras.layers.Flatten()
                dense1 = tf.keras.layers.Dense(4,activation=tf.keras.activations.softmax,kernel_regularizer=tf.keras.regularizers.l2(l=0.001))
                
                conv1_output = dropout(conv1(train_set_x))
                elu1_output = elu(batch_norm1(conv1_output , training = True))
                maxpool1_output = maxpool1(elu1_output)
                #conv2_output = conv2(maxpool_output)
                #elu2_output = elu(batch_norm2(conv2_output , training = True))
                #maxpool2_output = maxpool2(elu2_output)
                flatten_output = flatten(maxpool1_output)
                print(flatten_output)
                dense1_output = dense1(flatten_output)
                
            y_true = train_y
            #y_true = tf.concat([y_true,y_true,y_true,y_true] , axis=0)

            CrossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()
            train_loss = CrossEntropyLoss(y_true,dense1_output)
            
            m = tf.keras.metrics.CategoricalAccuracy()
            train_accuracy = tf.math.reduce_sum(tf.keras.metrics.categorical_accuracy(tf.one_hot(y_true,4),dense1_output))
            
            #Test accuracy 
            test_set_x = eagerSTFT.stft(test_x , window , pd["window_size"] , pd["overlap"])
            test_set_x = tf.transpose(test_set_x , perm = [0,3,2,1])
            
            if(pd["magnitude_only"] == True):
                test_set_x = tf.abs(test_set_x)
            else:
                test_set_x = tf.concat([ tf.abs(test_set_x) , tf.angle(test_set_x) ] , axis = 3)
            
            y_test = test_y
            
            with tf.variable_scope("eeg_network" , reuse=tf.AUTO_REUSE):
                conv1_test_output = conv1(test_set_x) 
                elu1_test_output = elu(batch_norm1(conv1_test_output , training = False))
                maxpool1_test_output = maxpool1(elu1_test_output)
                #conv2_test_output = conv2(maxpool_test_output)
                #elu2_test_output = elu(batch_norm2(conv2_test_output , training = False))
                #maxpool2_test_output = maxpool2(elu2_test_output)
                flatten_test_output = flatten(maxpool1_test_output)   
                dense1_test_output = dense1(flatten_test_output)
            
            test_loss = CrossEntropyLoss(y_test,dense1_test_output)
            
            #Compute mean of all predictions
            test_index = tf.argmax(dense1_test_output, axis=1)
            test_mean_prediction = tf.round(tf.math.reduce_mean(test_index))
            test_accuracy = tf.math.reduce_sum(tf.keras.metrics.categorical_accuracy(tf.one_hot(y_test[0],4),tf.one_hot(test_mean_prediction,4)))
            #####
            
            #Train Optimizer
            train_step = tf.train.AdamOptimizer(pd['learning_rate']).minimize(train_loss + tf.losses.get_regularization_loss())

        #tf.summary.scalar("train_loss",train_loss)
        #tf.summary.scalar("test_loss",test_loss)
        #tf.summary.scalar("train_acc",train_accuracy)
        #tf.summary.scalar("test_acc",test_accuracy)
        #merged = tf.summary.merge_all()
        
            
    with tf.Session(graph=brain_eeg,config=tf.ConfigProto(log_device_placement=True,allow_soft_placement=True)) as sess:
    
        train_set, valid_set, test_set = load_train_valid_test(train_filename=train_filename,test_filename=test_filename,low_cut_hz=low_cut_hz)
        
        train_set_x_list = []
        
        for i in range(train_set.X.shape[-1]-pd["sliding_window_size"]+1):
            train_set_x_list.append(train_set.X[:,:,i*pd["shift"]:i+pd["sliding_window_size"]])
            
        train_set_x = np.concatenate(train_set_x_list)  
        train_set_y = np.tile(train_set.y,reps=train_set.X.shape[-1]-pd["sliding_window_size"]+1)
        
        print(train_set_x.shape)
        print(train_set_y.shape)
        
        
        test_set_x_list = []
        for i in range(test_set.X.shape[0]):
            test_set_item = []
            for j in range(test_set.X.shape[-1]-pd["sliding_window_size"]+1):
                test_set_item.append(np.expand_dims(test_set.X[i,:,j*pd["shift"]:j+pd["sliding_window_size"]] , axis=0))
            
            test_set_x_list.append(np.concatenate(test_set_item))
            
        test_set_x = np.concatenate(test_set_x_list)
        test_set_y = np.repeat(test_set.y,repeats=test_set.X.shape[-1]-pd["sliding_window_size"]+1)
        
        print(test_set_x.shape)
        print(test_set_y.shape)
        
        train_size = train_set.X.shape[0]
        test_size = test_set.X.shape[0]
        
        print(train_size)
        print(test_size)
        
        tf.global_variables_initializer().run() 
        #summary_writer = tf.summary.FileWriter("./logs",sess.graph)
        
        for epoch in range(pd["iterations"]):
            sess.run([train_iter.initializer,test_iter.initializer], feed_dict={ train_placeholder_x: train_set_x, train_placeholder_y: train_set_y,test_placeholder_x: test_set_x, test_placeholder_y:test_set_y})
            #sess.run(test_iter.initializer, feed_dict={ test_placeholder_x: test_set.X, test_placeholder_y: test_set.y})
            
            print("Iteration : ", epoch)
            epoch_train_loss = 0
            epoch_test_loss = 0
            epoch_train_accuracy = 0
            epoch_test_accuracy = 0
            
            batch = 0
            
            while True:
                try:
                    _train_step = sess.run([train_step])      #Do the optimization alone for the epoch         
                    #print("Batch : " , batch)
                    #batch += 1
                    
                except tf.errors.OutOfRangeError:
                    #Calculate test and train loss and accuracy 
                    sess.run(train_iter.initializer, feed_dict={ train_placeholder_x: train_set_x, train_placeholder_y: train_set_y})
                    
                    while True:
                        try:
                            _train_loss , _train_accuracy, _train_x_shape = sess.run([train_loss,train_accuracy,train_x_shape]) 
                            epoch_train_loss += _train_loss
                            epoch_train_accuracy += _train_accuracy
                        except tf.errors.OutOfRangeError:
                            break
                        
                    while True:      
                        try:
                            _test_loss , _test_accuracy , _test_x_shape = sess.run([test_loss,test_accuracy,test_x_shape])
                            epoch_test_loss += _test_loss
                            epoch_test_accuracy += _test_accuracy
                        except tf.errors.OutOfRangeError:
                            break
                        
                    break
                
            print("Train Loss : " , epoch_train_loss)
            print("Train Accuracy : ", epoch_train_accuracy/(pd["repeats"]*train_size))
            print("Test Loss : " , epoch_test_loss)
            print("Test Accuracy : " ,epoch_test_accuracy/test_size)
            
''' 
        for i in range(train_set_x.shape[-1]):
            fig2 = plt.figure(1,figsize=(16, 12))
            ax1 = fig2.add_subplot(211)
            ax1.imshow(train_spectogram[56,:,:,i], cmap='hot', interpolation='nearest',origin='lower')
            ax2 = fig2.add_subplot(212)
            ax2.imshow(test_spectogram[54,:,:,i], cmap='hot', interpolation='nearest',origin='lower')
            
            fig2.savefig("./spectogram/" + str(i) + ".png")
            fig2.clf()
'''
        


    