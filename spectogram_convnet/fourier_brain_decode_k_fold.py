# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:46:33 2019

@author: gsoum
"""
#pylint: disable=C0103
import logging
import os.path
from collections import OrderedDict
import numpy as np

from braindecode.datasets.bbci import BBCIDataset
from braindecode.datautil.signalproc import highpass_cnt
from braindecode.datautil.splitters import split_into_two_sets
from braindecode.datautil.trial_segment import \
    create_signal_target_from_raw_mne
from braindecode.mne_ext.signalproc import mne_apply, resample_cnt
from braindecode.datautil.signalproc import exponential_running_standardize

import tensorflow as tf
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import eager_STFT as eagerSTFT


log = logging.getLogger(__name__)
log.setLevel('DEBUG')
#np.random.seed(42)

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

    # Trial interval, start at -500 already, since improved decoding for
    # networks
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


'''
    Data Augmentation Functions
    Source - https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data
    DA_Scaling - Scales each point by random noise
    DA_Permutation - Segments TIme Series into windows and shifts each point randomly
'''

def DA_Scaling(X, sigma=0.3):
    scalingFactor = np.random.normal(
        loc=1.0, scale=sigma, size=(
            1, X.shape[1]))  # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
    return X * myNoise


def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile:
        segs = np.zeros(nPerm + 1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength,
                                               X.shape[0] - minSegLength, nPerm - 1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:] - segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii] + 1], :]
        X_new[pp:pp + len(x_temp), :] = x_temp
        pp += len(x_temp)
    return(X_new)


if __name__ == "__main__":

    # Data Loading Parameters
    low_cut_hz = 0
    subject_id = 1

    pd = {}
    # Stft Parameters
    pd["window_type"] = "hann"
    pd["window_size"] = 64
    pd["overlap"] = 56

    # Conv Layers
    pd["filters"] = [4]
    pd["kernel"] = [3]

    # Train Data Parameters
    pd["magnitude_only"] = True
    pd["data_augmentation"] = False
    # Add 3 more noisy spectograms to the training set with Gaussian Noise
    pd["spectogram_noise"] = False

    # Learning Parameters
    pd["learning_rate"] = 0.001
    pd["iterations"] = 150
    pd["batch_size"] = 128
    pd["test_batch_size"] = 128
    pd["k_splits"] = 10

    pd["device_name"] = "/gpu:0"

    # Data Folder
    data_folder = './data/'
    train_filename = os.path.join(
        data_folder, 'train/{:d}.mat'.format(subject_id))
    test_filename = os.path.join(
        data_folder, 'test/{:d}.mat'.format(subject_id))

    # Create the dataset
    train_set, valid_set, test_set = load_train_valid_test(
        train_filename=train_filename, test_filename=test_filename, low_cut_hz=low_cut_hz)

    # Check number of samples
    train_size = train_set.X.shape[0]
    valid_size = valid_set.X.shape[0]
    test_size = test_set.X.shape[0]
    print("Train Samples :", train_size)
    print("Validation Samples :", valid_size)
    print("Test Samples :", test_size)

    if(pd["data_augmentation"]):
        train_set_temp = np.transpose(np.concatenate(
            [train_set.X, valid_set.X]), axes=[0, 2, 1])
        train_set_scaled = []
        train_set_permuted = []
        for i in range(train_set_temp.shape[0]):
            train_set_scaled.append(np.expand_dims(DA_Scaling(train_set_temp[i]),
                    axis=0))
            train_set_permuted.append(
                np.expand_dims(
                    DA_Permutation(
                        train_set_temp[i]),
                    axis=0))

        train_set_scaled = np.concatenate(train_set_scaled)
        train_set_scaled = np.transpose(train_set_scaled, axes=[0, 2, 1])

        train_set_permuted = np.concatenate(train_set_permuted)
        train_set_permuted = np.transpose(train_set_permuted, axes=[0, 2, 1])

        train_set_X = np.concatenate([np.concatenate(
            [train_set.X, valid_set.X]), train_set_scaled, train_set_permuted])
        train_set_y = np.tile(np.concatenate(
            [train_set.y, valid_set.y]), reps=3)

    else:
        train_set_X = np.concatenate([train_set.X, valid_set.X])
        train_set_y = np.concatenate([train_set.y, valid_set.y])

    if(pd["k_splits"] != 0):
        skf = StratifiedKFold(n_splits=pd["k_splits"])
        train_index_list = []
        test_index_list = []
        for train_index, test_index in skf.split(train_set_X, train_set_y):
            train_index_list.append(train_index)
            test_index_list.append(test_index)

    brain_eeg = tf.Graph()
    with brain_eeg.as_default():

        # Dataset Loading
        train_placeholder_x = tf.placeholder(
            tf.float32, shape=[None, 44, 1125])
        train_placeholder_y = tf.placeholder(tf.int32, shape=[None])

        test_placeholder_x = tf.placeholder(
            tf.float32, shape=[None, 44, 1125])
        test_placeholder_y = tf.placeholder(tf.int32, shape=[None])

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (train_placeholder_x, train_placeholder_y)).shuffle(50).batch(
            pd["batch_size"])
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (test_placeholder_x, test_placeholder_y)).batch(
            pd["test_batch_size"])

        train_iter = train_dataset.make_initializable_iterator()
        test_iter = test_dataset.make_initializable_iterator()

        train_x, train_y = train_iter.get_next()
        test_x, test_y = test_iter.get_next()

        train_x_shape = tf.shape(train_x)[0]
        test_x_shape = tf.shape(test_x)[0]

        # Perform Train STFT
        window = tf.constant(
            signal.get_window(
                pd["window_type"],
                pd["window_size"]),
            dtype=tf.float32)
        train_set_x = eagerSTFT.stft(
            train_x, window, pd["window_size"], pd["overlap"])
        train_set_x = tf.transpose(train_set_x, perm=[0, 3, 2, 1])

        # Get the magnitude only
        if(pd["magnitude_only"]):
            train_set_x = tf.abs(train_set_x)
        else:
            train_set_x = tf.concat(
                [tf.abs(train_set_x), tf.angle(train_set_x)], axis=3)

        # Train labels
        y_true = train_y

        # Adding noise to train set
        if pd["spectogram_noise"]:
            noise1 = tf.keras.layers.GaussianNoise(stddev=0.3)
            train_set_x_noise1 = noise1(train_set_x)
            train_set_x_noise2 = noise1(train_set_x)
            train_set_x_noise3 = noise1(train_set_x)
            train_set_x = tf.concat([train_set_x,
                                        train_set_x_noise1,
                                        train_set_x_noise2,
                                        train_set_x_noise3],
                                    axis=0)
            y_true = tf.concat([y_true, y_true, y_true, y_true], axis=0)

        # Forward Train Pass , Define the Layers
        with tf.variable_scope("eeg_network", reuse=tf.AUTO_REUSE):

            conv1 = tf.keras.layers.Conv2D(pd["filters"][0],pd["kernel"][0],kernel_regularizer=tf.keras.regularizers.l2(l=0.01))
            dropout = tf.keras.layers.Dropout(rate=0.5)
            batch_norm1 = tf.keras.layers.BatchNormalization(axis=-1)
            elu = tf.keras.layers.Activation("elu")
            maxpool1 = tf.keras.layers.MaxPool2D([2,5])
            flatten = tf.keras.layers.Flatten()
            dense2 = tf.keras.layers.Dense(8192,activation=tf.keras.activations.linear,kernel_regularizer=tf.keras.regularizers.l2(l=0.001))
            maxpool2 = tf.keras.layers.MaxPool1D(2)
            dense1 = tf.keras.layers.Dense(4,activation=tf.keras.activations.softmax,kernel_regularizer=tf.keras.regularizers.l2(l=0.001))

            conv1_output = (train_set_x)
            elu1_output = elu(conv1_output)
            maxpool1_output = maxpool1(elu1_output)
            flatten_output = flatten(maxpool1_output)
            dense2_output = dense2(flatten_output)
            reshaped_output = tf.reshape(dense2_output,[-1,8192,1])
            maxpool2_output = maxpool2(reshaped_output)
            flatten2_output = flatten(maxpool2_output)
            dense1_output = dense1(dense2_output)
            
        # Define Loss , Calculate Train Loss
        CrossEntropyLoss = tf.keras.losses.CategoricalCrossentropy()
        train_loss = CrossEntropyLoss(y_true, dense1_output)

        # Train Optimizer
        train_step = tf.train.AdamOptimizer(pd['learning_rate']).minimize(train_loss + tf.losses.get_regularization_loss())

        # Calculate Train Accuracy
        m = tf.keras.metrics.CategoricalAccuracy()
        train_accuracy = tf.math.reduce_sum(tf.keras.metrics.categorical_accuracy(tf.one_hot(y_true,4), dense1_output))

        # Test STFT
        test_set_x = eagerSTFT.stft(test_x, window, pd["window_size"], pd["overlap"])
        test_set_x = tf.transpose(test_set_x, perm=[0, 3, 2, 1])

        if(pd["magnitude_only"]):
            test_set_x = tf.abs(test_set_x)
        else:
            test_set_x = tf.concat(
                [tf.abs(test_set_x), tf.angle(test_set_x)], axis=3)

        y_test = test_y

        # Test Pass
        with tf.variable_scope("eeg_network", reuse=tf.AUTO_REUSE):
            conv1_test_output = (test_set_x)
            elu1_test_output = elu(conv1_test_output)
            maxpool1_test_output = maxpool1(elu1_test_output)
            flatten_test_output = flatten(maxpool1_test_output)
            dense2_test_output = dense2(flatten_test_output)
            reshaped_test_output = tf.reshape(dense2_test_output,[-1,8192,1])
            maxpool2_test_output = maxpool2(reshaped_test_output)
            flatten2_test_output = flatten(maxpool2_test_output)
            dense1_test_output = dense1(dense2_test_output)

        # Calculate Test Loss and Accuracy
        test_loss = CrossEntropyLoss(y_test, dense1_test_output)
        test_accuracy = tf.math.reduce_sum(tf.keras.metrics.categorical_accuracy(tf.one_hot(y_test,4), dense1_test_output))

    with tf.Session(graph=brain_eeg, config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:

        split_train_loss = []
        split_test_loss = []
        split_train_accuracy = []
        split_test_accuracy = []

        #summary_writer = tf.summary.FileWriter("./logs",sess.graph)
        for train_index, test_index in zip(train_index_list, test_index_list):

            print("New Split")

            train_split_X = train_set_X[train_index]
            train_split_y = train_set_y[train_index]
            train_split_samples = train_split_X.shape[0]

            test_split_X = train_set_X[test_index]
            test_split_y = train_set_y[test_index]
            test_split_samples = test_split_X.shape[0]

            tf.global_variables_initializer().run()  # Reset after every epoch

            for epoch in range(pd["iterations"]):
                # Initialize train and test iterators
                sess.run([train_iter.initializer,
                          test_iter.initializer],
                         feed_dict={train_placeholder_x: train_split_X,
                                    train_placeholder_y: train_split_y,
                                    test_placeholder_x: test_split_X,
                                    test_placeholder_y: test_split_y})

                print("Iteration : ", epoch)
                epoch_train_loss = 0
                epoch_test_loss = 0
                epoch_train_accuracy = 0
                epoch_test_accuracy = 0

                batch = 0

                while True:
                    try:
                        # Do the optimization alone for the epoch
                        _train_step = sess.run([train_step])
                        #print("Batch : " , batch)
                        #batch += 1

                    except tf.errors.OutOfRangeError:
                        # Calculate Train loss and accuracy after each epoch
                        sess.run(
                            train_iter.initializer,
                            feed_dict={
                                train_placeholder_x: train_split_X,
                                train_placeholder_y: train_split_y})

                        while True:
                            try:
                                _train_loss, _train_accuracy = sess.run(
                                    [train_loss, train_accuracy])
                                epoch_train_loss += _train_loss
                                epoch_train_accuracy += _train_accuracy
                            except tf.errors.OutOfRangeError:
                                break
                        # Calculate Test Loss and Accuracy after each epoch
                        while True:
                            try:
                                _test_loss, _test_accuracy = sess.run(
                                    [test_loss, test_accuracy])
                                epoch_test_loss += _test_loss
                                epoch_test_accuracy += _test_accuracy
                            except tf.errors.OutOfRangeError:
                                break

                        break

                print("Train Loss : ", epoch_train_loss)
                print(
                    "Train Accuracy : ",
                    epoch_train_accuracy /
                    train_split_samples)
                print("Test Loss : ", epoch_test_loss)
                print(
                    "Test Accuracy : ",
                    epoch_test_accuracy /
                    test_split_samples)

            split_train_loss.append(epoch_train_loss)
            split_test_loss.append(epoch_test_loss)
            split_train_accuracy.append(
                epoch_train_accuracy / train_split_samples)
            split_test_accuracy.append(
                epoch_test_accuracy / test_split_samples)

        np.savetxt("./train_loss.csv",split_train_loss,delimiter=",")
        np.savetxt("./test_loss.csv",split_test_loss,delimiter=",")
        np.savetxt("./train_accuracy.csv",split_train_accuracy,delimiter=",")
        np.savetxt("./test_accuracy.csv",split_test_accuracy,delimiter=",")

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
