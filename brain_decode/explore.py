import os
import ipdb
import logging
import numpy as np
from collections import OrderedDict
from util.bbci import BBCIDataset
from util.trial_segment import create_signal_target_from_raw_mne
from util.signalproc import resample_cnt, mne_apply, highpass_cnt
from util.signalproc import exponential_running_standardize
from util.splitters import split_into_two_sets
from util.sequential_segment import get_sequential_batches
log = logging.getLogger(__name__)
debug_here = ipdb.set_trace


def load_bbci_data(filename, low_cut_hz, debug=False):
    load_sensor_names = None
    if debug:
        load_sensor_names = ['C3', 'C4', 'C2']
    # we loaded all sensors to always get same cleaning results
    # independent of sensor selection
    # There is an inbuilt heuristic that tries to use only
    # EEG channels and that definitely
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
    cnt = resample_cnt(cnt, 256.0)  # 250
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


# Data Loading Parameters
low_cut_hz = 0
subject_id = 1

pd = {}
# Stft Parameters
pd["window_type"] = "learned_gaussian"
pd["window_size"] = 128
pd["overlap"] = 120

# Conv Layers
pd["filters"] = [4]
pd["kernel"] = [3]

# Train Data Parameters
pd["magnitude_only"] = True
pd["data_augmentation"] = False
pd["spectogram_noise"] = False  # Add 3 more noisy spectograms to the training set
# Learning Parameters
pd["learning_rate"] = 0.001
pd["iterations"] = 100
pd["batch_size"] = 64
pd["test_batch_size"] = 64
pd["k_splits"] = 10

pd["device_name"] = "/gpu:0"

# Data Folder
data_folder = './data/'
train_filename = os.path.join(data_folder, 'train/{:d}.mat'.format(subject_id))
test_filename = os.path.join(data_folder, 'test/{:d}.mat'.format(subject_id))

# Create the dataset
train_set, valid_set, test_set = load_train_valid_test(train_filename=train_filename,
                                                       test_filename=test_filename,
                                                       low_cut_hz=low_cut_hz)

# Check number of samples
train_size = train_set.X.shape[0]
valid_size = valid_set.X.shape[0]
test_size = test_set.X.shape[0]
print("Train Samples :", train_size)
print("Validation Samples :", valid_size)
print("Test Samples :", test_size)

if 1:
    # test the tf stft.
    import matplotlib.pyplot as plt
    import scipy.signal as scisig
    import tensorflow as tf
    tf.enable_eager_execution()
    import sys
    sys.path.insert(0, "../")
    import eager_STFT as stft

    nperseg = 128
    overlap = int(nperseg * 0.5)
    win = tf.constant(scisig.get_window('hann', nperseg), dtype=tf.float32)
    freq_sig = stft.stft(train_set.X, window=win,
                         nperseg=nperseg, noverlap=overlap)
    plt.imshow(np.abs(freq_sig.numpy()[0, 0, :, :]))
    plt.show()
