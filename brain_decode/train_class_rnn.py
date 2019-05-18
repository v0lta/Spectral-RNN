import os
import ipdb
import time
import pickle
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import OrderedDict
from rnn_single_label_classification_graph import SingleLabelClassificationGraph
from util.bbci import BBCIDataset
from util.trial_segment import create_signal_target_from_raw_mne
from util.signalproc import resample_cnt, mne_apply, highpass_cnt
from util.signalproc import exponential_running_standardize
from util.splitters import split_into_two_sets
from util.sequential_segment import get_sequential_batches
log = logging.getLogger(__name__)
debug_here = ipdb.set_trace


def pd_to_string(pd):
    pd_keys = pd.keys()
    param_str = ''
    for key in pd_keys:
        dict_el = pd[key]
        if type(dict_el) == bool:
            if dict_el:
                param_str += '_' + key
        elif type(dict_el) == str:
            param_str += '_' + dict_el
        else:
            param_str += '_' + key + '-' + str(dict_el)
    return param_str

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
    valid_set_fraction = 0.9
    train_set, valid_set = split_into_two_sets(full_train_set,
                                               valid_set_fraction)

    log.info("Train set with {:4d} trials".format(len(train_set.X)))
    if valid_set is not None:
        log.info("Valid set with {:4d} trials".format(len(valid_set.X)))
    log.info("Test set with  {:4d} trials".format(len(test_set.X)))

    return train_set, valid_set, test_set


#  define the experiment parameters.

pd = {}
# Stft Parameters
pd['window_function'] = 'learned_gaussian'
pd["window_size"] = 128
pd["overlap"] = int(pd['window_size']*0.5)
pd['fft_compression_rate'] = None

# RNN parameters
pd['label_total'] = 4
pd['num_units'] = 1024
pd['dense_units'] = 512

# Train Data Parameters
pd['magnitude_only'] = True
pd['channels'] = 44
# Learning Parameters
pd["learning_rate"] = 0.001
pd["epochs"] = 30
pd["batch_size"] = 50

pd["GPUs"] = [0]

# Data Loading Parameters
low_cut_hz = 0
subject_ids = 14

pd['data_folder'] = './data/'
pd['log_folder'] = './log/'

try:
    print('opening pickled version of the downsampled data.')
    train_sets, valid_sets, test_sets = pickle.load(
        open(pd['data_folder'] + 'full_downsampled.pkl', 'rb'))
except (OSError, IOError) as e:
    print('something went wrong', e)
    train_sets = []
    valid_sets = []
    test_sets = []
    for subject_id in range(1, subject_ids+1):
        # Data Folder

        train_filename = os.path.join(pd['data_folder'],
                                      'train/{:d}.mat'.format(subject_id))
        test_filename = os.path.join(pd['data_folder'],
                                     'test/{:d}.mat'.format(subject_id))

        # Create the dataset
        train_set, valid_set, test_set = load_train_valid_test(
            train_filename=train_filename, test_filename=test_filename,
            low_cut_hz=low_cut_hz)

        # Check number of samples
        train_size = train_set.X.shape[0]
        valid_size = valid_set.X.shape[0]
        test_size = test_set.X.shape[0]
        print("Train Samples :", train_size)
        print("Validation Samples :", valid_size)
        print("Test Samples :", test_size)
        train_sets.append(train_set)
        valid_sets.append(valid_set)
        test_sets.append(test_set)

    pickle.dump((train_sets, valid_sets, test_sets),
                open(pd['data_folder'] + "full_downsampled.pkl", "wb"))

train_X_full = []
train_y_full = []
val_X_full = []
val_y_full = []
test_X_full = []
test_y_full = []
for i in range(0, subject_ids):
    train_X_full.append(train_sets[i].X)
    train_y_full.append(train_sets[i].y)
    val_X_full.append(valid_sets[i].X)
    val_y_full.append(valid_sets[i].y)
    test_X_full.append(test_sets[i].X)
    test_y_full.append(test_sets[i].y)


train_X_full = np.concatenate(train_X_full, axis=0)
train_y_full = np.concatenate(train_y_full, axis=0)
val_X_full = np.concatenate(val_X_full, axis=0)
val_y_full = np.concatenate(val_y_full, axis=0)
test_X_full = np.concatenate(test_X_full, axis=0)
test_y_full = np.concatenate(test_y_full, axis=0)

assert train_X_full.shape[0] == train_y_full.shape[0]
assert val_X_full.shape[0] == val_y_full.shape[0]
assert test_X_full.shape[0] == test_y_full.shape[0]

pd['time'] = train_X_full.shape[-1]

# create the graph
cgraph = SingleLabelClassificationGraph(pd)
pd['parameter_total'] = cgraph.parameter_total

param_str = pd_to_string(pd)
print(param_str)

pd['time_str'] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
summary_writer = tf.summary.FileWriter(pd['log_folder'] + pd['time_str'] + param_str,
                                       graph=cgraph.graph)

# create a session and train dat thing
gpu_options = tf.GPUOptions(visible_device_list=str(pd['GPUs'])[1:-1])
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)

val_acc_lst = []
with tf.Session(graph=cgraph.graph, config=config) as sess:
    cgraph.init_op.run()
    train_size = train_X_full.shape[0]
    iterations = int(train_size/pd['batch_size'])

    valid_size = val_X_full.shape[0]
    if valid_size > 0:
        val_iterations = int(valid_size/pd['batch_size'])
        val_x = np.array_split(val_X_full, val_iterations, axis=0)
        val_y = np.array_split(val_y_full, val_iterations, axis=0)

    for e in range(0, pd['epochs']):
        # shuffle the train_pairs
        print('Shuffling ...')
        indices = np.array(range(0, train_size))
        np.random.shuffle(indices)
        train_x_shfl = []
        train_y_shfl = []
        for idx in indices:
            train_x_shfl.append(train_X_full[idx, :, :])
            train_y_shfl.append(train_y_full[idx])

        train_x_shfl = np.stack(train_x_shfl, axis=0)
        train_y_shfl = np.stack(train_y_shfl, axis=0)
        train_x = np.array_split(train_x_shfl, iterations, axis=0)
        train_y = np.array_split(train_y_shfl, iterations, axis=0)
        print('training_epoch', e)
        for i in range(0, iterations):
            x_in = train_x[i]
            y_in = train_y[i]

            feed_dict = {cgraph.data: x_in,
                         cgraph.labels: y_in}
            loss, out, labels, _ = \
                sess.run([cgraph.loss, cgraph.sig_out_center, cgraph.labels_one_hot,
                          cgraph.weight_update],
                         feed_dict=feed_dict)
            acc = np.mean(y_in == np.argmax(out, axis=-1))*100
            acc = np.round(acc, 1)
            loss = np.round(loss, 6)
            if i % 25 == 0:
                print('epoch', e, 'it', i, 'loss', loss, 'accuracy', acc)

        # validation round
        val_y_total = None
        val_out_max = None
        for vi in range(0, val_iterations):
                x_val_in = val_x[vi]
                y_val_in = val_y[vi]

                feed_dict = {cgraph.data: x_val_in,
                             cgraph.labels: y_val_in}
                out, np_global_step = \
                    sess.run([cgraph.sig_out_center, cgraph.global_step],
                             feed_dict=feed_dict)
                out_max = np.squeeze(np.argmax(out, axis=-1))
                if val_out_max is None:
                    val_out_max = out_max
                else:
                    val_out_max = np.concatenate([val_out_max, out_max], -1)

                if val_y_total is None:
                    val_y_total = y_val_in
                else:
                    val_y_total = np.concatenate([val_y_total, y_val_in], -1)
        assert (val_y_total == val_y_full).all()
        val_acc = np.mean(val_out_max == val_y_total)*100
        print('val acc', val_acc)
        acc_summary = tf.Summary.Value(tag='mse_net_test', simple_value=val_acc)
        acc_summary = tf.Summary(value=[acc_summary])
        summary_writer.add_summary(acc_summary, global_step=np_global_step)
