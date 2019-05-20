import os
import ipdb
import time
import pickle
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import OrderedDict
from cnn_label_graph import SingleLabelClassificationGraph
from train_helper import pd_to_string, load_train_valid_test

log = logging.getLogger(__name__)
debug_here = ipdb.set_trace


#  define the experiment parameters.

pd = {}
# Stft Parameters
pd['window_function'] = 'learned_gaussian'
pd["window_size"] = 158
pd["overlap"] = int(pd['window_size']*0.83)
pd['fft_compression_rate'] = 0

# CNN parameters
pd['label_total'] = 4
pd['cnn_layers'] = [{'f': (128),
                     'k': (4, 4),
                     's': (1, 2)},
                    {'f': (128),
                     'k': (3, 3),
                     's': (2, 2)},
                    {'f': (256),
                     'k': (3, 3),
                     's': (2, 2)},
                    {'f': (512),
                     'k': (3, 3),
                     's': (2, 2)},
                    {'f': (512),
                     'k': (3, 3),
                     's': (1, 1)}]

pd['dense_units'] = [1028]
pd['dropout'] = 0.8
pd['input_dropout'] = 0.8

# Train Data Parameters
pd['magnitude_only'] = True
pd['magnitude_and_phase'] = False
pd['channels'] = 44
# Learning Parameters
pd["learning_rate"] = 0.001
pd["epochs"] = 50
pd["batch_size"] = 100
pd['learning_rate_decay'] = None
pd['decay_steps'] = None

# Data Loading Parameters
low_cut_hz = 0
subject_ids = 14

pd['data_folder'] = './data/'
pd['log_folder'] = './log/cnn_drop_2/'

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

val_acc_lst = []
with tf.Session(graph=cgraph.graph) as sess:
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
            loss, out, labels, _, np_global_step, lr_summary = \
                sess.run([cgraph.loss, cgraph.sig_out, cgraph.labels_one_hot,
                          cgraph.weight_update, cgraph.global_step,
                          cgraph.learning_rate_summary],
                         feed_dict=feed_dict)
            acc = np.mean(y_in == np.argmax(out, axis=-1))*100
            acc = np.round(acc, 1)
            loss = np.round(loss, 6)
            if i % 25 == 0:
                print('epoch', e, 'it', i, 'loss', loss, 'accuracy', acc)
            summary_writer.add_summary(lr_summary, global_step=np_global_step)
            train_acc_summary = tf.Summary.Value(tag='train_acc', simple_value=acc)
            train_acc_summary = tf.Summary(value=[train_acc_summary])
            summary_writer.add_summary(train_acc_summary, global_step=np_global_step)

        # validation round
        val_y_total = None
        val_out_max = None
        for vi in range(0, val_iterations):
                x_val_in = val_x[vi]
                y_val_in = val_y[vi]

                feed_dict = {cgraph.data: x_val_in,
                             cgraph.labels: y_val_in}
                out, np_global_step, freq_abs_summary = \
                    sess.run([cgraph.sig_out_val, cgraph.global_step,
                              cgraph.freq_abs_summary],
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
        acc_summary = tf.Summary.Value(tag='val_acc', simple_value=val_acc)
        acc_summary = tf.Summary(value=[acc_summary])
        summary_writer.add_summary(acc_summary, global_step=np_global_step)
        # summary_writer.add_summary(freq_abs_summary, global_step=np_global_step)

    # test things.
    # TODO.
