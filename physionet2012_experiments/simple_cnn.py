import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.losses as losses
import tensorflow.train as train
import matplotlib.pyplot as plt

import physionet_handler
from IPython.core.debugger import Tracer
debug_here = Tracer()


def compute_parameter_total(trainable_variables):
    total_parameters = 0
    for variable in trainable_variables:
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print('var_name', variable.name, 'shape', shape, 'dim', len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print('parameters', variable_parameters)
        total_parameters += variable_parameters
    print('total:', total_parameters)
    return total_parameters


# TODO: Add imputation and STFT.
# TODO: Fix preprocessing.
# TODO: add everyone survives baseline.

# network parameters
epochs = 61
batch_size = 100
quantity_no = 42
activation = layers.ReLU()
learning_rate = 0.0005

train_set = './data/set-a/'
label_file = './data/set-a_outcome.txt'
val_set = './data/set-b/'
val_label_file = './data/set-b_outcome.txt'

filters = [15, 20, 25, 30]
kernel_size = [3, 3, 3, 3]
assert len(filters) == len(kernel_size)
strides = [1, 1, 1, 1, 1]
pool = [3, 3, 3, 3, 1]
dropout_rate = 0.95
padding = 'valid'
data_format = 'channels_last'

layer_lst = []
for layer_no in range(0, len(filters)):
    layer_lst.append(layers.Conv1D(filters[layer_no], kernel_size[layer_no],
                                   strides[layer_no], padding, data_format,
                                   activation=activation))
    layer_lst.append(layers.Dropout(dropout_rate))
    if pool[layer_no] != 1:
        layer_lst.append(layers.MaxPool1D(pool[layer_no],
                                          data_format=data_format))
# layer_lst.append(layers.Dense(80, activation=activation))
layer_lst.append(layers.Dense(1, activation=None))

graph = tf.Graph()
with graph.as_default():
    input_values = tf.placeholder(tf.float32,
                                  [batch_size, 203, 42])
    targets = tf.placeholder(tf.float32, [batch_size, 1])
    hidden_and_out = [input_values]
    with tf.variable_scope('classification_CNN'):
        for layer in layer_lst:
            print(hidden_and_out[-1])
            hidden_and_out.append(layer(hidden_and_out[-1]))
    out = tf.squeeze(hidden_and_out[-1], 1)
    sig_out = tf.nn.sigmoid(out)
    loss = losses.sigmoid_cross_entropy(targets, out)

    opt = train.RMSPropOptimizer(learning_rate)
    weight_update = opt.minimize(loss)

    test_hidden_and_out = [input_values]
    with tf.variable_scope('classification_CNN', reuse=True):
        for layer in layer_lst:
            if type(layer) is not layers.Dropout:
                print(test_hidden_and_out[-1])
                test_hidden_and_out.append(layer(test_hidden_and_out[-1]))
    test_out = tf.squeeze(test_hidden_and_out[-1], 1)
    test_sig_out = tf.nn.sigmoid(out)
    test_loss = losses.sigmoid_cross_entropy(targets, test_out)

    init_global = tf.initializers.global_variables()
    init_local = tf.initializers.local_variables()
    parameter_total = compute_parameter_total(tf.trainable_variables())


physionet = physionet_handler.PhysioHandler(set_path=train_set,
                                            label_file=label_file,
                                            batch_size=batch_size)
val_physionet = physionet_handler.PhysioHandler(set_path=val_set,
                                                label_file=val_label_file,
                                                max_length=physionet.max_length,
                                                mean_dict=physionet.mean_dict,
                                                std_dict=physionet.std_dict,
                                                batch_size=batch_size)


def scaled_accuracy(labels, predictions):
    return np.sum(np.abs(labels - predictions))/np.sum(labels)


with tf.Session(graph=graph) as sess:
    sess.run([init_global, init_local])
    print('parameter_total:', parameter_total)
    val_batches = val_physionet.get_batches()
    loss_mean_lst_train = []
    loss_mean_lst_val = []
    acc_mean_lst_train = []
    acc_mean_lst_val = []
    for e in range(0, epochs):
        # validate
        if e % 2 == 0:
            test_loss_lst = []
            test_acc_lst = []
            test_zero_acc_lst = []
            for j in range(len(val_batches[0])):
                x_val = val_batches[0][j]
                y_val = val_batches[1][j]
                feed_dict = {input_values: x_val,
                             targets: y_val}
                val_loss, val_out = sess.run([test_loss, test_sig_out],
                                             feed_dict=feed_dict)
                test_loss_lst.append(val_loss)
                test_acc_lst.append(scaled_accuracy(y_val, val_out))
            print('test loss', np.mean(test_loss_lst),
                  'test err ', np.mean(test_acc_lst))
            loss_mean_lst_val.append(np.mean(test_loss_lst))
            acc_mean_lst_val.append(np.mean(test_acc_lst))

        # get new shuffled batch.
        image_batches, target_batches = physionet.get_batches()
        assert len(image_batches) == len(target_batches)
        train_loss_lst = []
        train_acc_lst = []
        for i in range(len(image_batches)):
            x = image_batches[i]
            y = target_batches[i]
            feed_dict = {input_values: x,
                         targets: y}
            train_loss, sig_out_train, _ = \
                sess.run([loss, sig_out, weight_update],
                         feed_dict=feed_dict)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(scaled_accuracy(y, sig_out_train))
        if e % 2 == 0:
            print('train loss', np.mean(train_loss_lst),
                  'train err ', np.mean(train_acc_lst))
            print('epoch', e+1, 'of', epochs, 'done')
            loss_mean_lst_train.append(np.mean(train_loss_lst))
            acc_mean_lst_train.append(np.mean(train_acc_lst))
    # debug_here()
    plt.plot(loss_mean_lst_train, label='train_loss')
    plt.plot(acc_mean_lst_train, label='train_acc')
    plt.plot(loss_mean_lst_val, label='val_loss')
    plt.plot(acc_mean_lst_val, label='val_acc')
    plt.show()
