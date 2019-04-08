import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.losses as losses
import tensorflow.train as train

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
epochs = 501
batch_size = 100
quantity_no = 42
activation = layers.ReLU()
learning_rate = 0.001

train_set = './data/set-a/'
label_file = './data/set-a_outcome.txt'
physionet = physionet_handler.PhysioHandler(set_path=train_set,
                                            label_file=label_file,
                                            batch_size=batch_size)
val_set = './data/set-b/'
val_label_file = './data/set-b_outcome.txt'
val_physionet = physionet_handler.PhysioHandler(set_path=val_set,
                                                label_file=val_label_file,
                                                max_length=physionet.max_length,
                                                batch_size=batch_size)
filters = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
kernel_size = [12, 12, 8, 6, 6, 6, 6, 6, 6, 6]
assert len(filters) == len(kernel_size)
strides = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
padding = 'valid'
data_format = 'channels_last'

layer_lst = []
for layer_no in range(0, len(filters)):
    layer_lst.append(layers.Conv1D(filters[layer_no], kernel_size[layer_no],
                                   strides[layer_no], padding, data_format,
                                   activation=activation))
layer_lst.append(layers.Dense(400, activation=activation))
layer_lst.append(layers.Dense(1, activation=None))

graph = tf.Graph()
with graph.as_default():
    input_values = tf.placeholder(tf.float32,
                                  [batch_size, physionet.max_length,
                                   len(physionet.recorded_quantities)])
    targets = tf.placeholder(tf.float32, [batch_size, 1])
    hidden_and_out = [input_values]
    for layer in layer_lst:
        hidden_and_out.append(layer(hidden_and_out[-1]))

    out = tf.squeeze(hidden_and_out[-1], 1)
    loss = losses.sigmoid_cross_entropy(targets, out)

    opt = train.AdamOptimizer(learning_rate)
    weight_update = opt.minimize(loss)

    sig_out = tf.nn.sigmoid(out)
    acc = tf.metrics.accuracy(labels=targets, predictions=sig_out)
    init_global = tf.initializers.global_variables()
    init_local = tf.initializers.local_variables()
    parameter_total = compute_parameter_total(tf.trainable_variables())


with tf.Session(graph=graph) as sess:
    sess.run([init_global, init_local])
    print('parameter_total:', parameter_total)
    val_batches = val_physionet.get_batches()
    for e in range(0, epochs):
        # validate
        if e % 10 == 0:
            test_loss_lst = []
            test_acc_lst = []
            for j in range(len(val_batches[0])):
                x_val = val_batches[0][j]
                y_val = val_batches[1][j]
                feed_dict = {input_values: x_val,
                             targets: y_val}
                val_loss, val_out, val_acc = sess.run(
                    [loss, sig_out, acc], feed_dict=feed_dict)
                test_loss_lst.append(val_loss)
                test_acc_lst.append(val_acc)
            # print('test loss', np.mean(test_loss_lst))
            print('test acc ', np.mean(test_acc_lst))

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
            train_loss, train_acc, _ = sess.run([loss, acc, weight_update],
                                                feed_dict=feed_dict)
            train_loss_lst.append(train_loss)
            train_acc_lst.append(train_acc)
        if e % 10 == 0:
            # print('train loss', np.mean(train_loss))
            print('train acc ', np.mean(train_acc))
            print('epoch', e+1, 'of', epochs, 'done')
    debug_here()
    print('hoho')
