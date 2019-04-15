import tensorflow as tf
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.losses as losses
import tensorflow.train as train
import matplotlib.pyplot as plt
from air_data_handler import AirDataHandler
# from imputation_graphs import CNNImputationGraph
from imputation_graphs import DfnImputation


from IPython.core.debugger import Tracer
debug_here = Tracer()


# TODO: Add imputation and STFT.
# TODO: Fix preprocessing.

# network parameters
path_gt = './SampleData/pm25_ground.txt'
path_in = './SampleData/pm25_missing.txt'
batch_size = 25
sequence_length = 36
air_handler = AirDataHandler(path_in, path_gt,
                             batch_size=batch_size,
                             sequence_length=sequence_length)
epochs = 20000

filters = [5, 10, 15, 20, 25, 30, 35, 40]
kernel_size = [[6, 6], [6, 6], [6, 3], [6, 3], [3, 3], [3, 3], [2, 3],
               [1, 3]]
stride_size = [[2, 1], [2, 2], [2, 1], [2, 1], [1, 1], [1, 1], [1, 1],
               [1, 1]]
assert len(filters) == len(kernel_size)

dropout_rate = 0.5
padding = 'valid'
activation = layers.ReLU()

# graph = CNNImputationGraph(learning_rate=0.001,
#                            dropout_rate=dropout_rate,
#                            sequence_length=sequence_length)

graph = DfnImputation(learning_rate=0.0001,
                      dropout_rate=dropout_rate,
                      sequence_length=sequence_length)


def mean_relative_error(labels, predictions, mask=None):
    if mask is not None:
        mask_labels = np.where(mask == 1, labels,
                               np.zeros(shape=predictions.shape))
        mask_predictions = np.where(mask == 1, predictions,
                                    np.zeros(shape=predictions.shape))
        return np.sum(np.abs(mask_labels - mask_predictions))/np.sum(np.abs(mask_labels))
    else:
        return np.sum(np.abs(predictions - labels))/np.sum(np.abs(labels))


def mean_absolute_error(labels, predictions, mask=None):
    if mask is not None:
        mask_labels = np.where(mask == 1, labels,
                               np.zeros(shape=predictions.shape))
        mask_predictions = np.where(mask == 1, predictions,
                                    np.zeros(shape=predictions.shape))
        return np.sum(np.abs(mask_predictions - mask_labels))/np.prod(mask_labels.shape)
    else:
        return np.sum(np.abs(predictions - labels))/np.prod(labels.shape)


def mask_nan_to_num(input_array, dropout_rate=0.35):
    if dropout_rate > 0:
        random_array = np.random.uniform(0, 1, input_array.shape)
        input_array = np.where(random_array > dropout_rate, input_array, np.NaN)
    return np.concatenate([np.nan_to_num(input_array),
                           np.isnan(input_array).astype(np.float32)],
                          -1)


print_every = 25
with tf.Session(graph=graph.tf_graph) as sess:
    graph.init_global.run()
    graph.init_local.run()
    loss_lst = []
    val_loss_lst = []
    val_mre_lst = []
    for e in range(epochs):
        input_lst, target_lst = air_handler.get_epoch()
        assert len(input_lst) == len(target_lst)
        for i in range(len(input_lst)):
            input_array = np.expand_dims(np.transpose(input_lst[i], [0, 2, 1]), -1)
            target_array = np.expand_dims(np.transpose(target_lst[i], [0, 2, 1]), -1)
            input_array = mask_nan_to_num(input_array)
            target_array = mask_nan_to_num(target_array, dropout_rate=0)
            feed_dict = {graph.input_values: input_array,
                         graph.targets: np.expand_dims(target_array[:, :, :, 0], -1)}
            loss_np, input_loss_np, _ = sess.run([graph.loss, graph.input_loss,
                                                  graph.weight_update],
                                                 feed_dict=feed_dict)
            loss_lst.append(loss_np)
        if e % print_every == 0:
            # print('train', e, i, loss_np, input_loss_np, loss_np/input_loss_np*100)
            # do a validation pass over the data.
            norm_data_val, norm_data_gt_val = air_handler.get_validation_data()
            norm_data_val = mask_nan_to_num(norm_data_val, dropout_rate=0)
            norm_data_gt_val = mask_nan_to_num(norm_data_gt_val, dropout_rate=0)
            feed_dict = {graph.input_values: norm_data_val,
                         graph.targets: np.expand_dims(norm_data_gt_val[:, :, :, 0], -1)}
            loss_np_val, input_loss_np_val, test_out_np_val = \
                sess.run([graph.loss, graph.input_loss, graph.test_out],
                         feed_dict=feed_dict)
            # print('val  ', e, i, loss_np_val, input_loss_np_val,
            #       loss_np_val/input_loss_np_val*100)
            removed_mask = norm_data_val[:, :, :, 1] - norm_data_gt_val[:, :, :, 1]
            mean = 0
            std = 1
            mre_idt = mean_relative_error(norm_data_gt_val[:, :, :, 0]*std + mean,
                                          norm_data_val[:, :, :, 0]*std + mean,
                                          mask=removed_mask)
            mae_val = mean_absolute_error(norm_data_gt_val[:, :, :, 0]*std + mean,
                                          test_out_np_val*std + mean,
                                          mask=removed_mask)
            mre_val = mean_relative_error(norm_data_gt_val[:, :, :, 0]*std + mean,
                                          test_out_np_val*std + mean,
                                          mask=removed_mask)
            print(e, loss_np, loss_np_val, 'mre_val', mre_val, mre_idt)
            val_loss_lst.append(loss_np_val)
            val_mre_lst.append(mre_val)

    plt.loglog(loss_lst)
    plt.show()
    plt.loglog(val_loss_lst)
    plt.show()
    plt.loglog(val_mre_lst)
    plt.show()
    plt.imshow(np.concatenate([norm_data_val[0, :, :, 0],
                               test_out_np_val[0, :, :, 0],
                               norm_data_gt_val[0, :, :, 0],
                               np.abs(test_out_np_val[0, :, :, 0]
                                      - norm_data_val[0, :, :, 0]),
                               np.abs(test_out_np_val[0, :, :, 0]
                                      - norm_data_gt_val[0, :, :, 0]),
                               np.abs(norm_data_gt_val[0, :, :, 0]
                                      - norm_data_val[0, :, :, 0])], 0)
               )
    plt.show()
