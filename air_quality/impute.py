import time
import tensorflow as tf
import numpy as np
import scipy.signal as signal
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.losses as losses
import tensorflow.train as train
import matplotlib.pyplot as plt
from air_data_handler import AirDataHandler
# from imputation_graphs import CNNImputationGraph
from imputation_graphs import DfnImputation
from recurrent_imputation_graphs import RNNImputation
from imputation_graphs import AdvDfnImputation
from tensorboard_logger import TensorboardLogger

from IPython.core.debugger import Tracer
debug_here = Tracer()


# TODO: Add imputation and STFT.
# TODO: Fix preprocessing.

# network parameters
path_gt = './SampleData/pm25_ground.txt'
path_in = './SampleData/pm25_missing.txt'
batch_size = 25
sequence_length = 36
step_size = 1
air_handler = AirDataHandler(path_in, path_gt,
                             batch_size=batch_size,
                             sequence_length=sequence_length,
                             step_size=step_size)
epochs = 100
learning_rate = 0.0005
lr_decay = 0.9
lrd_steps = 25000
input_dropout = 0.4
padding = 'valid'
activation = layers.ReLU()

# graph = CNNImputationGraph(learning_rate=0.001,
#                            dropout_rate=dropout_rate,
#                            sequence_length=sequence_length)

# graph = DfnImputation(learning_rate=learning_rate,
#                       sequence_length=sequence_length,
#                       decay_rate=lr_decay,
#                       decay_steps=lrd_steps)

# graph = RNNImputation(learning_rate=0.001,
#                       sequence_length=sequence_length)

graph = AdvDfnImputation(learning_rate=0.0001,
                         sequence_length=sequence_length)


time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
param_str = 'e_' + str(epochs) + '_id_' + str(input_dropout)  \
    + '_dsz_' + str(step_size) \
    + '_lr_' + str(learning_rate) \
    + '_lrd_' + str(lr_decay) + '_lrds_' + str(lrd_steps) + '_pt_' \
    + str(graph.parameter_total) + '_' + time_str

tensorboard_logger = TensorboardLogger(path='./logs/adv2/' + param_str,
                                       graph=graph.tf_graph,
                                       no_log=False)


print_every = 2
with tf.Session(graph=graph.tf_graph) as sess:
    graph.init_global.run()
    graph.init_local.run()
    loss_lst = []
    val_loss_lst = []
    val_mre_lst = []
    loss_np = None
    input_lst_val, target_lst_val = air_handler.get_validation_data()
    for e in range(epochs):
        input_lst, target_lst = air_handler.get_epoch()
        if e % 2 == 0:
            adv_input_loss, adv_dfn_loss = graph.train_adv(
                sess, input_lst, target_lst, tensorboard_logger, input_dropout)
            print(e, 'adv', adv_input_loss, adv_dfn_loss)
        else:
            loss_np, input_loss_np, np_step = graph.train(
                sess, input_lst, target_lst, tensorboard_logger, input_dropout)
            loss_lst.append(loss_np)

        if e % print_every == 0:
            loss_np_val, mre_val, mre_idt, test_out_np_val = graph.val(
                sess, input_lst_val, target_lst_val, tensorboard_logger,
                air_handler, sequence_length)
            print(e, loss_np, loss_np_val, 'mre_val', mre_val, mre_idt)
            val_loss_lst.append(loss_np_val)
            val_mre_lst.append(mre_val)

    print('best', np.min(val_mre_lst))
    debug_here()
    print('test')
    # plt.plot(loss_lst)
    # plt.show()
    # plt.plot(val_loss_lst)
    # plt.show()
    # plt.plot(val_mre_lst)
    # plt.show()
    # plt.imshow(np.concatenate([norm_data_val[:72, :, 0],
    #                            test_out_np_val[:72, :, 0],
    #                            norm_data_gt_val[:72, :, 0],
    #                            np.abs(test_out_np_val[:72, :, 0]
    #                                   - norm_data_val[:72, :, 0]),
    #                            np.abs(test_out_np_val[:72, :, 0]
    #                                   - norm_data_gt_val[:72, :, 0]),
    #                            np.abs(norm_data_gt_val[:72, :, 0]
    #                                   - norm_data_val[:72, :, 0])], 0)
    #            )
    # plt.show()
