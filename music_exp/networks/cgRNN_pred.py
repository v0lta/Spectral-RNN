# Do the imports.
import io
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
import scipy.io.wavfile as sio
from tensorflow.core.framework import summary_pb2
import tensorflow.contrib.signal as tfsignal
import tensorflow.nn.rnn_cell as rnn_cell
import matplotlib.pyplot as plt
# from scipy.fftpack import fft
from tensorflow.contrib.rnn import LSTMStateTuple
from IPython.core.debugger import Tracer
from music_net_handler2 import MusicNet
sys.path.insert(0, "../")
import custom_cells as cc
import custom_optimizers as co
from RNN_wrapper import ResidualWrapper
from RNN_wrapper import RnnInputWrapper
from RNN_wrapper import LinearProjWrapper
import eager_STFT as eagerSTFT

# import custom_optimizers as co
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


# where to store the logfiles.
subfolder = 'cgRNN_prd_multi_1'

m = 128  # number of notes
sampling_rate = 11000  # samples/second
features_idx = 0    # first element of (X,Y) data tuple
labels_idx = 1      # second element of (X,Y) data tuple

# FFT parameters:
# window_size = 4096
sample_size = 512
pred_samples = int(sample_size/2.0)
fft = True
window = 'hann'
window_size = 128
overlap = int(window_size*0.75)
step_size = window_size - overlap
fft_pred_samples = pred_samples // step_size + 1

# Network parameters:
c = 1               # number of context vectors
fft_stride = 0      # deprecated TODO: remove!
batch_size = 5      # The number of data points to be processed in parallel.

num_units = 2048    # cell depth.
RNN = True
stiefel = False
dropout = False
cell_type = 'cgRNN'
sample_prob = 1.0
freq_loss = 'complex_abs'
epsilon = 1e-2
use_residuals = False
num_proj = int(window_size//2 + 1)  # the frequencies

# Training parameters:
init_learning_rate = 0.001
decay_rate = 0.96
decay_steps = 10000
epochs = 10
GPU = [0]

print('Setting up the tensorflow graph.')
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # We use c input windows to give the RNN acces to context.
    x = tf.placeholder(tf.float32, shape=[batch_size, sample_size])
    # The ground labelling is used during traning, wich random sampling
    # from the network output.

    print('data_nd_shape', x.shape)
    dtype = tf.float32
    x_exp = tf.expand_dims(x, axis=-1)
    data_encoder_time, data_decoder_time = tf.split(x_exp, [sample_size-pred_samples,
                                                    pred_samples],
                                                    axis=1)
    if fft:
        dtype = tf.complex64
        window = 'hann'

        def transpose_stft_squeeze(in_data, window):
            in_data = tf.transpose(in_data, [0, 2, 1])
            # eagerSTFT expects batch, dim, time
            in_data_fft = eagerSTFT.stft(in_data, window,
                                         window_size, overlap)
            in_data_fft = tf.transpose(in_data_fft, [0, 2, 3, 1])
            idft_shape = in_data_fft.shape.as_list()
            if idft_shape[-1] == 1:
                in_data_fft = tf.squeeze(in_data_fft, axis=-1)
            else:
                in_data_fft = tf.reshape(in_data_fft, [idft_shape[0],
                                                       idft_shape[1],
                                                       -1])
            return in_data_fft, idft_shape
        data_encoder_freq, _ = transpose_stft_squeeze(data_encoder_time,
                                                      window)
        data_decoder_freq, dec_shape = transpose_stft_squeeze(data_decoder_time,
                                                              window)
        fft_pred_samples = data_decoder_freq.shape[1].value
    if cell_type == 'cgRNN':
        cell = cc.StiefelGatedRecurrentUnit(num_units, num_proj=num_proj,
                                            complex_input=fft,
                                            complex_output=fft)
        cell = RnnInputWrapper(1.0, cell)
        if use_residuals:
            cell = ResidualWrapper(cell=cell)
    else:
        assert fft is False, "GRUs do not process complex inputs."
        gru = rnn_cell.GRUCell(num_units)
        cell = LinearProjWrapper(num_proj, cell=gru, sample_prob=sample_prob)
        cell = RnnInputWrapper(1.0, cell)
        if use_residuals:
            cell = ResidualWrapper(cell=cell)

    if fft:
        encoder_in = data_encoder_freq[:, :-1, :]
        encoder_out_gt = data_encoder_freq[:, 1:, :]
    else:
        encoder_in = data_encoder_time[:, :-1, :]
        encoder_out_gt = data_encoder_time[:, 1:, :]

    with tf.variable_scope("encoder_decoder") as scope:
        zero_state = cell.zero_state(batch_size, dtype=dtype)
        encoder_out, encoder_state = tf.nn.dynamic_rnn(cell, encoder_in,
                                                       initial_state=zero_state,
                                                       dtype=dtype)
        if not fft:
            decoder_in = tf.zeros([batch_size, pred_samples, 1])
            # TODO: add multi cell code!!
            if layers == 1:
                encoder_state = LSTMStateTuple(data_encoder_time[:, -1, :],
                                               encoder_state[-1])

        else:
            freqs = data_encoder_freq.shape[-1].value
            decoder_in = tf.zeros([batch_size, fft_pred_samples, freqs], dtype=dtype)
            if layers == 1:
                encoder_state = LSTMStateTuple(data_encoder_freq[:, -1, :],
                                               encoder_state[-1])

        cell.close()
        scope.reuse_variables()
        debug_here()
        decoder_out, _ = tf.nn.dynamic_rnn(cell, decoder_in,
                                           initial_state=encoder_state,
                                           dtype=dtype)

    if fft:
        if freq_loss == 'complex_abs':
            diff = data_decoder_freq - decoder_out
            prd_loss = tf.abs(tf.real(diff)) + tf.abs(tf.imag(diff))
            # tf.summary.histogram('complex_abs', prd_loss)
            # tf.summary.histogram('log_complex_abs', tf.log(prd_loss))
            prd_loss = tf.reduce_mean(prd_loss)
        if freq_loss == 'complex_square':
            diff = data_decoder_freq - decoder_out
            prd_loss = tf.real(diff)*tf.real(diff) + tf.imag(diff)*tf.imag(diff)
            tf.summary.histogram('complex_square', prd_loss)
            prd_loss = tf.reduce_mean(prd_loss)
        tf.summary.scalar('log_mse', prd_loss)

        def expand_dims_and_transpose(input_tensor):
            output = tf.expand_dims(input_tensor, -1)
            output = tf.transpose(output, [0, 3, 1, 2])
            return output

        decoder_out_freq = decoder_out
        encoder_out = expand_dims_and_transpose(encoder_out)
        decoder_out = expand_dims_and_transpose(decoder_out)
        encoder_out = eagerSTFT.istft(encoder_out, window,
                                      nperseg=window_size,
                                      noverlap=overlap)
        decoder_out = eagerSTFT.istft(decoder_out, window,
                                      nperseg=window_size,
                                      noverlap=overlap, epsilon=epsilon)
        data_encoder_gt = expand_dims_and_transpose(encoder_out_gt)
        data_decoder_gt = expand_dims_and_transpose(data_decoder_freq)
        data_encoder_gt = eagerSTFT.istft(data_encoder_gt, window,
                                          nperseg=window_size,
                                          noverlap=overlap)
        data_decoder_gt = eagerSTFT.istft(data_decoder_gt, window,
                                          nperseg=window_size,
                                          noverlap=overlap)
        encoder_out = tf.transpose(encoder_out, [0, 2, 1])
        decoder_out = tf.transpose(decoder_out, [0, 2, 1])
        data_encoder_gt = tf.transpose(data_encoder_gt, [0, 2, 1])
        data_decoder_gt = tf.transpose(data_decoder_gt, [0, 2, 1])
    else:
        data_encoder_gt = encoder_out_gt
        data_decoder_gt = data_decoder_time

    # debug_here()
    time_loss = tf.losses.mean_squared_error(
        tf.real(data_decoder_time), tf.real(decoder_out[:, :pred_samples, :]))
    loss = prd_loss

    # debug_here()
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_steps, decay_rate,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)
    optimizer = tf.train.RMSPropOptimizer(learning_rate)
    gvs = optimizer.compute_gradients(loss)

    with tf.variable_scope("clip_grads"):
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

    # grad_summary = tf.histogram_summary(grads)
    # training_op = optimizer.minimize(loss, global_step=global_step)
    training_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
    tf.summary.scalar('time_loss', time_loss)
    tf.summary.scalar('training_loss', loss)

    init_op = tf.global_variables_initializer()
    summary_sum = tf.summary.merge_all()
    total_parameters = compute_parameter_total(tf.trainable_variables())
    saver = tf.train.Saver()

# Load the data.
# debug_here()
print('Loading music-Net...')
musicNet = MusicNet(c, fft_stride, sample_size, sampling_rate=sampling_rate)
batched_time_music_lst, batcheded_time_labels_lst = musicNet.get_test_batches(batch_size)

print('parameters:', 'm', m, 'sampling_rate', sampling_rate, 'c', c,
      'sample_size', sample_size, 'window_size', window_size,
      'window_size', window_size, 'fft_stride', fft_stride,
      'learning_rate', init_learning_rate,
      'learning_rate_decay', decay_rate, 'epochs', epochs,
      'GPU', GPU, 'dropout', dropout,
      'parameter_total', total_parameters)


def lst_to_str(lst):
    string = ''
    for lst_el in lst:
        string += str(lst_el) + '_'
    return string


time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
param_str = 'lr_' + str(init_learning_rate) + '_lrd_' + str(decay_rate) \
            + '_lrdi_' + str(decay_steps) + '_ep_' + str(epochs) \
            + '_bs_' + str(batch_size) + '_ws_' + str(window_size) \
            + '_ov_' + str(overlap) + '_sample_size_' + str(sample_size) \
            + '_fft_stride_' + str(fft_stride) + '_fs_' + str(sampling_rate) \
            + '_rc_' + str(use_residuals)
param_str += '_loss_' + freq_loss \
             + '_dropout_' + str(dropout) \
             + '_cs_' + str(num_units) \
             + '_c_' + str(c) \
             + '_totparam_' + str(total_parameters)
savedir = './logs' + '/' + subfolder + '/' + time_str \
          + '_' + param_str
# debug_here()
print(savedir)
summary_writer = tf.summary.FileWriter(savedir, graph=train_graph)

square_error = []
average_precision = []
gpu_options = tf.GPUOptions(visible_device_list=str(GPU)[1:-1])
config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False,
                        gpu_options=gpu_options)
with tf.Session(graph=train_graph, config=config) as sess:
    start = time.time()
    print('Initialize...')
    init_op.run(session=sess)

    print('Training...')
    for e in range(epoch):
        for i, batch in enumerate(train_batches):
            if i % 100 == 0 and (i != 0 or len(square_error) == 0):
                batch_time_music_test, batched_time_labels_test = \
                    musicNet.get_batch(musicNet.test_data, musicNet.test_ids,
                                       batch_size)
                feed_dict = {x: np.squeeze(batch_time_music_test, axis=1)}
                L_np, test_summary_eval, global_step_eval = sess.run([loss, summary_sum,
                                                                     global_step],
                                                                     feed_dict=feed_dict)
                square_error.append(L_np)
                summary_writer.add_summary(test_summary_eval, global_step=global_step_eval)

            # if i % 5000 == 0:
        if i % 1000 == 0 and i > 0:
            # run trough the entire test set.
            yflat = np.array([])
            yhatflat = np.array([])
            losses_lst = []

            # record the outputs
            data_decoder_time_np_lst = []
            decoder_out_np_lst = []

            for j in range(len(batched_time_music_lst)):
                batch_time_music = batched_time_music_lst[j]
                batched_time_labels = batcheded_time_labels_lst[j]
                feed_dict = {x: np.squeeze(batch_time_music, axis=1)}
                np_loss, np_global_step, data_decoder_time_np, decoder_out_np, \
                    data_decoder_freq_np, decoder_out_freq_np =  \
                    sess.run([loss, global_step, data_decoder_time, decoder_out,
                              data_decoder_freq, decoder_out_freq], feed_dict=feed_dict)
                losses_lst.append(np_loss)
                data_decoder_time_np_lst.append(data_decoder_time_np)
                decoder_out_np_lst.append(decoder_out_np)

            end = time.time()
            print(i, '\t', round(np.mean(losses_lst), 8),
                     '\t', round(end-start, 8))
            saver.save(sess, savedir + '/weights', global_step=np_global_step)
            # add average precision to tensorboard...
            mean_loss = summary_pb2.Summary.Value(tag="test_loss",
                                                  simple_value=np.mean(losses_lst))
            summary = summary_pb2.Summary(value=[mean_loss])
            summary_writer.add_summary(summary, global_step=np_global_step)
            start = time.time()

            # debug_here()
            # visualize output.
            plt.figure()
            plt.plot(decoder_out_np_lst[671][0, :, 0])
            plt.plot(data_decoder_time_np_lst[671][0, :, 0])
            plt.plot(np.abs(decoder_out_np_lst[671][0, :, 0]
                            - data_decoder_time_np_lst[671][0, :, 0]))

            plt.title("Prediction vs. ground truth")
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            summary_image = tf.Summary.Image(
                encoded_image_string=buf.getvalue(),
                height=int(plt.rcParams["figure.figsize"][0]*100),
                width=int(plt.rcParams["figure.figsize"][1]*100))
            summary_image = tf.Summary.Value(tag='prediction_error',
                                             image=summary_image)
            summary_image = tf.Summary(value=[summary_image])
            summary_writer.add_summary(summary_image, global_step=np_global_step)
            plt.close()
            buf.close()

            data_decoder_time_np_lst = []
            decoder_out_np_lst = []

        batch_time_music, batched_time_labels = \
            musicNet.get_batch(musicNet.train_data, musicNet.train_ids, batch_size)
        feed_dict = {x: np.squeeze(batch_time_music, axis=1)}
        loss_np, out_net, _, summaries, np_global_step = \
            sess.run([loss, decoder_out, training_op, summary_sum, global_step],
                     feed_dict=feed_dict)
        summary_writer.add_summary(summaries, global_step=np_global_step)
        # if i % 10 == 0:
        #     print('loss', loss)

    # save the network
    saver.save(sess, savedir + '/weights/', global_step=np_global_step)
    pickle.dump(average_precision, open(savedir + "/avgprec.pkl", "wb"))
    pickle.dump([data_decoder_time_np, decoder_out_np],
                open(savedir + "/np_pred.pkl", "wb"))
    sio.write(savedir + '/gt', 1000, data_decoder_time_np[0, :, 0]*100)
    sio.write(savedir + '/net_out', 1000, decoder_out_np[0, :, 0]*100)
