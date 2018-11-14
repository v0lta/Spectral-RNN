# Do the imports.
import sys
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2
import tensorflow.contrib.signal as tfsignal
# from scipy.fftpack import fft
from sklearn.metrics import average_precision_score
from IPython.core.debugger import Tracer
from music_net_handler import MusicNet
sys.path.insert(0, "../")
import custom_cells as cc
import eagerSTFT

# import custom_optimizers as co
debug_here = Tracer()


# where to store the logfiles.
subfolder = 'cgRNN_prd_tst'

m = 128         # number of notes
sampling_rate = 11000      # samples/second
features_idx = 0    # first element of (X,Y) data tuple
labels_idx = 1      # second element of (X,Y) data tuple

# Network parameters:
c = 1               # number of context vectors
batch_size = 5      # The number of data points to be processed in parallel.

cell_size = 1024    # cell depth.
RNN = True
stiefel = True
dropout = False

sample_size = 16384
pred_samples = int(sample_size/3)

# FFT parameters:
# window_size = 4096
fft = True
window = 'hann'
window_size = 2048
overlap = int(window_size*0.75)
step_size = window_size - overlap
fft_pred_samples = pred_samples // step_size + 1


# Training parameters:
learning_rate = 0.0001
learning_rate_decay = 0.9
decay_iterations = 5000
iterations = 25000
GPU = [0]


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


print('Setting up the tensorflow graph.')
train_graph = tf.Graph()
with train_graph.as_default():
    global_step = tf.Variable(0, trainable=False, name='global_step')
    # We use c input windows to give the RNN acces to context.
    x = tf.placeholder(tf.float32, shape=[batch_size, c, sample_size])
    # The ground labelling is used during traning, wich random sampling
    # from the network output.

    print('data_nd_shape', x.shape)
    dtype = tf.float32
    data_encoder_time, data_decoder_time = tf.split(x, [sample_size-pred_samples,
                                                    pred_samples],
                                                    axis=1)
    if fft:
        dtype = tf.complex64
        window = 'hann'

        def transpose_stft_squeeze(in_data, window):
            # debug_here()
            tmp_in_data = tf.transpose(in_data, [0, 2, 1])
            in_data_fft = eagerSTFT.stft(tmp_in_data, window,
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
    elif cell_type == 'orthogonal':
        if fft:
            uRNN = cc.UnitaryCell(num_units, activation=cc.mod_relu,
                                  real=not fft, num_proj=num_proj,
                                  complex_input=fft)
            cell = RnnInputWrapper(1.0, uRNN)
            if use_residuals:
                cell = ResidualWrapper(cell=cell)
        else:
            oRNN = cc.UnitaryCell(num_units, activation=cc.relu,
                                  real=True, num_proj=num_proj)
            cell = RnnInputWrapper(1.0, oRNN)
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
        zero_state = LSTMStateTuple(encoder_in[:, 0, :], zero_state[1])
        # debug_here()
        encoder_out, encoder_state = tf.nn.dynamic_rnn(cell, encoder_in,
                                                       initial_state=zero_state,
                                                       dtype=dtype)
        if not fft:
            decoder_in = tf.zeros([batch_size, pred_samples, 1])
            encoder_state = LSTMStateTuple(data_encoder_time[:, -1, :],
                                           encoder_state[-1])
        else:
            freqs = data_encoder_freq.shape[-1].value
            decoder_in = tf.zeros([batch_size, fft_pred_samples, freqs], dtype=dtype)
            encoder_state = LSTMStateTuple(data_encoder_freq[:, -1, :],
                                           encoder_state[-1])
        cell.close()
        scope.reuse_variables()
        decoder_out, _ = tf.nn.dynamic_rnn(cell, decoder_in,
                                           initial_state=encoder_state,
                                           dtype=dtype)

    if fft:
        if (freq_loss == 'mse') or (freq_loss == 'mse_time'):
            prd_loss = tf.losses.mean_squared_error(tf.real(data_decoder_freq),
                                                    tf.real(decoder_out)) \
                + tf.losses.mean_squared_error(tf.imag(data_decoder_freq),
                                               tf.imag(decoder_out))
            tf.summary.scalar('mse', prd_loss)

        elif (freq_loss == 'ad') or (freq_loss == 'ad_time'):
            prd_loss = tf.losses.absolute_difference(tf.real(data_decoder_freq),
                                                     tf.real(decoder_out)) \
                + tf.losses.absolute_difference(tf.imag(data_decoder_freq),
                                                tf.imag(decoder_out))
            tf.summary.scalar('ad', prd_loss)
        elif freq_loss == 'norm_ad':
            prd_loss = tf.losses.absolute_difference(tf.real(data_decoder_freq),
                                                     tf.real(decoder_out)) \
                + tf.losses.absolute_difference(tf.imag(data_decoder_freq),
                                                tf.imag(decoder_out)) \
                + tf.linalg.norm(decoder_out, ord=1)
            tf.summary.scalar('norm_ad', prd_loss)
        elif freq_loss == 'log_ad':
            def log_epsilon(fourier_coeff):
                epsilon = 1e-7
                return tf.log(tf.to_float(fourier_coeff) + epsilon)

            prd_loss = log_epsilon(tf.losses.absolute_difference(
                tf.real(data_decoder_freq), tf.real(decoder_out))) \
                + log_epsilon(tf.losses.absolute_difference(
                    tf.imag(data_decoder_freq), tf.imag(decoder_out)))
            tf.summary.scalar('log_ad', prd_loss)
        elif (freq_loss == 'log_mse') or (freq_loss == 'log_mse_time'):
            # avoid taking a loss of zero using an epsilon.
            def log_epsilon(fourier_coeff):
                epsilon = 1e-7
                return tf.log(tf.to_float(fourier_coeff) + epsilon)

            prd_loss = log_epsilon(tf.losses.mean_squared_error(
                tf.real(data_decoder_freq),
                tf.real(decoder_out))) \
                + log_epsilon(tf.losses.mean_squared_error(
                    tf.imag(data_decoder_freq),
                    tf.imag(decoder_out)))
            tf.summary.scalar('log_mse', prd_loss)

        def expand_dims_and_transpose(input_tensor):
            if spikes_instead_of_states:
                output = tf.expand_dims(input_tensor, -1)
            else:
                its = input_tensor.shape.as_list()
                output = tf.reshape(input_tensor, its[:2] + [-1] + [3])
            output = tf.transpose(output, [0, 3, 1, 2])
            return output
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
    if not fft:
        loss = time_loss
    else:
        if (freq_loss == 'ad_time') or \
           (freq_loss == 'log_mse_time') or \
           (freq_loss == 'mse_time'):
            print('using freq and time based loss.')
            loss = 0.01*prd_loss + time_loss
        else:
            loss = prd_loss

    # debug_here()
    learning_rate = tf.train.exponential_decay(init_learning_rate, global_step,
                                               decay_steps, decay_rate,
                                               staircase=True)
    tf.summary.scalar('learning_rate', learning_rate)

    if cell_type == 'orthogonal' or cell_type == 'cgRNN':
        optimizer = co.RMSpropNatGrad(learning_rate, global_step=global_step)
    else:
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
      'learning_rate', learning_rate,
      'learning_rate_decay', learning_rate_decay, 'iterations', iterations,
      'GPU', GPU, 'dropout', dropout,
      'parameter_total', parameter_total)


def lst_to_str(lst):
    string = ''
    for lst_el in lst:
        string += str(lst_el) + '_'
    return string


time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
param_str = 'lr_' + str(learning_rate) + '_lrd_' + str(learning_rate_decay) \
            + '_lrdi_' + str(decay_iterations) + '_it_' + str(iterations) \
            + '_bs_' + str(batch_size) + '_ws_' + str(window_size) \
            + 'fft_stride' + str(fft_stride) + '_fs_' + str(sampling_rate)
param_str += '_loss_' + str(L.name[:-8]) \
             + '_dropout_' + str(dropout) \
             + '_cs_' + str(cell_size) \
             + '_c_' + str(c) \
             + '_totparam_' + str(parameter_total)
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
    for i in range(iterations):
        if i % 100 == 0 and (i != 0 or len(square_error) == 0):
            batch_time_music_test, batched_time_labels_test = \
                musicNet.get_batch(musicNet.test_data, musicNet.test_ids,
                                   batch_size)
            feed_dict = {x: batch_time_music_test,
                         y_gt: batched_time_labels_test}
            L_np, test_summary_eval, global_step_eval = sess.run([L_test, test_summary,
                                                                 global_step],
                                                                 feed_dict=feed_dict)
            square_error.append(L_np)
            summary_writer.add_summary(test_summary_eval, global_step=global_step_eval)

        # if i % 5000 == 0:
        if i % 5000 == 0 and i > 0:
            # run trough the entire test set.
            yflat = np.array([])
            yhatflat = np.array([])
            losses_lst = []
            for j in range(len(batched_time_music_lst)):
                batch_time_music = batched_time_music_lst[j]
                batched_time_labels = batcheded_time_labels_lst[j]
                feed_dict = {x: batch_time_music,
                             y_gt: batched_time_labels}
                loss, Yhattest, np_global_step =  \
                    sess.run([L_test, y_test, global_step], feed_dict=feed_dict)
                losses_lst.append(loss)
                center = int(c/2.0)
                yhatflat = np.append(yhatflat, Yhattest[:, center, :].flatten())
                yflat = np.append(yflat, batched_time_labels[:, center, :].flatten())
            average_precision.append(average_precision_score(yflat,
                                                             yhatflat))
            end = time.time()
            print(i, '\t', round(np.mean(losses_lst), 8),
                     '\t', round(average_precision[-1], 8),
                     '\t', round(end-start, 8))
            saver.save(sess, savedir + '/weights', global_step=np_global_step)
            # add average precision to tensorboard...
            acc_value = summary_pb2.Summary.Value(tag="Accuracy",
                                                  simple_value=average_precision[-1])
            summary = summary_pb2.Summary(value=[acc_value])
            summary_writer.add_summary(summary, global_step=np_global_step)

            start = time.time()

        batch_time_music, batched_time_labels = \
            musicNet.get_batch(musicNet.train_data, musicNet.train_ids, batch_size)
        feed_dict = {x: batch_time_music,
                     y_gt: batched_time_labels}
        loss, out_net, out_gt, _, summaries, np_global_step = \
            sess.run([L, y, y_gt, training_step, summary_op, global_step],
                     feed_dict=feed_dict)
        summary_writer.add_summary(summaries, global_step=np_global_step)
        # if i % 10 == 0:
        #     print('loss', loss)

    # save the network
    saver.save(sess, savedir + '/weights/', global_step=np_global_step)
    pickle.dump(average_precision, open(savedir + "/avgprec.pkl", "wb"))
