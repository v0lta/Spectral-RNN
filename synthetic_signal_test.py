import io
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import custom_cells as cc
import custom_optimizers as co
from RNN_wrapper import RnnWrapper
from RNN_wrapper import RnnInputWrapper
import tensorflow.nn.rnn_cell as rnn_cell
from tensorflow.contrib.rnn import LSTMStateTuple
from IPython.core.debugger import Tracer
# import tensorflow.contrib.signal as tfsig
import eager_STFT as eagerSTFT
import scipy.signal as scisig

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


def run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                   num_units, sample_prob, pred_samples, num_proj, learning_rate,
                   iterations, GPUs, batch_size, tmax, delta_t, steps, fft,
                   window_function=None, window_size=None, overlap=None,
                   step_size=None, fft_pred_samples=None, freq_loss=None):
    graph = tf.Graph()
    with graph.as_default():
        global_step = tf.Variable(0, name='global_step', trainable=False)

        if spikes_instead_of_states:
            data_nd = eagerSTFT.generate_data(tmax, delta_t, batch_size=batch_size)[0]
        else:
            data_nd = eagerSTFT.generate_data(tmax, delta_t, batch_size=batch_size)[1]
        print('data_nd_shape', data_nd.shape)
        dtype = tf.float32
        data_encoder_time, data_decoder_time = tf.split(data_nd, [steps-pred_samples,
                                                        pred_samples],
                                                        axis=1)
        if fft:
            dtype = tf.complex64
            window = tf.constant(scisig.get_window('hann', window_size),
                                 dtype=tf.float32)

            def transpose_stft_squeeze(in_data, window):
                tmp_in_data = tf.transpose(in_data, [0, 2, 1])
                in_data_fft = eagerSTFT.stft(tmp_in_data, window,
                                             window_size, overlap)

                in_data_fft = tf.transpose(in_data_fft, [0, 2, 3, 1])
                in_data_fft = tf.squeeze(in_data_fft, axis=-1)
                return in_data_fft
            data_encoder_freq = transpose_stft_squeeze(data_encoder_time, window)
            data_decoder_freq = transpose_stft_squeeze(data_decoder_time, window)
            fft_pred_samples = data_decoder_freq.shape[1].value

        if cell_type == 'cgRNN':
            cell = cc.StiefelGatedRecurrentUnit(num_units, num_proj=num_proj,
                                                complex_input=fft,
                                                complex_output=fft)
            cell = RnnInputWrapper(cell=cell, sample_prob=sample_prob)
        elif cell_type == 'orthogonal':
            if fft:
                uRNN = cc.UnitaryCell(num_units, activation=cc.mod_relu,
                                      real=not fft, num_proj=num_proj,
                                      complex_input=fft)
                cell = RnnInputWrapper(cell=uRNN, sample_prob=sample_prob)
            else:
                oRNN = cc.UnitaryCell(num_units, activation=cc.relu,
                                      real=True, num_proj=num_proj)
                cell = RnnInputWrapper(cell=oRNN, sample_prob=sample_prob)
        else:
            assert fft is False, "GRUs do not process complex inputs."
            gru = rnn_cell.GRUCell(num_units)
            cell = RnnWrapper(num_proj, cell=gru, sample_prob=sample_prob)

        if fft:
            encoder_in = data_encoder_freq[:, :-1, :]
            encoder_out_gt = data_encoder_freq[:, 1:, :]
        else:
            encoder_in = data_encoder_time[:, :-1, :]
            encoder_out_gt = data_encoder_time[:, 1:, :]

        with tf.variable_scope("encoder_decoder") as scope:
            zero_state = cell.zero_state(batch_size)
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
            if freq_loss == 'mse':
                prd_loss = tf.losses.mean_squared_error(tf.real(data_decoder_freq),
                                                        tf.real(decoder_out)) \
                    + tf.losses.mean_squared_error(tf.imag(data_decoder_freq),
                                                   tf.imag(decoder_out))
            elif freq_loss == 'ad':
                prd_loss = tf.losses.absolute_difference(tf.real(data_decoder_freq),
                                                         tf.real(decoder_out)) \
                    + tf.losses.absolute_difference(tf.imag(data_decoder_freq),
                                                    tf.imag(decoder_out))

            def expand_dims_and_transpose(input_tensor):
                output = tf.expand_dims(input_tensor, -1)
                output = tf.transpose(output, [0, 3, 1, 2])
                return output
            # debug_here()
            encoder_out = expand_dims_and_transpose(encoder_out)
            decoder_out = expand_dims_and_transpose(decoder_out)
            encoder_out = eagerSTFT.istft(encoder_out, window=window,
                                          nperseg=window_size,
                                          noverlap=overlap)
            decoder_out = eagerSTFT.istft(decoder_out, window=window,
                                          nperseg=window_size,
                                          noverlap=overlap)
            data_encoder_gt = expand_dims_and_transpose(encoder_out_gt)
            data_decoder_gt = expand_dims_and_transpose(data_decoder_freq)
            data_encoder_gt = eagerSTFT.istft(data_encoder_gt, window=window,
                                              nperseg=window_size,
                                              noverlap=overlap)
            data_decoder_gt = eagerSTFT.istft(data_decoder_gt, window=window,
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
        time_loss = tf.losses.mean_squared_error(tf.real(data_decoder_time),
                                                 tf.real(decoder_out))
        if not fft:
            loss = time_loss
        else:
            loss = prd_loss

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
        pred_loss_summary = tf.summary.scalar('time_loss', time_loss)
        if fft:
            pred_loss_summary = tf.summary.scalar(freq_loss, prd_loss)

        init_op = tf.global_variables_initializer()
        summary_sum = tf.summary.merge_all()
        total_parameters = compute_parameter_total(tf.trainable_variables())
        saver = tf.train.Saver()

    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    param_str = '_' + cell.to_string() + '_fft_' + str(fft) + \
        '_bs_' + str(batch_size) + \
        '_ps_' + str(pred_samples) + \
        '_lr_' + str(learning_rate) + '_sp_' + str(sample_prob) + \
        '_pt_' + str(total_parameters)

    if fft:
        param_str += '_wf_' + str(window_function)
        param_str += '_ws_' + str(window_size)
        param_str += '_ol_' + str(overlap)
        param_str += '_ffts_' + str(step_size)
        param_str += '_fftp_' + str(fft_pred_samples)
        param_str += '_fl_' + freq_loss

    if spikes_instead_of_states:
        param_str += '_1d'
    else:
        param_str += '_3d'
    print(param_str)
    # debug_here()
    summary_writer = tf.summary.FileWriter(base_dir + time_str + param_str,
                                           graph=graph)

    # train this.
    gpu_options = tf.GPUOptions(visible_device_list=str(GPUs)[1:-1])
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False,
                            gpu_options=gpu_options)
    with tf.Session(graph=graph, config=config) as sess:
        print('initialize....')
        init_op.run()
        for it in range(0, iterations):
            start = time.time()
            np_loss, summary_to_file, np_global_step, _, \
                data_encoder_np, encoder_out_np, data_decoder_np, decoder_out_np, \
                data_nd_np = \
                sess.run([loss, summary_sum, global_step, training_op,
                          data_encoder_gt, encoder_out,
                          data_decoder_gt, decoder_out, data_nd])
            stop = time.time()
            print(it, 'loss', np_loss, 'time [s]', stop-start,
                  'time until done [min]', (iterations-it)*(stop-start)/60.0)
            # debug_here()
            summary_writer.add_summary(summary_to_file, global_step=np_global_step)
            if it % 100 == 0:
                plt.figure()
                if spikes_instead_of_states:
                    plt.plot(decoder_out_np[0, :, 0])
                    plt.plot(data_decoder_np[0, :, 0])
                    plt.plot(np.abs(decoder_out_np[0, :, 0] - data_decoder_np[0, :, 0]))
                else:
                    plt.plot(decoder_out_np[0, :, :].flatten())
                    plt.plot(data_decoder_np[0, :, :].flatten())
                    plt.plot(np.abs(decoder_out_np[0, :, :].flatten()
                             - data_decoder_np[0, :, :].flatten()))
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

                # plt.figure()
                # if spikes_instead_of_states:
                #     plt.plot(encoder_out_np[0, :, 0])
                #     plt.plot(data_encoder_np[0, :, 0])
                #     plt.plot(np.abs(encoder_out_np[0, :, 0] - data_encoder_np[0, :, 0]))
                # else:
                #     plt.plot(encoder_out_np[0, :, :].flatten())
                #     plt.plot(data_encoder_np[0, :, :].flatten())
                #     plt.plot(np.abs(encoder_out_np[0, :, :].flatten()
                #                     - data_encoder_np[0, :, :].flatten()))
                # plt.title("fit vs. ground truth")
                # buf = io.BytesIO()
                # plt.savefig(buf, format='png')
                # buf.seek(0)
                # summary_image = tf.Summary.Image(
                #     encoded_image_string=buf.getvalue(),
                #     height=int(plt.rcParams["figure.figsize"][0]*100),
                #     width=int(plt.rcParams["figure.figsize"][1]*100))
                # summary_image = tf.Summary.Value(tag='fit',
                #                                  image=summary_image)
                # summary_image = tf.Summary(value=[summary_image])
                # summary_writer.add_summary(summary_image, global_step=np_global_step)
                # plt.close()
                # buf.close()

                # if not spikes_instead_of_states:
                #     fig = plt.figure()
                #     ax = fig.gca(projection='3d')
                #     ax.plot(encoder_out_np[0, :, 0],
                #             encoder_out_np[0, :, 1],
                #             encoder_out_np[0, :, 2])
                #     ax.plot(encoder_out_gt_np[0, :, 0],
                #             encoder_out_gt_np[0, :, 1],
                #             encoder_out_gt_np[0, :, 2])
                #     plt.title("fit vs. ground truth 3d")
                #     buf = io.BytesIO()
                #     plt.savefig(buf, format='png')
                #     buf.seek(0)
                #     summary_image = tf.Summary.Image(
                #         encoded_image_string=buf.getvalue(),
                #         height=int(plt.rcParams["figure.figsize"][0]*100),
                #         width=int(plt.rcParams["figure.figsize"][1]*100))
                #     summary_image = tf.Summary.Value(tag='prediction_error_3d',
                #                                      image=summary_image)
                #     summary_image = tf.Summary(value=[summary_image])
                #     summary_writer.add_summary(summary_image,
                #                                global_step=np_global_step)
                #     plt.close()
                #     buf.close()


spikes_instead_of_states = True
base_dir = 'logs_tf_fft_large_pa_64_ws_128/'
if spikes_instead_of_states:
    dimensions = 1
else:
    dimensions = 3
cell_type = 'cgRNN'
num_units = 220
if cell_type == 'uRNN':
    circ_h = 4
    conv_h = 5
    input_samples = 17
    num_units = 150
if cell_type == 'orthogonal':
    num_units = 50
sample_prob = 1.0
pred_samples = 256
num_proj = dimensions
learning_rate = 0.001
iterations = 25000
GPUs = [0]
batch_size = 200

# data parameters
tmax = 10.24
# tmax = 10.23
delta_t = 0.01
steps = int(tmax/delta_t)+1

# fft parameters
fft = True
if fft:
    window_function = 'hann'
    window_size = 128
    overlap = int(window_size*0.5)
    step_size = window_size - overlap
    fft_pred_samples = pred_samples // step_size + 1
    num_proj = int(window_size//2 + 1)  # the frequencies
    freq_loss = 'ad'  # 'mse, ad'
    num_units = 150

# num_units = 1024
# num_units = 1108
# fft = False
# num_proj = dimensions
# run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
#                num_units, sample_prob, pred_samples, num_proj, learning_rate,
#                iterations, GPUs, batch_size, tmax, delta_t, steps, fft,
#                window_function, window_size, overlap, step_size, fft_pred_samples,
#                freq_loss)

if 1:
    for length_factor in [0, 1, 2, 3, 4]:
        tmp_fft = True
        tmp_num_units = 1024
        tmp_pred_samples = (length_factor*64) + pred_samples
        tmp_tmax = (length_factor*1.28*2) + tmax
        tmp_steps = int(tmp_tmax/delta_t) + 1
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       tmp_num_units, sample_prob, tmp_pred_samples, num_proj,
                       learning_rate, iterations, GPUs, batch_size, tmp_tmax, delta_t,
                       tmp_steps, tmp_fft, window_function, window_size, overlap,
                       step_size, fft_pred_samples, freq_loss)

    for length_factor in [0, 1, 2, 3, 4]:
        tmp_fft = False
        num_proj = dimensions
        tmp_num_units = 1108
        tmp_pred_samples = (length_factor*64) + pred_samples
        tmp_tmax = (length_factor*1.28*2) + tmax
        tmp_steps = int(tmp_tmax/delta_t) + 1
        run_experiment(spikes_instead_of_states, base_dir, dimensions, cell_type,
                       num_units, sample_prob, tmp_pred_samples, num_proj, learning_rate,
                       iterations, GPUs, batch_size, tmp_tmax, delta_t, tmp_steps,
                       tmp_fft, window_function, window_size, overlap, step_size,
                       fft_pred_samples, freq_loss)
