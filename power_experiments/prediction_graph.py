import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import scipy.signal as scisig
import sys
from IPython.core.debugger import Pdb
debug_here = Pdb().set_trace
sys.path.insert(0, "../")
import custom_cells as ccell
import custom_optimizers as co
from RNN_wrapper import ResidualWrapper
from RNN_wrapper import RnnInputWrapper
from RNN_wrapper import LinearProjWrapper
import eager_STFT as eagerSTFT
import tensorflow.nn.rnn_cell as rnn_cell
from tensorflow.contrib.rnn import LSTMStateTuple
import window_learning as wl


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


class FFTpredictionGraph(object):
    '''
    Create a fourier prediction graph.
    Arguments:
        pd prediction parameter dict.
        generator: A generator object used
                   to generate synthetic data.
    '''

    def __init__(self, pd, generator=None):
        self.pd = pd
        self.graph = tf.Graph()
        with self.graph.as_default():
            global_step = tf.Variable(0, name='global_step', trainable=False)
            if generator:
                print('Running synthetic experiment')
                data_nd_norm = generator()
                data_nd = data_nd_norm
            else:
                data_mean = tf.constant(pd['power_handler'].mean, tf.float32,
                                        name='data_mean')
                data_std = tf.constant(pd['power_handler'].std, tf.float32,
                                       name='data_std')

                data_nd = tf.placeholder(tf.float32, [pd['batch_size'],
                                                      pd['input_samples'], 1])
                self.data_nd = data_nd
                data_nd_norm = (data_nd - data_mean)/data_std

            print('data_nd_shape', data_nd_norm.shape)
            dtype = tf.float32
            data_encoder_time, data_decoder_time = tf.split(data_nd_norm,
                                                            [pd['input_samples']
                                                             - pd['pred_samples'],
                                                             pd['pred_samples']],
                                                            axis=1)

            if pd['fft']:
                dtype = tf.complex64
                if pd['window_function'] == 'learned_gaussian':
                    window = wl.gaussian_window(pd['window_size'])
                elif pd['window_function'] == 'learned_plank':
                    window = wl.plank_taper(pd['window_size'])
                elif pd['window_function'] == 'learned_tukey':
                    window = wl.tukey_window(pd['window_size'])
                elif pd['window_function'] == 'learned_gauss_plank':
                    window = wl.gauss_plank_window(pd['window_size'])
                else:
                    window = scisig.get_window(window=pd['window_function'],
                                               Nx=pd['window_size'])
                    window = tf.constant(window, tf.float32)

                def transpose_stft_squeeze(in_data, window, pd):
                    '''
                    Compute a windowed stft and do low pass filtering if
                    necessary.
                    '''
                    tmp_in_data = tf.transpose(in_data, [0, 2, 1])
                    in_data_fft = eagerSTFT.stft(tmp_in_data, window,
                                                 pd['window_size'], pd['overlap'])
                    freqs = int(in_data_fft.shape[-1])
                    idft_shape = in_data_fft.shape.as_list()
                    if idft_shape[1] == 1:
                        # in the one dimensional case squeeze the dim away.
                        in_data_fft = tf.squeeze(in_data_fft, axis=1)
                        if pd['fft_compression_rate']:
                            compressed_freqs = int(freqs/pd['fft_compression_rate'])
                            print('fft_compression_rate', pd['fft_compression_rate'],
                                  'freqs', freqs,
                                  'compressed_freqs', compressed_freqs)
                            # remove frequencies from the last dimension.
                            return in_data_fft[..., :compressed_freqs], idft_shape, freqs
                        else:
                            return in_data_fft, idft_shape, freqs
                    else:
                        # arrange as batch time freq dim
                        in_data_fft = tf.transpose(in_data_fft, [0, 2, 3, 1])
                        raise NotImplementedError

                data_encoder_freq, _, enc_freqs = \
                    transpose_stft_squeeze(data_encoder_time, window, pd)
                data_decoder_freq, dec_shape, dec_freqs = \
                    transpose_stft_squeeze(data_decoder_time, window, pd)
                assert enc_freqs == dec_freqs, 'encoder-decoder frequencies must agree'
                fft_pred_samples = data_decoder_freq.shape[1].value


            elif pd['linear_reshape']:
                encoder_time_steps = data_encoder_time.shape[1].value//pd['step_size']
                data_encoder_time = tf.reshape(data_encoder_time, [pd['batch_size'],
                                                                   encoder_time_steps,
                                                                   pd['step_size']])
                if pd['downsampling'] > 1:
                    data_encoder_time = data_encoder_time[:, :, ::pd['downsampling']]
                decoder_time_steps = data_decoder_time.shape[1].value//pd['step_size']

            if pd['cell_type'] == 'cgRNN':
                if pd['stiefel']:
                    cell = ccell.StiefelGatedRecurrentUnit(pd['num_units'],
                                                           num_proj=pd['num_proj'],
                                                           complex_input=pd['fft'],
                                                           complex_output=pd['fft'],
                                                           activation=ccell.mod_relu,
                                                           stiefel=pd['stiefel'])
                else:
                    cell = ccell.StiefelGatedRecurrentUnit(pd['num_units'],
                                                           num_proj=pd['num_proj'],
                                                           complex_input=pd['fft'],
                                                           complex_output=pd['fft'],
                                                           activation=ccell.hirose,
                                                           stiefel=pd['stiefel'])
                cell = RnnInputWrapper(1.0, cell)
                if pd['use_residuals']:
                    cell = ResidualWrapper(cell=cell)
            elif pd['cell_type'] == 'gru':
                gru = rnn_cell.GRUCell(pd['num_units'])
                if pd['fft'] is True:
                    dtype = tf.float32
                    # concatenate real and imaginary parts.
                    data_encoder_freq = tf.concat([tf.real(data_encoder_freq),
                                                   tf.imag(data_encoder_freq)],
                                                  axis=-1)
                    cell = LinearProjWrapper(pd['num_proj']*2, cell=gru,
                                             sample_prob=pd['sample_prob'])
                else:
                    cell = LinearProjWrapper(pd['num_proj'], cell=gru,
                                             sample_prob=pd['sample_prob'])
                cell = RnnInputWrapper(1.0, cell)
                if pd['use_residuals']:
                    cell = ResidualWrapper(cell=cell)
            else:
                print('cell type not supported.')

            if pd['fft']:
                encoder_in = data_encoder_freq
            else:
                encoder_in = data_encoder_time

            with tf.variable_scope("encoder_decoder") as scope:

                zero_state = cell.zero_state(pd['batch_size'], dtype=dtype)
                zero_state = LSTMStateTuple(encoder_in[:, 0, :], zero_state[1])
                # debug_here()
                encoder_out, encoder_state = tf.nn.dynamic_rnn(cell, encoder_in,
                                                               initial_state=zero_state,
                                                               dtype=dtype)
                if not pd['fft']:
                    if pd['linear_reshape']:
                        decoder_in = tf.zeros([pd['batch_size'], decoder_time_steps, 1])
                    else:
                        decoder_in = tf.zeros([pd['batch_size'], pd['pred_samples'], 1])
                    encoder_state = LSTMStateTuple(data_encoder_time[:, -1, :],
                                                   encoder_state[-1])
                else:
                    freqs = encoder_in.shape[-1].value
                    decoder_in = tf.zeros([pd['batch_size'],
                                           fft_pred_samples,
                                           freqs], dtype=dtype)

                    encoder_state = LSTMStateTuple(data_encoder_freq[:, -1, :],
                                                   encoder_state[-1])
                cell.close()
                scope.reuse_variables()
                # debug_here()
                decoder_out, _ = tf.nn.dynamic_rnn(cell, decoder_in,
                                                   initial_state=encoder_state,
                                                   dtype=dtype)

                if pd['fft'] and pd['cell_type'] == 'gru':
                    # assemble complex output.
                    decoder_freqs_t2 = decoder_out.shape[-1].value
                    decoder_out = tf.complex(decoder_out[:, :, :int(decoder_freqs_t2/2)],
                                             decoder_out[:, :, int(decoder_freqs_t2/2):])
                    encoder_out = tf.complex(encoder_out[:, :, :int(decoder_freqs_t2/2)],
                                             encoder_out[:, :, int(decoder_freqs_t2/2):])
                    encoder_in = tf.complex(
                        encoder_in[:, :, :int(decoder_freqs_t2/2)],
                        encoder_in[:, :, int(decoder_freqs_t2/2):])

            if pd['fft']:
                if (pd['freq_loss'] == 'complex_abs') \
                   or (pd['freq_loss'] == 'complex_abs_time'):
                    diff = data_decoder_freq - decoder_out
                    prd_loss = tf.abs(tf.real(diff)) + tf.abs(tf.imag(diff))
                    # tf.summary.histogram('complex_abs', prd_loss)
                    # tf.summary.histogram('log_complex_abs', tf.log(prd_loss))
                    prd_loss = tf.reduce_mean(prd_loss)
                    tf.summary.scalar('f_complex_abs', prd_loss)
                if (pd['freq_loss'] == 'complex_square') \
                   or (pd['freq_loss'] == 'complex_square_time'):
                    diff = data_decoder_freq - decoder_out
                    prd_loss = tf.real(diff)*tf.real(diff) + tf.imag(diff)*tf.imag(diff)
                    # tf.summary.histogram('complex_square', prd_loss)
                    prd_loss = tf.reduce_mean(prd_loss)
                    tf.summary.scalar('f_complex_square', prd_loss)

                def expand_dims_and_transpose(input_tensor, pd, freqs):
                    output = tf.expand_dims(input_tensor, 1)
                    if pd['fft_compression_rate']:
                        zero_coeffs = freqs - int(input_tensor.shape[-1])
                        zero_stack = tf.zeros(output.shape[:-1].as_list()
                                              + [zero_coeffs], tf.complex64)
                        output = tf.concat([output, zero_stack], -1)
                    return output
                decoder_out = expand_dims_and_transpose(decoder_out, pd, dec_freqs)
                decoder_out = eagerSTFT.istft(decoder_out, window,
                                              nperseg=pd['window_size'],
                                              noverlap=pd['overlap'],
                                              epsilon=pd['epsilon'])
                # data_encoder_gt = expand_dims_and_transpose(encoder_in, pd, enc_freqs)
                decoder_out = tf.transpose(decoder_out, [0, 2, 1])
            elif pd['linear_reshape']:
                if pd['downsampling'] > 1:
                    decoder_out_t = tf.transpose(decoder_out, [0, 2, 1])
                    decoder_out_t = eagerSTFT.interpolate(decoder_out_t, pd['step_size'])
                    decoder_out = tf.transpose(decoder_out_t, [0, 2, 1])
                decoder_out = tf.reshape(decoder_out,
                                         [pd['batch_size'],
                                          pd['pred_samples'], 1])

            time_loss = tf.losses.mean_squared_error(
                tf.real(data_decoder_time),
                tf.real(decoder_out[:, :pd['pred_samples'], :]))
            if not pd['fft']:
                loss = time_loss
            else:
                if (pd['freq_loss'] == 'ad_time') or \
                   (pd['freq_loss'] == 'log_mse_time') or \
                   (pd['freq_loss'] == 'mse_time') or \
                   (pd['freq_loss'] == 'log_mse_mse_time') or \
                   (pd['freq_loss'] == 'complex_square_time') or \
                   (pd['freq_loss'] == 'complex_abs_time'):
                    print('using freq and time based loss.')
                    lambda_t = 1
                    loss = prd_loss*lambda_t + time_loss
                    tf.summary.scalar('lambda_t', lambda_t)
                elif (pd['freq_loss'] is None):
                    print('time loss only')
                    loss = time_loss
                else:
                    loss = prd_loss

            learning_rate = tf.train.exponential_decay(pd['init_learning_rate'],
                                                       global_step,
                                                       pd['decay_steps'],
                                                       pd['decay_rate'],
                                                       staircase=True)
            tf.summary.scalar('learning_rate', learning_rate)

            if (pd['cell_type'] == 'orthogonal' or pd['cell_type'] == 'cgRNN') \
               and (pd['stiefel'] is True):
                optimizer = co.RMSpropNatGrad(learning_rate, global_step=global_step)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(loss)

            with tf.variable_scope("clip_grads"):
                capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]

            # grad_summary = tf.histogram_summary(grads)
            # training_op = optimizer.minimize(loss, global_step=global_step)
            self.training_op = optimizer.apply_gradients(capped_gvs,
                                                         global_step=global_step)
            tf.summary.scalar('time_loss', time_loss)
            tf.summary.scalar('training_loss', loss)

            self.init_op = tf.global_variables_initializer()
            self.summary_sum = tf.summary.merge_all()
            self.total_parameters = compute_parameter_total(tf.trainable_variables())
            self.saver = tf.train.Saver()
            self.loss = loss
            self.global_step = global_step
            self.decoder_out = decoder_out
            self.data_nd = data_nd
            self.data_encoder_time = data_encoder_time
            self.data_decoder_time = data_decoder_time
            if pd['fft']:
                self.window = window
