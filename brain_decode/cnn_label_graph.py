import sys
import ipdb
import numpy as np
import tensorflow as tf
import tensorflow.nn.rnn_cell as rnn_cell
import scipy.signal as scisig
sys.path.insert(0, "../")
import window_learning as wl
import eager_STFT as stft
debug_here = ipdb.set_trace


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


class SingleLabelClassificationGraph(object):
    """Come up with a single label for the classification
       problem. """

    def __init__(self, pd):
        self.pd = pd
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            self.data = tf.placeholder(tf.float32, [None, pd['channels'], pd['time']])
            self.labels = tf.placeholder(tf.int32, [None])
            self.labels_one_hot = tf.one_hot(self.labels, pd['label_total'])

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

            def stft_filter(in_data, window, pd):
                '''
                Compute a windowed stft and do low pass filtering if
                necessary.

                Params:
                    in_data: Tensor of shape [None, channels, time]
                    window: Tensor of shape [nperseg]
                    pd: the parameter dictionary

                Returns:
                    freq_data: A complex Tensor of shape [None, channels, time, freq]
                '''
                in_data_fft = stft.stft(in_data, window,
                                        pd['window_size'], pd['overlap'])
                freqs = int(in_data_fft.shape[-1])
                idft_shape = in_data_fft.shape.as_list()
                if pd['fft_compression_rate']:
                    compressed_freqs = int(freqs/pd['fft_compression_rate'])
                    print('fft_compression_rate', pd['fft_compression_rate'],
                          'freqs', freqs,
                          'compressed_freqs', compressed_freqs)
                    # remove frequencies from the last dimension.
                    return in_data_fft[..., :compressed_freqs], idft_shape, freqs
                else:
                    return in_data_fft, idft_shape, freqs

            data_freq, idft_shape, freqs = stft_filter(self.data, window, pd)

            # add a visualization of the spectrogram to tensorboard
            data_freq_abs = tf.math.abs(data_freq)
            to_tensorboard = tf.expand_dims(data_freq_abs[:, 0, :, :], [-1])
            self.freq_abs_summary = tf.summary.image('spectogram', to_tensorboard)

            # evaluate the bidirectional rnn.
            # concatenate the complex numbers away or use absolute value.
            if pd['magnitude_only']:
                data_freq_rnn = data_freq_abs
            elif pd['magnitude_and_phase']:
                data_freq_psi = tf.math.angle(data_freq)
                data_freq_rnn = tf.concat([data_freq_abs, data_freq_psi], 1)
            else:
                # concatenate real and imaginary parts along the channel dimension.
                data_freq_rnn = tf.concat([tf.real(data_freq), tf.imag(data_freq)], 1)

            # flatten the channels
            # time_steps = data_freq_rnn.shape.as_list()[2]
            data_freq_rnn = tf.transpose(data_freq_rnn, [0, 2, 3, 1])
            print('data_freq shape', data_freq_rnn.shape)
            # run the rnn
            # bidirectional_layer = tf.keras.layers.Bidirectional(gru_cell)
            # outputs, _ = bidirectional_layer(data_freq_rnn)
            # tf-keras is terrible it's time to move to pytorch.

            def run_cnn(data_freq_rnn, train):
                # data_freq_rnn = tf.expand_dims(data_freq_rnn, -1)
                if train and pd['input_dropout']:
                    print('input_dropout', pd['input_dropout'])
                    cnn_input = tf.nn.dropout(data_freq_rnn, pd['input_dropout'])
                else:
                    cnn_input = data_freq_rnn

                for layer_no, cnn_params in enumerate(pd['cnn_layers']):
                    hidden = tf.layers.conv2d(
                        cnn_input, cnn_params['f'],
                        cnn_params['k'], cnn_params['s'],
                        padding='valid', data_format='channels_last',
                        activation=tf.nn.relu, name='cnn_' + str(layer_no))
                    print('cnn_layer', layer_no, hidden.shape)
                    if train and pd['dropout']:
                        print('dropout', pd['dropout'])
                        hidden = tf.nn.dropout(hidden, pd['dropout'])
                    cnn_input = hidden

                debug_here()
                dense_in = tf.reshape(
                    hidden, [-1, hidden.shape[1]*hidden.shape[2]*hidden.shape[3]])
                for layer_no, dense_unit in enumerate(pd['dense_units']):
                    with tf.variable_scope('dense_layer_' + str(layer_no)):
                        print('dense_in', layer_no, dense_in.shape)
                        hidden = tf.layers.dense(dense_in, dense_unit,
                                                 tf.nn.relu, name='hidden_dense')
                    if train and pd['dropout']:
                        hidden = tf.nn.dropout(hidden,
                                               pd['dropout'])
                    dense_in = hidden

                out = tf.layers.dense(hidden, pd['label_total'], None,
                                      name='linear_out')
                return out

            with tf.variable_scope('mutlilayer_cnn') as vs:
                self.out = run_cnn(data_freq_rnn, True)
                vs.reuse_variables()
                self.out_val = run_cnn(data_freq_rnn, False)

            # select the center value
            self.sig_out = tf.nn.sigmoid(self.out)
            self.sig_out_val = tf.nn.sigmoid(self.out_val)
            # self.sig_out_center_val = self.sig_out_center
            self.loss = tf.losses.sigmoid_cross_entropy(
                logits=self.out, multi_class_labels=self.labels_one_hot)

            if pd['learning_rate_decay']:
                learning_rate = tf.train.exponential_decay(
                    learning_rate=pd['learning_rate'], global_step=self.global_step,
                    decay_rate=pd['learning_rate_decay'], decay_steps=pd['decay_steps'],
                    staircase=True)
            else:
                learning_rate = pd['learning_rate']
            self.learning_rate_summary = tf.summary.scalar('learning_rate',
                                                           learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.weight_update = optimizer.minimize(self.loss,
                                                    global_step=self.global_step)
            self.init_op = tf.global_variables_initializer()
            self.parameter_total = compute_parameter_total(tf.trainable_variables())
