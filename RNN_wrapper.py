import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
from IPython.core.debugger import Tracer
debug_here = Tracer()


class RnnWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, num_proj, sample_prob, cell, reuse=None, name='GRU_wrapper'):
        self._cell = cell
        self._num_proj = num_proj
        self._sample_prob = sample_prob
        print('GRU wrapped')
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

    def to_string(self):
        return 'wrapped_gru' + '_cell_size_' + str(self._cell._num_units) + \
               '_num_proj_' + str(self._num_proj)

    def close(self):
        self._sample_prob = 0.0

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._num_proj

    def zero_state(self, batch_size, dtype=tf.float32):
        out = tf.zeros([batch_size] + [self.output_size])
        first_state = tf.zeros([batch_size] + [self._cell.state_size])
        return LSTMStateTuple(out, first_state)

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""
        previous_output, previous_state = state
        if self._sample_prob == 0.0:
            # network is closed.
            inputs = previous_output
            print('cell fully closed.')
        elif self._sample_prob == 1.0:
            print('cell fully open.')
        else:
            def sel_input():
                return inputs

            def sel_prev_output():
                return previous_output

            condition = tf.greater(self._sample_prob, tf.random_uniform([]))
            inputs = tf.cond(condition, sel_input, sel_prev_output)

        # Run the rnn as usual
        output, new_state = self._cell(inputs, state[-1], scope)

        output_shape = output.get_shape().as_list()
        with tf.variable_scope('output_proj'):
            w_proj = tf.get_variable('projection_weights',
                                     [output_shape[-1], self._num_proj])
            b_proj = tf.get_variable('projection_bias', self._num_proj,
                                     initializer=tf.zeros_initializer())
            output = tf.matmul(output, w_proj) + b_proj
        return output, LSTMStateTuple(output, new_state)


class RnnInputWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, sample_prob, cell, reuse=None, name='GRU_wrapper'):
        self._cell = cell
        self._sample_prob = sample_prob
        print('GRU wrapped')
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not a RNNCell.")

    def to_string(self):
        return '_input_wrapper_' + self._cell.to_string()

    def close(self):
        self._sample_prob = 0.0

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def zero_state(self, batch_size, dtype=tf.float32):
        return self._cell.zero_state(batch_size)

    def __call__(self, inputs, state, scope=None):
        """Run the cell and add a residual connection."""
        previous_output, previous_state = state
        if self._sample_prob == 0.0:
            # network is closed.
            inputs = previous_output
            print('cell fully closed.')
        elif self._sample_prob == 1.0:
            print('cell fully open.')
        else:
            def sel_input():
                return inputs

            def sel_prev_output():
                return previous_output

            condition = tf.greater(self._sample_prob, tf.random_uniform([]))
            inputs = tf.cond(condition, sel_input, sel_prev_output)
        return self._cell(inputs, state, scope)
