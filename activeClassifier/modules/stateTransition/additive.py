import tensorflow as tf

from activeClassifier.modules.stateTransition.base import StateTransition


class StAdditive(StateTransition):
    def __init__(self, FLAGS, batch_sz, conv_shape_z):
        super().__init__(FLAGS, batch_sz)

        self._conv_shape_z = conv_shape_z
        self._cell = _rnn_cell_Additive(conv_shape_z)

    def _get_cell_input(self, z, glimpse_idx, last_action, next_action):
        return tf.reshape(z, [self._B] + self._conv_shape_z)

    def _get_zero_cell_output(self, batch_sz):
        return self._cell.zero_state(batch_sz, dtype=tf.float32)


class _rnn_cell_Additive:
    '''Simple additive cell: new inputs are just added to the state.'''
    def __init__(self, input_shape):
        self._input_shape = input_shape

    @property
    def state_size(self):
        return self._input_shape

    @property
    def output_size(self):
        return self._input_shape

    @property
    def trainable_variables(self):
        return []

    def zero_state(self, batch_sz, dtype):
        return tf.zeros([batch_sz] + self.state_size, dtype=dtype)

    def __call__(self, inputs, state):
        output = state + inputs
        return output, output