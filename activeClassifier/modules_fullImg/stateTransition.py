import tensorflow as tf


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


class StateTransitionAdditive:
    def __init__(self, FLAGS, batch_sz, reprNet):
        self._B = batch_sz
        self._num_classes_kn = FLAGS.num_classes_kn

        self.reprNet = reprNet
        self._cell = _rnn_cell_Additive(reprNet.output_shape)

    def __call__(self, new_obs, location, prev_state, KLdiv, time, newly_done):
        """
        Returns:
            next_state
        """
        r = self.reprNet.calc_repr(new_obs, loc=location)

        next_s_output, next_s_state = self._cell(r, prev_state['s_state'])

        fb, c = self._believe_update(prev_state['fb'], KLdiv, time)

        next_state = {'c'        : prev_state['c'],
                      's'        : next_s_output,
                      's_state'  : next_s_state,
                      'fb'       : fb,
                      'uk_belief': prev_state['uk_belief']}

        return next_state

    def _believe_update(self, current_fb, KLdiv, time):
            # TODO: uk_belief
            assert KLdiv.get_shape().ndims == 2  # [B, hyp]

            fb = current_fb + KLdiv

            c = tf.nn.softmax(1 * -fb / (time + 1), axis=1)
            c = tf.stop_gradient(c)

            return fb, c

    @property
    def initial_state(self):
        return {'c'        : tf.fill([self._B, self._num_classes_kn], 1. / self._num_classes_kn),
                's'        : self._cell.zero_state(self._B, tf.float32),  # output TODO: only h if usig an LSTM cell
                's_state'  : self._cell.zero_state(self._B, tf.float32),  # full state
                'fb'       : tf.zeros([self._B, self._num_classes_kn]),
                'uk_belief': tf.zeros([self._B])}

