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

        self._img_shape = FLAGS.img_shape
        self._scale_sizes = FLAGS.scale_sizes
        ix, iy = tf.meshgrid(tf.range(self._img_shape[1]), tf.range(self._img_shape[0]))
        self._ix, self._iy = tf.cast(ix[tf.newaxis], tf.float32), tf.cast(iy[tf.newaxis], tf.float32)  # each [1, 28, 28]

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

        # TODO: could be done more efficiently
        updated_seen = self._update_seen(prev_state['seen'], location)

        next_state = {'c'        : c,
                      's'        : next_s_output,
                      's_state'  : next_s_state,
                      'fb'       : fb,
                      'uk_belief': prev_state['uk_belief'],
                      'seen'     : updated_seen}

        return next_state

    def _believe_update(self, current_fb, KLdiv, time):
            # TODO: uk_belief
            assert KLdiv.get_shape().ndims == 2  # [B, hyp]

            fb = current_fb + KLdiv

            c = tf.nn.softmax(1 * -fb / (time + 1), axis=1)
            c = tf.stop_gradient(c)

            return fb, c

    def _update_seen(self, seen, loc):
        """
        Args:
            loc: [B, loc_dim], where loc_dim = (y, x)
        """
        # from [-1, 1] into [0, img_shape - 1] range (-1 because of 0-indexing)
        loc_pixel_x = ((self._img_shape[1] - 1) * loc[:, 1] + self._img_shape[1] - 1) / 2
        loc_pixel_y = ((self._img_shape[0] - 1)* loc[:, 0] + self._img_shape[0] - 1) / 2
        x_boundry = (loc_pixel_x - self._scale_sizes[0] / 2, loc_pixel_x + self._scale_sizes[0] / 2)
        y_boundry = (loc_pixel_y - self._scale_sizes[0] / 2, loc_pixel_y + self._scale_sizes[0] / 2)

        new = ((self._ix >= x_boundry[0][:, tf.newaxis, tf.newaxis]) & (self._ix <= x_boundry[1][:, tf.newaxis, tf.newaxis])
               & (self._iy >= y_boundry[0][:, tf.newaxis, tf.newaxis]) & (self._iy <= y_boundry[1][:, tf.newaxis, tf.newaxis]))
        return tf.logical_or(new, seen)

    @property
    def initial_state(self):
        return {'c'        : tf.fill([self._B, self._num_classes_kn], 1. / self._num_classes_kn),
                's'        : self._cell.zero_state(self._B, tf.float32),  # output TODO: only h if usig an LSTM cell
                's_state'  : self._cell.zero_state(self._B, tf.float32),  # full state
                'fb'       : tf.zeros([self._B, self._num_classes_kn]),
                'uk_belief': tf.zeros([self._B]),
                'seen'     : tf.zeros([self._B] + self._img_shape[:2], dtype=tf.bool)}

