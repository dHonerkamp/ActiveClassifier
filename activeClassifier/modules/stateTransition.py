import numpy as np
import tensorflow as tf


class StateTransition:
    def __init__(self, FLAGS, rnn_units, batch_sz, conv_shape_z):
        self._num_classes = FLAGS.num_classes
        self._num_classes_kn = FLAGS.num_classes_kn
        self._loc_dim = FLAGS.loc_dim
        self._B = batch_sz

        self._convLSTM = FLAGS.convLSTM
        self._conv_shape_z = conv_shape_z
        self._conv_shape = [7, 7, self._conv_shape_z[-1]]
        self._conv_shrink = np.array(FLAGS.img_shape[:-1]) // np.array(self._conv_shape[:2])
        assert len(np.unique(self._conv_shrink)) == 1, 'Not shrunk by the same factor across the x and y axis'
        self._conv_shrink = self._conv_shrink[0]
        assert ((np.array(FLAGS.img_shape[:-1]) % np.array(self._conv_shape[:2])) == 0).all(), 'For now use even divisible state size, so there is a clear mapping of locations onto a cell.'
        if FLAGS.convLSTM:
            self._cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=self._conv_shape, output_channels=conv_shape_z[-1], kernel_shape=[2, 2])
            assert len(FLAGS.scale_sizes) == 1, 'glimpse_idx not taking into account multiple scales so far!'
        else:
            self._cell = tf.nn.rnn_cell.GRUCell(rnn_units)

    def __call__(self, inputs, prev_state):
        """
        Returns:
            output, next_state
        """
        z, action, glimpse_idx = inputs

        if self._convLSTM:
            # stitch z onto an empty canvas according to the lcoation of the glimpse
            z = tf.reshape(z, [self._B] + self._conv_shape_z)

            glimpse_idx = glimpse_idx[:, 1:3, 1:3] // self._conv_shrink[np.newaxis]
            cell_input = tf.scatter_nd(glimpse_idx, z, shape=[self._B] + self._conv_shape)
        else:
            cell_input = z

        next_s_output, next_s_state = self._cell(cell_input, prev_state['s_state'])

        next_state = {'c': prev_state['c'],
                      'l': action,
                      's': tf.layers.flatten(next_s_output),
                      's_state': next_s_state,
                      'fb': prev_state['fb'],
                      'uk_belief': prev_state['uk_belief']}

        return next_state

    def initial_state(self, batch_sz, initial_location):
        if self._convLSTM:
            output = self._cell.zero_state(batch_sz, dtype=tf.float32).h
        else:
            output = self._cell.zero_state(batch_sz, dtype=tf.float32)

        return {'c': tf.fill([batch_sz, self._num_classes_kn], 1. / self._num_classes_kn),
                'l': initial_location,
                's': tf.layers.flatten(output),  # output
                's_state': self._cell.zero_state(batch_sz, dtype=tf.float32),  #
                'fb': tf.zeros([batch_sz, self._num_classes_kn]),
                'uk_belief': tf.zeros([batch_sz])}

    @property
    def state_size(self):
        if type(self._cell.state_size) == int:  # GRU cell
            cell_size = self._cell.state_size
        elif len(self._cell.state_size) == 2:  #LSTMStateTuple
            cell_size = self._cell.state_size[0].to_list()
        else:
            raise ValueError('Unknown state size: {}'.format(self._cell.state_size))
        return {'c': self._num_classes_kn,
                'l': self._loc_dim,
                's': cell_size,  # output
                'fb': self._num_classes_kn,
                'uk_belief': 1}


# class StateTransition_AC:
#     def __init__(self, FLAGS, rnn_units, embed_size):
#         self._num_classes = FLAGS.num_classes
#         self._loc_dim = FLAGS.loc_dim
#         self._cell = tf.nn.rnn_cell.GRUCell(rnn_units)
#         self._kwargs = dict(units=embed_size, activation=tf.nn.relu)
#
#     def __call__(self, inputs, prev_state, labels):
#         """
#         Returns:
#             output, next_state
#         """
#         z, labels, action = inputs
#
#         inputs = tf.concat([z, labels], axis=1)
#         next_s = self._cell(inputs, prev_state['s'])
#
#         next_state = {'c': prev_state['c'],
#                       'l': action,
#                       's': next_s}
#
#         return next_state
#
#     # def zero_state(self, batch_size, dtype):
#     #     return {name: tf.zeros([batch_size, size], dtype) for name, size in self.state_size}
#
#     def initial_state(self, batch_sz, initial_location):
#         return {'c': tf.fill([batch_sz, self._num_classes], 1. / self._num_classes),
#                 'l': initial_location,
#                 's': tf.zeros([batch_sz, self._cell.state_size])}
#
#     @property
#     def state_size(self):
#         return {'c': self._num_classes,
#                 'l': self._loc_dim,
#                 's': self._cell.state_size}
