import numpy as np
import tensorflow as tf

from activeClassifier.modules.stateTransition.base import StateTransition


class StConvLSTM(StateTransition):
    def __init__(self, FLAGS, batch_sz, conv_shape_z):
        super().__init__(FLAGS, batch_sz)

        self._conv_shape_z = conv_shape_z
        self._conv_shape = [7, 7, self._conv_shape_z[-1]]
        self._conv_shrink = np.array(FLAGS.img_shape[:-1]) // np.array(self._conv_shape[:2])
        assert len(np.unique(self._conv_shrink)) == 1, 'Not shrunk by the same factor across the x and y axis'
        self._conv_shrink = self._conv_shrink[0]
        assert ((np.array(FLAGS.img_shape[:-1]) % np.array(
                self._conv_shape[:2])) == 0).all(), 'For now use even divisible state size, so there is a clear mapping of locations onto a cell.'
        self._cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=self._conv_shape, output_channels=conv_shape_z[-1], kernel_shape=[2, 2])
        assert len(FLAGS.scale_sizes) == 1, 'glimpse_idx not taking into account multiple scales so far!'

    def _get_cell_input(self, z, glimpse_idx, action):
        # stitch z onto an empty canvas according to the lcoation of the glimpse
        z = tf.reshape(z, [self._B] + self._conv_shape_z)

        glimpse_idx = glimpse_idx[:, 1:3, 1:3] // self._conv_shrink[np.newaxis]
        cell_input = tf.scatter_nd(glimpse_idx, z, shape=[self._B] + self._conv_shape)
        return cell_input

    def _get_zero_cell_output(self, batch_sz):
        return self._cell.zero_state(batch_sz, dtype=tf.float32).h
