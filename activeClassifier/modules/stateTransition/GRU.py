import tensorflow as tf

from activeClassifier.modules.stateTransition.base import StateTransition

class StGRU(StateTransition):
    def __init__(self, FLAGS, batch_sz, conv_shape_z):
        super().__init__(FLAGS, batch_sz)

        self._cell = tf.nn.rnn_cell.GRUCell(FLAGS.size_rnn)

    def _get_cell_input(self, z, glimpse_idx):
        return z

    def _get_zero_cell_output(self, batch_sz):
        return self._cell.zero_state(batch_sz, dtype=tf.float32)
