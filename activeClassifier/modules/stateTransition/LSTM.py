import tensorflow as tf

from activeClassifier.modules.stateTransition.base import StateTransition

class StLSTM(StateTransition):
    def __init__(self, FLAGS, batch_sz):
        super().__init__(FLAGS, batch_sz)
        cell_size = int(FLAGS.rnn_cell.replace('LSTM', ''))
        self._cell = tf.nn.rnn_cell.LSTMCell(cell_size)

    def _get_cell_input(self, z, glimpse_idx, action):
        input = tf.concat([z, action], axis=-1)
        return input

    def _get_zero_cell_output(self, batch_sz):
        return self._cell.zero_state(batch_sz, dtype=tf.float32).h
