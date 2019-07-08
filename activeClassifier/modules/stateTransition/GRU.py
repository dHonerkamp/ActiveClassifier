import tensorflow as tf

from activeClassifier.modules.stateTransition.base import StateTransition

class StGRU(StateTransition):
    def __init__(self, FLAGS, batch_sz):
        super().__init__(FLAGS, batch_sz)
        cell_size = int(FLAGS.rnn_cell.replace('GRU', ''))
        self._cell = tf.nn.rnn_cell.GRUCell(cell_size)

    def _get_cell_input(self, z, glimpse_idx, last_action, next_action):
        # input = tf.concat([z, next_action], axis=-1)  # cannot control the outputs based on inputs. So don't give it the next action
        input = tf.concat([z, last_action], axis=-1)
        return input

    def _get_zero_cell_output(self, batch_sz):
        return self._cell.zero_state(batch_sz, dtype=tf.float32)
