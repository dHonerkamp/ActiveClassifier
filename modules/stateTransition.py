import tensorflow as tf


class StateTransition:
    def __init__(self, FLAGS, rnn_units):
        self._num_classes = FLAGS.num_classes
        self._num_classes_kn = FLAGS.num_classes_kn
        self._loc_dim = FLAGS.loc_dim
        self._cell = tf.nn.rnn_cell.GRUCell(rnn_units)

    def __call__(self, inputs, prev_state):
        """
        Returns:
            output, next_state
        """
        z, action = inputs

        next_s_output, next_s_state = self._cell(z, prev_state['s_state'])

        next_state = {'c': prev_state['c'],
                      'l': action,
                      's': next_s_output,
                      's_state': next_s_state,
                      'fb': prev_state['fb'],
                      'uk_belief': prev_state['uk_belief']}

        return next_state

    def initial_state(self, batch_sz, initial_location):
        return {'c': tf.fill([batch_sz, self._num_classes_kn], 1. / self._num_classes_kn),
                'l': initial_location,
                's': tf.zeros([batch_sz, self._cell.state_size]),
                's_state': tf.zeros([batch_sz, self._cell.state_size]),
                'fb': tf.zeros([batch_sz, self._num_classes_kn]),
                'uk_belief': tf.zeros([batch_sz])}

    @property
    def state_size(self):
        return {'c': self._num_classes_kn,
                'l': self._loc_dim,
                's': self._cell.state_size,
                'fb': self._num_classes_kn,
                'uk_belief': 1}


class StateTransition_AC:
    def __init__(self, FLAGS, rnn_units, embed_size):
        self._num_classes = FLAGS.num_classes
        self._loc_dim = FLAGS.loc_dim
        self._cell = tf.nn.rnn_cell.GRUCell(rnn_units)
        self._kwargs = dict(units=embed_size, activation=tf.nn.relu)

    def __call__(self, inputs, prev_state, labels):
        """
        Returns:
            output, next_state
        """
        z, labels, action = inputs

        inputs = tf.concat([z, labels], axis=1)
        next_s = self._cell(inputs, prev_state['s'])

        next_state = {'c': prev_state['c'],
                      'l': action,
                      's': next_s}

        return next_state

    # def zero_state(self, batch_size, dtype):
    #     return {name: tf.zeros([batch_size, size], dtype) for name, size in self.state_size}

    def initial_state(self, batch_sz, initial_location):
        return {'c': tf.fill([batch_sz, self._num_classes], 1. / self._num_classes),
                'l': initial_location,
                's': tf.zeros([batch_sz, self._cell.state_size])}

    @property
    def state_size(self):
        return {'c': self._num_classes,
                'l': self._loc_dim,
                's': self._cell.state_size}
