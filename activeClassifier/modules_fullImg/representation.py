import numpy as np
import tensorflow as tf

from activeClassifier.tools.tf_tools import FiLM_layer


class Representation:
    def __init__(self, FLAGS, name='reprNet'):
        self.name = name
        self.use_conv = False
        self._kwargs = dict(units=FLAGS.num_hidden_fc, activation=tf.nn.relu)
        self.size_r = FLAGS.size_r
        self._conv_shape = [7, 7, self.size_r // 49]
        assert np.prod(self._conv_shape) == self.size_r

    def calc_repr(self, glimpse, loc):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            input = glimpse
            hidden = tf.layers.dense(input, **self._kwargs)
            hidden =  FiLM_layer(loc, hidden, conv_input=self.use_conv)
            r = tf.layers.dense(hidden, units=self.size_r, activation=None, name='mu')

        return tf.reshape(r, [-1] + self.output_shape)

    @property
    def output_shape(self):
        return self._conv_shape