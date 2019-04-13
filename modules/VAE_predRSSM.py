import tensorflow as tf
from tensorflow import distributions as tfd

from tools.tf_tools import TINY, output_into_gaussian_params, create_MLP, FiLM_layer
from tools.MSE_distribution import MSEDistribution

class Encoder:
    """
    Infer hidden encodings z, either from prior or posterior
    """
    def __init__(self, FLAGS, patch_shape, name='VAEEncoder'):
        self.name = name
        self.size_z = FLAGS.size_z
        self._min_stddev = 1e-5
        self._kwargs = dict(units=FLAGS.num_hidden_fc, activation=tf.nn.relu)
        self.use_conv = FLAGS.use_conv
        self.patch_shape = patch_shape
        self.calc_prior = tf.make_template(self.name + '/prior', self._prior)
        self.calc_post  = tf.make_template(self.name + '/posterior', self._posterior)

    def _prior(self, c, s, l):
        inputs = tf.concat([c, s, l], axis=1)
        hidden = tf.layers.dense(inputs, **self._kwargs)
        mu = tf.layers.dense(hidden, self.size_z, None)
        sigma = tf.layers.dense(hidden, self.size_z, tf.nn.softplus)
        sigma += self._min_stddev

        sample = tfd.Normal(loc=mu, scale=sigma).sample()

        return {'mu': mu,
                'sigma': sigma,
                'sample': sample}

    def _posterior(self, glimpse, l):
        if self.use_conv:
            # Glimpse is flattened[batch_sz, num_scales, H, W, C].
            # Bring into[B, num_scales * scale_sz[0], scale_sz[0], C] (retina scales stacked vertically)
            glimpse = tf.reshape(glimpse, [tf.shape(glimpse)[0]] + [self.patch_shape[0] * self.patch_shape[1], self.patch_shape[2], self.patch_shape[3]])
            hidden = tf.layers.conv2d(glimpse, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            hidden = tf.layers.conv2d(hidden, filters=32, kernel_size=[2, 2], padding='valid', activation=tf.nn.relu)
        else:
            hidden = tf.layers.dense(glimpse, **self._kwargs)
        hidden =  FiLM_layer(l, hidden, conv_input=self.use_conv)

        if self.use_conv:
            hidden = tf.layers.flatten(hidden)

        mu = tf.layers.dense(hidden, units=self.size_z, activation=None)
        sigma = tf.layers.dense(hidden, units=self.size_z, activation=tf.nn.softplus)
        sigma += self._min_stddev

        sample = tfd.Normal(loc=mu, scale=sigma).sample()

        return {'mu': mu,
                'sigma': sigma,
                'sample': sample}

    @property
    def output_size(self):
        return self.size_z


class Decoder:
    def __init__(self, FLAGS, size_glimpse_out, name='VAEDecoder'):
        self.name = name
        self.size_glimpse_out = size_glimpse_out
        self._gl_std = FLAGS.gl_std
        self._min_stddev = 1e-5
        self._kwargs = dict(units=FLAGS.num_hidden_fc, activation=tf.nn.relu)
        self.use_conv = FLAGS.use_conv

    def decode(self, inputs, true_glimpse=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.concat(inputs, axis=1)
            if self.use_conv:
                hidden = tf.layers.dense(inputs, units=5*5*32, activation=tf.nn.relu)
                hidden = tf.reshape(hidden, [tf.shape(hidden)[0], 5, 5, 32])
                hidden = tf.layers.conv2d_transpose(hidden, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
                mu_logits = tf.layers.conv2d_transpose(hidden, filters=1, kernel_size=[2, 2], padding='valid', activation=None)
                mu_logits = tf.layers.flatten(mu_logits)
                if self._gl_std == -1:
                    sigma = tf.layers.conv2d_transpose(hidden, filters=1, kernel_size=[2, 2], padding='valid', activation=tf.nn.softplus)
                    sigma += self._min_stddev
                    sigma = tf.layers.flatten(sigma)
            else:
                hidden = tf.layers.dense(inputs, **self._kwargs)
                mu_logits = tf.layers.dense(hidden, self.size_glimpse_out, None)
                if self._gl_std == -1:
                    sigma = tf.layers.dense(hidden, self.size_glimpse_out, tf.nn.softplus)
                    sigma += self._min_stddev

            mu_prob = tf.nn.sigmoid(mu_logits)

            if self._gl_std != -1:
                sigma = tf.fill(tf.shape(mu_prob), tf.cast(self._gl_std, tf.float32))

            if true_glimpse is not None:
                if self._gl_std == -1:
                    dist = tf.distributions.Normal(loc=mu_prob, scale=sigma)
                    loss = -dist.log_prob(true_glimpse)
                    loss = tf.reduce_sum(loss, axis=-1)
                else:
                    dist = MSEDistribution(mu_prob)
                    loss = -dist.log_prob(true_glimpse)
                    loss = tf.reduce_sum(loss, axis=-1)
            else:
                loss = None

        return {'sample': mu_prob,
                'logits': mu_logits,
                'params': tf.concat([mu_prob, sigma], axis=1),
                'loss'  : loss}

    @property
    def output_size(self):
        return self.size_glimpse_out