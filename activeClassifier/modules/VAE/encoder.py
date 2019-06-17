import numpy as np
import tensorflow as tf
from tensorflow import distributions as tfd
import logging
logger = logging.getLogger(__name__)

from activeClassifier.tools.tf_tools import FiLM_layer, pseudo_LogRelaxedBernoulli, exponential_mov_avg


class Encoder:
    """
    Infer hidden encodings z, either from prior or posterior
    """
    def __init__(self, FLAGS, patch_shape, is_training, name='VAEEncoder'):
        self.name = name
        self.size_z = FLAGS.size_z
        self.z_dist = FLAGS.z_dist
        self._min_stddev = 1e-5
        self._kwargs = dict(units=FLAGS.num_hidden_fc, activation=tf.nn.relu)
        self.use_conv = FLAGS.use_conv
        self.patch_shape = patch_shape
        self.calc_prior = tf.make_template(self.name + '/prior', self._prior)
        self.calc_post  = tf.make_template(self.name + '/posterior', self._posterior)

        # only relevant for bernoulli latent variables / gumbel relaxation
        # following https://arxiv.org/pdf/1611.00712.pdf
        self.temp_prior = 2 / 3
        self.temp_post = 1 / 2
        # exponential moving averages over mini batches
        self.ema_prior = tf.Variable(tf.zeros([self.size_z]), trainable=False, name='ExpMovAvg_RelaxedBernoulli_prior')
        self.ema_post  = tf.Variable(tf.zeros([self.size_z]), trainable=False, name='ExpMovAvg_RelaxedBernoulli_post')
        self.is_training = is_training
        self.z_B_center = FLAGS.z_B_center
        self.conv_shape_z = [None]

    def _sample(self, mu, sigma, temp=None):
        if self.z_dist == 'N':
            return tfd.Normal(loc=mu, scale=sigma).sample(), None
        elif self.z_dist == 'B':
            # return tfd.Bernoulli(logits=mu, dtype=tf.float32).sample()  # not reparameterized
            # return RelaxedBernoulli(temperature=temp, logits=mu).sample()
            log_sample = pseudo_LogRelaxedBernoulli(logits=mu, temperature=temp).sample()
            return tf.nn.sigmoid(log_sample), log_sample

    def _centering(self, mu, ema):
        """Center the logits. If training: update the exponential moving average. Decay as in https://arxiv.org/pdf/1611.00712.pdf."""
        if (self.z_dist == 'B') and self.z_B_center:
            mu -= ema
            new_ema = tf.cond(self.is_training,
                              lambda: exponential_mov_avg(tracked_var=tf.reduce_mean(mu, axis=0), avg_var=ema, decay=0.9),
                              lambda: ema,
                              name='update_ema_cond')
            assignment = tf.assign(ema, new_ema)
            with tf.control_dependencies([assignment]):
                mu = tf.identity(mu)
        return mu

    def _unflatten_glimpse(self, glimpse):
        """
        Glimpse is flattened[batch_sz, num_scales, H, W, C].
        Bring into [B, num_scales * scale_sz[0], scale_sz[0], C] (retina scales stacked vertically)
        """
        shp = [tf.shape(glimpse)[0]] + [self.patch_shape[0] * self.patch_shape[1], self.patch_shape[2], self.patch_shape[3]]
        return tf.reshape(glimpse, shp)

    def _prior(self, inputs, out_shp=tuple([-1])):
        """Inputs: hyp, s, l"""
        inputs_concat = tf.concat(inputs, axis=1)
        hidden = tf.layers.dense(inputs_concat, **self._kwargs)
        hidden = tf.concat([hidden, inputs[1], inputs[2]], axis=1)  # skip connection for non-class conditional (enable reading from memory for already seen)
        hidden = tf.layers.dense(hidden, **self._kwargs)
        mu = tf.layers.dense(hidden, self.size_z, None)

        if self.z_dist == 'N':
            sigma = tf.layers.dense(hidden, self.size_z, tf.nn.softplus, name='sigma')
            sigma += self._min_stddev
        else:
            sigma = None

        # TODO: NOT SURE IF EMA WORKS A HUNDRED PERCENT CORRECTLY
        mu = self._centering(mu, self.ema_prior)

        sample, log_sample = self._sample(mu, sigma, temp=self.temp_prior)

        out = {'mu': mu,
               'sigma': sigma,
               'sample': sample,
               'log_sample': log_sample}
        if out_shp is None:
            out_shp = [-1]
        out = {k: tf.reshape(v, list(out_shp) + self.output_shape_flat) if (v is not None) else None for k, v in out.items()}
        return out

    def _posterior(self, glimpse, l, s):
        if self.use_conv:
            glimpse = self._unflatten_glimpse(glimpse)
            hidden = tf.layers.conv2d(glimpse, filters=32, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)
            hidden = tf.layers.conv2d(hidden, filters=32, kernel_size=[2, 2], padding='valid', activation=tf.nn.relu)
        else:
            input = glimpse
            # input = tf.concat([glimpse, s], axis=-1)
            hidden = tf.layers.dense(input, **self._kwargs)
        hidden =  FiLM_layer(l, hidden, conv_input=self.use_conv)

        if self.use_conv:
            hidden = tf.layers.flatten(hidden)

        mu = tf.layers.dense(hidden, units=self.size_z, activation=None, name='mu')

        if self.z_dist == 'N':
            sigma = tf.layers.dense(hidden, units=self.size_z, activation=tf.nn.softplus, name='sigma')
            sigma += self._min_stddev
        else:
            sigma = tf.zeros_like(mu)

        mu = self._centering(mu, self.ema_post)

        sample, log_sample = self._sample(mu, sigma, temp=self.temp_post)

        out = {'mu': mu,
               'sigma': sigma,
               'sample': sample,
               'log_sample': log_sample}
        out = {k: tf.reshape(v, [-1] + self.output_shape_flat) if (v is not None) else None for k, v in out.items()}
        return out

    @property
    def output_shape_flat(self):
        """Flattened! shape. Using unflattened shape could break reduce_...(axis=-1) functions. Rather reshape it into output_shape in case needed"""
        return [self.size_z]

    @property
    def output_shape(self):
        return [self.size_z]


class EncoderConv(Encoder):
    def __init__(self, FLAGS, patch_shape, is_training, name='VAEEncoder'):
        super().__init__(FLAGS, patch_shape, is_training, name)
        assert self.patch_shape == [1, 8, 8, 1], 'Not implemented for these scales with patch shape {}'.format(self.patch_shape)

        self.n_filters = FLAGS.n_filters_encoder  # number of output filters
        self.conv_shape_z = [2, 2, self.n_filters]
        self.size_z = np.prod(self.conv_shape_z)  # flattened shape

    # def _prior(self, inputs, out_shp=tuple([-1])):
    #     """Inputs: hyp, s, l"""
    #     # TODO: think whether there is a good way to do this with convolutions (add hyp, loc as a channel or smth)

    def _posterior(self, glimpse, l, s):
        glimpse_img = self._unflatten_glimpse(glimpse)  # [B, num_scales * scale_sz[0], scale_sz[0], C]
        hidden = tf.layers.conv2d(glimpse_img, filters=self.n_filters, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)  # [6, 6]
        hidden = tf.layers.conv2d(hidden, filters=self.n_filters, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)  # [4, 4]
        mu = tf.layers.conv2d(hidden, filters=self.n_filters, kernel_size=[3, 3], padding='valid', activation=None, name='mu')  # [2, 2]

        if self.z_dist == 'N':
            sigma = tf.layers.conv2d(hidden, filters=self.n_filters, kernel_size=[3, 3], padding='valid', activation=tf.nn.softplus, name='sigma')
            assert sigma.shape[1:] == mu.shape[1:], 'mu: {} vs sigma: {}'.format(mu.shape, sigma.shape)
            sigma += self._min_stddev
        else:
            sigma= None

        mu = self._centering(mu, self.ema_post)
        assert mu.shape[1:] == self.conv_shape_z, 'mu shape {} vs conv_shape_z {}'.format(mu.shape[1:], self.conv_shape_z)

        sample, log_sample = self._sample(mu, sigma, temp=self.temp_post)

        out = {'mu': mu,
                'sigma': sigma,
                'sample': sample,
                'log_sample': log_sample}
        out = {k: tf.reshape(v, [-1] + self.output_shape_flat) if (v is not None) else None for k, v in out.items()}
        return out

    @property
    def output_shape(self):
        return self.conv_shape_z


class EncoderConvTower(EncoderConv):
    def __init__(self, FLAGS, patch_shape, is_training, name='VAEEncoder'):
        super().__init__(FLAGS, patch_shape, is_training, name)
        self.conv_shape_z = [6, 6, self.n_filters]
        self.size_z = np.prod(self.conv_shape_z)  # flattened shape

    # def _prior(self, inputs, out_shp=tuple([-1])):
    #     """Inputs: hyp, s, l"""
    #     # TODO: think whether there is a good way to do this with convolutions (add hyp, loc as a channel or smth)

    def _posterior(self, glimpse, l):
        glimpse_img = self._unflatten_glimpse(glimpse)  # [B, num_scales * scale_sz[0], scale_sz[0], C]
        hidden = tf.layers.conv2d(glimpse_img, filters=self.n_filters, kernel_size=[3, 3], padding='valid', activation=tf.nn.relu)  # [6, 6]
        l_conv = tf.tile(l[:, tf.newaxis, tf.newaxis, :], [1, hidden.shape[1], hidden.shape[2], 1])  # [B, conv, conv, loc]
        hidden = tf.concat([hidden, l_conv], axis=-1)  # filters + loc_dim
        hidden = tf.layers.conv2d(hidden, filters=self.n_filters, kernel_size=[2, 2], padding='valid', activation=tf.nn.relu)  # [6, 6]
        mu = tf.layers.conv2d(hidden, filters=self.n_filters, kernel_size=[2, 2], padding='valid', activation=None, name='mu')  # [6, 6]

        if self.z_dist == 'N':
            sigma = tf.layers.conv2d(hidden, filters=self.n_filters, kernel_size=[2, 2], padding='valid', activation=tf.nn.softplus, name='sigma')
            assert sigma.shape[1:] == mu.shape[1:], 'mu: {} vs sigma: {}'.format(mu.shape, sigma.shape)
            sigma += self._min_stddev
        else:
            sigma = None

        mu = self._centering(mu, self.ema_post)
        assert mu.shape[1:] == self.conv_shape_z, 'mu shape {} vs conv_shape_z {}'.format(mu.shape[1:], self.conv_shape_z)

        sample, log_sample = self._sample(mu, sigma, temp=self.temp_post)

        out = {'mu'        : mu,
               'sigma'     : sigma,
               'sample'    : sample,
               'log_sample': log_sample}
        out = {k: tf.reshape(v, [-1] + self.output_shape_flat) if (v is not None) else None for k, v in out.items()}
        return out
