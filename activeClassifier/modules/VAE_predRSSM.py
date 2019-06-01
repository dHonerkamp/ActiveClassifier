import tensorflow as tf
from tensorflow import distributions as tfd

from tools.tf_tools import FiLM_layer, pseudo_LogRelaxedBernoulli, exponential_mov_avg
from tools.MSE_distribution import MSEDistribution


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

    def _prior(self, inputs, out_shp=None):
        inputs = tf.concat(inputs, axis=1)
        hidden = tf.layers.dense(inputs, **self._kwargs)
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
        if out_shp is not None:
            out = {k: tf.reshape(v, out_shp) if (v is not None) else None for k, v in out.items()}
        return out

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

        if self.z_dist == 'N':
            sigma = tf.layers.dense(hidden, units=self.size_z, activation=tf.nn.softplus, name='sigma')
            sigma += self._min_stddev
        else:
            sigma = tf.zeros_like(mu)

        mu = self._centering(mu, self.ema_post)

        sample, log_sample = self._sample(mu, sigma, temp=self.temp_post)

        return {'mu': mu,
                'sigma': sigma,
                'sample': sample,
                'log_sample': log_sample}

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
                    sigma = None
            else:
                hidden = tf.layers.dense(inputs, **self._kwargs)
                mu_logits = tf.layers.dense(hidden, self.size_glimpse_out, None)
                if self._gl_std == -1:
                    sigma = tf.layers.dense(hidden, self.size_glimpse_out, tf.nn.softplus)
                    sigma += self._min_stddev
                else:
                    sigma = None

            mu_prob = tf.nn.sigmoid(mu_logits)

            if true_glimpse is not None:
                loss = self._nll_loss(mu_prob, sigma, true_glimpse)
            else:
                loss = None

        return {'sample': mu_prob,
                'logits': mu_logits,
                'mu_prob': mu_prob,
                'sigma': sigma,
                'loss'  : loss}

    def _nll_loss(self, mu_prob, sigma, true_glimpse):
        if self._gl_std == -1:
            dist = tf.distributions.Normal(loc=mu_prob, scale=sigma)
            loss = -dist.log_prob(true_glimpse)
            loss = tf.reduce_sum(loss, axis=-1)
        else:
            dist = MSEDistribution(mu_prob)
            loss = -dist.log_prob(true_glimpse)
            loss = tf.reduce_sum(loss, axis=-1)
        return loss


    @property
    def output_size(self):
        return self.size_glimpse_out