import tensorflow as tf
import logging
logger = logging.getLogger(__name__)

from activeClassifier.tools.MSE_distribution import MSEDistribution


class Decoder:
    def __init__(self, FLAGS, size_glimpse_out, name='VAEDecoder'):
        self.name = name
        self._size_glimpse_out = size_glimpse_out
        if FLAGS.pixel_obs_discrete:
            self._size_glimpse_out *= FLAGS.pixel_obs_discrete
        self.pixel_obs_discrete = FLAGS.pixel_obs_discrete
        self._gl_std = FLAGS.gl_std
        self._min_stddev = 1e-5
        self._kwargs = dict(units=FLAGS.num_hidden_fc, activation=tf.nn.relu)

    def decode(self, inputs, true_glimpse=None, out_shp=tuple([-1])):
        # TODO: HOW (AND IF) LOCATION SHOULD BE INCORPORATED IN THE INPUTS (z IS ALREADY KINDOF LOCATION DEPENDENT)
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.concat(inputs, axis=1)
            hidden = tf.layers.dense(inputs, **self._kwargs)
            mu_logits = tf.layers.dense(hidden, self._size_glimpse_out, None)
            if self._gl_std == -1:
                sigma = tf.layers.dense(hidden, self._size_glimpse_out, tf.nn.softplus)
                sigma += self._min_stddev
            else:
                sigma = None

            if self.pixel_obs_discrete:
                mu_logits = tf.reshape(mu_logits, [-1, self._size_glimpse_out // self.pixel_obs_discrete, self.pixel_obs_discrete])
                sigma = tf.reshape(sigma, [-1, self._size_glimpse_out // self.pixel_obs_discrete, self.pixel_obs_discrete]) if sigma else None
                mu_prob = tf.nn.softmax(mu_logits)
            else:
                mu_prob = tf.nn.sigmoid(mu_logits)

            if true_glimpse is not None:
                loss = self._nll_loss(mu_logits, mu_prob, sigma, true_glimpse)
                loss = tf.reshape(loss, list(out_shp))
            else:
                loss = None

        out = {'sample': mu_prob,
                'mu_logits': mu_logits,
                'mu_prob': mu_prob,
                'sigma': sigma,
               }
        out = {k: tf.reshape(v, list(out_shp) + self.output_shape) if (v is not None) else None for k, v in out.items()}
        out['loss'] = loss
        return out

    def _nll_loss(self, mu_logits, mu_prob, sigma, true_glimpse):
        if self.pixel_obs_discrete:
            true_glimpse_1hot = tf.cast(self.pixel_obs_discrete * true_glimpse, tf.int32)
            true_glimpse_1hot = tf.one_hot(true_glimpse_1hot, depth=self.pixel_obs_discrete)

            loss = tf.nn.softmax_cross_entropy_with_logits(labels=true_glimpse_1hot, logits=mu_logits)
            loss /= self.pixel_obs_discrete  # normalise by number of dimensions
        else:
            if self._gl_std == -1:
                dist = tf.distributions.Normal(loc=mu_prob, scale=sigma)
            else:
                dist = MSEDistribution(mu_prob)
            loss = -dist.log_prob(true_glimpse)
        loss = tf.reduce_sum(loss, axis=-1)
        return loss

    @property
    def output_shape(self):
        if self.pixel_obs_discrete:
            return [self._size_glimpse_out // self.pixel_obs_discrete, self.pixel_obs_discrete]
        else:
            return [self._size_glimpse_out]
