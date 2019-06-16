import tensorflow as tf
import logging
logger = logging.getLogger(__name__)


class PolicyNetwork:
    def __init__(self, FLAGS, batch_sz, name='LocationNetwork'):
        self.name = name
        self.max_loc_rng = FLAGS.max_loc_rng
        self.init_loc_rng = FLAGS.init_loc_rng
        self.B = batch_sz
        self.loc_dim = FLAGS.loc_dim
        self.img_shape =  FLAGS.img_shape
        self._units = FLAGS.num_hidden_fc

        if (FLAGS.loc_std != FLAGS.loc_std_min):
            self.std = tf.train.exponential_decay(FLAGS.loc_std, tf.train.get_global_step(),
                                                  decay_steps=FLAGS.train_batches_per_epoch,
                                                  decay_rate=0.9)
            self.std = tf.maximum(self.std, FLAGS.loc_std_min)
        else:
            self.std = FLAGS.loc_std

        self.calc_encoding = tf.make_template(self.name, self._input_encoding)

    def _input_encoding(self, inputs):
        hidden = tf.layers.dense(inputs, self._units, activation=tf.nn.relu)
        return tf.layers.dense(hidden, self.loc_dim, activation=tf.nn.tanh)

    def next_actions(self, inputs, is_training, n_policies):
        """
        NOTE: Does not propagate back into the inputs. Gradients for the policyNet weights only flow through the loc_mean (which should only be used in the REINFORCE_loss calculations)
        NOTE: tf by default backpropagates through sampling if using tf.distributions.Normal()
        """
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.stop_gradient(tf.concat(inputs, axis=1))
            loc_mean = self.calc_encoding(inputs)    # [B * n_policies, loc_dim]
            loc_mean = tf.clip_by_value(loc_mean, self.max_loc_rng * -1, self.max_loc_rng * 1)

            loc_mean = tf.reshape(loc_mean, [self.B, n_policies, self.loc_dim]) if n_policies > 1 else tf.reshape(loc_mean, [self.B, self.loc_dim])

            with tf.name_scope('sample_locs'):
                loc = tf.cond(is_training,
                              lambda: tf.distributions.Normal(loc=tf.stop_gradient(loc_mean), scale=self.std).sample(),
                              lambda: tf.identity(loc_mean),
                              name='sample_loc_cond')

        return loc, loc_mean  # [B, n_policies, loc_dims] or [B, loc_dims]

    def REINFORCE_losses(self, returns, baselines, locs, loc_means):
        with tf.name_scope('loc_losses'):
            # NOTE: advantages / baselines start at t=1!
            # last baseline is based on final state, thereby not used anymore (baseline for location t has to always come from the state t - 1)
            advantages = returns - baselines
            # only want gradients flow through the suggested mean
            # includes first location, but irrelevant for gradients as not depending on parameters
            z = (tf.stop_gradient(locs) - loc_means) / self.std  # [T, batch_sz, loc_dims]
            loc_loglik = -0.5 * tf.reduce_mean(tf.square(z), axis=-1)

            # do not propagate back through advantages
            loc_losses = -1. * loc_loglik * tf.stop_gradient(advantages)

        return loc_losses

    def random_loc(self, rng=1., shp=None):
        with tf.name_scope(self.name):
            rng = min(rng, self.max_loc_rng)
            if shp is None:
                shp = [self.B, self.loc_dim]
            else:
                shp = shp + [self.loc_dim]
            loc = tf.random_uniform(shp, minval=rng * -1., maxval=rng * 1.)
        return loc, loc

    def uniform_loc_10(self, n_policies):
        """Spread out locations for evaluation, for n_policies=10"""
        assert n_policies == 10

        with tf.name_scope(self.name):
            loc = []
            loc.append([-1., -1.])  # a corner
            row = [-0.5, 0., 0.5]
            for i in row:
                for j in row:
                    loc.append([i, j])
            loc = tf.constant(loc)
            loc = tf.tile(loc[tf.newaxis, :, :], [self.B, 1, 1])
        return loc, loc

    def find_seen_locs(self, locations, num_glimpses, closeness=3):
        """
        Args:
            closeness: in pixels
        """
        if (self.img_shape[0] != self.img_shape[1]):
            logger.warning('Location closeness assumes square images')
        closeness /= (self.img_shape[0] / 2)  # into [0, 2] range (as locations are in [-1, 1] range). Assuming square image

        seen = [tf.zeros(self.B, dtype=tf.bool)]  # nothing has been seen yet at t=0
        for t in range(1, num_glimpses):  # TODO: if using dynamic glimpses changes this to a tf.while loop over self.num_glimpses_dyn
            loc_dist = tf.norm(locations[t][tf.newaxis] - locations[:t], axis=-1, ord='euclidean')  # [t, B]
            is_close = tf.reduce_any(loc_dist <= closeness, axis=0)  # [B]
            seen.append(is_close)
        seen_locs = tf.stack(seen, axis=0)  # [T, B]
        return seen_locs

    def inital_loc(self):
        return self.random_loc(rng=self.init_loc_rng)

    @property
    def output_size(self):
        return self.loc_dim

