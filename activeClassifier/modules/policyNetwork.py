import tensorflow as tf


class PolicyNetwork:
    def __init__(self, FLAGS, batch_sz, n_policies, name='LocationNetwork'):
        self.name = name
        self.std = FLAGS.loc_std
        self.max_loc_rng = FLAGS.max_loc_rng
        self.batch_sz = batch_sz
        self.loc_dim = FLAGS.loc_dim
        self._units = FLAGS.num_hidden_fc

        self.calc_encoding = tf.make_template(self.name, self._input_encoding)

    def _input_encoding(self, inputs):
        hidden = tf.layers.dense(inputs, self._units, activation=tf.nn.relu)
        return tf.layers.dense(hidden, self.loc_dim, activation=tf.nn.tanh)

    def next_actions(self, time, inputs, is_training, n_policies, policy_dep_input=None):
        """
        NOTE: Does not propagate back into the inputs. Gradients for the policyNet weights only flow through the loc_mean (which should only be used in the REINFORCE_loss calculations)
        NOTE: tf backpropagates through sampling if using tf.distributions.Normal()"""
        if n_policies > 1:
            assert policy_dep_input is not None

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if inputs is None:
                inputs = []
            if time is not None:
                inputs.append(tf.fill([self.batch_sz, 1], tf.cast(time, tf.float32)))

            loc_mean = []
            for k in range(n_policies):
                specific_inputs = inputs.copy()
                if policy_dep_input is not None:
                    specific_inputs += [policy_dep_input[k]]
                specific_inputs = tf.stop_gradient(tf.concat(specific_inputs, axis=1))
                loc_mean.append(self.calc_encoding(specific_inputs))
            loc_mean = tf.stack(loc_mean, axis=1)

            loc_mean = tf.reshape(loc_mean, [self.batch_sz * n_policies, self.loc_dim])  # [batch_sz * n_policies, FLAGS.loc_dims]
            loc_mean = tf.clip_by_value(loc_mean, self.max_loc_rng * -1, self.max_loc_rng * 1)

            with tf.name_scope('sample_locs'):
                loc = tf.cond(is_training,
                              lambda: tf.distributions.Normal(loc=tf.stop_gradient(loc_mean), scale=self.std).sample(),
                              lambda: tf.identity(loc_mean),
                              name='sample_loc_cond')

        if n_policies > 1:
            return tf.reshape(loc, [self.batch_sz, n_policies, self.loc_dim]), tf.reshape(loc_mean, [self.batch_sz, n_policies, self.loc_dim])
        else:
            return loc, loc_mean

    def REINFORCE_losses(self, returns, baselines, locs, loc_means):
        with tf.name_scope('loc_losses'):
            # NOTE: advantages / baselines start at t=1!
            # last baseline is based on final state, thereby not used anymore (baseline for location t has to always come from the state t - 1)
            advantages = returns - baselines
            # only want gradients flow through the suggested mean
            # includes first location, but irrelevant for gradients as not depending on parameters
            z = (tf.stop_gradient(locs) - loc_means) / self.std  # [T, batch_sz, loc_dims]
            loc_loglik = -0.5 * tf.reduce_mean(tf.square(z), axis=2)  # not using reduce_sum as number of glimpses can differ

            # do not propagate back through advantages
            loc_losses = -1. * loc_loglik * tf.stop_gradient(advantages)

        return loc_losses

    def random_loc(self, rng=1.):
        rng = min(rng, self.max_loc_rng)
        with tf.name_scope(self.name):
            loc = tf.random_uniform([self.batch_sz, self.loc_dim], minval=rng * -1.,  maxval=rng * 1.)

        return loc, loc

    def inital_loc(self):
        return self.random_loc()
