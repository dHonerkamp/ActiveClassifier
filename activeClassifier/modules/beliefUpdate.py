import tensorflow as tf
from tensorflow import distributions as tfd
# from tensorflow.contrib.distributions import RelaxedBernoulli, Logistic

from tools.tf_tools import FiLM_layer, pseudo_LogRelaxedBernoulli


class BeliefUpdate:
    def __init__(self, FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name='BeliefUpdate'):
        self.m = submodules
        self.name = name
        self.n_policies = n_policies
        self.B = batch_size
        self.uk_label = FLAGS.uk_label
        self.labels = labels
        self.normalise_fb = FLAGS.normalise_fb
        self.num_classes_kn = FLAGS.num_classes_kn
        self.z_dist = FLAGS.z_dist

        self.uk_cycling = FLAGS.uk_cycling
        if self.uk_cycling:
            khot_vector = tf.reduce_any(tf.cast(tf.one_hot(current_cycl_uk, depth=self.num_classes_kn), tf.bool), axis=0, keep_dims=True)
            self.current_cycl_uk_mask = tf.tile(khot_vector, [self.B, 1])  # [B, num_classes_kn]

    def update(self, current_state, new_observation, exp_zs_prior, time, newly_done):
        """Given a new observation, and the last believes over the state, update the believes over the states.
        The sufficient statistic of the old state in this case is z, as the VAEencoder is class-specific.

        Returns:
            c: [B, num_classes} believe over classes based on past observations
            zs_post: [B, num_classes, size_z] inferred zs conditional on each class
            glimpse_nll_stacked: [B, num_classes] likelihood of each past observation conditional on each class
            """
        with tf.name_scope(self.name + '/'):
            # Infer posterior z for all hypotheses
            with tf.name_scope('poterior_inference/'):
                # TODO: SHOULD POSTERIOR GET THE current_state['s']?
                z_post = self.m['VAEEncoder'].calc_post(new_observation,
                                                        current_state['l'])
                # COULD ALSO PASS current_state['s'], BUT THAT MEANS MODEL CAN USE THINGS THAT THE PRIOR DOES NOT PREDICT AND EASILY GET GOOD PREDICTIONS AND RECONSTRUCTIONS
                reconstr_post = self.m['VAEDecoder'].decode([z_post['sample'], current_state['l']],
                                                            true_glimpse=new_observation)  # ^= filtering, given that transitions are deterministic

            # believes over the classes based on all past observations (uniformly weighted)
            with tf.name_scope('prediction_feedback/'):
                # 2 possibilties to infer state from received observations:
                # i)  judge by likelihood of the observations under each hypothesis
                # ii) train a separate model (e.g. LSTM) for infering states
                KLdiv = self.calc_KLdiv(z_prior=exp_zs_prior, z_post=z_post)

            # aggregate feedback
            if self.normalise_fb == 1:
                # predError = batch_min_normalization(KLdiv, epsilon=0.1) - 1.  # SUFFERS FROM ERRORS BEING MUCH LOWER IF LOOKING INTO THE CORNERS
                bl_surprise = self._surprise_bl([current_state['l'], current_state['s']])
                predError = tf.maximum(KLdiv / (tf.stop_gradient(bl_surprise) + 0.01), 1.) - 1.
            elif self.normalise_fb == 2:
                bl_surprise = self._surprise_bl([current_state['l'], current_state['s']])
                predError = tf.maximum(KLdiv - (tf.stop_gradient(bl_surprise)), 0.)
            else:
                predError, bl_surprise = KLdiv, tf.zeros([self.B])
            # TODO: INITIAL FB ADDS LOT OF NOISE AS IT OFTEN IS JUST EMPTY SPACE (MUCH LOWER ERROR). MAYBE IGNORE IT IN THE AGGREGATION? OR MAKE 1ST GLIMPSE PLANNED
            if time == 1:
                current_state['fb'] = predError
            else:
                current_state['fb'] += predError
            current_state, loss = self.update_fn(current_state, KLdiv, time, newly_done)

            return (current_state,
                    z_post['sample'],  # [B, z]
                    reconstr_post['loss'],  # [B]
                    reconstr_post['sample'],  # [B, glimpse]
                    KLdiv,  # [B, num_classes]
                    loss,  # [B]
                    bl_surprise,  # [B]
                   )

    def calc_KLdiv(self, z_prior, z_post):
        post_mu    = tf.tile(z_post['mu'][:, tf.newaxis, :], [1, self.n_policies, 1])

        if self.z_dist == 'N':
            post_sigma = tf.tile(z_post['sigma'][:, tf.newaxis, :], [1, self.n_policies, 1])

            dist_prior = tfd.Normal(loc=z_prior['mu'], scale=z_prior['sigma'])
            dist_post = tfd.Normal(loc=post_mu, scale=post_sigma)
            KLdiv = dist_post.kl_divergence(dist_prior)  # [B, hyp, z]
        elif self.z_dist == 'B':
            post_log_sample = tf.tile(z_post['log_sample'][:, tf.newaxis, :], [1, self.n_policies, 1])

            # dist_prior = tfd.Bernoulli(logits=z_prior['mu'])
            # dist_post = tfd.Bernoulli(logits=post_mu)
            dist_prior = pseudo_LogRelaxedBernoulli(logits=z_prior['mu'], temperature=self.m['VAEEncoder'].temp_prior)
            dist_post  = pseudo_LogRelaxedBernoulli(logits=post_mu, temperature=self.m['VAEEncoder'].temp_post)
            log_diff = dist_post.log_prob(post_log_sample) - dist_prior.log_prob(post_log_sample)
            KLdiv = dist_post.prob(post_log_sample) * log_diff  # [B, hyp, z]
        else:
            raise ValueError('Unknown z_dist: {}'.format(self.z_dist))

        KLdiv = tf.reduce_sum(KLdiv, axis=2)

        if self.uk_cycling:
            # mask the prediction error of the current uk classes with the highest prediction error of the observation
            KLdiv = tf.where(self.current_cycl_uk_mask, tf.tile(tf.reduce_max(KLdiv, axis=1, keep_dims=True), [1, self.num_classes_kn]), KLdiv)

        return KLdiv


    def update_fn(self, current_state, predError, time, newly_done):
        current_state, loss = None, None
        return current_state, loss

    def _surprise_bl(self, inputs):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.concat(inputs, axis=1)
            baseline = tf.layers.dense(tf.stop_gradient(inputs), units=1, activation=None, bias_initializer=tf.constant_initializer(1.), name='bl_surprise')
        return baseline


class PredErrorUpdate(BeliefUpdate):
    def __init__(self, FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name='BeliefUpdate'):
        super().__init__(FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name)

    def update_fn(self, current_state, predError, time, newly_done):
        # c = expanding_mean(tf.nn.softmax(-predError, axis=1), current_state['c'], time)
        c = tf.nn.softmax(-current_state['fb'] / (time+1), axis=1)

        if self.uk_label is not None:
            avg_fb = current_state['fb'] / tf.cast((time + 1), tf.float32)
            surprise = tf.reduce_min(avg_fb, axis=1, keep_dims=True)  # best fb across hyp

            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                uk_belief_logits = tf.layers.dense(tf.stop_gradient(surprise), units=1, activation=None, bias_initializer=tf.constant_initializer(-6), name='uk_belief')
                uk_belief_logits = tf.squeeze(uk_belief_logits, 1)
                current_state['uk_belief'] = tf.stop_gradient(tf.nn.sigmoid(uk_belief_logits))
        else:
            uk_belief_logits = None

        # TODO: THINK ABOUT THESE stop_gradient
        current_state['c'] = tf.stop_gradient(c)
        return current_state, self.loss(uk_belief_logits)

    def loss(self, logits):
        # TODO: DON'T TRAIN IF IN PRETRAINING WITHOUT ANY UKS
        if logits is not None:
            # TODO: ALTERNATIVELY USE A POLICY GRADIENT LOSS INCL. BASELINE
            is_uk = tf.equal(self.labels, self.uk_label)
            binary_labels = tf.cast(is_uk, tf.float32)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=binary_labels, logits=logits)
            return losses
        else:
            return tf.zeros([self.B])


class FullyConnectedUpdate(BeliefUpdate):
    def __init__(self, FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name='BeliefUpdate'):
        super().__init__(FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name)

    def update_fn(self, current_state, predError, time, newly_done):
        # TODO: AGGREGATE THE RAW VALUES OR SOFTMAX AT EACH STEP?
        avg_fb = current_state['fb'] / tf.cast((time + 1), tf.float32)

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = tf.stop_gradient(tf.concat([avg_fb, current_state['s']], axis=1))
            hidden = tf.layers.dense(inputs, units=self.num_classes_kn)
            c_logits = FiLM_layer(tf.concat([avg_fb, tf.fill([self.B, 1], tf.cast(time, tf.float32))], axis=1), hidden)
            c = tf.nn.softmax(c_logits)

        current_state['c'] = tf.stop_gradient(c)
        return current_state, self.loss(logits=c_logits)

    def loss(self, logits):
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)


class RAMUpdate(BeliefUpdate):
    def __init__(self, FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name='BeliefUpdate'):
        super().__init__(FLAGS, submodules, n_policies, batch_size, labels, current_cycl_uk, name)

    def update_fn(self, current_state, predError, time, newly_done):
        """Propagating xent gradients into current_state['s']"""
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            inputs = current_state['s']
            c_logits = tf.layers.dense(inputs, units=self.num_classes_kn)
             # TODO; ONLY TRAIN ON FINAL DECISION LOSS
            c = tf.nn.softmax(c_logits)

        current_state['c'] = tf.stop_gradient(c)
        return current_state, self.loss(logits=c_logits, newly_done=newly_done)

    def loss(self, logits, newly_done):
        """Only train on last decision"""
        loss = 100 * tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=logits)
        return tf.where(newly_done, loss, tf.zeros_like(loss))
