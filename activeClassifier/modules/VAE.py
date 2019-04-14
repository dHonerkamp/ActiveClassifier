import tensorflow as tf
from tools.tf_tools import output_into_gaussian_params, create_MLP


class GlimpseEncoder:
    def __init__(self, FLAGS, glimpse_size):
        fc_specs_encode = ([FLAGS.num_hidden_fc, tf.nn.relu],
                           [FLAGS.num_hidden_fc, tf.nn.tanh],
                           )
        self.encode_layers = create_MLP(fc_specs_encode)

        fc_specs_decode = ([FLAGS.num_hidden_fc, tf.nn.relu],
                           [glimpse_size, tf.nn.sigmoid],
                           )
        self.decode_layers = create_MLP(fc_specs_decode)

    def encode(self, observation):
        enc = observation
        for l in self.encode_layers:
            enc = l(enc)
        return enc

    def decode(self, enc):
        glimpse = enc
        for l in self.decode_layers:
            glimpse = l(enc)
        return glimpse

    @property
    def encode_size(self):
        return self.encode_layers[-1].units

    @property
    def decode_size(self):
        return self.decode_layers[-1].units



class Encoder:
    """
    Generate encodings z, drawn from posterior during training, from prior during inference.
    """
    def __init__(self, FLAGS, patch_shape_flat, name='GlimpseEncoder'):
        self.name = name
        self.size_z = FLAGS.size_z
        # self.prior_type = FLAGS.prior
        # self.infoVAE = FLAGS.infoVAE
        self.patch_shape_flat = patch_shape_flat
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            # self._init_prior(FLAGS)
            self.layers_post = self._init_posterior(FLAGS)

            self.fc_film_gamma = tf.layers.Dense(self.layers_post[-2].units, name='fc_film_gamma')
            self.fc_film_beta = tf.layers.Dense(self.layers_post[-2].units, name='fc_film2_beta')

    # def _init_prior(self, FLAGS):
    #     with tf.name_scope('prior'):
    #         input_sz = FLAGS.size_rnn_state + FLAGS.loc_dim + FLAGS.num_classes
    #         num_hidden = 2 * FLAGS.size_rnn_state
    #         self.layers_prior = []
    #         self.layers_prior.append(tf.keras.layers.Reshape([1, 1, input_sz]))
    #         self.layers_prior.append(tf.layers.Conv2D(num_hidden, [1, 1], activation=tf.nn.relu))
    #         self.layers_prior.append(tf.layers.Flatten())
    #
    #         fc_specs = ([num_hidden, tf.nn.relu],
    #                     [num_hidden // 2, tf.nn.relu],
    #                     [2 * FLAGS.size_z, None])
    #         self.layers_prior += create_MLP(fc_specs)

    def _init_posterior(self, FLAGS):
        # input_sz = self.patch_shape_flat + FLAGS.loc_dim + FLAGS.num_classes + FLAGS.size_z
        layers_post = []
        # self.layers_post.append(tf.keras.layers.Reshape([1, 1, input_sz]))
        # self.layers_post.append(tf.layers.Conv2D(num_hidden, [1, 1], activation=tf.nn.relu))
        # self.layers_post.append(tf.layers.Flatten())

        fc_specs = ([FLAGS.num_hidden_fc, tf.nn.relu],
                    [FLAGS.num_hidden_fc // 2, tf.nn.relu],
                    [2 * FLAGS.size_z, None])
        layers_post += create_MLP(fc_specs)
        return layers_post

    # def prior_inference(self, cell_output, loc, label):
    #     with tf.name_scope('prior'):
    #         if self.prior_type == 'N01':
    #             mu = tf.fill([tf.shape(loc)[0], self.size_z], 0.)
    #             sigma = tf.fill([tf.shape(loc)[0], self.size_z], 1.)
    #             z = tf.concat([mu, sigma], axis=1)
    #         elif self.prior_type == 'filter':
    #             z = tf.concat([cell_output, loc, label], axis=1)
    #             for l in self.layers_prior:
    #                 z = l(z)
    #     return output_into_gaussian_params(z,
    #                                        min_std=0.01)  # min_std to ensure no vanishing variance. As infoVAE authors.

    def _context_enc_FiLM(self, label, loc, img_code):
        inputs = tf.concat([loc, label], axis=1)
        gamma = self.fc_film_gamma(inputs)
        beta  = self.fc_film_beta(inputs)
        return tf.nn.tanh(gamma * img_code + beta)

    @staticmethod
    def _img_enc(layers, img_patch_flat):
        hidden = img_patch_flat
        for l in layers:
            hidden = l(hidden)
        return hidden

    def posterior_inference(self, label, last_z, loc, next_glimpse):  #, prior_mu, prior_sigma):
        with tf.variable_scope(self.name + '/posterior', reuse=tf.AUTO_REUSE):
            img_enc = self._img_enc(self.layers_post[:-1], next_glimpse)
            enc = self._context_enc_FiLM(label, loc, img_enc)

            inputs = tf.concat([enc, label, last_z], axis=1)
            z = self.layers_post[-1](inputs)
            dist, mu, sigma = output_into_gaussian_params(z, min_std=0.01)
        return {'mu': mu, 'sigma': sigma, 'sample': dist.sample()}

    def kl_div_normal(self, q_mu, q_sigma, p_mu, p_sigma):
        """
        Return KL(q || p) for two normal distributions. (For VAE: KL(post || prior)).
        """
        p_sigma_sqr = tf.square(p_sigma)
        q_sigma_sqr = tf.square(q_sigma)

        numerator = tf.square(p_mu - q_mu) + p_sigma_sqr - q_sigma_sqr
        denominator = 2. * q_sigma_sqr

        kls = (numerator / denominator) + tf.log(q_sigma) - tf.log(p_sigma)
        return self.size_z * tf.reduce_mean(kls)

    # def __call__(self, cell_output, next_glimpse, loc, label, is_training):
    #     with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
    #         prior, prior_mu, prior_sigma = self._prior_inference(cell_output, loc, label)
    #         posterior, post_mu, post_sigma = self._posterior_inference(cell_output, loc, label, next_glimpse,
    #                                                                    prior_mu, prior_sigma)
    #
    #         prior_sample = prior.sample()
    #         post_sample = posterior.sample()
    #
    #         # inefficiency: could only evaluate prior for inference. Both for stats for now
    #         z = tf.cond(is_training,
    #                     lambda: tf.identity(post_sample),
    #                     lambda: tf.identity(prior_sample))
    #
    #     return z, tf.stack([prior_sample, post_sample], axis=0), tf.stack(
    #             [prior_mu, post_mu, prior_sigma, post_sigma], axis=0)

    @property
    def trainable(self):
        return [l.trainable_variables for l in self.layers_post]


class Decoder:
    """
    Source mixture loss: https://github.com/hardmaru/WorldModelsExperiments
    """
    def __init__(self, FLAGS, size_glimpse_out, name='GlimpseDecoder'):
        self.name = name
        self.size_glimpse_out = size_glimpse_out
        self.gl_std = FLAGS.gl_std

        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            self._init_fc(FLAGS)

    def _init_fc(self, FLAGS):
        if FLAGS.gl_std == -1:
            final_sz = 2 * self.size_glimpse_out
        else:
            final_sz = self.size_glimpse_out

        self.layers = []
        self.layers.append(tf.layers.Dense(FLAGS.num_hidden_fc, activation=tf.nn.relu))
        self.layers.append(tf.layers.Dense(final_sz, activation=tf.nn.tanh))

    def decode(self, label, last_s, z, loc, glimpse=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            output = tf.concat([label, last_s, z, loc], axis=1)
            for l in self.layers:
                output = l(output)

            if self.gl_std == -1:
                dist, mu, sigma = output_into_gaussian_params(output)
            else:
                mu = output
                sigma = tf.fill(tf.shape(output), self.gl_std)
                dist = tf.distributions.Normal(loc=mu, scale=sigma)

            if glimpse is not None:
                loss = -dist.log_prob(glimpse)
                loss = tf.reduce_sum(loss, axis=-1)
                # sample = dist.sample()
                # loss = mse_losses(output, next_glEnc)
            else:
                loss = None

            return {'sample': mu,
                    'params': tf.concat([mu, sigma], axis=1),
                    'loss': loss}

    @property
    def trainable(self):
        return [l.trainable_variables for l in self.layers]

    @property
    def output_size(self):
        return self.size_glimpse_out
