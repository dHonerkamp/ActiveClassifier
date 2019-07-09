import tensorflow as tf
from tensorflow.contrib.rnn import Conv2DLSTMCell

from activeClassifier.tools.MSE_distribution import MSEDistribution
from activeClassifier.tools.tf_tools import repeat_axis
from activeClassifier.modules.planner.basePlanner import BasePlanner


class Generator:
    def __init__(self, FLAGS, batch_sz, input_shape, y_MC, name='generator'):
        self.name = name
        self.B = batch_sz
        self.input_shape = input_shape
        self.img_shp = FLAGS.img_shape
        self.num_classes_kn = FLAGS.num_classes_kn
        self.debug = FLAGS.debug

        self.convLSTM_L = FLAGS.convLSTM_L
        self.convLSTM_filters = FLAGS.convLSTM_filters
        self.convLSTM = Conv2DLSTMCell(input_shape=self.input_shape, output_channels=self.convLSTM_filters, kernel_shape=[5, 5])

        # mapping onto z, mu
        self.z_filters = FLAGS.z_filters
        self.mu_layer = tf.layers.Conv2D(self.z_filters, kernel_size=[5, 5], padding='SAME', activation=None, name='mu_layer')
        self.sigma_layer = tf.layers.Conv2D(self.z_filters, kernel_size=[5, 5], padding='SAME', activation=tf.nn.softplus, name='sigma_layer')

        # u: 7x7 into 28x28
        self._upsample = tf.layers.Conv2DTranspose(filters=self.img_shp[-1], kernel_size=[4, 4], strides=(4, 4), padding='VALID', name='upsample')
        self.mu_layer_u = tf.layers.Conv2D(filters=self.img_shp[-1], kernel_size=[1, 1], padding='SAME', activation=None, name='mu_layer_u')

        # class conditional stuff
        self.hyp = BasePlanner._hyp_tiling(FLAGS.num_classes_kn, self.B)  # [B * hyp, num_classes]

        # for masked nll loss
        self.correct_hypoths = tf.cast(tf.one_hot(y_MC, depth=FLAGS.num_classes_kn), tf.bool)  # [B, hyp]

        # decaying std
        if (FLAGS.gl_std != 1):
            self.gl_std = tf.train.polynomial_decay(FLAGS.gl_std, tf.train.get_global_step(),
                                                  decay_steps=FLAGS.num_epochs * FLAGS.train_batches_per_epoch,
                                                  end_learning_rate=FLAGS.gl_std_min)
        else:
            self.gl_std = None

    def _sample(self, h, min_std=1e-5):
        mu = self.mu_layer(h)
        sigma = self.sigma_layer(h)
        sigma += min_std

        dist = tf.distributions.Normal(loc=mu, scale=sigma, validate_args=self.debug, allow_nan_stats=~self.debug)
        params = tf.stack([mu, sigma], axis=0)
        return params, dist, dist.sample()

    def predict(self, r, hyp, prior_h, prior_z, out_shp=None):
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            if out_shp is None:
                out_shp = [self.B]

            hyp_conv_shp = tf.tile(hyp[:, tf.newaxis, tf.newaxis, :], [1, self.input_shape[0], self.input_shape[1], 1])  # [B, state, state, hyp]

            # keeping track
            hs, zs, KLdiv = [], [], tf.zeros(tf.shape(r)[0])

            # initial values. use tf.shape(r)[0] instead of self.B as input might be tiled for all hyp
            state = self.convLSTM.zero_state(tf.shape(r)[0], tf.float32)  # h, c
            u = tf.zeros([tf.shape(r)[0], self.img_shp[0], self.img_shp[1], self.z_filters])
            z_sample = tf.zeros([tf.shape(r)[0], self.input_shape[0], self.input_shape[1], self.z_filters])

            for l in range(self.convLSTM_L):
                # TODO: give hyp at every step or only in beginning (as to enable model to ignore it for certain locations)
                # TODO: also include previous (prior's) h as input?
                inputs = tf.concat([hyp_conv_shp, r, z_sample, prior_h[l]], axis=-1)

                h, state = self.convLSTM(inputs, state)
                z_params, z_dist, z_sample = self._sample(h)

                u += self._upsample(h)

                prior_dist = tf.distributions.Normal(loc=prior_z[l, 0], scale=prior_z[l, 1], validate_args=self.debug, allow_nan_stats=~self.debug)
                KLdiv += tf.reduce_sum(z_dist.kl_divergence(prior_dist), axis=[1, 2, 3])  # [B]

                hs.append(h)
                zs.append(z_params)

            xhat_logits = self.mu_layer_u(u)
            xhat_probs = tf.nn.sigmoid(xhat_logits)

            xhat = {'mu_logits': tf.reshape(xhat_logits, out_shp + self.img_shp),  # [B, img_shp]
                    'mu_probs': tf.reshape(xhat_probs, out_shp + self.img_shp),  # [B, img_shp]
                    'sigma': None,
                    }

            VAE_results = {'h': tf.stack(hs, axis=0),  # [L, B, H, W, n_filters]
                           'z': tf.stack(zs, axis=0),  # [L, mu/sigma, B, H, W, z_filter]
                           'KLdiv': tf.reshape(KLdiv, out_shp),  # [B]
                           }

        return VAE_results, xhat

    def _mask_loss(self, nll, seen, class_believes, mask_type):
        """
        Mask unseen regions of incorrect hyp by setting them to zero

        Args:
            nll: [B, hyp, H, W, C]
            seen: [B, hyp, H, W]
            class_believes: [B, hyp]
        """
        mask = self.correct_hypoths[:, :, tf.newaxis, tf.newaxis]  # [B, hyp, H, W]

        if mask_type == 'probable_seen':
            probable_hyps = (class_believes >= (1. / self.num_classes_kn))
            probable_hyps = tf.stop_gradient(probable_hyps )
            seen_and_probable = tf.logical_and(seen[:, tf.newaxis, :, :], probable_hyps[:, :, tf.newaxis, tf.newaxis])  # [B, hyp, H, W]
            mask = tf.logical_or(mask, seen_and_probable)  # [B, hyp, H, W]
            mask = mask[:, :, :, :, tf.newaxis]
        elif mask_type == 'seen':
            mask = tf.logical_or(mask, seen[:, tf.newaxis, :, :])  # [B, hyp, H, W]
            mask = mask[:, :, :, :, tf.newaxis]
        else:
            # broadcast into right shape
            mask = tf.logical_and(tf.ones_like(nll, dtype=tf.bool), mask[:, :, :, :, tf.newaxis])

        nll_masked = tf.where(mask, nll, tf.zeros_like(nll))  # [B, hyp, H, W, C]
        return nll_masked

    def nll_loss(self, xhat_probs, true_img, out_shp, seen, class_believes):
        if len(out_shp) > 1:  # broadcasting for nll loss
            true_img = tf.reshape(true_img, [self.B] + (len(out_shp) - 1) * [1] + true_img.get_shape().as_list()[1:])
        # TODO: alternatively use annealing or learned std
        if self.gl_std is None:
            dist = MSEDistribution(xhat_probs)
        else:  # decaying std
            dist = tf.distributions.Normal(loc=xhat_probs, scale=self.gl_std)
        nll = -dist.log_prob(true_img)

        # TODO: implement properly
        mask_type = 'seen'
        nll_masked = self._mask_loss(nll, seen, class_believes, mask_type=mask_type)

        nll_summed = tf.reduce_sum(nll_masked, axis=[-3, -2, -1])  # [B] or out_shp (reducing H, W, C)

        return nll_summed  # [B]


    def class_cond_predictions(self, r, prior_h, prior_z, img, seen, class_believes):
        r_repeated = repeat_axis(r, axis=0, repeats=self.num_classes_kn)
        out_shp = [self.B, self.num_classes_kn]
        VAE_results, obs_prior = self.predict(r=r_repeated,
                                              hyp=self.hyp,
                                              prior_h=prior_h,
                                              prior_z=prior_z,
                                              out_shp=out_shp)  # [B, hyp, height, width, C]
        VAE_results['nll'] = self.nll_loss(xhat_probs=obs_prior['mu_probs'],
                                                  true_img=img,
                                                  out_shp=out_shp,
                                                  seen=seen,
                                                  class_believes=class_believes)

        return VAE_results, obs_prior

    # @property
    # def output_shape(self):
    #     return

    @property
    def zero_state(self):
        return {'h': tf.zeros([self.convLSTM_L, self.B * self.num_classes_kn, self.input_shape[0], self.input_shape[1], self.convLSTM_filters]),
                # use ones as it includes the std. which has to be > 0. Would o/w fail 'validate_args' in tf.Normal. Should never be an input and first KLdiv does not get included in loss
                'z': tf.ones([self.convLSTM_L, 2, self.B * self.num_classes_kn, self.input_shape[0], self.input_shape[1], self.z_filters])}