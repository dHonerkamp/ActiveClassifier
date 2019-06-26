import tensorflow as tf
from tensorflow import distributions as tfd

from activeClassifier.modules.planner.ActInfPlanner import ActInfPlanner
from activeClassifier.modules.policyNetwork import PolicyNetwork
from activeClassifier.tools.tf_tools import repeat_axis, binary_entropy, differential_entropy_normal, differential_entropy_diag_normal


def maximum_patch(img, kernel_size):
    filter = tf.ones(kernel_size + [1, 1])
    return tf.nn.conv2d(img, filter, strides=(1, 1, 1, 1), padding='SAME')


class Base:
    def __init__(self, FLAGS, batch_sz):
        self.B = batch_sz
        self.num_glimpses = FLAGS.num_glimpses
        self.num_classes_kn = FLAGS.num_classes_kn
        self.uk_label = FLAGS.uk_label
        self.img_shp = FLAGS.img_shape

    def planning_step(self, current_state, obs_prior, time):
        raise NotImplementedError("Abstract method")


    def _best_believe(self, state):
        best_belief = tf.argmax(state['c'], axis=1, output_type=tf.int32)
        # TODO: IF THRESHOLD FOR UK CLASSIFICATION IS NOT LEARNED (EG. JUST WORSE THAN THE 99TH PERCENTILE OF THE PAST OR SMTH), THEN WE COULD ALSO OUTPUT UNKNOWN
        # TODO: CLFS WHEN TRAINING ON ONLY KNOWNS (WHICH WOULD HOPEFULLY DETECT WEIRD EXAMPLES / OUTLIERS)
        if self.uk_label is not None:
            best_belief = tf.where(tf.greater(state['uk_belief'], tf.reduce_max(state['c'], axis=1)), tf.fill([self.B], self.uk_label), best_belief)
        return best_belief

    @property
    def zero_records(self):
        # TODO: replace with actual shapes
        return {'G'                : tf.zeros([self.B]),  # [B, Hpixel, Wpixel]
                'exp_exp_obs'      : tf.zeros([self.B]),  # [B, Hpixel, Wpixel, Cpixel]
                'H_exp_exp_obs'    : tf.zeros([self.B]),  # [B, Hpixel, Wpixel]
                'exp_H'            : tf.zeros([self.B]),  # [B, Hpixel, Wpixel]
                }


class ActInfPlanner_fullImg(Base):
    def __init__(self, FLAGS, batch_sz, C):
        super().__init__(FLAGS, batch_sz)

        self.n_policies = 1  # taking next glimpse (+ decision action)
        self.C = C
        self.alpha = FLAGS.precision_alpha

        self.glimpse_shp = 2 * [FLAGS.scale_sizes[0]]
        assert len(FLAGS.scale_sizes) == 1, 'Not implemented for multiple scales'
        self.dist_pixel = 'B' if (self.img_shp[-1] == 1) else 'Cat'

    def planning_step(self, current_state, obs_prior, time):
        """
        Args:
          current_state: [B, H, W, C]
          class_believes: [B, num_classes_kn
        """
        # TODO: random policy option here

        H_prior = ActInfPlanner._discrete_entropy_agg(logits=obs_prior['mu_logits'], d=self.dist_pixel, agg=False)  # [B, hyp, Hpixel, Wpixel, Cpixel]
        exp_H_prior = tf.reduce_sum(current_state['c'][:, :, tf.newaxis, tf.newaxis, tf.newaxis] * H_prior, axis=1)  # [B, Hpixel, Wpixel, Cpixel]

        exp_obs_prior = tf.reduce_sum(current_state['c'][:, :, tf.newaxis, tf.newaxis, tf.newaxis] * obs_prior['mu_probs'], axis=1)  # [B, Hpixel, Wpixel, Cpixel]
        H_exp_prior = ActInfPlanner._discrete_entropy_agg(probs=exp_obs_prior, d=self.dist_pixel, agg=False)  # [B, Hpixel, Wpixel, Cpixel]

        G = H_exp_prior - exp_H_prior  # [B, Hpixel, Wpixel, Cpixel]

        # find next location: place with highest G
        max_G = maximum_patch(G, self.glimpse_shp)  # must have a channel dimension, # [B, Hpixel, Wpixel, Cpixel]
        # max_G = tf.squeeze(max_G, axis=3)  # get rid of channel dim again

        # unravel_index returns a tuple of (row_idx, col_idx), but as a single tensor
        row_col_tuple = tf.unravel_index(tf.argmax(tf.layers.flatten(max_G), axis=-1, output_type=tf.int32),
                                          dims=self.glimpse_shp)
        next_loc_pixel = tf.transpose(row_col_tuple, [1, 0])
        # into [-1, 1] range
        next_loc = PolicyNetwork.normalise_loc(next_loc_pixel, self.img_shp)
        next_loc = tf.cast(next_loc, tf.float32)  # from float64

        # extract predicted glimpses from this loc. Edge locations will never be a maximum, so padding is irrelevant
        # TODO: (unless G <0!)
        # TODO: expects 4D. Need to first combine the first 2 axes [B, hyp] into 1
        # exp_glimpse_obs = {k: tf.image.extract_glimpse(v, self.glimpse_shp, next_loc) for k, v in obs_prior.items()}

        # TODO: G for decision, G for no decision = tf.reduce_max(max_G) + prior_preferences
        decision = tf.fill([self.B], -1)  # for now always take all glimpses

        records = {'G'                : tf.squeeze(G, -1),  # [B, Hpixel, Wpixel]
                   'exp_exp_obs'      : exp_obs_prior,  # [B, Hpixel, Wpixel, Cpixel]
                   'H_exp_exp_obs'    : tf.squeeze(H_exp_prior, -1),  # [B, Hpixel, Wpixel]
                   'exp_H'            : tf.squeeze(exp_H_prior, -1),  # [B, Hpixel, Wpixel]
                   }
        return decision, next_loc, records

    def _action_selection(self, current_state, G, time, is_training):
        # TODO: use pi = softmax(-F - gamma*G) instead?
        gamma = 1
        pi_logits = tf.nn.log_softmax(gamma * G)
        # Incorporate past into the decision. But what to use for the 2 decision actions?
        # pi_logits = tf.nn.log_softmax(tf.log(new_state['c']) + gamma * G)

        # TODO: precision?
        # TODO: which version of action selection?
        # Visual foraging code: a_t ~ softmax(alpha * log(softmax(-F - gamma * G))) with alpha=512
        # Visual foraging paper: a_t = min_a[ o*_t+1 * [log(o*_t+1) - log(o^a_t+1)] ]
        # Maze code: a_t ~ softmax(gamma * G) [summed over policies with the same next action]
        selected_action_idx = tf.cond(is_training,
                                      lambda: tfd.Categorical(logits=self.alpha * pi_logits, allow_nan_stats=False).sample(),
                                      lambda: tf.argmax(G, axis=1, output_type=tf.int32),
                                      name='sample_action_cond')
        # give back the action itself, not its index. Differentiate between decision and location actions
        best_belief = self._best_believe(current_state)
        dec = tf.equal(selected_action_idx, self.n_policies)  # the last action is the decision
        selected_action_idx = tf.where(tf.stop_gradient(dec), tf.fill([self.B], 0), selected_action_idx)  # replace decision indeces (which exceed the shape of selected_action), so we can use gather on the locations

        decision = tf.cond(tf.equal(time, self.num_glimpses - 1),
                           lambda: best_belief,  # always take a decision at the last time step
                           lambda: tf.where(dec, best_belief, tf.fill([self.B], -1)),
                           name='last_t_decision_cond')
        return decision, selected_action_idx


class RandomLocPlanner(Base):
    def __init__(self, FLAGS, batch_sz):
        super().__init__(FLAGS, batch_sz)

        self.loc_dim = FLAGS.loc_dim
        self.max_loc_rng = FLAGS.max_loc_rng

    def planning_step(self, current_state, obs_prior, time):
        next_loc = tf.random_uniform([self.B, self.loc_dim], minval=self.max_loc_rng * -1., maxval=self.max_loc_rng * 1.)

        best_belief = self._best_believe(current_state)

        decision = tf.cond(tf.equal(time, self.num_glimpses - 1),
                           lambda: best_belief,  # always take a decision at the last time step
                           lambda: tf.fill([self.B], -1),
                           name='last_t_decision_cond')

        return decision, next_loc, self.zero_records
