import tensorflow as tf

from activeClassifier.tools.tf_tools import repeat_axis


class BasePlanner:
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition):
        self.B = batch_sz
        self.num_classes = FLAGS.num_classes
        self.num_classes_kn = FLAGS.num_classes_kn
        self.uk_label = FLAGS.uk_label
        self.num_glimpses = FLAGS.num_glimpses
        self.size_z = FLAGS.size_z
        self.loc_dim = FLAGS.loc_dim
        self.patch_shape_flat = patch_shape_flat
        self.stateTransition = stateTransition
        self.rl_reward = FLAGS.rl_reward
        self.m = submodules

    def random_policy(self, rng=1.):
        rnd_loc, rnd_loc = self.m['policyNet'].random_loc(rng)
        return tf.fill([self.B], -1), rnd_loc, rnd_loc, self.zero_records

    def planning_step(self, current_state, time, is_training, rnd_loc_eval):
        raise NotImplementedError("Abstract method")

    def _best_believe(self, state):
        best_belief = tf.argmax(state['c'], axis=1, output_type=tf.int32)
        # TODO: IF THRESHOLD FOR UK CLASSIFICATION IS NOT LEARNED (EG. JUST WORSE THAN THE 99TH PERCENTILE OF THE PAST OR SMTH), THEN WE COULD ALSO OUTPUT UNKNOWN
        # TODO: CLFS WHEN TRAINING ON ONLY KNOWNS (WHICH WOULD HOPEFULLY DETECT WEIRD EXAMPLES / OUTLIERS)
        if self.uk_label is not None:
            best_belief = tf.where(tf.greater(state['uk_belief'], tf.reduce_max(state['c'], axis=1)), tf.fill([self.B], self.uk_label), best_belief)
        return best_belief

    @staticmethod
    def _hyp_tiling(n_classes, n_tiles):
        one_hot = tf.one_hot(tf.range(n_classes), depth=n_classes)  # [n_classes, n_classes]
        return tf.tile(one_hot, [n_tiles, 1])  # [n_tiles*n_classes, n_classes]

    def single_policy_prediction(self, state, next_action):
        hyp = self._hyp_tiling(n_classes=self.num_classes_kn, n_tiles=self.B * 1)  # [B * 1 * n_classes_kn, n_classes_kn]
        s_tiled = repeat_axis(state['s'], axis=0, repeats=self.num_classes_kn)  # [B, rnn] -> [B * hyp, rnn]
        next_action_tiled = repeat_axis(next_action, axis=0, repeats=self.num_classes_kn)  # [B, loc] -> [B * hyp, loc]

        exp_obs_enc = self.m['VAEEncoder'].calc_prior([hyp, s_tiled, next_action_tiled], out_shp=[self.B, self.num_classes_kn])
        return exp_obs_enc

    def initial_planning(self, initial_state):
        selected_action, selected_action_mean = self.m['policyNet'].inital_loc()
        selected_exp_obs_enc = self.single_policy_prediction(initial_state, selected_action)  # [B, num_classes, m['VAEEncoder'].output_size]
        decision = tf.fill([self.B], -1)

        return decision, selected_action, selected_action_mean, selected_exp_obs_enc, self.zero_records

    @property
    def zero_records(self):
        return {'G'                : tf.zeros([self.B, self.n_policies + 1]),
                'exp_obs'          : tf.zeros([self.B, self.n_policies, self.num_classes_kn] + self.exp_obs_shape),
                'exp_exp_obs'      : tf.zeros([self.B, self.n_policies] + self.exp_obs_shape),
                'H_exp_exp_obs'    : tf.zeros([self.B, self.n_policies]),
                'exp_H'            : tf.zeros([self.B, self.n_policies]),
                'potential_actions': tf.zeros([self.B, self.n_policies, self.loc_dim]),
                'selected_action_idx': tf.zeros([self.B], dtype=tf.int32),
                'rewards_Gobs'     : tf.zeros([self.B, self.n_policies])}
