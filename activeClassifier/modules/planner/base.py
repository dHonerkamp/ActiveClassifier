import tensorflow as tf

from tools.tf_tools import repeat_axis


class Base:
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition):
        self.B = batch_sz
        self.num_classes = FLAGS.num_classes
        self.num_classes_kn = FLAGS.num_classes_kn
        self.uk_label = FLAGS.uk_label
        self.size_z = FLAGS.size_z
        self.loc_dim = FLAGS.loc_dim
        self.patch_shape_flat = patch_shape_flat
        self.stateTransition = stateTransition
        self.m = submodules

    def random_policy(self):
        return [tf.fill([self.B], -1)] + list(self.m['policyNet'].random_loc()) + [self.zero_records]

    def planning_step(self, current_state, z_samples, time, is_training):
        return

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
        return tf.tile(one_hot, [n_tiles, 1])  # [B*n_tiles, n_classes]

    def single_policy_prediction(self, state, z_samples=None, next_actions=None):
        if z_samples is not None:
            state = self.stateTransition([z_samples, next_actions], state)

        hyp = self._hyp_tiling(n_classes=self.num_classes_kn, n_tiles=self.B * 1)  # [B * 1 * n_classes_kn, n_classes_kn]
        new_s_tiled = repeat_axis(state['s'], axis=0, repeats=self.num_classes_kn)  # [B, rnn] -> [B * hyp, rnn]
        new_l_tiled = repeat_axis(state['l'], axis=0, repeats=self.num_classes_kn)  # [B, loc] -> [B * hyp, loc]

        exp_obs = self.m['VAEEncoder'].calc_prior([hyp, new_s_tiled, new_l_tiled], out_shp=[self.B, self.num_classes_kn, self.m['VAEEncoder'].output_size])

        return exp_obs

    def initial_planning(self):
        selected_action, selected_action_mean = self.m['policyNet'].inital_loc()
        new_state = self.stateTransition.initial_state(self.B, selected_action)
        selected_exp_obs = self.single_policy_prediction(new_state)  # [B, num_classes, m['VAEEncoder'].output_size]
        decision = tf.fill([self.B], -1)

        return new_state, decision, selected_action, selected_action_mean, selected_exp_obs, self.zero_records

    @property
    def zero_records(self):
        return {'G'                : tf.zeros([self.B, self.n_policies + 1]),
                'exp_obs'          : tf.zeros([self.B, self.n_policies, self.num_classes_kn, self.m['VAEEncoder'].output_size]),
                'exp_exp_obs'      : tf.zeros([self.B, self.n_policies, self.m['VAEEncoder'].output_size]),
                'H_exp_exp_obs'    : tf.zeros([self.B, self.n_policies]),
                'exp_H'            : tf.zeros([self.B, self.n_policies]),
                'potential_actions': tf.zeros([self.B, self.n_policies, self.loc_dim])}
