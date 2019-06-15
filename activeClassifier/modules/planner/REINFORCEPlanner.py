import tensorflow as tf

from activeClassifier.modules.planner.base import Base


class REINFORCEPlanner(Base):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition, is_pre_phase, labels):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)
        self.n_policies = 1
        self._is_pre_phase = is_pre_phase
        self.exp_obs_shape = submodules['VAEEncoder'].output_shape
        # self.exp_obs_shape = self.m['VAEDecoder'].output_shape if FLAGS.use_pixel_obs_FE else self.m['VAEEncoder'].output_shape

        # SEEMS TO LEARN TO CHEAT IF DOING THIS
        # if use_true_labels:
        #     self.policy_dep_input = tf.one_hot(y_MC, depth=FLAGS.num_classes)[tf.newaxis, :, :]  # first dimension is n_policies
        # else:
        #     self.policy_dep_input = None
        self.lbls = labels


    def planning_step(self, current_state, z_samples, glimpse_idx, time, is_training, rnd_loc_eval):
        # TODO: CHECK EFFECT ON PERFORMANCE FROM INCLUDING THIS OR NOT (WHETHER INCLUDING IT MAKES THE MODEL FOCUS LESS ON LEARNING RECONSTRUCTIONS)
        if self._is_pre_phase and (self.rl_reward == 'clf'):
            # TODO: ADJUST FOR UK CLASSES
            # policy_dep_input = tf.one_hot(tf.argmax(current_state['c'], axis=1), depth=self.num_classes_kn)
            policy_dep_input = tf.one_hot(self.lbls, depth=self.num_classes_kn)
            # inputs = [current_state['s'], tf.fill([self.B, 1], tf.cast(time, tf.float32))]
            inputs = [current_state['s'], policy_dep_input]
        else:
            inputs = [current_state['s'], current_state['c']]
        next_action, next_action_mean = self.m['policyNet'].next_actions(inputs=inputs, is_training=is_training, n_policies=self.n_policies)
        next_exp_obs = self.single_policy_prediction(current_state, z_samples, next_action, glimpse_idx)

        if time < (self.num_glimpses - 1):
            decision = tf.fill([self.B], -1)
        else:
            decision = self._best_believe(current_state)

        return decision, next_action, next_action_mean, next_exp_obs, self.zero_records
