import tensorflow as tf

from modules.planner.base import Base


class REINFORCEPlanner(Base):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition, is_pre_phase, labels):
        self.n_policies = 1
        self.num_glimpses = FLAGS.num_glimpses
        self._is_pre_phase = is_pre_phase

        # SEEMS TO LEARN TO CHEAT IF DOING THIS
        # if use_true_labels:
        #     self.policy_dep_input = tf.one_hot(y_MC, depth=FLAGS.num_classes)[tf.newaxis, :, :]  # first dimension is n_policies
        # else:
        #     self.policy_dep_input = None
        self.lbls = labels

        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)

    def planning_step(self, current_state, z_samples, time, is_training):
        # TODO: CHECK EFFECT ON PERFORMANCE FROM INCLUDING THIS OR NOT (WHETHER INCLUDING IT MAKES THE MODEL FOCUS LESS ON LEARNING RECONSTRUCTIONS)
        if self._is_pre_phase:
            # TODO: ADJUST FOR UK CLASSES
            # policy_dep_input = tf.one_hot(tf.argmax(current_state['c'], axis=1),
            #                               depth=self.num_classes_kn)[tf.newaxis, :, :]
            policy_dep_input = tf.one_hot(self.lbls,
                                          depth=self.num_classes_kn)[tf.newaxis, :, :]
            # inputs = [current_state['s'], tf.fill([self.B, 1], tf.cast(time, tf.float32))]
            inputs = [current_state['s']]
            next_action, next_action_mean = self.m['policyNet'].next_actions(inputs=inputs,
                                                                             is_training=is_training,
                                                                             n_policies=self.n_policies,
                                                                             policy_dep_input=policy_dep_input)
        else:
            next_action, next_action_mean = self.m['policyNet'].next_actions(inputs=[current_state['c'], current_state['s']],
                                                                             is_training=is_training,
                                                                             n_policies=self.n_policies,
                                                                             policy_dep_input=None)
        next_exp_obs = self.single_policy_prediction(current_state, z_samples, next_action)

        if time < (self.num_glimpses - 1):
            decision = tf.fill([self.B], -1)
        else:
            decision = self._best_believe(current_state)

        return decision, next_action, next_action_mean, next_exp_obs, self.zero_records
