import tensorflow as tf

from activeClassifier.modules.planner.basePlanner import BasePlanner


class RandomPlanner(BasePlanner):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)
        self.n_policies = 1

    def planning_step(self, current_state, time, is_training, rnd_loc_eval):
        next_action, next_action_mean= self.m['policyNet'].random_loc()
        next_exp_obs = self.single_policy_prediction(current_state, next_action)

        if time < (self.num_glimpses - 1):
            decision = tf.fill([self.B], -1)
        else:
            decision = self._best_believe(current_state)

        return decision, next_action, next_action_mean, next_exp_obs, self.zero_records
