import tensorflow as tf
from tensorflow import distributions as tfd

from activeClassifier.tools.tf_tools import repeat_axis, entropy


class Planner:
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat):
        self.B = batch_sz
        self.num_classes = FLAGS.num_classes
        self.size_z = FLAGS.size_z
        self.loc_dim = FLAGS.loc_dim
        self.patch_shape_flat = patch_shape_flat
        self.m = submodules

    def random_policy(self):
        return [tf.fill([self.B], -1)] + list(self.m['policyNet'].random_loc()) + [self.zero_records]

    def planning_step(self, current_state, z_samples, time, is_training):
        return

    @property
    def zero_records(self):
        return


class ActInfPlanner(Planner):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, C, stateTransition):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat)
        self.n_policies = FLAGS.num_classes
        self.C = C
        self.stateTransition = stateTransition

        # No need to tile at every step
        self.hyp = tf.tile(tf.one_hot(tf.range(FLAGS.num_classes), depth=FLAGS.num_classes),
                           [batch_sz * self.n_policies, 1])  # [B * n_policies * hyp, num_classes]

    def planning_step(self, current_state, z_samples, time, is_training):
        """Perform one planning step.
        Args:
            current state

        Returns:
            Next state
        """
        with tf.name_scope('Planning_loop'):  # loop over policies, parallised into [B * self.n_policies, ...]
            # TODO: define inputs for policyNet. Remember to use tf.stop_gradient!
            # TODO: MIGHT HAVE TO CHANGE new_l_tiled IF POLICIES DEPEND ON CLASS-CONDITIONAL Z
            next_actions, next_actions_mean = self.m['policyNet'].next_actions(time=tf.stop_gradient(time), inputs=None, is_training=is_training)
            # action specific state transition
            # TODO: USE SAME Z AS IN MAIN LOOP: SAMPLES OR (MEAN, STD)?
            new_state = self.stateTransition([z_samples, next_actions], current_state)

            with tf.name_scope('Hypotheses_loop'):  # for every action: loop over hypotheses
                new_c_tiled = tf.tile(new_state['c'], [self.n_policies, 1])  # [B * n_policies, hyp]
                new_s_tiled = tf.reshape(repeat_axis(new_state['s'], axis=0, repeats=self.n_policies),
                                         [self.B * self.num_classes * self.n_policies, self.size_z])  # [B, hyp, z] -> [[n_policies] * B, hyp, z] -> [[n_policies * hyp] * B, z]
                # TODO: THIS MIGHT HAVE TO CHANGE IF POLICIES DEPEND ON CLASS-CONDITIONAL Z
                new_l_tiled = repeat_axis(tf.reshape(new_state['l'], [self.B * self.n_policies, self.loc_dim]),
                                          axis=0, repeats=self.num_classes)  # [B, n_policies, loc] -> [B * n_policies, loc] -> [[B * hyp] * n_policies, loc]

                exp_obs_prior = self.m['decoder'].decode(self.hyp,
                                                         new_s_tiled,
                                                         new_l_tiled)

                # expected entropy
                exp_obs = tf.reshape(exp_obs_prior['sample'], [self.B, self.n_policies, self.num_classes, -1])  # [B, n_policies, hyp, glimpse]
                exp_obs_flat = tf.reshape(exp_obs, [self.B * self.n_policies * self.num_classes, -1])
                H = entropy(probs=exp_obs_flat)  # [B * n_policies * hyp]
                H = tf.reshape(H, [self.B * self.n_policies, self.num_classes])
                exp_H = tf.reduce_sum(new_c_tiled * H, axis=1)  # [B * n_policies]

            # Entropy of the expectation
            exp_exp_obs = tf.einsum('bh,bkhg->bkg', new_state['c'], exp_obs)  # [B, n_policies, glimpse]
            exp_exp_obs_flat = tf.reshape(exp_exp_obs, [self.B * self.n_policies, -1])
            H_exp_exp_obs = entropy(probs=exp_exp_obs_flat)  # [B * n_policies]

            # For all non-decision actions the probability of classifying is 0, hence the probability of an observation is 1
            preference_error = 1. * self.C[time, 0]

            G = H_exp_exp_obs - exp_H
            G = tf.reshape(G, [self.B, self.n_policies]) + preference_error

            # decision actions: G = extrinsic value, as they have no epistemic value
            # TODO: SHOULD THIS DIFFER FOR MULTI-STEP PLANNING? E.G. SHOULD THE STATE BELIEVES INCORPORATE THE FORECASTED OBSERVATIONS?
            believe_correct = tf.reduce_max(new_state['c'], axis=1)
            extr_value = believe_correct * self.C[time, 1] + (1. - believe_correct) * self.C[time, 2]
            G = tf.concat([G, extr_value[:, tf.newaxis]], axis=1)

            # TODO: use pi = softmax(-F - gamma*G) instead?
            pi_logits = G

            # action selection
            # TODO: precision?
            # TODO: which version of action selection?
            # Visual foraging code: a_t ~ softmax(-F - gamma * G)
            # Visual foraging paper: a_t = min_a[ o*_t+1 * [log(o*_t+1) - log(o^a_t+1)] ]
            # Maze code: a_t ~ softmax(gamma * G) [summed over policies with the same next action]
            selected_action_idx = tf.cond(is_training,
                                          lambda: tfd.Categorical(logits=pi_logits, allow_nan_stats=False).sample(),
                                          lambda: tf.argmax(G, axis=1, output_type=tf.int32),
                                          name='sample_action_cond')
            # give back the action itself, not its index. Differentiate between decision and location actions
            dec = tf.equal(selected_action_idx, self.n_policies)  # the last action is the decision
            decision = tf.where(dec, tf.argmax(new_state['c'], axis=1, output_type=tf.int32), tf.fill([self.B], -1))
            selected_action_idx = tf.where(tf.stop_gradient(dec), tf.fill([self.B], 0),
                                           selected_action_idx)  # replace decision indeces (which exceed the shape of selected_action), so we can use gather on the locations

        coords = tf.stack(tf.meshgrid(tf.range(self.B)) + [selected_action_idx], axis=1)
        selected_action      = tf.gather_nd(next_actions, coords)
        selected_action_mean = tf.gather_nd(next_actions_mean, coords)
        selected_exp_obs     = tf.gather_nd(exp_obs, coords)

        records = {'G'                : G,
                   'exp_obs'          : exp_obs,  # [B, n_policies, num_classes, -1]
                   'exp_exp_obs'      : exp_exp_obs,  # [B, n_policies, -1]
                   'H_exp_exp_obs'    : tf.reshape(H_exp_exp_obs, [self.B, self.n_policies]),
                   'exp_H'            : tf.reshape(exp_H, [self.B, self.n_policies]),
                   'potential_actions': next_actions}
        return decision, selected_action, selected_action_mean, selected_exp_obs, records

    @property
    def zero_records(self):
        return {'G'                : tf.zeros([self.B, self.n_policies + 1]),
                'exp_obs'          : tf.zeros([self.B, self.n_policies, self.num_classes, self.m['glimpseEncoder'].encode_size]),
                'exp_exp_obs'      : tf.zeros([self.B, self.n_policies, self.m['glimpseEncoder'].encode_size]),
                'H_exp_exp_obs'    : tf.zeros([self.B, self.n_policies]),
                'exp_H'            : tf.zeros([self.B, self.n_policies]),
                'potential_actions': tf.zeros([self.B, self.n_policies, self.loc_dim])}


class REINFORCEPlanner(Planner):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat)
        self.n_policies = 1
        self.num_glimpses = FLAGS.num_glimpses

    def planning_step(self, current_c, zs_post, z_samples, last_action, time, is_training):
        next_action, next_action_mean = self.m['policyNet'].next_actions(time=time, inputs=[current_c], is_training=is_training)

        if time < (self.num_glimpses - 1):
            decision = tf.fill([self.B], -1)
        else:
            decision = tf.argmax(current_c, axis=1, output_type=tf.int32)

        return decision, next_action, next_action_mean, self.zero_records

    @property
    def zero_records(self):
        return {'G'                : tf.zeros([self.B, self.n_policies + 1]),
                'exp_obs'          : tf.zeros([self.B, self.n_policies, self.num_classes, self.m['glimpseEncoder'].encode_size]),
                'exp_exp_obs'      : tf.zeros([self.B, self.n_policies, self.m['glimpseEncoder'].encode_size]),
                'H_exp_exp_obs'    : tf.zeros([self.B, self.n_policies]),
                'exp_H'            : tf.zeros([self.B, self.n_policies]),
                'potential_actions': tf.zeros([self.B, self.n_policies, self.loc_dim])}
