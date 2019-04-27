import tensorflow as tf
from tensorflow import distributions as tfd

from tools.tf_tools import repeat_axis, entropy, differential_entropy_normal, differential_entropy_diag_normal


class Planner:
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

    def single_policy_prediction(self, state, z_samples=None, next_actions=None):
        if z_samples is not None:
            state = self.stateTransition([z_samples, next_actions], state)

        hyp = tf.tile(tf.one_hot(tf.range(self.num_classes_kn), depth=self.num_classes_kn), [self.B, 1])
        new_s_tiled = repeat_axis(state['s'], axis=0, repeats=self.num_classes_kn)  # [B, rnn] -> [B * hyp, rnn]
        new_l_tiled = repeat_axis(state['l'], axis=0, repeats=self.num_classes_kn)  # [B, loc] -> [B * hyp, loc]

        exp_obs = self.m['VAEEncoder'].calc_prior(hyp,
                                                  new_s_tiled,
                                                  new_l_tiled)

        return {k: tf.reshape(v, [self.B, self.num_classes_kn, -1]) for k, v in exp_obs.items()}

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


class ActInfPlanner(Planner):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition, C):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)
        self.n_policies = FLAGS.num_classes_kn
        self.num_classes_kn = FLAGS.num_classes_kn
        self.B = batch_sz
        self.C = C
        self.z_dist = FLAGS.z_dist

        # No need to tile at every step
        self.hyp = tf.tile(tf.one_hot(tf.range(FLAGS.num_classes_kn), depth=FLAGS.num_classes_kn),
                           [batch_sz * self.n_policies, 1])  # [B * n_policies * hyp, num_classes]
        self.policy_dep_input =  tf.tile(tf.one_hot(tf.range(FLAGS.num_classes_kn), depth=FLAGS.num_classes_kn)[:, tf.newaxis, :],
                                         [1, batch_sz, 1])  # [hyp, B, num_classes]

    def planning_step(self, current_state, z_samples, time, is_training):
        """Perform one planning step.
        Args:
            current state

        Returns:
            Next state
        """
        with tf.name_scope('Planning_loop/'):  # loop over policies, parallised into [B * self.n_policies, ...]
            # TODO: define inputs for policyNet
            next_actions, next_actions_mean = self.m['policyNet'].next_actions(time=time, inputs=[current_state['c'], current_state['s']],
                                                                               is_training=is_training, n_policies=self.n_policies, policy_dep_input=self.policy_dep_input)
            # action specific state transition
            new_state = self.stateTransition([z_samples, next_actions], current_state)

            with tf.name_scope('Hypotheses_loop/'):  # for every action: loop over hypotheses
                new_s_tiled = repeat_axis(new_state['s'], axis=0, repeats=self.n_policies * self.num_classes_kn)  # [B, rnn] -> [B * n_policies * hyp, rnn]
                # TODO: THIS MIGHT HAVE TO CHANGE IF POLICIES DEPEND ON CLASS-CONDITIONAL Z
                new_l_tiled = repeat_axis(tf.reshape(new_state['l'], [self.B * self.n_policies, self.loc_dim]),
                                          axis=0, repeats=self.num_classes_kn)  # [B, n_policies, loc] -> [B * n_policies, loc] -> [B * n_policies * hyp, loc]

                exp_obs_prior = self.m['VAEEncoder'].calc_prior(self.hyp,
                                                                new_s_tiled,
                                                                new_l_tiled)

                # expected entropy
                # TODO: HOW TO DEFINE ENTROPY??? (FIRST THING: PREDICTED Z ARE DEFINITELY NO PROBS, IF ANYTHING THEY'D BE LOGITS)
                # TODO: COULD USE SIGMA AS MEASURE OF HOW UNCERTAIN THIS FEATURE IS
                exp_obs = tf.reshape(exp_obs_prior['mu'], [self.B, self.n_policies, self.num_classes_kn, self.m['VAEEncoder'].output_size])  # [B, n_policies, hyp, glimpse]
                if self.z_dist == 'B':
                    exp_obs_flat = tf.reshape(exp_obs, [self.B * self.n_policies * self.num_classes_kn, self.m['VAEEncoder'].output_size])
                    H = entropy(logits=exp_obs_flat)  # [B * n_policies * hyp]
                elif self.z_dist == 'N':
                    exp_obs_sigma = tf.reshape(exp_obs_prior['sigma'], [self.B, self.n_policies, self.num_classes_kn, self.m['VAEEncoder'].output_size])  # [B, n_policies, hyp, glimpse]
                    exp_obs_sigma_flat = tf.reshape(exp_obs_sigma, [self.B * self.n_policies * self.num_classes_kn, self.m['VAEEncoder'].output_size])
                    # H = differential_entropy_normal(exp_obs_sigma_flat)  # [B * n_policies * hyp]
                    H = differential_entropy_diag_normal(exp_obs_sigma_flat)  # [B * n_policies * hyp]

                H = tf.reshape(H, [self.B * self.n_policies, self.num_classes_kn])
                new_c_tiled = repeat_axis(new_state['c'], axis=0, repeats=self.n_policies)  # [B, hyp] -> [B * n_policies, hyp]
                exp_H = tf.reduce_sum(new_c_tiled * H, axis=1)  # [B * n_policies]

            # Entropy of the expectation
            exp_exp_obs = tf.einsum('bh,bkhg->bkg', new_state['c'], exp_obs)  # [B, n_policies, glimpse]
            exp_exp_obs_flat = tf.reshape(exp_exp_obs, [self.B * self.n_policies, self.m['VAEEncoder'].output_size])
            H_exp_exp_obs = entropy(logits=exp_exp_obs_flat)  # [B * n_policies]

            # For all non-decision actions the probability of classifying is 0, hence the probability of an observation is 1
            preference_error = 1. * self.C[time, 0]

            G = H_exp_exp_obs - exp_H
            G = tf.reshape(G, [self.B, self.n_policies]) + preference_error

            # decision actions: G = extrinsic value, as they have no epistemic value
            # TODO: SHOULD THIS DIFFER FOR MULTI-STEP PLANNING? E.G. SHOULD THE STATE BELIEVES INCORPORATE THE FORECASTED OBSERVATIONS?
            # TODO: SHOULD PROBABLY TAKE INTO ACCOUNT THE UK BELIEF
            believe_correct = tf.reduce_max(new_state['c'], axis=1)
            dec_extr_value = believe_correct * self.C[time, 1] + (1. - believe_correct) * self.C[time, 2]
            dec_H = entropy(probs=tf.stack([believe_correct, 1 - believe_correct], axis=1))  # probs will be zero for anything but the feedback outcomes, i.e. irrelevant for entropy
            dec_G = dec_extr_value - dec_H
            G = tf.concat([G, dec_G[:, tf.newaxis]], axis=1)

            # G = tf.Print(G, [time, 'H', tf.reduce_mean(exp_H), 'Hexpexp', tf.reduce_mean(H_exp_exp_obs), 'ext', tf.reduce_mean(preference_error), 'G', tf.reduce_mean(G, axis=0)], summarize=30)

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
            best_belief = self._best_believe(new_state)
            dec = tf.equal(selected_action_idx, self.n_policies)  # the last action is the decision
            decision = tf.where(dec, best_belief, tf.fill([self.B], -1))
            selected_action_idx = tf.where(tf.stop_gradient(dec), tf.fill([self.B], 0),
                                           selected_action_idx)  # replace decision indeces (which exceed the shape of selected_action), so we can use gather on the locations

        coords = tf.stack(tf.meshgrid(tf.range(self.B)) + [selected_action_idx], axis=1)
        selected_action      = tf.gather_nd(next_actions, coords)
        selected_action_mean = tf.gather_nd(next_actions_mean, coords)
        selected_exp_obs     = {k: tf.gather_nd(tf.reshape(v, [self.B, self.n_policies, self.num_classes_kn, -1]), coords) for k, v in exp_obs_prior.items()}

        records = {'G'                : G,
                   'exp_obs'          : exp_obs,  # [B, n_policies, num_classes, -1]
                   'exp_exp_obs'      : exp_exp_obs,  # [B, n_policies, -1]
                   'H_exp_exp_obs'    : tf.reshape(H_exp_exp_obs, [self.B, self.n_policies]),
                   'exp_H'            : tf.reshape(exp_H, [self.B, self.n_policies]),
                   'potential_actions': next_actions}
        return decision, selected_action, selected_action_mean, selected_exp_obs, records


class REINFORCEPlanner(Planner):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition, is_pre_phase):
        self.n_policies = 1
        self.num_glimpses = FLAGS.num_glimpses
        self._is_pre_phase = is_pre_phase
        # SEEMS TO LEARN TO CHEAT IF DOING THIS
        # if use_true_labels:
        #     self.policy_dep_input = tf.one_hot(y_MC, depth=FLAGS.num_classes)[tf.newaxis, :, :]  # first dimension is n_policies
        # else:
        #     self.policy_dep_input = None

        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)

    def planning_step(self, current_state, z_samples, time, is_training):
        # TODO: CHECK EFFECT ON PERFORMANCE FROM INCLUDING THIS OR NOT (WHETHER INCLUDING IT MAKES THE MODEL FOCUS LESS ON LEARNING RECONSTRUCTIONS)
        if self._is_pre_phase:
            # TODO: ADJUST FOR UK CLASSES
            policy_dep_input = tf.one_hot(tf.argmax(current_state['c'], axis=1),
                                          depth=self.num_classes_kn)[tf.newaxis, :, :]
        else:
            policy_dep_input = None

        next_action, next_action_mean = self.m['policyNet'].next_actions(time=time,
                                                                         inputs=[current_state['c'], current_state['s']],
                                                                         is_training=is_training,
                                                                         n_policies=self.n_policies,
                                                                         policy_dep_input=policy_dep_input)
        next_exp_obs = self.single_policy_prediction(current_state, z_samples, next_action)

        if time < (self.num_glimpses - 1):
            decision = tf.fill([self.B], -1)
        else:
            decision = self._best_believe(current_state)

        return decision, next_action, next_action_mean, next_exp_obs, self.zero_records
