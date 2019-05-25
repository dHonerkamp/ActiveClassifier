import tensorflow as tf
from tensorflow import distributions as tfd

from modules.planner.base import Base
from tools.tf_tools import repeat_axis, binary_entropy, differential_entropy_normal, differential_entropy_diag_normal


class ActInfPlanner(Base):
    def __init__(self, FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition, C):
        super().__init__(FLAGS, submodules, batch_sz, patch_shape_flat, stateTransition)
        self.max_glimpses = FLAGS.num_glimpses
        self.n_policies = FLAGS.num_classes_kn
        self.num_classes_kn = FLAGS.num_classes_kn
        self.z_dist = FLAGS.z_dist
        self.B = batch_sz
        self.C = C
        self.alpha = FLAGS.precision_alpha

        # No need to tile at every step
        self.hyp = self._hyp_tiling(FLAGS.num_classes_kn, self.B * self.n_policies)  # [B * n_policies * hyp, num_classes]


        self.policy_dep_input =  tf.tile(tf.one_hot(tf.range(FLAGS.num_classes_kn), depth=FLAGS.num_classes_kn)[:, tf.newaxis, :],
                                         [1, batch_sz, 1])  # [hyp, B, num_classes]

    @staticmethod
    def _binary_entropy_agg(logits):
        # TODO: DOES MEAN MAKE SENSE? (at least better than sum, as indifferent to size_z)
        H = binary_entropy(logits=logits)  # [B * n_policies * hyp, z]
        return tf.reduce_mean(H, axis=-1)  # [B * n_policies * hyp]

    def _exp_H(self, exp_obs_mu, exp_obs_sigma, c_believes):
        """
        Args:
            exp_obs_mu, exp_obs_sigma: [B, n_policies, hyp, glimpse]
            c_believes: [B, hyp]
        Returns:
            exp_H: [B, n_policies]
        """
        # TODO: HOW TO DEFINE ENTROPY OVER INDEPENDENT BERNOULLI (WITHOUT EXPLODING COMBINATORICS) OR NORMAL DIST
        if self.z_dist == 'B':
            H = self._binary_entropy_agg(logits=exp_obs_mu)  # [B, n_policies, hyp]
        elif self.z_dist == 'N':
            H = differential_entropy_diag_normal(exp_obs_sigma)  # [B, n_policies, hyp]
        else:
            raise ValueError('Unknown z_dist: {}'.format(self.z_dist))

        new_c_tiled = c_believes[:, tf.newaxis, :]  # [B, hyp] -> [B, 1, hyp] -> broadcasting into [B, n_policies, hyp]
        exp_H = tf.reduce_sum(new_c_tiled * H, axis=2)  # [B, n_policies]
        return exp_H

    def _H_exp_exp(self, exp_obs, c_believes):
        exp_exp_obs = tf.einsum('bh,bkhg->bkg', c_believes, exp_obs)  # [B, n_policies, glimpse]
        if self.z_dist == 'B':
            H_exp_exp_obs = self._binary_entropy_agg(logits=exp_exp_obs)  # [B, n_policies]
        else:
            # TODO: HOW TO DO THIS FOR NORMAL DISTRIBUTED CODE?
            raise ValueError('H_exp_exp not implemented for this distribution: {}'.format(self.z_dist))
        return H_exp_exp_obs, exp_exp_obs

    def calc_G_obs_prePreferences(self, exp_obs, exp_obs_sigma, c_believes):
        """
        Args:
            exp_obs_mu, exp_obs_sigma: [B, n_policies, hyp, glimpse]
            c_believes: [B, hyp]
        Returns:
            G, exp_exp_obs, exp_H, H_exp_exp_obs: [B, n_policies]
        """
        # expected entropy
        exp_H = self._exp_H(exp_obs, exp_obs_sigma, c_believes)
        # Entropy of the expectation
        H_exp_exp_obs, exp_exp_obs = self._H_exp_exp(exp_obs, c_believes)
        G = H_exp_exp_obs - exp_H
        return G, exp_exp_obs, exp_H, H_exp_exp_obs

    def _G_decision(self, time, c_believes):
        # decision actions: will always assign 0 probabiity to any of the non-decsion actions, i.e. those won't impact the entropy
        # TODO: SHOULD THIS DIFFER FOR MULTI-STEP PLANNING? E.G. SHOULD THE STATE BELIEVES INCORPORATE THE FORECASTED OBSERVATIONS?
        # TODO: SHOULD PROBABLY TAKE INTO ACCOUNT THE UK BELIEF
        believe_correct = tf.reduce_max(c_believes, axis=1)
        dec_extr_value = believe_correct * self.C[time, 1] + (1. - believe_correct) * self.C[time, 2]
        dec_H = binary_entropy(probs=believe_correct)  # probs will be zero for anything but the feedback outcomes, i.e. irrelevant for entropy
        dec_G = dec_extr_value - dec_H
        return dec_G

    def planning_step(self, current_state, z_samples, time, is_training):
        """Perform one planning step.
        Args:
            current state

        Returns:
            Next state
        """
        with tf.name_scope('Planning_loop/'):  # loop over policies, parallised into [B * self.n_policies, ...]
            # TODO: define inputs for policyNet (and use the same in reinforce-planner if using it for pre-training)`
            # inputs = [current_state['s'], tf.fill([self.B, 1], tf.cast(time, tf.float32))]
            inputs = [current_state['s']]
            next_actions, next_actions_mean = self.m['policyNet'].next_actions(inputs=inputs,
                                                                               is_training=is_training,
                                                                               n_policies=self.n_policies,
                                                                               policy_dep_input=self.policy_dep_input)
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
                exp_obs = tf.reshape(exp_obs_prior['mu'], [self.B, self.n_policies, self.num_classes_kn, self.m['VAEEncoder'].output_size])  # [B, n_policies, hyp, glimpse]
                if self.z_dist != 'B':
                    exp_obs_sigma = tf.reshape(exp_obs_prior['sigma'], [self.B, self.n_policies, self.num_classes_kn, self.m['VAEEncoder'].output_size])  # [B, n_policies, hyp, glimpse]
                else:
                    exp_obs_sigma = None

            G_obs, exp_exp_obs, exp_H, H_exp_exp_obs = self.calc_G_obs_prePreferences(exp_obs, exp_obs_sigma, new_state['c'])

            # For all non-decision actions the probability of classifying is 0, hence the probability of an observation is 1
            preference_error_obs = 1. * self.C[time, 0]
            G_obs += preference_error_obs

            # decision actions
            G_dec = self._G_decision(time, new_state['c'])
            G = tf.concat([G_obs, G_dec[:, tf.newaxis]], axis=1)

            # G = tf.Print(G, [time, 'H', tf.reduce_mean(exp_H), 'Hexpexp', tf.reduce_mean(H_exp_exp_obs), 'ext', tf.reduce_mean(preference_error), 'G', tf.reduce_mean(G, axis=0)], summarize=30)

            # TODO: use pi = softmax(-F - gamma*G) instead?
            gamma = 1
            pi_logits = tf.nn.log_softmax(gamma * G)
            # Incorporate past into the decision. But what to use for the 2 decision actions?
            # pi_logits = tf.nn.log_softmax(tf.log(new_state['c']) + gamma * G)

            # action selection
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
            best_belief = self._best_believe(new_state)
            dec = tf.equal(selected_action_idx, self.n_policies)  # the last action is the decision

            decision = tf.cond(tf.equal(time, self.max_glimpses - 1),
                               lambda: best_belief,  # always take a decision at the last time step
                               lambda: tf.where(dec, best_belief, tf.fill([self.B], -1)),
                               name='last_timestep_decision_cond')
            selected_action_idx = tf.where(tf.stop_gradient(dec), tf.fill([self.B], 0),
                                           selected_action_idx)  # replace decision indeces (which exceed the shape of selected_action), so we can use gather on the locations

        coords = tf.stack(tf.meshgrid(tf.range(self.B)) + [selected_action_idx], axis=1)
        selected_action      = tf.gather_nd(next_actions, coords)
        selected_action_mean = tf.gather_nd(next_actions_mean, coords)
        selected_exp_obs     = {k: tf.gather_nd(tf.reshape(v, [self.B, self.n_policies, self.num_classes_kn, -1]), coords) if (v is not None) else None
                                for k, v in exp_obs_prior.items()}

        records = {'G'                : G,
                   'exp_obs'          : exp_obs,  # [B, n_policies, num_classes, -1]
                   'exp_exp_obs'      : exp_exp_obs,  # [B, n_policies, -1]
                   'H_exp_exp_obs'    : H_exp_exp_obs,
                   'exp_H'            : exp_H,
                   'potential_actions': next_actions}
        return decision, selected_action, selected_action_mean, selected_exp_obs, records
